"""Budgeted frozen metadata tables below memo; no activation/full-weight cache."""
import bytepair_launch as bp_kernel
import word4_launch as word_kernel
import word8_launch as eight_kernel
import copy,time
import torch
from row_grouped_dispatch_v41 import RowReuse as Previous
from bp_identity import BYTEPAIR,WORD4,WORD8,CODEBOOK,SCALE,FUSION,normalize,geometry,fused_scale_sides
import nf4_forward_const_codebook_v60 as codebook_kernel
import nf4_row_scale_candidate_v44 as kernel
prepare_scales=kernel.prepare_scales
launch_scale=kernel.launch
from nf4_scale_fused64_v46 import launch as launch_scale_fused

class RowReuse(Previous):
    def __init__(self,module,choose,blocks,memo):
        super().__init__(module,choose,blocks,memo)
        previous=self.installed
        self.tables={};self.scale_policy=None;self.scale_counts=dict.fromkeys(self.counts,0)
        self.prepare_calls=0;self.prepare_seconds=0.;self.prepared_bytes_total=0;self.scale_bytes=0;self.scale_peak_bytes=0
        self.scale_fused_counts=dict.fromkeys(self.counts,0)
        self.bp_sources={};self.bp_source_ids={};self.bp_parent=None;self.bp_tables=None;self.bp_roles=None;self.bp_enabled=False
        self.bp_counts={r+'_forward':0 for r in ('gate_proj','up_proj','down_proj')}
        self.codebooks={};self.codebook_bindings={};self.codebook_enabled=False
        self.codebook_counts=dict.fromkeys(self.counts,0);self.codebook_prepare_calls=0
        self._codebook_parents=(self.codebooks,self.codebook_bindings)
        self.word_sources={};self.word_source_ids={};self.word_enabled=False
        self.word_counts={r+"_forward":0 for r in ("gate_proj","up_proj","down_proj")}
        self.eight_sources={};self.eight_source_ids={};self.eight_enabled=False
        self.eight_counts={r+"_forward":0 for r in ("gate_proj","up_proj","down_proj")}
        self.configure(choose())
        def run(a,p,ends,rows,transposed):
            cfg=self.choose();policy=cfg.get(SCALE);side='forward' if transposed else 'backward'
            item=self.roles.get(id(p))
            if not policy or item is None or item[1] not in policy['roles'] or policy['direction'] not in ('both',side):
                return previous(a,p,ends,rows,transposed)
            if item[0] is not p or policy!=self.scale_policy or id(p) not in self.tables:raise ValueError('Scale cache not configured for this source')
            key=item[1]+'_'+side;g=geometry(cfg,item[1],transposed)
            if cfg.get(FUSION,{}).get(item[1],{}).get(side+'_kernel'):
                if key not in self.device_branch_counts:self.device_branch_counts[key]=torch.zeros(2,device=a.device,dtype=torch.int64)
                out=launch_scale_fused(a,p,self.tables[id(p)],ends,rows,transposed,g,write_once=cfg.get('nf4_static_write_once',False),branch_counts=self.device_branch_counts[key])
                self.fusion_counts[key]+=1;self.scale_fused_counts[key]+=1
            elif transposed and cfg.get(BYTEPAIR,False):
                table=self.bound_bp_source(p)
                fixed={k:g[k] for k in ('bn','bk','warps','stages','waves')}
                if cfg.get(WORD4,False):
                    self.bound_word_source(p)
                    if cfg.get(WORD8,False):
                        self.bound_eight_source(p)
                        out=eight_kernel.launch(a,p,table,ends,rows,transposed,fixed,write_once=cfg.get('nf4_static_write_once',False))
                        self.eight_counts[key]+=1
                    else:
                        out=word_kernel.launch(a,p,table,ends,rows,transposed,fixed,write_once=cfg.get('nf4_static_write_once',False))
                    self.word_counts[key]+=1
                else:
                    out=bp_kernel.launch(a,p,table,ends,rows,transposed,fixed,write_once=cfg.get('nf4_static_write_once',False))
                self.bp_counts[key]+=1
            elif transposed and cfg.get(CODEBOOK,False):
                book=self.bound_codebook(p)
                out=codebook_kernel.launch(a,p,self.tables[id(p)],ends,rows,transposed,g,codebook=book,write_once=cfg.get('nf4_static_write_once',False))
                self.codebook_counts[key]+=1
            else:
                out=launch_scale(a,p,self.tables[id(p)],ends,rows,transposed,g,write_once=cfg.get('nf4_static_write_once',False))
            self.scale_counts[key]+=1;self.counts[key]+=1
            if side+'_kernel' in cfg.get('nf4_row_reuse_waves',{}).get(item[1],{}):self.wave_counts[key]+=1
            return out
        self.installed=run;module.grouped_mm=run

    def _configure_scales(self,config):
        self.boundary();policy=normalize(config).get(SCALE)
        if policy==self.scale_policy:
            for key,table in self.tables.items():kernel.validate_table(self.roles[key][0],table)
            return
        self.tables.clear();self.scale_bytes=0;self.scale_policy=None
        if not policy:return
        selected=[(key,p) for key,(p,role) in self.roles.items() if role in policy['roles']]
        dtype=torch.float32 if policy['precision']=='fp32' else torch.bfloat16
        needed=sum(p.experts*p.shape[0]*p.shape[1]//p.blocksize*(4 if dtype==torch.float32 else 2) for _,p in selected)
        assert needed<=policy['max_bytes'],'SCALE_TABLE_BUDGET_EXCEEDED'
        device=selected[0][1].data.device
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter()
        try:
            for key,p in selected:
                table=prepare_scales(p,dtype);kernel.validate_table(p,table)
                self.tables[key]=table;self.scale_bytes+=table.nbytes;self.prepare_calls+=1
                assert self.scale_bytes<=policy['max_bytes'],'SCALE_TABLE_BUDGET_EXCEEDED'
            if device.type=='cuda':torch.cuda.synchronize(device)
            assert self.scale_bytes==needed,'SCALE_BYTES_MISMATCH'
        except BaseException:
            self.tables.clear();self.scale_bytes=0;raise
        finally:self.prepare_seconds+=time.perf_counter()-started
        self.prepared_bytes_total+=needed;self.scale_peak_bytes=max(self.scale_peak_bytes,needed);self.scale_policy=copy.deepcopy(policy)

    def scale_statistics(self):
        return dict(policy=copy.deepcopy(self.scale_policy),tables=len(self.tables),live_metadata_bytes=self.scale_bytes,
            peak_metadata_bytes=self.scale_peak_bytes,prepare_calls=self.prepare_calls,prepare_seconds=self.prepare_seconds,
            prepared_bytes_total=self.prepared_bytes_total,activation_cache=False,decoded_full_weight_cache=False,
            frozen_metadata_across_updates=True,live_graphs=self.live_graphs)

    def record_scale_window(self,row,before):
        self.boundary();cfg=row['config'];policy=cfg.get(SCALE)
        row['scale_cache']=self.scale_statistics()
        row['row_scale_calls']={k:v-before[k] for k,v in self.scale_counts.items()}
        if policy:
            assert 0<self.scale_bytes<=policy['max_bytes'],'SCALE_TABLE_NOT_LIVE'
            for role in policy['roles']:
                for side in ('forward','backward'):
                    if policy['direction'] in ('both',side):assert row['row_scale_calls'][role+'_'+side]>0,'ROW_SCALE_NOT_EXECUTED'
        else:assert self.scale_bytes==0 and sum(row['row_scale_calls'].values())==0,'SCALE_DISABLED_NOT_CLEARED'

    def record_scale_fused_window(self,row,before):
        selected=fused_scale_sides(row['config'])
        row['row_scale_fused_calls']={k:v-before[k] for k,v in self.scale_fused_counts.items()}
        for key,count in row['row_scale_fused_calls'].items():
            if key in selected:
                assert count>0 and row['row_scale_calls'][key]==count and row['row_fusion_calls'][key]==count,'SCALE_FUSED_NOT_DISPATCHED'
                assert row['row_fusion_device_tiles'][key]['fused_output_tiles']>0,'SCALE_FUSED_DEVICE_BRANCH_NOT_EXECUTED'
            else:assert count==0,'SCALE_FUSED_UNSELECTED_DIRECTION'


    def bound_codebook(self,p):
        if (not self.codebook_enabled or self.codebooks is not self._codebook_parents[0]
                or self.codebook_bindings is not self._codebook_parents[1]):
            raise RuntimeError('Codebook parent changed or disabled; rebuild at boundary')
        book=self.codebooks.get(id(p));bound=self.codebook_bindings.get(id(p))
        if (bound is None or book is not bound[0] or book.values is not bound[1]
                or book.source is not bound[2]):
            raise RuntimeError('Codebook selected payload changed; rebuild at boundary')
        codebook_kernel.validate_codebook(p,book)
        return book

    def configure_bp_sources(self,c):
        self.boundary()
        self.bp_enabled=bool(c.get(BYTEPAIR,False))
        self.bp_sources={}
        self.bp_source_ids={}
        self.bp_parent=self.bp_sources;self.bp_ids_parent=self.bp_source_ids
        self.bp_tables=self.tables
        self.bp_roles=self.roles
        if self.bp_enabled:
            for key,(p,role) in self.roles.items():
                table=self.tables[key]
                kernel.validate_table(p,table)
                item=(p,table,table.values,table.source,table.payload,self.roles[key])
                self.bp_sources[key]=item
                self.bp_source_ids[key]=id(item)

    def bound_bp_source(self,p):
        if (not self.bp_enabled or self.bp_sources is not self.bp_parent
                or self.tables is not self.bp_tables or self.roles is not self.bp_roles
                or self.bp_source_ids is not self.bp_ids_parent):
            raise RuntimeError('Paired-byte source parent changed')
        item=self.bp_sources.get(id(p));table=self.tables.get(id(p))
        if (item is None or id(item)!=self.bp_source_ids.get(id(p)) or item[0] is not p
                or table is not item[1] or table.values is not item[2]
                or table.source is not item[3] or table.payload is not item[4]
                or self.roles.get(id(p)) is not item[5]):
            raise RuntimeError('Paired-byte selected source replaced')
        kernel.validate_table(p,table)
        return table

    def configure_word_sources(self,c):
        self.boundary();self.word_enabled=bool(c.get(WORD4,False))
        self.word_sources={};self.word_source_ids={}
        self.word_parents=(self.word_sources,self.word_source_ids,self.roles)
        if self.word_enabled:
            for key,(p,role) in self.roles.items():
                self.bound_bp_source(p)
                item=(p,p.scale_state,kernel.source_identity(p),self.roles[key])
                self.word_sources[key]=item;self.word_source_ids[key]=id(item)

    def bound_word_source(self,p):
        if (not self.word_enabled or self.word_sources is not self.word_parents[0]
                or self.word_source_ids is not self.word_parents[1] or self.roles is not self.word_parents[2]):
            raise RuntimeError('Backward source parent changed or disabled')
        item=self.word_sources.get(id(p))
        if (item is None or id(item)!=self.word_source_ids.get(id(p)) or item[0] is not p
                or p.scale_state is not item[1] or self.roles.get(id(p)) is not item[3]
                or kernel.source_identity(p)!=item[2]):
            raise RuntimeError('Backward packed source or version changed')
        self.bound_bp_source(p)
        return item

    def configure_eight_sources(self,c):
        self.boundary();self.eight_enabled=bool(c.get(WORD8,False))
        self.eight_sources={};self.eight_source_ids={}
        self.eight_parents=(self.eight_sources,self.eight_source_ids,self.roles)
        if self.eight_enabled:
            for key,(p,role) in self.roles.items():
                self.bound_word_source(p)
                item=(p,p.scale_state,kernel.source_identity(p),self.roles[key])
                self.eight_sources[key]=item;self.eight_source_ids[key]=id(item)

    def bound_eight_source(self,p):
        if (not self.eight_enabled or self.eight_sources is not self.eight_parents[0]
                or self.eight_source_ids is not self.eight_parents[1] or self.roles is not self.eight_parents[2]):
            raise RuntimeError('Backward source parent changed or disabled')
        item=self.eight_sources.get(id(p))
        if (item is None or id(item)!=self.eight_source_ids.get(id(p)) or item[0] is not p
                or p.scale_state is not item[1] or self.roles.get(id(p)) is not item[3]
                or kernel.source_identity(p)!=item[2]):
            raise RuntimeError('Backward packed source or version changed')
        self.bound_word_source(p)
        return item

    def configure(self,config):
        cfg=normalize(config);self.boundary()
        self._configure_scales(cfg)
        self.configure_bp_sources(cfg)
        self.configure_word_sources(cfg)
        self.configure_eight_sources(cfg)
        enabled=cfg.get(CODEBOOK,False)
        if enabled and self.codebook_enabled:
            for p,role in self.roles.values():self.bound_codebook(p)
            return
        self.codebooks.clear();self.codebook_bindings.clear();self.codebook_enabled=False
        if not enabled:return
        for key,(p,role) in self.roles.items():
            book=codebook_kernel.prepare_codebook(p)
            self.codebooks[key]=book
            self.codebook_bindings[key]=(book,book.values,book.source)
            self.codebook_prepare_calls+=1
        self.codebook_enabled=True

    def record_codebook_window(self,row,before):
        self.boundary()
        counts={k:v-before[k] for k,v in self.codebook_counts.items()}
        enabled=row['config'].get(CODEBOOK,False)
        for key,count in counts.items():
            if enabled and key.endswith('_forward'):assert count>0,'CONST_CODEBOOK_NOT_DISPATCHED'
            else:assert count==0,'CONST_CODEBOOK_UNSELECTED_DIRECTION'
        row['constant_codebook']=dict(enabled=enabled,calls=counts,books=len(self.codebooks),
            logical_fp32_bytes=len(self.codebooks)*16*4,host_values=True,
            values_read_only_at_boundary=True,prepare_calls_cumulative=self.codebook_prepare_calls,
            logical_bytes_are_not_HBM_or_time=True)

    def close(self):
        super().close();self.eight_sources.clear();self.eight_source_ids.clear();self.eight_enabled=False;self.word_sources.clear();self.word_source_ids.clear();self.word_enabled=False;self.bp_sources.clear();self.bp_source_ids.clear();self.bp_enabled=False;self.tables.clear();self.scale_bytes=0;self.scale_policy=None
        self.codebooks.clear();self.codebook_bindings.clear();self.codebook_enabled=False
