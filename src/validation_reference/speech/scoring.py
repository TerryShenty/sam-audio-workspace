from functools import lru_cache
from pathlib import Path
import numpy as np
import soundfile as sf
from media_utils import decode_audio_tensor
import gt_metric_protocol as metrics
KEYS=['si_sdr_db','aligned_si_sdr_db','si_sdri_db','activity_primary_f1']

@lru_cache(maxsize=300)
def mono(path):
    try:
        x,sr=sf.read(path,always_2d=True,dtype='float64')
        x=metrics.resample_mono(x,sr)
    except sf.LibsndfileError:
        x=decode_audio_tensor(Path(path),metrics.TARGET_SR).mean(0).numpy().astype(np.float64)
    assert np.isfinite(x).all()
    return x

@lru_cache(maxsize=150)
def reference_inputs(gt_path,mix_path):
    ref=mono(gt_path);mix=mono(mix_path);original_gt_samples=len(ref)
    ref=metrics.fit_length(ref,len(mix))
    am,_,_=metrics.align_small_lag(ref,mix)
    return ref,mix,original_gt_samples,metrics.standard_si_sdr(ref,am)

def score(row,pred):
    ref,mix,original_gt_samples,msi=reference_inputs(row['gt_audio_path'],row['mixture_audio_path'])
    assert np.isfinite(pred).all()
    assert abs(len(pred)-len(mix))<=4800,(row['row_index'],len(pred),len(mix))
    pred=metrics.fit_length(pred.astype(np.float64),len(ref));mix=metrics.fit_length(mix,len(ref))
    aligned,lag,corr=metrics.align_small_lag(ref,pred)
    silent=metrics.db(metrics.rms(pred))<metrics.SILENT_ABSOLUTE_DBFS or metrics.db(metrics.rms(pred)/max(metrics.rms(ref),metrics.EPS))<metrics.SILENT_RELATIVE_TO_GT_DB
    result={k:row[k] for k in ['row_index','clip_id','dataset','task_type','domain','scene_difficulty','referring_difficulty']}
    result.update(si_sdr_db=metrics.standard_si_sdr(ref,pred),aligned_si_sdr_db=metrics.standard_si_sdr(ref,aligned),mixture_aligned_si_sdr_db=msi,alignment_lag_ms=lag*1000/metrics.TARGET_SR,alignment_corr=corr,silent_failure=bool(silent),original_gt_samples=original_gt_samples,evaluation_samples=len(ref))
    result['si_sdri_db']=result['aligned_si_sdr_db']-msi
    result.update(metrics.activity_metrics(ref,aligned,'true_gt' if row['dataset'] in metrics.TRUE_GT_DATASETS else 'near_gt',bool(silent)))
    assert all(np.isfinite(result[k]) for k in KEYS)
    return result

