import torch,triton
from nf4_row_scale_candidate_v44 import validate_table
from word4_decode_kernel import _mm
def launch(a, p, table, ends, rows, transposed, geometry, *, write_once=False):
    if transposed is not True or p.shape[1]%4 or p.blocksize%4 or p.data.data_ptr()%2 or p.data.storage_offset()%2 or not p.data.is_contiguous() or type(write_once) is not bool:
        raise ValueError('Paired-byte forward requires K/block multiple4 and aligned contiguous packed bytes and explicit write mode')
    if geometry != dict(bn=64,bk=32,warps=4,stages=1,waves=32):
        raise ValueError('Exact accepted geometry required')
    scales = validate_table(p, table)
    if scales.dtype!=torch.bfloat16:raise ValueError('Accepted BF16 scale table required')
    n, k = p.shape if transposed else p.shape[::-1]
    if not (a.is_cuda and a.dtype == torch.bfloat16 and a.is_contiguous()
            and a.ndim == 2 and a.shape[1] == k and a.shape[0] >= rows > 0
            and rows % 32 == 0 and ends.is_contiguous()
            and ends.dtype == torch.int32 and ends.numel() == p.experts
            and ends.device == a.device == p.data.device):
        raise ValueError('Require static row32 BF16 GPU inputs')
    if (geometry['bn'] not in (64, 128) or geometry['bk'] != 32
            or geometry['warps'] != 4 or geometry['stages'] != 1
            or geometry['waves'] not in (2, 4, 8, 16, 32)):
        raise ValueError('Only prepared two-row BK32 geometry')
    bn, bk = geometry['bn'], geometry['bk']
    ctas = min(triton.cdiv(rows, 64) * triton.cdiv(n, bn),
               torch.cuda.get_device_properties(a.device).multi_processor_count * geometry['waves'])
    y = (a.new_empty if write_once else a.new_zeros)(a.shape[0], n)
    _mm[(ctas,)](a, p.data, scales, p.code, ends, y, n, k, p.experts, rows,
                p.shape[0] * p.shape[1], transposed, p.blocksize,
                p.scale_state.blocksize, bn, bk, ctas, a.shape[0], write_once,
                num_warps=geometry['warps'], num_stages=geometry['stages'],
                enable_fp_fusion=False)
    return y
