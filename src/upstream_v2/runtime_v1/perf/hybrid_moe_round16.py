"""Use online NF4 GEMM only for measured normal B2; exact preallocation elsewhere."""
import buffers_moe_round9 as buffers
import nf4_moe_round13b as nf4
import nf4_grouped_round13b as kernel
PREALLOC_ONLY=False

def forward(self,hidden_states):
    batch,length=hidden_states.shape[:2]
    n=batch*length
    if not PREALLOC_ONLY and batch==2 and length<=1024 and 1024<n<=2048:
        kernel.BM=32;kernel.BN=64;kernel.BK=32;kernel.WARPS=4
        nf4.PAD=32;nf4.FP32_OUTPUT=False
        return nf4.forward(self,hidden_states)
    buffers.VARIANT=1
    return buffers.forward(self,hidden_states)
