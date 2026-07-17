# test_rmsnorm.py
import pytest
import torch
from rmsnorm import NaiveRMSNorm, FusedRMSNorm

@pytest.mark.parametrize("shape", [(4, 512), (16, 2048, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)

    dim = shape[-1]

    x = torch.randn(shape, device='cuda', dtype=dtype)

    naive_norm = NaiveRMSNorm(dim).cuda().to(dtype)
    fused_norm = FusedRMSNorm(dim).cuda().to(dtype)

    with torch.no_grad():
        fused_norm.weight.copy_(naive_norm.weight)

    res_naive = naive_norm(x)
    res_fused = fused_norm(x)

    atol = 1e-5 if dtype == torch.float32 else 1e-2
    rtol = 1e-5 if dtype == torch.float32 else 1e-2

    torch.testing.assert_close(res_fused, res_naive, atol=atol, rtol=rtol)
