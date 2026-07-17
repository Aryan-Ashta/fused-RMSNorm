import pytest
import torch
from rmsnorm import NaiveRMSNorm, FusedRMSNorm

class NaiveRMSNormFunction(torch.autograd.Function): # new custom baseline
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])

        mean_sq = torch.mean(x_2d ** 2, dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_sq + eps)
        x_hat = x_2d * rstd
        y = x_hat * weight

        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.orig_shape = orig_shape
        return y.view(*orig_shape)

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rstd = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        M, N = x_2d.shape

        dy_2d = dy.reshape(-1, N)

        dw = (dy_2d * (x_2d * rstd)).sum(dim=0)

        sum_dy_w_x = (dy_2d * weight * x_2d).sum(dim=-1, keepdim=True)
        term1 = dy_2d * weight * rstd
        term2 = x_2d * (rstd ** 3) * sum_dy_w_x / N
        dx = term1 - term2

        return dx.view(*orig_shape), dw, None


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


@pytest.mark.parametrize("shape", [(4, 128), (8, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_rmsnorm_backward_correctness(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    dim = shape[-1]

    # Set requires_grad=True to track gradients
    x_naive = torch.randn(shape, device='cuda', dtype=dtype, requires_grad=True)
    w_naive = torch.randn((dim,), device='cuda', dtype=dtype, requires_grad=True)

    x_fused = x_naive.detach().clone().requires_grad_(True)
    w_fused = w_naive.detach().clone().requires_grad_(True)

    # Forward pass
    res_naive = NaiveRMSNormFunction.apply(x_naive, w_naive)
    res_fused = FusedRMSNorm.apply(x_fused, w_fused)

    dy = torch.randn_like(res_naive)

    # Backward pass
    res_naive.backward(dy)
    res_fused.backward(dy)

    atol = 1e-4 if dtype == torch.float32 else 1e-2
    rtol = 1e-4 if dtype == torch.float32 else 1e-2

    torch.testing.assert_close(x_fused.grad, x_naive.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(w_fused.grad, w_naive.grad, atol=atol, rtol=rtol)
