import torch
import torch.nn as nn
import triton
from kernels import rmsnorm_fw_kernel

class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        orig_shape = x.shape
        x_2d = x.view(-1, orig_shape[-1])
        M, N = x_2d.shape #handle input tensors

        y = torch.empty_like(x_2d) # output tensor allocation
        rstd = torch.empty((M,),dtype=torch.float32, device=x.device) # allocate tensor to save rstd

        def grid(meta):
            return (M,)

        rmsnorm_fw_kernel[grid](
            x_2d, y, weight, rstd,
            x_2d.stride(0), N, eps,
        )

        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return y.view(*orig_shape)

        @staticmethod
        def backward(ctx, dy):
            raise NotImplementedError("backward pass is not implemented yet")

class FusedRMSNorm(nn.Module):
    # Fused RMS Norm
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FusedRMSNormFunction.apply(x, self.weight, self.eps)


# profiler and baseline for existing PyTorch RMSNorm

class NaiveRMSNorm(nn.Module):
    # standard eager PyTorch RMSNorm
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight=nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True) # MS calc
        inv_std = torch.rsqrt(variance + self.eps) # Reciprocal square root with epsilon
        return self.weight * (x*inv_std) # normalize and scale by weight

class TorchCompileRMSNorm(nn.Module):
    # Baseline using torch.compile
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self._norm = NaiveRMSNorm(dim, eps)

        self.compiled_norm = torch.compile(self._norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.compiled_norm(x)
