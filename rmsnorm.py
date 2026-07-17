import torch
import torch.nn as nn

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
