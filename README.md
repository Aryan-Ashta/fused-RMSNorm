# Fused RMSNorm GPU Kernel (Triton)

A Triton implementation of the Root Mean Square Normalization (RMSNorm) forward and backward passes. This project demonstrates low-level GPU programming, memory-coalescing, tiling strategies, and kernel fusion to eliminate High Bandwidth Memory (HBM) round-trips compared to naive PyTorch eager execution.

The forward pass achieves **relative performance parity** with `torch.compile` (TorchInductor), peaking at **~240 GB/s** of effective memory bandwidth on a NVIDIA T4 GPU running through Google Colab (est. 320 GB/s bandwidth). The backward pass achieves a **2.7x speedup** over PyTorch Eager, but does not beat `torch.compile` due to redundancy in my backwards pass implementation.

---

## Performance Summary

### Forward Pass
* **PyTorch Eager:** ~62 GB/s (bottlenecked by 3-5 distinct sequential kernel launches and intermediate HBM reads/writes).
* **Torch Compile / Custom Triton:** **~240 GB/s** (bound by physical hardware memory bandwidth limits; 3.8x faster than Eager).

### Backward Pass
* **PyTorch Eager:** ~20 GB/s.
* **Custom Triton:** **~54 GB/s** (2.7x speedup over Eager).
* **Torch Compile:** ~65–75 GB/s.

---

## Core Architecture & Fusion Strategy

RMSNorm is fundamentally a memory-bound operation. The mathematical formulation is:

$$y = \frac{x}{\sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2 + \epsilon}} \odot \gamma$$

In a naive PyTorch eager execution model, this operation launches separate kernels for squaring elements, calculating the row-wise mean, adding epsilon, computing the reciprocal square root, multiplying by the input, and scaling by the weight vector $\gamma$. 

### Optimization Techniques Implemented:
* **Fused Execution:** Combines the entire mathematical sequence into a single kernel launch per row, forcing all intermediate states to live inside fast On-Chip SRAM registers instead of HBM.
* **Persistent Reductions:** Exploits thread-block level caching for the variance calculation, ensuring data is loaded exactly once per row.
* **FP32 Accumulation:** Loads and stores tensors in `float16` or `bfloat16` while executing algebraic accumulations in `float32` to preserve numerical stability and combat rounding drift.
* **Autotuning:** Integrated `triton.autotune` to dynamically sweep optimal configurations for `BLOCK_SIZE` based on the input matrix dimensions.

---

## Architectural Deep Dive: The Backward Pass Bottleneck

While the forward pass achieves parity with PyTorch's automated compiler, the manual backward pass exhibits a ~30% performance delta compared to `torch.compile`. 

### Why `torch.compile` Wins on Backward
My custom Triton kernel handles the weight gradient by executing a direct `tl.atomic_add` on global memory pointers from within each parallel block. At high thread counts, this triggers severe **global atomic lock contention** in HBM, stalling execution blocks while they wait to write to the shared $N$-dimensional vector.

TorchInductor (`torch.compile`) circumvents this bottleneck by automatically split-compiling the backward pass. It caches block-level partial gradients into temporary structural workspace buffers, then dispatches a highly vectorized, separate reduction kernel to finalize $d\gamma$, eliminating thread serialization.

---

## Repository Structure

```text
├── data/
│   └── rmsnorm-backward-bandwidth.csv
│   └── rmsnorm-backward-bandwidth.png
│   └── rmsnorm-forward-bandwidth.csv
│   └── rmsnorm-forward-bandwidth.png
├── benchmark.py
├── kernels.py         # Core Triton forward & backward kernel implementations
├── rmsnorm.py
├── test_rmsnorm.py
├── README.md
└── requirements.txt
```
---

## Getting Started
### Prerequisites
- NVIDIA GPU (Ampere or newer recommended)
- CUDA Toolkit installedInstallation
```bash
pip install -r requirements.txt
```
### Running Tests
Validates the custom Triton kernel outputs against the PyTorch reference implementation across float32, float16, and bfloat16 with rigid numeric tolerances:
```bash
pytest test_rmsnorm.py
```
### Running the Profiling Benchmark
Sweeps hidden dimensions from $N = 256$ to $N = 4096$, calculating the exact effective memory bandwidth achieved:
```bash
python benchmark.py
```
