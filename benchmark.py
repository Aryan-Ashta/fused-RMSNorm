import torch
import triton
import triton.testing as tt

from rmsnorm import FusedRMSNorm, NaiveRMSNorm

# performance tracking configurations
@tt.perf_report([
    tt.Benchmark(
        x_names=['N'],                          # Sweep across hidden dimension sizes (N)
        x_vals=[256 * i for i in range(1, 17)], # From 256 to 4096
        line_arg='provider',                    # The variable changing between lines
        line_vals=['naive', 'compile', 'triton'],
        line_names=['PyTorch Eager', 'Torch Compile', 'Fused Triton'],
        styles=[('blue', '-'), ('orange', '-'), ('green', '-')],
        ylabel='GB/s',                         # Focus on Memory Bandwidth
        plot_name='rmsnorm-forward-bandwidth',   # Name for the saved chart image
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'forward'} # Default settings
    ),
    tt.Benchmark(
        x_names=['N'],
        x_vals=[256 * i for i in range(1, 17)],
        line_arg='provider',
        line_vals=['naive', 'compile', 'triton'],
        line_names=['PyTorch Eager', 'Torch Compile', 'Fused Triton'],
        styles=[('blue', '--'), ('orange', '--'), ('green', '--')],
        ylabel='GB/s',
        plot_name='rmsnorm-backward-bandwidth',
        args={'M': 4096, 'dtype': torch.float16, 'mode': 'backward'} # Separate configuration for backward
    )
])
def benchmark(M, N, dtype, provider, mode):
    # 1. initialize data tensors
    x = torch.randn((M, N), device='cuda', dtype=dtype, requires_grad=True)
    weight = torch.ones((N,), device='cuda', dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)

    # 2. instantiate implementations
    naive_norm = NaiveRMSNorm(N).to('cuda', dtype=dtype)
    fused_norm = FusedRMSNorm(N).to('cuda', dtype=dtype)
    compiled_norm = torch.compile(naive_norm)

    # 3. choose target function
    if provider == 'naive':
        module = naive_norm
    elif provider == 'compile':
        module = compiled_norm
    elif provider == 'triton':
        module = fused_norm

    if mode == 'forward':
        fn = lambda: module(x)
    else:
        def fn_bw():
            y = module(x)
            y.backward(dy, retain_graph=True)
            if x.grad is not None:
                x.grad.zero_()
            if module.weight.grad is not None:
                module.weight.grad.zero_()
        fn = fn_bw

    # 4. calc total memory footprint transferred to/from HBM
    element_size = x.element_size()
    if mode == 'forward':
        # Read x (M*N) + Write y (M*N)
        gb = (2 * M * N * element_size) / 1e9
    else:
        # Read dy (M*N) + Read x (M*N) + Read weight (N) + Write dx (M*N) + Write dw (N)
        gb = (3 * M * N * element_size) / 1e9

    # 5. measure time in ms
    ms = tt.do_bench(fn)

    return gb / (ms / 1000.0)

if __name__ == '__main__':
    print("Running integrated RMSNorm Profiling Sweep...")
    benchmark.run(show_plots=True, save_path='.')
