import torch
import triton
import matplotlib.pyplot as plt
from rmsnorm import NaiveRMSNorm, TorchCompileRMSNorm, FusedRMSNorm

plt.style.use('seaborn-v0_8-whitegrid')

def run_benchmark():
    M = 2048
    configs = [512, 1024, 2048, 4096, 8192]
    dtype = torch.float16
    device = "cuda"

    naive_times = []
    compile_times = []
    triton_times = []

    naive_bws = []
    compile_bws = []
    triton_bws = []

    print(f"Benchmarking RMSNorm (M={M}, dtype={dtype})")
    print(f"{'N':<10} | {'Naive (ms)':<12} | {'Compile (ms)':<12} | {'Triton (ms)':<12}")
    print("-" * 60)

    for N in configs:
        x = torch.randn((M, N), device=device, dtype=dtype)

        naive_mod = NaiveRMSNorm(N).to(device).to(dtype)
        compile_mod = TorchCompileRMSNorm(N).to(device).to(dtype)
        fused_mod = FusedRMSNorm(N).to(device).to(dtype)

        # warm up compilation + autotuner
        for _ in range(5):
            _ = naive_mod(x)
            _ = compile_mod(x)
            _ = fused_mod(x)

        # measure execution time in ms
        ms_naive = triton.testing.do_bench(lambda: naive_mod(x))
        ms_compile = triton.testing.do_bench(lambda: compile_mod(x))
        ms_triton = triton.testing.do_bench(lambda: fused_mod(x))

        # calc memory bandwidth
        total_bytes = (2 * M * N + N) * x.element_size()

        bw_naive = total_bytes / (ms_naive * 1e6)
        bw_compile = total_bytes / (ms_compile * 1e6)
        bw_triton = total_bytes / (ms_triton * 1e6)

        # store results
        naive_times.append(ms_naive)
        compile_times.append(ms_compile)
        triton_times.append(ms_triton)

        naive_bws.append(bw_naive)
        compile_bws.append(bw_compile)
        triton_bws.append(bw_triton)

        print(f"{N:<10} | {ms_naive:<12.4f} | {ms_compile:<12.4f} | {ms_triton:<12.4f}")

    # latency plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(configs, naive_times, marker='o', label='PyTorch Eager')
    plt.plot(configs, compile_times, marker='s', label='torch.compile')
    plt.plot(configs, triton_times, marker='^', label='Fused Triton')
    plt.xlabel('Hidden Dimension (N)')
    plt.ylabel('Execution Time (ms)')
    plt.title('RMSNorm Latency Comparison')
    plt.legend()

    # plot bandwidth in comparison to hard peak
    plt.subplot(1, 2, 2)
    plt.plot(configs, naive_bws, marker='o', label='PyTorch Eager')
    plt.plot(configs, compile_bws, marker='s', label='torch.compile')
    plt.plot(configs, triton_bws, marker='^', label='Fused Triton')

    peak_bw = 335.0  # Default estimate for Colab T4 GPU (335 GB/s)

    plt.axhline(y=peak_bw, color='r', linestyle='--', label=f'Peak Peak HBM ({peak_bw} GB/s)')

    plt.xlabel('Hidden Dimension (N)')
    plt.ylabel('Achieved Bandwidth (GB/s)')
    plt.title('RMSNorm HBM Bandwidth Utilization')
    plt.legend()

    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    print("\nBenchmark complete! Plot saved as 'benchmark_results.png'.")

if __name__ == "__main__":
    run_benchmark()
