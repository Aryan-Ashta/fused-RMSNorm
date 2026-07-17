import torch
from rmsnorm import NaiveRMSNorm

def profile_naive_baseline():
    B, SEQ, H = 16, 2048, 4096 # Setup LLM Shape
    print(f"allocating baseline tensor of shape {(B, SEQ, H)} ...")

    x = torch.randn(B, SEQ, H, device='cuda', dtype=torch.float32)
    model = NaiveRMSNorm(H).cuda()

    # Warmup to discard CUDA context initialization overhead
    print("warming up GPU ...")
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    print("profiling naive RMSNorm eager execution ...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        out = model(x)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this script on an NVIDIA GPU environment (like Google Colab).")
    else:
        profile_naive_baseline()
