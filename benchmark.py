import torch
from splash_attention import sparse_attention_naive, splash_attention

import matplotlib.pyplot as plt


def benchmark(fn, x, name="", repeat=50, warmup=10):

    # -------- Peak Memory --------
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    fn(*x)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()

    # -------- Speed --------
    # warmup
    for _ in range(warmup):
        fn(*x)
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(repeat):
        start.record()
        fn(*x)
        end.record()

        torch.cuda.synchronize()

        times.append(start.elapsed_time(end))

    avg_time = sum(times) / len(times)

    print(f"{name}:")
    print(f"  speed: {avg_time:.3f} ms")
    print(f"  peak memory: {peak / 1024**2:.2f} MB\n")

    return avg_time, peak


# Example usage ----------------

x = list(range(5, 12))
lens = [2 ** n for n in range(5, 12)]

peaks_splash = []
peaks_naive = []

times_splash = []
times_naive = []

for _len in lens:
    q = k = v = torch.randn((10, 10, _len, 10), device="cuda")

    inp = (q, k, v, 0, False, False, False)

    result_splash = benchmark(splash_attention, inp, name=f"Splash len: {_len}")
    times_splash.append(result_splash[0])
    peaks_splash.append(result_splash[1] / 2 ** 30)
    result_naive = benchmark(sparse_attention_naive, inp, name=f"Naive len: {_len}")
    times_naive.append(result_naive[0])
    peaks_naive.append(result_naive[1] / 2 ** 30)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Execution Times Comparison
ax1.semilogx(lens, times_splash, 'o-', label='Splash Attention', linewidth=2, markersize=8)
ax1.semilogx(lens, times_naive, 's-', label='Naive Attention', linewidth=2, markersize=8)
ax1.set_xlabel('Sequence Length', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)
ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: Memory Peak Comparison
ax2.semilogx(lens, peaks_splash, 'o-', label='Splash Attention', linewidth=2, markersize=8)
ax2.semilogx(lens, peaks_naive, 's-', label='Naive Attention', linewidth=2, markersize=8)
ax2.set_xlabel('Sequence Length', fontsize=12)
ax2.set_ylabel('Peak Memory (GB)', fontsize=12)
ax2.set_title('Memory Peak Comparison', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('attention_comparison.png', dpi=300, bbox_inches='tight')
