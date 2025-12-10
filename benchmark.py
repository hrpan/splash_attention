import torch
from splash_attention import sparse_attention_naive, splash_attention


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

lens = [2 ** n for n in range(5, 12)]

for _len in lens:
    q = k = v = torch.randn((10, 10, _len, 10), device="cuda")

    inp = (q, k, v, 0, False, False, False)

    benchmark(splash_attention, inp, name=f"Splash len: {_len}")
    benchmark(sparse_attention_naive, inp, name=f"Naive len: {_len}")
