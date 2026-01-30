import statistics
import torch
from collections import defaultdict

# -------------------------------------------------
# config
# -------------------------------------------------
DTYPES = (torch.float32, torch.bfloat16, torch.float16)
SIZE_BYTES_LIST = (1 << 20, 16 << 20, 256 << 20, 1 << 30)  # 1MiB,16MiB,256MiB,1GiB

# name map: op -> human func
OP_FUNC_MAP = {
    "d2d_copy": "c=a",
    "add": "c=a+b",
    "triad": "c=a+αb",
    "scale": "c*=α",
    "write-only": "c=0",
    "strided4": "c[::4]=a[::4]",
    "H2D_32MiB": "host->dev 32MiB",
    "D2H_32MiB": "dev->host 32MiB",
}


# -------------------------------------------------
# helpers
# -------------------------------------------------
def device_info(device=0):
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    return props.name, props.total_memory

def bytes_to_numel(bytes_target, dtype):
    elem_size = torch.tensor([], dtype=dtype).element_size()
    return bytes_target // elem_size

def run_kernel(fn, bytes_per_iter, iters=50, measures=5):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    results = []

    # warmup
    for _ in range(5):
        fn()
    torch.cuda.synchronize()

    for _ in range(measures):
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()

        secs = start.elapsed_time(end) / 1000.0  # s
        gbs = (bytes_per_iter * iters) / secs / (1024 ** 3)
        results.append(gbs)

    avg = statistics.mean(results)
    std = statistics.stdev(results) if len(results) > 1 else 0.0
    return avg, std


# -------------------------------------------------
# per-device suite
# -------------------------------------------------
def gpu_mem_bw_suite_for_device(device, aggregator):
    torch.cuda.set_device(device)
    dev_name, total_mem = device_info(device)
    print(f"=== GPU {device}: {dev_name} ({total_mem/1024**3:.1f} GiB) ===")

    for dtype in DTYPES:
        # skip unsupported
        try:
            torch.empty(1, dtype=dtype, device="cuda")
        except RuntimeError:
            continue

        print(f"\n-- dtype={dtype} --")

        for size_bytes in SIZE_BYTES_LIST:
            # don't exceed ~1/3 VRAM for 3 tensors
            if size_bytes * 3 > total_mem:
                continue

            numel = bytes_to_numel(size_bytes, dtype)
            if numel == 0:
                continue

            print(f"  size ≈ {size_bytes / (1024**2):6.1f} MiB  (numel={numel})")

            a = torch.empty(numel, dtype=dtype, device="cuda").uniform_()
            b = torch.empty(numel, dtype=dtype, device="cuda").uniform_()
            c = torch.empty(numel, dtype=dtype, device="cuda").zero_()
            torch.cuda.synchronize()

            elem = a.element_size()

            # 1) d2d
            def d2d_copy():
                c.copy_(a)
            avg, std = run_kernel(d2d_copy, bytes_per_iter=2 * numel * elem)
            print(f"d2d_copy          : {avg:7.2f} ± {std:4.2f} GB/s")
            aggregator[("d2d_copy", str(dtype), size_bytes)].append(avg)

            # 2) add
            def vec_add():
                torch.add(a, b, out=c)
            avg, std = run_kernel(vec_add, bytes_per_iter=3 * numel * elem)
            print(f"add (c=a+b)       : {avg:7.2f} ± {std:4.2f} GB/s")
            aggregator[("add", str(dtype), size_bytes)].append(avg)

            # 3) triad
            alpha = 1.2345
            def triad():
                torch.add(a, b, alpha=alpha, out=c)
            avg, std = run_kernel(triad, bytes_per_iter=3 * numel * elem)
            print(f"triad (c=a+αb)    : {avg:7.2f} ± {std:4.2f} GB/s")
            aggregator[("triad", str(dtype), size_bytes)].append(avg)

            # 4) scale
            def scale_inplace():
                c.mul_(alpha)
            avg, std = run_kernel(scale_inplace, bytes_per_iter=2 * numel * elem)
            print(f"scale (c*=α)      : {avg:7.2f} ± {std:4.2f} GB/s")
            aggregator[("scale", str(dtype), size_bytes)].append(avg)

            # 5) write-only
            def write_only():
                c.zero_()
            avg, std = run_kernel(write_only, bytes_per_iter=1 * numel * elem)
            print(f"write-only        : {avg:7.2f} ± {std:4.2f} GB/s")
            aggregator[("write-only", str(dtype), size_bytes)].append(avg)

            # 6) strided
            if numel // 4 > 0:
                view = a[::4]
                view_out = c[::4]
                eff_numel = view.numel()
                def strided():
                    view_out.copy_(view)
                avg, std = run_kernel(strided, bytes_per_iter=2 * eff_numel * elem)
                print(f"strided copy (4)  : {avg:7.2f} ± {std:4.2f} GB/s")
                aggregator[("strided4", str(dtype), size_bytes)].append(avg)

    # optional host<->device
    try:
        print("\n-- host <-> device (32MiB) --")
        pinned = torch.empty(32 << 20, dtype=torch.uint8).pin_memory()
        dev = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")

        def h2d():
            dev.copy_(pinned, non_blocking=True)
        avg, std = run_kernel(h2d, bytes_per_iter=32 << 20)
        print(f"H2D 32MiB         : {avg:7.2f} ± {std:4.2f} GB/s")
        aggregator[("H2D_32MiB", "uint8", 32 << 20)].append(avg)

        def d2h():
            pinned.copy_(dev, non_blocking=True)
        avg, std = run_kernel(d2h, bytes_per_iter=32 << 20)
        print(f"D2H 32MiB         : {avg:7.2f} ± {std:4.2f} GB/s")
        aggregator[("D2H_32MiB", "uint8", 32 << 20)].append(avg)
    except RuntimeError:
        pass


# -------------------------------------------------
# driver
# -------------------------------------------------
def run_all_devices(theoretical_gbs=8000.0):
    n = torch.cuda.device_count()
    if n == 0:
        print("No CUDA/ROCm devices found.")
        return

    aggregator = defaultdict(list)

    for dev in range(n):
        gpu_mem_bw_suite_for_device(dev, aggregator)

    # CSV header
    print("\nop,func,dtype,size,mean(GB/s),median(GB/s),device_count,theoretical(GB/s),efficiency")
    # CSV rows
    for key, values in sorted(aggregator.items()):
        op, dtype_str, size_bytes = key
        mean_v = statistics.mean(values)
        median_v = statistics.median(values)
        count_v = len(values)
        size_mb = size_bytes / (1024 ** 2)
        func = OP_FUNC_MAP.get(op, "")
        eff = mean_v / theoretical_gbs if theoretical_gbs > 0 else ""
        # print CSV
        print(f"{op},{func},{dtype_str},{size_mb:.1f}MiB,{mean_v:.2f},{median_v:.2f},{count_v},{theoretical_gbs:.1f},{eff:.4f}")


if __name__ == "__main__":
    # change this if your HBM theoretical is different per node
    run_all_devices(theoretical_gbs=8000.0)