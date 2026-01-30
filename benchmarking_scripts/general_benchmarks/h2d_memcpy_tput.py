#!/usr/bin/env python3
"""
Benchmarking transfer speeds from CPU to GPU and back for both
pinned and pageable memory. Can be run as a standalone script
or imported.
"""
import statistics
import sys
from typing import Tuple

import torch


def profile_copies(
    h_array_up: torch.FloatTensor,
    h_array_down: torch.FloatTensor,
    d_array: torch.FloatTensor,
    n_elements: int,
    desc: str,
    n_copy_iters: int = 100,
    n_warmup_iters: int = 10,
) -> Tuple[float, float]:
    """Performs D2H and H2D copy profiling.

    Args:
        h_array_up: Host array for H2D copy measurement.
        h_array_down: Host array for D2H copy measurement.
        d_array: Device array utilized in both copy measurements.
        n_elements: Total number of elements allocated.
        desc: Description of copy type.
        n_copy_iters: Number of profiled copy iterations.
        n_warmup_iters: Number of warmup iterations to load GPU context.

    Returns:
        Mean bandwidth from sample runs.
    """
    print(f"\n{desc} transfer results:")

    total_bytes = n_elements * h_array_up.element_size()
    print(f"Total bytes: {total_bytes / (1024 * 1024)}MB")
    print(f"Performing {n_warmup_iters} warmups and {n_copy_iters} benchmarks.")

    stream = torch.cuda.Stream()

    # Warmup iterations.
    with torch.cuda.stream(stream):
        for _ in range(n_warmup_iters):
            d_array.copy_(h_array_up)
            h_array_down.copy_(d_array)
    stream.synchronize()

    # H2D profiling.
    h2d_results = []
    for _ in range(n_copy_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with torch.cuda.stream(stream):
            d_array.copy_(h_array_up)
        end.record(stream)
        stream.synchronize()
        h2d_results.append(start.elapsed_time(end))

    h2d_bandwidth = [total_bytes * 1e-6 / t for t in h2d_results]
    print(
        f"  Host to Device bandwidth (GB/s): {statistics.mean(h2d_bandwidth):.2f} ± {statistics.stdev(h2d_bandwidth):.2f}"
    )

    # D2H profiling.
    d2h_results = []
    for _ in range(n_copy_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with torch.cuda.stream(stream):
            h_array_down.copy_(d_array)
        end.record(stream)
        stream.synchronize()
        d2h_results.append(start.elapsed_time(end))

    d2h_bandwidth = [total_bytes * 1e-6 / t for t in d2h_results]
    print(
        f"  Device to Host bandwidth (GB/s): {statistics.mean(d2h_bandwidth):.2f} ± {statistics.stdev(d2h_bandwidth):.2f}"
    )

    if not torch.all(h_array_up.eq(h_array_down)):
        print(f"*** {desc} transfers failed.")

    return statistics.mean(h2d_bandwidth), statistics.mean(d2h_bandwidth)


def init_data_and_profile(
    n_elements: int = 32 * (1024**2),
    warmup_iters: int = 10,
    copy_iters: int = 100,
) -> Tuple[float, float]:
    """Initializes data and runs profiling.

    Args:
        size_mb: Size of copy in megabytes.
        warmup_iters: Number of warmup iterations to load GPU context.
        copy_iters: Benchmark number of copy iterations.
    Returns:
        pinned_h2d: pinned h2d transfer rates in GB/s
        pinned_d2h: pinned d2h transfer rates in GB/s
    """
    # 4 Bytes for float32.
    total_bytes = n_elements * 4

    # Host arrays. Pinned memory cannot be swapped to disk. CUDA requires
    # that memory be resident in physical (not virtual) RAM prior to use.
    # If memory is paged, this requires staging into an extra copy buffer prior
    # to utilization on GPU. The consequence of this is more impactful on small
    # memory pools or large workloads.
    h_array_up_pageable = torch.arange(n_elements, dtype=torch.float32)
    h_array_up_pinned = torch.arange(n_elements, dtype=torch.float32).pin_memory()
    h_array_down_pageable = torch.arange(n_elements, dtype=torch.float32)
    h_array_down_pinned = torch.arange(n_elements, dtype=torch.float32).pin_memory()

    # Device array.
    d_array = torch.empty(n_elements, dtype=torch.float32, device="cuda")

    # Output device info and transfer size. Probably can either perform this
    # on all available devices as an extension.
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Transfer size: {total_bytes / (1024 * 1024)}MB")

    # Perform copies and report bandwidth.
    pinned_h2d, pinned_d2h = profile_copies(
        h_array_up_pinned,
        h_array_down_pinned,
        d_array,
        n_elements,
        "Pinned",
        copy_iters,
        warmup_iters,
    )
    _, _ = profile_copies(
        h_array_up_pageable,
        h_array_down_pageable,
        d_array,
        n_elements,
        "Pageable",
        copy_iters,
        warmup_iters,
    )
    # Cleanup.
    del h_array_down_pageable
    del h_array_down_pinned
    del h_array_up_pageable
    del h_array_up_pinned
    del d_array
    # Sync after cleanup.
    torch.cuda.synchronize()
    print("\n")
    return round(pinned_h2d, 2), round(pinned_d2h, 2)


def main() -> int:
    """Dummy entrypoint to run as script."""
    import argparse

    parser = argparse.ArgumentParser(description="Single GPU memcpy benchmark")
    parser.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=32 * 1024**2,
        help="Number of elements to use in copy (32M).",
    )
    parser.add_argument(
        "-w",
        "--num-warmups",
        type=int,
        default=10,
        help="Number of warmup iterations for loading GPU context (10).",
    )
    parser.add_argument(
        "-c",
        "--num-copies",
        type=int,
        default=100,
        help="Number of benchmark copies to run (100).",
    )
    args = parser.parse_args()
    init_data_and_profile(args.num_elements, args.num_warmups, args.num_copies)
    return 0


if __name__ == "__main__":
    sys.exit(main())