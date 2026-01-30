import argparse
import os
import statistics
import time
import torch
from collections import defaultdict
from itertools import product

# -------------------------------------------------
# THEORETICAL PEAKS (per dtype) — from your table
# -------------------------------------------------
THEORETICAL_BY_DTYPE = {
    # matrix (PFLOPS) -> TFLOPS
    "torch.float16": 5.0332 * 1000.0,     # FP16 MATRIX
    "torch.bfloat16": 10.0664 * 1000.0,   # BFLOAT16
    # HPC peaks (already TFLOPS)
    "torch.float32": 157.3,               # FP32 MATRIX
    "torch.float64": 78.6,                # FP64 MATRIX
}

# -------------------------------------------------
# what dtypes to try
# -------------------------------------------------
DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)

# -------------------------------------------------
# base sizes
# we'll generate: squares + rectangular permutations
# -------------------------------------------------
BASE_SIZES = [2048, 4096, 6144]
# Larger sizes for multi-GPU runs
LARGE_SIZES_4GPU = [8192, 10240]  # For 4+ GPUs
LARGE_SIZES_8GPU = [12288, 16384]  # For 8 GPUs

def build_shapes(num_gpus=1):
    """
    Build matrix shapes. Adds larger sizes for 4+ and 8 GPU runs.
    """
    shapes = []
    sizes_to_use = BASE_SIZES.copy()
    
    # Add larger sizes for 4+ GPUs
    if num_gpus >= 4:
        sizes_to_use.extend(LARGE_SIZES_4GPU)
    
    # Add even larger sizes for 8 GPUs
    if num_gpus == 8:
        sizes_to_use.extend(LARGE_SIZES_8GPU)

    # 1) keep original squares
    for s in sizes_to_use:
        shapes.append((s, s, s))

    # 2) add permutations M,K,N from base sizes
    #    but don't add exact duplicates of the squares we already added
    for M, K, N in product(sizes_to_use, repeat=3):
        if (M, K, N) not in shapes:
            shapes.append((M, K, N))

    return shapes

INNER_ITERS = 10
MEASUREMENTS = 3
MEM_SAFETY = 0.5  # use up to 50% of VRAM for A,B,C

OP_FUNC_MAP = {
    "matmul": "c=ab",
}

# -------------------------------------------------
# helpers
# -------------------------------------------------
def get_device_info(device: int):
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    return props.name, props.total_memory

def fits_in_memory(m, k, n, dtype, total_mem_bytes, safety=0.5):
    elem_size = torch.tensor([], dtype=dtype).element_size()
    needed = (m * k + k * n + m * n) * elem_size
    return needed <= total_mem_bytes * safety

def set_tf32(enabled: bool):
    # guarded for ROCm
    if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "matmul"):
            try:
                torch.backends.cuda.matmul.allow_tf32 = enabled
            except Exception:
                pass
        if hasattr(torch.backends.cuda, "cudnn"):
            try:
                torch.backends.cuda.cudnn.allow_tf32 = enabled
            except Exception:
                pass

def benchmark_matmul_multi_gpu(m, k, n, dtype, num_gpus, use_tf32=False,
                               iters=INNER_ITERS, measurements=MEASUREMENTS):
    """
    Benchmark matmul across multiple GPUs by splitting along the M dimension.
    Each GPU computes a portion of the result matrix.
    Uses CUDA streams to enable true parallel execution across GPUs.
    """
    set_tf32(use_tf32)
    
    # Split M dimension across GPUs
    m_per_gpu = m // num_gpus
    remainder = m % num_gpus
    
    # Create tensors and streams on each GPU
    devices = list(range(num_gpus))
    a_tensors = []
    b_tensors = []
    c_tensors = []
    streams = []
    
    for i, dev in enumerate(devices):
        # Calculate the slice of M for this GPU
        m_start = i * m_per_gpu + min(i, remainder)
        m_end = m_start + m_per_gpu + (1 if i < remainder else 0)
        m_local = m_end - m_start
        
        # Create stream for this GPU (on the correct device)
        stream = torch.cuda.Stream(device=dev)
        streams.append(stream)
        
        # Create local tensors directly on the device
        a_local = torch.randn((m_local, k), device=f"cuda:{dev}", dtype=dtype)
        b_local = torch.randn((k, n), device=f"cuda:{dev}", dtype=dtype)
        c_local = torch.empty((m_local, n), device=f"cuda:{dev}", dtype=dtype)
        
        a_tensors.append(a_local)
        b_tensors.append(b_local)
        c_tensors.append(c_local)
    
    # Warmup - launch all operations concurrently
    for _ in range(3):
        # Launch all operations first (asynchronous)
        for dev_idx, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                torch.matmul(a_tensors[dev_idx], b_tensors[dev_idx], out=c_tensors[dev_idx])
        # Then synchronize
        for stream in streams:
            stream.synchronize()
    
    # flops per GEMM (total across all GPUs)
    if dtype.is_complex:
        flops_per_matmul = 8.0 * m * n * k
    else:
        flops_per_matmul = 2.0 * m * n * k
    
    results = []
    for _ in range(measurements):
        # Synchronize all devices before timing
        for stream in streams:
            stream.synchronize()
        
        # Use wall-clock time for accurate multi-GPU timing
        start_time = time.perf_counter()
        
        # Launch iterations concurrently across all GPUs
        # For each iteration, launch on all GPUs simultaneously
        for _ in range(iters):
            # Launch matmul on all GPUs simultaneously
            for dev_idx, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    torch.matmul(a_tensors[dev_idx], b_tensors[dev_idx], out=c_tensors[dev_idx])
        
        # Synchronize all streams after launching all operations
        for stream in streams:
            stream.synchronize()
        
        end_time = time.perf_counter()
        elapsed_s = end_time - start_time
        
        total_flops = flops_per_matmul * iters
        tflops = total_flops / elapsed_s / 1e12
        results.append(tflops)
    
    return results

def benchmark_matmul(m, k, n, dtype, use_tf32=False, num_gpus=1,
                     iters=INNER_ITERS, measurements=MEASUREMENTS):
    """
    Benchmark matmul on single or multiple GPUs.
    """
    if num_gpus == 1:
        # Single GPU path (original implementation)
        set_tf32(use_tf32)

        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((k, n), device="cuda", dtype=dtype)
        c = torch.empty((m, n), device="cuda", dtype=dtype)

        # warmup
        for _ in range(3):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()

        # flops per GEMM
        if dtype.is_complex:
            flops_per_matmul = 8.0 * m * n * k
        else:
            flops_per_matmul = 2.0 * m * n * k

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        results = []
        for _ in range(measurements):
            torch.cuda.synchronize()
            start.record()
            for _ in range(iters):
                torch.matmul(a, b, out=c)
            end.record()
            torch.cuda.synchronize()

            elapsed_s = start.elapsed_time(end) / 1000.0
            total_flops = flops_per_matmul * iters
            tflops = total_flops / elapsed_s / 1e12
            results.append(tflops)

        return results
    else:
        # Multi-GPU path
        return benchmark_matmul_multi_gpu(m, k, n, dtype, num_gpus, use_tf32,
                                         iters, measurements)

# -------------------------------------------------
# per-device suite (single GPU)
# -------------------------------------------------
def matmul_suite_for_device(device: int, aggregator: dict, num_gpus: int = 1):
    torch.cuda.set_device(device)
    name, total_mem = get_device_info(device)
    
    if num_gpus == 1:
        print(f"=== GPU {device}: {name} ({total_mem/1024**3:.1f} GiB) ===")
    else:
        print(f"=== GPUs {device}-{device+num_gpus-1}: {name} ({total_mem/1024**3:.1f} GiB per GPU, {num_gpus} GPUs) ===")

    # Build shapes (includes larger sizes for 4+ and 8 GPUs)
    shapes = build_shapes(num_gpus)

    for dtype in DTYPES:
        # skip unsup dtypes
        try:
            torch.empty(1, dtype=dtype, device="cuda")
        except RuntimeError:
            continue

        print(f"\n-- dtype={dtype} --")

        for (m, k, n) in shapes:
            # For multi-GPU, check memory on each GPU (they each hold a portion)
            if num_gpus == 1:
                if not fits_in_memory(m, k, n, dtype, total_mem, safety=MEM_SAFETY):
                    continue
            else:
                # For multi-GPU, each GPU holds m/num_gpus rows of A, full B, and m/num_gpus rows of C
                m_per_gpu = m // num_gpus
                # Check memory: m_per_gpu*k (A slice) + k*n (full B) + m_per_gpu*n (C slice)
                if not fits_in_memory(m_per_gpu, k, n, dtype, total_mem, safety=MEM_SAFETY):
                    continue

            print(f"  shape=({m},{k})x({k},{n}) [num_gpus={num_gpus}]")

            use_tf32 = (dtype is torch.float32)

            try:
                runs = benchmark_matmul(m, k, n, dtype, use_tf32=use_tf32, num_gpus=num_gpus)
            except RuntimeError as e:
                print(f"  FAILED {m}x{k}x{n} for {dtype}: {e}")
                continue

            mean_v = statistics.mean(runs)
            std_v = statistics.stdev(runs) if len(runs) > 1 else 0.0
            print(f"    matmul: {mean_v:7.2f} ± {std_v:4.2f} TFLOP/s (TF32={use_tf32}, GPUs={num_gpus})")

            key = ("matmul", str(dtype), m, k, n, num_gpus)
            aggregator[key].append(mean_v)

# -------------------------------------------------
# driver
# -------------------------------------------------
def run_all_devices_matmul(num_gpus=1, output_file=None, print_header=True):
    n = torch.cuda.device_count()
    if n == 0:
        print("No CUDA/ROCm devices found.")
        return
    
    if num_gpus > n:
        print(f"Warning: Requested {num_gpus} GPUs but only {n} available. Using {n} GPUs.")
        num_gpus = n
    
    # Validate num_gpus is 1, 2, 4, or 8
    if num_gpus not in [1, 2, 4, 8]:
        print(f"Error: num_gpus must be 1, 2, 4, or 8. Got {num_gpus}.")
        return

    aggregator = defaultdict(list)

    if num_gpus == 1:
        # Run on each GPU individually
        for dev in range(n):
            matmul_suite_for_device(dev, aggregator, num_gpus=1)
    else:
        # Run multi-GPU benchmarks starting from GPU 0
        # Process GPUs in groups of num_gpus
        for start_dev in range(0, n, num_gpus):
            end_dev = min(start_dev + num_gpus, n)
            actual_num_gpus = end_dev - start_dev
            
            if actual_num_gpus < num_gpus:
                print(f"Skipping GPU group starting at {start_dev}: need {num_gpus} GPUs, have {actual_num_gpus}")
                continue
            
            # Use the first GPU in the group as the primary device
            matmul_suite_for_device(start_dev, aggregator, num_gpus=num_gpus)

    # Generate CSV lines
    csv_lines = []
    header = "op,func,dtype,M,K,N,num_gpus,mean(TFLOP/s),median(TFLOP/s),device_count,theoretical(TFLOP/s),efficiency"
    
    # Only include header if this is the first write (not appending)
    if print_header:
        csv_lines.append(header)
        print(f"\n{header}")

    for key, values in sorted(aggregator.items()):
        if len(key) == 6:
            op, dtype_str, m, k, n, num_gpus_key = key
        else:
            # Backward compatibility: old format without num_gpus
            op, dtype_str, m, k, n = key
            num_gpus_key = 1
        
        mean_v = statistics.mean(values)
        median_v = statistics.median(values)
        device_count = len(values)

        func = OP_FUNC_MAP.get(op, "")

        # For multi-GPU, theoretical peak is scaled by num_gpus
        theoretical = THEORETICAL_BY_DTYPE.get(dtype_str, 0.0)
        theoretical_total = theoretical * num_gpus_key
        eff = (mean_v / theoretical_total) if theoretical_total and theoretical_total > 0 else ""

        csv_line = (
            f"{op},{func},{dtype_str},{m},{k},{n},{num_gpus_key},"
            f"{mean_v:.2f},{median_v:.2f},{device_count},"
            f"{theoretical_total:.1f},{eff}"
        )
        csv_lines.append(csv_line)
        print(csv_line)
    
    # Write to file if specified
    if output_file:
        file_exists = os.path.exists(output_file)
        mode = 'a' if file_exists else 'w'
        with open(output_file, mode) as f:
            f.write('\n'.join(csv_lines) + '\n')
        if mode == 'w':
            print(f"\nSummary written to {output_file}")
        else:
            print(f"\nResults appended to {output_file}")
    
    return csv_lines

def run_all_configurations(output_file=None):
    """
    Run benchmarks for 1, 2, 4, and 8 GPUs sequentially, appending results to the same file.
    """
    n = torch.cuda.device_count()
    if n == 0:
        print("No CUDA/ROCm devices found.")
        return
    
    # Clear output file if it exists
    if output_file and os.path.exists(output_file):
        os.remove(output_file)
        print(f"Cleared existing file: {output_file}")
    
    gpu_configs = [1, 2, 4, 8]
    
    for idx, num_gpus in enumerate(gpu_configs):
        if num_gpus > n:
            print(f"\nSkipping {num_gpus} GPUs: only {n} GPUs available")
            continue
        
        print(f"\n{'='*80}")
        print(f"Running benchmark with {num_gpus} GPU(s)")
        print(f"{'='*80}")
        
        # Only print header for the first configuration
        print_header = (idx == 0)
        run_all_devices_matmul(num_gpus=num_gpus, output_file=output_file, print_header=print_header)
    
    if output_file:
        print(f"\n{'='*80}")
        print(f"All benchmarks completed. Results saved to {output_file}")
        print(f"{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark matmul operations on GPU(s)")
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        choices=[1, 2, 4, 8],
        help="Number of GPUs to use for parallelization (default: 1, or use --run-all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output file for CSV summary (default: print to stdout only)"
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run benchmarks for 1, 2, 4, and 8 GPUs sequentially, appending to output file"
    )
    
    args = parser.parse_args()
    
    if args.run_all:
        if not args.output:
            print("Error: --output must be specified when using --run-all")
            exit(1)
        run_all_configurations(output_file=args.output)
    else:
        num_gpus = args.num_gpus if args.num_gpus is not None else 1
        run_all_devices_matmul(num_gpus=num_gpus, output_file=args.output, print_header=True)