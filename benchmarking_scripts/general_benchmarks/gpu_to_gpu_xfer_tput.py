#!/usr/bin/env python3
import torch
import time

def benchmark_gpu_to_gpu_transfer(size_gb=1, num_iterations=10, warmup_iterations=5):
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # Create a tensor of the specified size
    size_bytes = size_gb * 1024 * 1024 * 1024
    data = torch.rand(size_bytes // 4, dtype=torch.float32)  # 4 bytes per float32

    for src_gpu in range(num_gpus):
        for dst_gpu in range(num_gpus):
            if src_gpu != dst_gpu:
                print(f"\nTransferring from GPU {src_gpu} to GPU {dst_gpu}")
                
                # Move data to source GPU
                src_data = data.to(f'cuda:{src_gpu}')
                torch.cuda.synchronize(src_gpu)
                
                # Warm-up transfer
                for _ in range(warmup_iterations):
                    dst_data = src_data.to(f'cuda:{dst_gpu}')
                    torch.cuda.synchronize(dst_gpu)

                # Benchmark
                start_time = time.perf_counter()
                for _ in range(num_iterations):
                    dst_data = src_data.to(f'cuda:{dst_gpu}')
                    torch.cuda.synchronize(dst_gpu)
                    # Ensure the data is actually used
                    # _ = dst_data.sum().item()
                end_time = time.perf_counter()

                # Calculate and print results
                total_time = end_time - start_time
                avg_time = total_time / num_iterations
                bandwidth_gb = size_gb / avg_time
                print(f"Average transfer time: {avg_time:.4f} seconds")
                print(f"Bandwidth: {bandwidth_gb:.2f} GB/s")

if __name__ == "__main__":
    benchmark_gpu_to_gpu_transfer()