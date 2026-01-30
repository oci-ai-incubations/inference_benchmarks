#!/bin/bash

export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export VLLM_USE_AITER_TRITON_ROPE=1
export TRITON_HIP_ASYNC_COPY_BYPASS_PERMUTE=1
export TRITON_HIP_USE_ASYNC_COPY=1
export TRITON_HIP_USE_BLOCK_PINGPONG=1
export TRITON_HIP_ASYNC_FAST_SWIZZLE=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_RMSNORM=1

# Common compilation config
COMPILATION_CONFIG='{"compile_sizes": [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], "cudagraph_capture_sizes":[8192,4096,2048,1024,1008,992,976,960,944,928,912,896,880,864,848,832,816,800,784,768,752,736,720,704,688,672,656,640,624,608,592,576,560,544,528,512,496,480,464,448,432,416,400,384,368,352,336,320,304,288,272,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1], "full_cuda_graph": true}'

# Benchmark configurations: tensor_parallel input_len output_len [output_json]
# If output_json is not provided, it will be auto-generated as: gpt-oss-120b-{input_len}-{output_len}-tp{tensor_parallel}.json
CONFIGS=(
    "8 4096 2048"
    "8 8192 2048"
    "8 16384 2048"
    "8 128 1024"
    "8 512 4096"
    "8 1024 16384"
    "8 2048 4096"
    "8 2048 8192"
    "8 2048 16384"

)

# Loop through configurations
for config in "${CONFIGS[@]}"; do
    read -r tensor_parallel input_len output_len <<< "$config"
    
    # Generate output JSON filename if not provided
    output_json="/workdir/llama-3.3-70b-fp8-${input_len}-${output_len}-tp${tensor_parallel}.json"
    
    # Generate log filename
    log_file="/workdir/llama-3.3-70b-fp8-${input_len}-${output_len}-tp${tensor_parallel}.log"
    
    echo "Running benchmark: TP=${tensor_parallel}, Input=${input_len}, Output=${output_len}"
    echo "  Output JSON: ${output_json}"
    echo "  Log file: ${log_file}"
    echo "  Started at: $(date)"
    
    # Run benchmark and log both stdout and stderr to file, also show on console
    vllm bench throughput \
        --no-enable-prefix-caching --disable-log-requests \
        --tensor-parallel "${tensor_parallel}" \
        --dtype auto \
        --max-model-len 131072 \
        --max-num-seqs 512 \
        --distributed-executor-backend mp \
        --kv-cache-dtype fp8 \
        --max-seq-len-to-capture 65536 \
        --max-num-batched-tokens 131072 \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --n 1 \
        --num-prompts 1000 \
        --gpu-memory-utilization 0.95 \
        --async-scheduling \
        --output-json "${output_json}" \
        --model /models/models/amd/Llama-3.3-70B-Instruct-FP8-KV \
        2>&1 | tee "${log_file}"
    
    echo "  Completed at: $(date)"
    echo ""
done