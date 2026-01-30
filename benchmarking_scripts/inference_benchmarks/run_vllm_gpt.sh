#!/bin/bash

model=/models/openai/gpt-oss-120b
COMPILATION_CONFIG='{"compile_sizes": [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], "cudagraph_capture_sizes":[8192,4096,2048,1024,1008,992,976,960,944,928,912,896,880,864,848,832,816,800,784,768,752,736,720,704,688,672,656,640,624,608,592,576,560,544,528,512,496,480,464,448,432,416,400,384,368,352,336,320,304,288,272,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1], "cudagraph_mode": "FULL_AND_PIECEWISE"}'

export VLLM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_FUSED_MOE_A16W4=1


CONFIGS=(
    "8 128 128 8192 8192 1024"
    "8 128 1024 8192 8192 1024"
    "8 128 2048 8192 8192 1024"
    "8 512 4096 8192 8192 1024"
    "8 2048 128 8192 8192 1024"
    "8 2048 2048 8192 8192 1024"
    "8 2048 4096 8192 8192 1024"
    "8 4096 2048 8192 8192 1024"
    "8 8192 2048 12288 12288 1024"
    "8 16384 2048 20480 20480 1024"
    "8 1024 16384 19456 19456 1024"
    "8 2048 8192 12288 12288 1024"
    "8 2048 16384 20480 20480 1024"
    "8 16384 256 18688 18688 1024"
)


# Loop through configurations
for config in "${CONFIGS[@]}"; do
    read -r tensor_parallel input_len output_len max_model_len max_seq_len_to_capture max_num_seqs <<< "$config"
    
    # Generate output JSON filename if not provided
    output_json="/workdir/updated-gptoss/gpt-oss-120b-${input_len}-${output_len}-tp${tensor_parallel}.json"
    
    # Generate log filename
    log_file="/workdir/updated-gptoss/gpt-oss-120b-${input_len}-${output_len}-tp${tensor_parallel}.log"
    
    echo "Running benchmark: TP=${tensor_parallel}, Input=${input_len}, Output=${output_len}, Max Model Len=${max_model_len}, Max Seq Len to Capture=${max_seq_len_to_capture}, Max Num Seqs=${max_num_seqs}"
    echo "  Output JSON: ${output_json}"
    echo "  Log file: ${log_file}"
    echo "  Started at: $(date)"
    
    # Run benchmark and log both stdout and stderr to file, also show on console
    vllm bench throughput \
        --compilation-config "${COMPILATION_CONFIG}" \
        --no-enable-prefix-caching --disable-log-requests \
        --tensor-parallel "${tensor_parallel}" \
        --block-size 64 \
        --swap-space 16 \
        --backend vllm \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --n 1 \
        --num-prompts 1000 \
        --max-model-len "${max_model_len}" \
        --max-seq-len-to-capture "${max_seq_len_to_capture}" \
        --max-num-seqs "${max_num_seqs}" \
        --gpu-memory-utilization 0.95 \
        --async-scheduling \
        --output-json "${output_json}" \
        --model "${model}" 
        2>&1 | tee "${log_file}"
    
    echo "  Completed at: $(date)"
    echo ""
done