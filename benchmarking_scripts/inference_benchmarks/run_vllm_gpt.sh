#!/bin/bash

model=/models/openai/gpt-oss-120b

export VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1

CONFIGS=(
    "4 128 128 8192 8192 1024"
    "4 128 1024 8192 8192 1024"
    "4 128 2048 8192 8192 1024"
    "4 512 4096 8192 8192 1024"
    "4 2048 128 8192 8192 1024"
    "4 2048 2048 8192 8192 1024"
    "4 2048 4096 8192 8192 1024"
    "4 4096 2048 8192 8192 1024"
    "4 8192 2048 12288 12288 1024"
    "4 16384 2048 20480 20480 1024"
    "4 1024 16384 19456 19456 1024"
    "4 2048 8192 12288 12288 1024"
    "4 2048 16384 20480 20480 1024"
    "4 16384 256 18688 18688 1024"
)


# Loop through configurations
for config in "${CONFIGS[@]}"; do
    read -r tensor_parallel input_len output_len max_model_len max_seq_len_to_capture max_num_seqs <<< "$config"
    
    # Generate output JSON filename if not provided
    output_json="/workdir/updated-gptoss/gpt-oss-120b-${input_len}-${output_len}-tp${tensor_parallel}.json"
    
    # Generate log filename
    log_file="/workdir/updated-gptoss/gpt-oss-120b-${input_len}-${output_len}-tp${tensor_parallel}.log"
    
    echo "Running benchmark: TP=${tensor_parallel}, Input=${input_len}, Output=${output_len}, Max Model Len=${max_model_len}, Max Seq 
Len to Capture=${max_seq_len_to_capture}, Max Num Seqs=${max_num_seqs}"
    echo "  Output JSON: ${output_json}"
    echo "  Log file: ${log_file}"
    echo "  Started at: $(date)"
    
    # Run benchmark and log both stdout and stderr to file, also show on console
    vllm bench throughput \
        --no-enable-prefix-caching \
	    --kv-cache-dtype fp8 \
	    --max-cudagraph-capture-size 2048 \
	    --stream-interval 20 \
        --tensor-parallel-size "${tensor_parallel}" \
        --backend vllm \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --n 1 \
        --num-prompts 1000 \
        --max-model-len "${max_model_len}" \
        --max-num-seqs "${max_num_seqs}" \
	    --max-num-batched-tokens "${max_seq_len_to_capture}" \
        --gpu-memory-utilization 0.95 \
        --output-json "${output_json}" \
        --model "${model}" 
        2>&1 | tee "${log_file}"
    
    echo "  Completed at: $(date)"
    echo ""
done
