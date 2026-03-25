# Setup on RoCM

Testing the following models with each subsection below:
- gpt-oss-120b
- QWEN3 32B
- DeepSeek R1 671B FP4
- Llama3.1-70B

Before anything else, ensure your node is setup correctly. Use the following image below.

Don't click the link or you will download the 30GB file, instead copy the link and import it as a custom image:

https://objectstorage.ap-kulai-1.oraclecloud.com/p/r7NmOiphWU9Pm9G7yBSkGIYRT5EXCjSNL2BYqso7R-s2zYBoTPmdwn3uyJ-pCvGb/n/hpctraininglab/b/Sudhir-Bucket/o/Canonical-Ubuntu-24.04-2026.02.28-0-MOFED-2410_1140-AMD-ROCM-72-2026.03.13-0


Then, once the node is up, install and configure docker:

```bash
sudo snap install docker
sudo groupadd docker
sudo usermod -aG docker ubuntu
sudo adduser $USER docker
newgrp docker
sudo snap connect docker:home
sudo snap disable docker
sudo snap enable docker
sudo snap services docker
```

Common RAID Configs explained:
- RAID0: striping with 0 redundancy - highest performance but no fault tolerance and full usable disk space. drive failure results in data loss
- RAID1: mirroring - writes identical data to two or more disks, providing high redundancy. 2 drives must fail for data loss to occur. 50% usable disk space (IE 28 TB total, 14TB usable)
- RAID5: striping with parity - Distributes data and parity info across 3 or more drives. Balances performance, storage, and redundancy allowing system to function if one drive fails.
- RAID6: striping with double parity - similar to RAID5, but uses two distributed parity blocks allowing for up to two drive failures without data loss
- RAID 10: striped mirrors - combines RAID1 and RAID0, requiring at least four drives. It mirrors data and then stripes it, providing high speed and redundancy but less usable space.

After your node is up, the rest of this guide uses a RAID 0 array to host models. To change this in the guide, change the `--level` in the `mdadm create` command. [setup_raid](./setup_raid.md)

When completed, create a models directory and chmod it:
```bash
sudo mkdir /mnt/nvme/models
sudo chmod 777 /mnt/nvme/models
```

## openai/gpt-oss-120b
https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-vllm-gpt-oss-120b.html

```bash
model=openai/gpt-oss-120b

pip install huggingface_hub
HF_HUB_ENABLE_HF_TRANSFER=1 \
HF_HOME=/data/huggingface-cache \
HF_TOKEN="<HF_TOKEN>" \ # Replace with your HF_TOKEN Hugging Face access token.
huggingface-cli download ${model} --exclude "original/*" --local-dir /mnt/nvme/models/${model} --max-workers 32

docker run -it   --ipc=host   --network=host   --privileged   --cap-add=CAP_SYS_ADMIN   --device=/dev/kfd   --device=/dev/dri   --cap-add=SYS_PTRACE   --security-opt seccomp=unconfined   -v /mnt/nvme/models:/models   -v $HOME:/workdir   -e HF_HUB_OFFLINE=1   rocm/7.x-preview:rocm7.2_preview_ubuntu_22.04_vlm_0.10.1_instinct_20251029

docker run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /mnt/nvme/models:/models \
  -v $HOME:/workdir \
  -e HF_HUB_OFFLINE=1 \
  rocm/7.x-preview:rocm7.2_preview_ubuntu_22.04_vlm_0.10.1_instinct_20251029
```

### From inside the container

```bash
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

outdir=/workdir/updated-gptoss/
mkdir -p $outdir

# Loop through configurations
for config in "${CONFIGS[@]}"; do
    read -r tensor_parallel input_len output_len max_model_len max_seq_len_to_capture max_num_seqs <<< "$config"
    
    # Generate output JSON filename if not provided
    output_json="$outdir/gpt-oss-120b-${input_len}-${output_len}-tp${tensor_parallel}.json"
    
    # Generate log filename
    log_file="$outdir/gpt-oss-120b-${input_len}-${output_len}-tp${tensor_parallel}.log"
    
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
```

## QWEN3 32B

https://rocm.blogs.amd.com/artificial-intelligence/scaling-ai-inference/README.html#qwen3-235b-a22b-instruct-2507

```bash
model=Qwen/Qwen3-32B

pip install huggingface_hub
HF_HUB_ENABLE_HF_TRANSFER=1 \
HF_TOKEN="<HF_TOKEN>" \ # Replace with your HF_TOKEN Hugging Face access token.
huggingface-cli download ${model} --exclude "original/*" --local-dir /mnt/nvme/models/${model} --max-workers 32

docker run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /mnt/nvme/models:/models \
  -v $HOME:/workdir \
  -e HF_HUB_OFFLINE=1 \
  amdsiloai/vllm:20251208-qwen3-1999bf5
```

### Inside the container
```bash
#!/bin/bash

model=/models/Qwen/Qwen3-32B
COMPILATION_CONFIG='{"custom_ops": ["-rms_norm", "-quant_fp8"], "cudagraph_mode": "FULL_AND_PIECEWISE"}'

export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1
export SAFETENSORS_FAST_GPU=1

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

outdir=/workdir/updated-qwen3-32b/
mkdir -p $outdir

# Loop through configurations
for config in "${CONFIGS[@]}"; do
    read -r tensor_parallel input_len output_len max_model_len max_num_batched_tokens max_num_seqs <<< "$config"
    
    # Generate output JSON filename if not provided
    output_json="$outdir/qwen3-32b-${input_len}-${output_len}-tp${tensor_parallel}.json"
    
    # Generate log filename
    log_file="$outdir/qwen3-32b-${input_len}-${output_len}-tp${tensor_parallel}.log"
    
    echo "Running benchmark: TP=${tensor_parallel}, Input=${input_len}, Output=${output_len}, Max Model Len=${max_model_len}, Max Num Batched Tokens=${max_num_batched_tokens} Max Num Seqs=${max_num_seqs}"
    echo "  Output JSON: ${output_json}"
    echo "  Log file: ${log_file}"
    echo "  Started at: $(date)"
    
    # Run benchmark and log both stdout and stderr to file, also show on console
    vllm bench throughput \
        --compilation-config "${COMPILATION_CONFIG}" \
        --no-enable-prefix-caching \
        --tensor-parallel "${tensor_parallel}" \
        --backend vllm \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --n 1 \
        --num-prompts 1000 \
        --max-model-len "${max_model_len}" \
        --max-num-seqs "${max_num_seqs}" \
        --max-num-batched-tokens "${max_num_batched_tokens}" \
        --gpu-memory-utilization 0.95 \
        --async-scheduling \
        --output-json "${output_json}" \
        --model "${model}" 
        2>&1 | tee "${log_file}"
    
    echo "  Completed at: $(date)"
    echo ""
done
```

## amd/Llama-3.3-70B-Instruct-FP8-KV

https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-vllm-llama-3.3-70b-fp8.html

```bash
model=amd/Llama-3.3-70B-Instruct-FP8-KV

pip install huggingface_hub
HF_HOME=/mnt/nvme/models/${model} \
huggingface-cli download ${model} --exclude "original/*" --local-dir /mnt/nvme/models/${model} --max-workers 32

docker run -it \
  --ipc=host \
  --network=host \
  --privileged \
  --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd \
  --device=/dev/dri \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /mnt/nvme/models:/models \
  -v $HOME:/workdir \
  -e HF_HUB_OFFLINE=1 \
  rocm/7.x-preview:rocm7.2_preview_ubuntu_22.04_vlm_0.10.1_instinct_20251029
```

Now, inside the container:

```bash
#!/bin/bash

model=/models/amd/Llama-3.3-70B-Instruct-FP8-KV

export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4


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

outdir=/workdir/updated-llama-33-70b-fp8/
mkdir -p $outdir

# Loop through configurations
for config in "${CONFIGS[@]}"; do
    read -r tensor_parallel input_len output_len max_model_len max_seq_len_to_capture max_num_seqs <<< "$config"
    
    # Generate output JSON filename if not provided
    output_json="$outdir/llama-33-70b-fp8-${input_len}-${output_len}-tp${tensor_parallel}.json"
    
    # Generate log filename
    log_file="$outdir/llama-33-70b-fp8-${input_len}-${output_len}-tp${tensor_parallel}.log"
    
    echo "Running benchmark: TP=${tensor_parallel}, Input=${input_len}, Output=${output_len}, Max Model Len=${max_model_len}, Max Seq Len to Capture=${max_seq_len_to_capture}, Max Num Seqs=${max_num_seqs}"
    echo "  Output JSON: ${output_json}"
    echo "  Log file: ${log_file}"
    echo "  Started at: $(date)"
    
    # Run benchmark and log both stdout and stderr to file, also show on console
    vllm bench throughput \
        --no-enable-prefix-caching --disable-log-requests \
        --tensor-parallel "${tensor_parallel}" \
        --swap-space 64 \
        --backend vllm \
        --input-len "${input_len}" \
        --output-len "${output_len}" \
        --n 1 \
        --num-prompts 1000 \
        --max-model-len "${max_model_len}" \
        --kv-cache-dtype fp8 \
        --max-seq-len-to-capture "${max_seq_len_to_capture}" \
        --max-num-batched-tokens 131072 \
        --max-num-seqs "${max_num_seqs}" \
        --gpu-memory-utilization 0.95 \
        --async-scheduling \
        --output-json "${output_json}" \
        --model "${model}" 
        2>&1 | tee "${log_file}"
    
    echo "  Completed at: $(date)"
    echo ""
done
```
