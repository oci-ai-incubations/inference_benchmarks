# Setup and run

https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-gpt-oss-on-trtllm.html

## Download the model
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install huggingface_hub

hf auth login
<hf token>

hf download nvidia/Llama-3.3-70B-Instruct-NVFP4 --local-dir /mnt/nvme/models/nvidia/Llama-3.3-70B-Instruct-NVFP4 --max-workers 32
hf download openai/gpt-oss-120b --local-dir /mnt/nvme/models/openai/gpt-oss-120b --max-workers 32
```

## Download the docker container

Change the vllm version to the latest which can be found by going to their GitHub / releases

```bash
### Not Grace Blackwell
docker pull --platform linux/amd64 vllm/vllm-openai:v0.17.0

### Grace Blackwell
docker pull --platform linux/aarch64 vllm/vllm-openai:v0.17.0
```

## Running the container
After the image is pulled, we can run the relevant scripts, but first edit the commands to match what vLLM recommends for the serving which can generally be found here:

- [gpt-oss-120b](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#recipe-for-nvidia-blackwell-hopper-hardware)
- [llama-3.3-70b-instruct](https://docs.vllm.ai/projects/recipes/en/latest/Llama/Llama3.3-70B.html)

```bash
docker run -v $PWD:/workdir -v /mnt/nvme/models:/models --workdir /workdir --shm-size 32G --ipc=host --gpus all --entrypoint "/bin/bash" --rm -it vllm/vllm-openai:v0.17.0

# In the container