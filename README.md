# Inference Benchmarking Results

This repo serves as a warehouse for benchmarking across different LLM models, GPUs, and serving frameworks using [OCI AI Blueprints](https://github.com/oracle-quickstart/oci-ai-blueprints). We will perform both online and offline benchmarks.

 - Online: Models are served over an endpoint and benchmarks are launched using a modified llmperf from another machine in the same region as the GPU nodes
 - Offline: Benchmarks are run directly on the BM host

Offline benchmarks serve the purpose of ground truth performance - since there is no network overhead, this is a good proxy for raw framework performance on the shape.

Online benchmarks allow testing of various input / output lengths to show how frameworks serve traffic over internet, and is much closer to a real world scenario.

Each directory will contain the blueprint required to reproduce the results shown in the directory.

Benchmarking Results will be stored according to the following directory structure:

```bash
models/
|
|__<model_name1>
   |
   |__<GPU_SHAPE1>
   |  |
   |  |__vllm
   |  |
   |  |__sglang
   |  |
   |  |__nim
   |   
   |__<GPU_SHAPE2>
   |
   |...
|
|__<model_name2>
   |
   |__<GPU_SHAPE1>
   |...
```

