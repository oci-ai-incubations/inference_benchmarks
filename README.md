# Inference Benchmarking Results with OCI AI Blueprints

## Motivation
We know firsthand that benchmarks for self-hosted models can be incredibly hard to come by. We also know that OCI's bare-metal GPU nodes are running at peak-performance as is our RoCE network and NVMe storage. Because of this, we can attribute any performance differences directly to how a framework runs on the hardware. By making these benchmarks available to the community - and showing you exactly how to reproduce them - we intend to foster an open community around inference performance while empowering you to make the right choice for your serving needs.

We also intend to show you how easy it is to self-host models using OCI AI Blueprints - from configurations utilizing multi-instance GPU (MIG) to single-node multi-GPU, to multi-node serving. It is just an API call away!

## Repository Description
This repo serves as a warehouse for benchmarking across different LLM models, GPUs, and serving frameworks using [OCI AI Blueprints](https://github.com/oracle-quickstart/oci-ai-blueprints). We will perform both online and offline benchmarks.

 - Online: Models are served over an openai compatible endpoint and benchmarks are launched using a modified llmperf from another machine in the same region as the GPU nodes
 - Offline: Benchmarks are run directly on the BM host

Offline benchmarks serve the purpose of ground truth performance - since there is no network overhead, this is a good proxy for raw framework performance on the node.

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

## Benchmarking Overview

Benchmarks are performed starting from the smallest tensor parallelism (tp) that will allow a model to fit on a single node up until maximum tp via doubling.

For example, if a model fits on 1 GPU with tp1, and the node has 8 GPUs, then we will run:
 - tp1
 - tp2
 - tp4
 - tp8

For multinode benchmarks utilizing RDMA networking backend, we will use pipeline parallelism (pp)+tp to distribute the model over both nodes. For now, we serve directly with the framework and don't support inference platforms such as NVIDIA dynamo or llm-d, although we may provide support for those in the future. If we do, they will show up here.

Below is the table of input / output token lengths we benchmark based on frequently asked use cases:

| Scenario | Input Token Length | Output Token Length |
| :------: | :----------------: | :-----------------: |
| Chatbot             | 128     | 128                 |
| Generation Heavy    | 128     | 2048                |
| Summarization Heavy | 2048    | 128                 |
| Long input / output | 2048    | 2048                |

For online benchmarks, we run these scenarios at the following concurrencies to test a variety of sustained loads:
 - 1
 - 5
 - 10
 - 25
 - 50

## Results and Visualizations

Both our offline and online benchmarking utilities push results to your MLFlow server that is deployed with your Blueprints platform. For online benchmarking, you can vizualize vLLM results in the pre-packaged vLLM Grafana dashboard in real time. Some of the more interesting online dashboard metrics that aren't directly captured in benchmarking output are:
  - Cache Utilization
  - Queue Time
  - Requests Prefill and Decode Time

These all provide further insights into optimization strategies that aren't directly apparent from benchmarking results.

## Collaborations
If you are a serving framework or platform developer or a model provider, please reach out to one of the emails below to discuss potential opportunities for collaboration.

## Questions or Issues?

If you are a customer and have an interest in trying these benchmarks yourself with Blueprints, reach out to:
 - [dennis.kennetz@oracle.com](mailto:dennis.kennetz@oracle.com)
 - [vishnu.kammari@oracle.com](mailto:vishnu.kammari@oracle.com)
 - [grant.neuman@oracle.com](mailto:grant.neuman@oracle.com)

If you are having issues specifically related to these benchmarks, please post an issue on this GitHub. For Blueprints related issues, please post an issue to [OCI AI Blueprints](https://github.com/oracle-quickstart/oci-ai-blueprints).

Lastly, we try to explain the rationale for our serving configurations inside the model / GPU / framework README, but if you have a question feel free to email or post an issue.