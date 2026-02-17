# For running decode benchmarks

Git clone and checkout the following repo:

https://github.com/tlrmchlsmth/llm-d/tree/varun/rocm_wide_ep-benchmarking

- `cd guides/wide-ep-lws`
- update the NAMESPACE in the Justfile
- update the replica size in `decode_bench_qwen.yaml` based on how many nodes you want to run
- do `just start` to start the deployment
- look at the `parallel-guidellm` recipe in the Justfile and that is what is used to trigger the benchmarks


For information on runs done here, download [bench data](./OCI%20AMD%20-%20WIDEEP%20DECODE%20PERF%20RUNS.xlsx).
