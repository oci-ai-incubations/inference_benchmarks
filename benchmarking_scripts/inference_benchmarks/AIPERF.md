# Benchmarking with NIM and aiperf

## Serve gpt-oss-120b

To deploy the gpt-oss-120b nim, follow

https://build.nvidia.com/openai/gpt-oss-120b/deploy

I recommend starting the docker container in a tmux session like:

```bash
tmux
export NGC_API_KEY=<NGC_API_KEY> # ping dennis if you don't have one
export LOCAL_NIM_CACHE=/mnt/nvme/models
docker run -it --rm --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -p 8000:8000 \
  nvcr.io/nim/openai/gpt-oss-120b:latest
```

When it is finished serving, you are ready to setup and run aiperf

## Setup and run aiperf locally (again with tmux):

```bash
tmux
mkdir genai_perf_results
cd genai_perf_results/
export RELEASE="25.05"
export WORKDIR=$PWD
docker run -it --net=host --gpus=all -v $WORKDIR:/workdir nvcr.io/nvidia/tritonserver:${RELEASE}-py3-sdk

cd /workdir
# Now inside container
pip install aiperf -t pythonpath

# Quick test
export PATH=$PWD/pythonpath/bin:$PATH
export PYTHONPATH=$PWD/pythonpath:$PYTHONPATH
export MODEL=openai/gpt-oss-120b
export INPUT_SEQUENCE_LENGTH=200
export INPUT_SEQUENCE_STD=10
export OUTPUT_SEQUENCE_LENGTH=200
export CONCURRENCY=10
export REQUEST_COUNT=$(($CONCURRENCY * 3))

aiperf profile \
  -m $MODEL \
  --endpoint-type chat \
  --streaming \
  -u localhost:8000 \
  --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
  --synthetic-input-tokens-stddev $INPUT_SEQUENCE_STD \
  --concurrency $CONCURRENCY \
  --request-count $REQUEST_COUNT \
  --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
  --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
  --extra-inputs ignore_eos:true \
  --tokenizer $MODEL \
  --profile-export-file ${INPUT_SEQUENCE_LENGTH}_${OUTPUT_SEQUENCE_LENGTH}.json

                                            NVIDIA AIPerf | LLM Metrics                                            
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃                               Metric ┃      avg ┃      min ┃      max ┃      p99 ┃      p90 ┃      p50 ┃    std ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│             Time to First Token (ms) │   129.82 │   106.44 │   163.39 │   163.37 │   162.41 │   118.92 │  21.77 │
│            Time to Second Token (ms) │     5.61 │     5.48 │     5.76 │     5.76 │     5.75 │     5.58 │   0.09 │
│      Time to First Output Token (ms) │   972.82 │   422.43 │ 1,262.38 │ 1,252.29 │ 1,190.15 │ 1,062.71 │ 244.74 │
│                 Request Latency (ms) │ 1,292.13 │ 1,236.06 │ 1,337.28 │ 1,337.24 │ 1,336.42 │ 1,282.19 │  28.24 │
│             Inter Token Latency (ms) │     6.03 │     5.88 │     6.18 │     6.18 │     6.14 │     6.03 │   0.11 │
│     Output Token Throughput Per User │   165.86 │   161.84 │   169.97 │   169.92 │   169.72 │   165.84 │   2.90 │
│                    (tokens/sec/user) │          │          │          │          │          │          │        │
│      Output Sequence Length (tokens) │   193.77 │   191.00 │   197.00 │   197.00 │   197.00 │   192.00 │   2.89 │
│       Input Sequence Length (tokens) │   201.43 │   177.00 │   218.00 │   217.71 │   211.40 │   203.00 │   9.67 │
│ Output Token Throughput (tokens/sec) │ 1,493.78 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│    Request Throughput (requests/sec) │     7.71 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
│             Request Count (requests) │    30.00 │      N/A │      N/A │      N/A │      N/A │      N/A │    N/A │
└──────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴────────┘

CLI Command: aiperf profile -m 'openai/gpt-oss-120b' --endpoint-type 'chat' --streaming -u 'localhost:8000' 
--synthetic-input-tokens-mean 200 --synthetic-input-tokens-stddev 10 --concurrency 10 --request-count 30 --output-tokens-mean 200 
--extra-inputs 'min_tokens:200' --extra-inputs 'ignore_eos:true' --tokenizer 'openai/gpt-oss-120b' --profile-export-file 
'200_200.json'
Benchmark Duration: 3.89 sec
CSV Export: /workdir/artifacts/openai_gpt-oss-120b-openai-chat-concurrency10/200_200.csv
JSON Export: /workdir/artifacts/openai_gpt-oss-120b-openai-chat-concurrency10/200_200.json
Log File: /workdir/artifacts/openai_gpt-oss-120b-openai-chat-concurrency10/logs/aiperf.log
```

## Run script for sweeps

In the same `/workdir` create a script called `sweeps.sh` with the following content:
```bash
declare -A useCases

# Populate the array with use case descriptions and their specified input/output lengths
useCases["chatbot"]="128/128"
useCases["Text classification"]="200/5"
useCases["heavyGeneration"]="128/2048"
useCases["summarization"]="2048/128"
useCases["largeBalanced"]="2048/2048"
useCases["longPrompt1"]="1024/4096"
useCases["longPrompt2"]="4096/1024"


# Function to execute AIPerf with the input/output lengths as arguments
runBenchmark() {
   local description="$1"
   local lengths="${useCases[$description]}"
   IFS='/' read -r inputLength outputLength <<< "$lengths"

   echo "Running AIPerf for $description with input length $inputLength and output length $outputLength"
   #Runs
   for concurrency in 1 2 4 8 16 32 64 128 256; do

       local INPUT_SEQUENCE_LENGTH=$inputLength
       local INPUT_SEQUENCE_STD=0
       local OUTPUT_SEQUENCE_LENGTH=$outputLength
       local CONCURRENCY=$concurrency
       local REQUEST_COUNT=$(($CONCURRENCY * 100))
       local MODEL=openai/gpt-oss-120b

       aiperf profile \
           -m $MODEL \
           --endpoint-type chat \
           --streaming \
           -u localhost:8000 \
           --synthetic-input-tokens-mean $INPUT_SEQUENCE_LENGTH \
           --synthetic-input-tokens-stddev $INPUT_SEQUENCE_STD \
           --concurrency $CONCURRENCY \
           --request-count $REQUEST_COUNT \
           --output-tokens-mean $OUTPUT_SEQUENCE_LENGTH \
           --extra-inputs min_tokens:$OUTPUT_SEQUENCE_LENGTH \
           --extra-inputs ignore_eos:true \
           --tokenizer $MODEL \
           --artifact-dir artifact/ISL${INPUT_SEQUENCE_LENGTH}_OSL${OUTPUT_SEQUENCE_LENGTH}/CON${CONCURRENCY}

   done
}

# Iterate over all defined use cases and run the benchmark script for each
for description in "${!useCases[@]}"; do
   runBenchmark "$description"
done
```

Then run all the use cases!
```bash
chmod +x sweeps.sh
./sweeps.sh
```
