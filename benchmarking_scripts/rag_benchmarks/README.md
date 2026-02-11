# NVIDIA RAG Blueprint Benchmarking Suite

This benchmarking suite provides comprehensive performance evaluation capabilities for the NVIDIA RAG Blueprint, measuring performance across multiple dimensions including latency, throughput, accuracy, and resource utilization.

## Features

- **Performance Benchmarking**: Measure latency, throughput, and resource usage
- **Accuracy Evaluation**: Evaluate response quality using ground truth data
- **Concurrent Testing**: Test system behavior under load
- **Visualization**: Generate plots and reports for analysis
- **Configurable Tests**: Multiple preset configurations for different scenarios
- **Detailed Logging**: Comprehensive metrics and error tracking

## Quick Start

### 1. Install Dependencies

```bash
# Method 1: Using the automatic installer (recommended)
./install_dependencies.sh

# Method 2: Manual installation
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install rouge-score sentence-transformers
```

### 2. Ensure RAG Services are Running

Make sure your NVIDIA RAG Blueprint services are running:

```bash
# Start vector database
docker compose -f ../deploy/compose/vectordb.yaml up -d

# Start NIMs (if using on-prem)
USERID=$(id -u) docker compose -f ../deploy/compose/nims.yaml up -d

# Start ingestion services
docker compose -f ../deploy/compose/docker-compose-ingestor-server.yaml up nv-ingest-ms-runtime redis -d
```

### 3. Run a Quick Benchmark

```bash
# Method 1: Using the automatic runner (recommended)
./run_benchmark_with_env.sh --preset quick

# Method 2: Manual activation
source .venv/bin/activate
cd benchmark
python run_benchmark.py --preset quick

# Run standard benchmark (100 requests, 10 concurrent)
./run_benchmark_with_env.sh --preset standard

# Run stress test (500 requests, 20 concurrent)
./run_benchmark_with_env.sh --preset stress
```

## Available Preset Configurations

| Preset | Requests | Concurrent | Use Case |
|--------|----------|------------|----------|
| `quick` | 20 | 3 | Development testing |
| `standard` | 100 | 10 | General performance evaluation |
| `stress` | 500 | 20 | Load testing |
| `accuracy` | 50 | 5 | Accuracy-focused evaluation |
| `performance` | 200 | 15 | Performance optimization |

## Detailed Usage

### Performance Benchmarking

The main benchmarking script (`rag_benchmark.py`) provides comprehensive performance testing:

```python
from rag_benchmark import RAGBenchmarker, BenchmarkConfig

# Create custom configuration
config = BenchmarkConfig(
    collection_name="my_test_collection",
    num_requests=100,
    concurrent_requests=10,
    test_documents=["path/to/document1.pdf", "path/to/document2.docx"],
    enable_reranker=True,
    vdb_top_k=100,
    reranker_top_k=10
)

# Run benchmark
benchmarker = RAGBenchmarker(config)
await benchmarker.run_full_benchmark_suite()
```

### Accuracy Evaluation

Use the accuracy evaluator to measure response quality:

```python
from accuracy_evaluator import AccuracyEvaluator

# Create evaluator
evaluator = AccuracyEvaluator("my_collection")

# Load evaluation data
samples = evaluator.load_evaluation_data("evaluation_data.json")

# Run evaluation
metrics = await evaluator.run_evaluation(samples)

# Generate report
evaluator.generate_report(metrics, "accuracy_results.json")
```

### Custom Configuration

Create custom benchmark configurations:

```bash
# Save current configuration
python run_benchmark.py --preset standard --save-config my_config.json

# Run with custom configuration
python run_benchmark.py --config my_config.json
```

## Metrics Measured

### Performance Metrics

- **Latency**: Average, P50, P95, P99 latencies
- **Throughput**: Requests per second (RPS)
- **Resource Usage**: CPU, memory, and GPU utilization
- **Token Processing**: Input/output tokens per request
- **Error Rates**: Success/failure rates

### Accuracy Metrics

- **Answer Similarity**: Semantic similarity between predicted and ground truth
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L F1 scores
- **Retrieval Metrics**: Precision and recall for document retrieval
- **Context Relevance**: Relevance of retrieved context
- **BERT Score**: Semantic similarity using BERT embeddings

## Output Files

The benchmarking suite generates several output files:

- `benchmark_results/`: Directory containing all results
- `benchmark_*.json`: Detailed metrics for each test
- `benchmark_plots_*.png`: Visualization plots
- `benchmark_summary_*.json`: Summary of all tests
- `accuracy_evaluation_results.json`: Accuracy evaluation results

## Configuration Options

### BenchmarkConfig Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `collection_name` | "benchmark_collection" | Vector database collection name |
| `num_requests` | 100 | Total number of requests to send |
| `concurrent_requests` | 10 | Number of concurrent requests |
| `warmup_requests` | 10 | Number of warmup requests |
| `timeout_seconds` | 30 | Request timeout |
| `enable_reranker` | True | Enable reranking |
| `enable_vlm` | False | Enable vision-language model |
| `vdb_top_k` | 100 | Vector database top-k |
| `reranker_top_k` | 10 | Reranker top-k |
| `chunk_size` | 512 | Document chunk size |
| `chunk_overlap` | 150 | Chunk overlap |

## Advanced Usage

### Custom Test Queries

```python
config = BenchmarkConfig(
    test_queries=[
        "What is the main topic?",
        "Can you summarize the key points?",
        "What are the main findings?",
        # Add your custom queries
    ]
)
```

### Resource Monitoring

The benchmark automatically monitors system resources:

```python
# Access resource metrics
metrics = await benchmarker.concurrent_benchmark()
print(f"CPU Usage: {metrics.cpu_usage_percent:.1f}%")
print(f"Memory Usage: {metrics.memory_usage_mb:.1f}%")
```

### Custom Evaluation Data

Create evaluation data for accuracy testing:

```json
[
  {
    "question": "What is the main topic?",
    "ground_truth_answer": "The document discusses RAG systems.",
    "expected_context": ["RAG systems", "retrieval", "generation"],
    "category": "general",
    "difficulty": "easy"
  }
]
```

## Troubleshooting

### Common Issues

1. **Services Not Running**: Ensure all RAG services are healthy
   ```bash
   docker ps
   # Check that all containers are running
   ```

2. **Connection Errors**: Verify service endpoints
   ```bash
   # Check Milvus connection
   curl http://localhost:19530/health
   ```

3. **Memory Issues**: Reduce concurrent requests or document size
   ```python
   config = BenchmarkConfig(
       concurrent_requests=5,  # Reduce from 10
       chunk_size=256  # Reduce from 512
   )
   ```

4. **Timeout Errors**: Increase timeout or reduce request complexity
   ```python
   config = BenchmarkConfig(
       timeout_seconds=60,  # Increase from 30
       vdb_top_k=50  # Reduce from 100
   )
   ```

### Performance Optimization

1. **GPU Utilization**: Monitor GPU usage and adjust batch sizes
2. **Memory Management**: Tune chunk sizes and concurrent requests
3. **Network Latency**: Use local services when possible
4. **Database Optimization**: Tune vector database parameters

## Integration with CI/CD

Add benchmarking to your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run RAG Benchmark
  run: |
    python benchmark/run_benchmark.py --preset standard
    python benchmark/accuracy_evaluator.py
```

## Contributing

To extend the benchmarking suite:

1. Add new metrics to `BenchmarkMetrics` class
2. Implement new benchmark types in `RAGBenchmarker`
3. Create custom evaluation metrics in `AccuracyEvaluator`
4. Add visualization functions for new metrics

## License

This benchmarking suite follows the same license as the NVIDIA RAG Blueprint. 