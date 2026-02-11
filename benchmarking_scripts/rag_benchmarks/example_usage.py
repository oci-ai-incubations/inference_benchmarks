#!/usr/bin/env python3
"""
Example usage of the NVIDIA RAG Blueprint Benchmarking Suite

This script demonstrates how to use the benchmarking tools with different
configurations and scenarios.
"""

import asyncio
import json
from pathlib import Path
from rag_benchmark import RAGBenchmarker, BenchmarkConfig
from accuracy_evaluator import AccuracyEvaluator


async def example_performance_benchmark():
    """Example of running performance benchmarks"""
    print("=== Performance Benchmarking Example ===")
    
    # Create a custom configuration for performance testing
    config = BenchmarkConfig(
        collection_name="example_performance_test",
        num_requests=50,
        concurrent_requests=5,
        warmup_requests=5,
        test_documents=[
            "../data/multimodal/woods_frost.docx",
            "../data/multimodal/multimodal_test.pdf"
        ],
        enable_reranker=True,
        enable_vlm=False,
        vdb_top_k=100,
        reranker_top_k=10,
        output_dir="example_results/performance"
    )
    
    # Create and run benchmarker
    benchmarker = RAGBenchmarker(config)
    await benchmarker.run_full_benchmark_suite()


async def example_accuracy_evaluation():
    """Example of running accuracy evaluation"""
    print("\n=== Accuracy Evaluation Example ===")
    
    # Create evaluator
    evaluator = AccuracyEvaluator("example_performance_test")
    
    # Create sample evaluation data
    samples = evaluator.create_sample_evaluation_data()
    
    # Run evaluation
    metrics = await evaluator.run_evaluation(samples)
    
    # Generate report
    evaluator.generate_report(metrics, "example_results/accuracy_evaluation.json")


async def example_comparison_benchmark():
    """Example of comparing different configurations"""
    print("\n=== Configuration Comparison Example ===")
    
    # Test different configurations
    configurations = {
        "with_reranker": BenchmarkConfig(
            collection_name="comparison_test",
            num_requests=30,
            concurrent_requests=3,
            enable_reranker=True,
            vdb_top_k=100,
            reranker_top_k=10,
            output_dir="example_results/with_reranker"
        ),
        "without_reranker": BenchmarkConfig(
            collection_name="comparison_test",
            num_requests=30,
            concurrent_requests=3,
            enable_reranker=False,
            vdb_top_k=10,
            output_dir="example_results/without_reranker"
        ),
        "high_concurrency": BenchmarkConfig(
            collection_name="comparison_test",
            num_requests=30,
            concurrent_requests=10,
            enable_reranker=True,
            vdb_top_k=50,
            reranker_top_k=5,
            output_dir="example_results/high_concurrency"
        )
    }
    
    results = {}
    
    for name, config in configurations.items():
        print(f"\nTesting configuration: {name}")
        benchmarker = RAGBenchmarker(config)
        
        # Run only concurrent benchmark for comparison
        metrics = await benchmarker.concurrent_benchmark()
        results[name] = metrics
        
        # Generate report
        benchmarker.generate_report(metrics)
    
    # Compare results
    print("\n=== Configuration Comparison Results ===")
    print(f"{'Configuration':<20} {'Avg Latency (ms)':<15} {'Throughput (RPS)':<15} {'Success Rate (%)':<15}")
    print("-" * 70)
    
    for name, metrics in results.items():
        success_rate = (metrics.successful_requests / metrics.total_requests) * 100
        print(f"{name:<20} {metrics.avg_latency_ms:<15.1f} {metrics.throughput_rps:<15.2f} {success_rate:<15.1f}")


async def example_custom_queries():
    """Example of using custom test queries"""
    print("\n=== Custom Queries Example ===")
    
    # Define custom queries for your specific use case
    custom_queries = [
        "What are the main features of this system?",
        "How does the system handle errors?",
        "What are the performance characteristics?",
        "What are the system requirements?",
        "How can I optimize the system?",
        "What are the limitations?",
        "How does it compare to other solutions?",
        "What are the best practices?",
        "How do I troubleshoot issues?",
        "What are the future improvements planned?"
    ]
    
    config = BenchmarkConfig(
        collection_name="custom_queries_test",
        num_requests=40,
        concurrent_requests=4,
        test_queries=custom_queries,
        test_documents=["../data/multimodal/woods_frost.docx"],
        output_dir="example_results/custom_queries"
    )
    
    benchmarker = RAGBenchmarker(config)
    await benchmarker.run_full_benchmark_suite()


async def example_resource_monitoring():
    """Example of monitoring system resources during benchmarks"""
    print("\n=== Resource Monitoring Example ===")
    
    import psutil
    
    # Monitor system resources before benchmark
    print("System resources before benchmark:")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent:.1f}%")
    
    # Run benchmark with resource monitoring
    config = BenchmarkConfig(
        collection_name="resource_test",
        num_requests=20,
        concurrent_requests=2,
        output_dir="example_results/resource_monitoring"
    )
    
    benchmarker = RAGBenchmarker(config)
    metrics = await benchmarker.concurrent_benchmark()
    
    # Display resource usage during benchmark
    print(f"\nResource usage during benchmark:")
    print(f"CPU Usage: {metrics.cpu_usage_percent:.1f}%")
    print(f"Memory Usage: {metrics.memory_usage_mb:.1f}%")
    
    # Monitor system resources after benchmark
    print(f"\nSystem resources after benchmark:")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    print(f"Memory Usage: {psutil.virtual_memory().percent:.1f}%")


async def example_save_and_load_config():
    """Example of saving and loading benchmark configurations"""
    print("\n=== Configuration Management Example ===")
    
    # Create a custom configuration
    config = BenchmarkConfig(
        collection_name="config_test",
        num_requests=25,
        concurrent_requests=3,
        enable_reranker=True,
        vdb_top_k=75,
        reranker_top_k=8,
        test_documents=["../data/multimodal/woods_frost.docx"]
    )
    
    # Save configuration to file
    config_dict = {
        "collection_name": config.collection_name,
        "num_requests": config.num_requests,
        "concurrent_requests": config.concurrent_requests,
        "enable_reranker": config.enable_reranker,
        "vdb_top_k": config.vdb_top_k,
        "reranker_top_k": config.reranker_top_k,
        "test_documents": config.test_documents
    }
    
    with open("example_results/saved_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print("Configuration saved to example_results/saved_config.json")
    
    # Load configuration from file
    with open("example_results/saved_config.json", "r") as f:
        loaded_config_dict = json.load(f)
    
    loaded_config = BenchmarkConfig(**loaded_config_dict)
    loaded_config.output_dir = "example_results/loaded_config"
    
    print("Configuration loaded and running benchmark...")
    
    # Run benchmark with loaded configuration
    benchmarker = RAGBenchmarker(loaded_config)
    await benchmarker.concurrent_benchmark()


async def main():
    """Run all examples"""
    print("NVIDIA RAG Blueprint Benchmarking Suite - Examples")
    print("=" * 60)
    
    # Create output directory
    Path("example_results").mkdir(exist_ok=True)
    
    try:
        # Run examples
        await example_performance_benchmark()
        await example_accuracy_evaluation()
        await example_comparison_benchmark()
        await example_custom_queries()
        await example_resource_monitoring()
        await example_save_and_load_config()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Check the 'example_results' directory for output files.")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure your RAG services are running before executing examples.")


if __name__ == "__main__":
    asyncio.run(main()) 