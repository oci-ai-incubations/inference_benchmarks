#!/usr/bin/env python3
"""
Simple benchmark runner for NVIDIA RAG Blueprint

This script provides an easy way to run benchmarks with different configurations
and compare results.
"""

import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any

from rag_benchmark import RAGBenchmarker, BenchmarkConfig


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)


def save_config(config: BenchmarkConfig, filename: str):
    """Save configuration to JSON file"""
    config_dict = {
        "collection_name": config.collection_name,
        "num_requests": config.num_requests,
        "concurrent_requests": config.concurrent_requests,
        "warmup_requests": config.warmup_requests,
        "timeout_seconds": config.timeout_seconds,
        "test_queries": config.test_queries,
        "test_documents": config.test_documents,
        "enable_reranker": config.enable_reranker,
        "enable_vlm": config.enable_vlm,
        "vdb_top_k": config.vdb_top_k,
        "reranker_top_k": config.reranker_top_k,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "output_dir": config.output_dir,
        "save_detailed_logs": config.save_detailed_logs,
        "generate_plots": config.generate_plots
    }
    
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)


async def run_benchmark(config: BenchmarkConfig):
    """Run benchmark with given configuration"""
    benchmarker = RAGBenchmarker(config)
    await benchmarker.run_full_benchmark_suite()


def create_preset_configs():
    """Create preset benchmark configurations"""
    configs = {
        "quick": BenchmarkConfig(
            collection_name="benchmark_quick",
            num_requests=20,
            concurrent_requests=3,
            warmup_requests=3,
            test_documents=["../data/multimodal/woods_frost.docx"],
            output_dir="./benchmark_results/quick"
        ),
        "standard": BenchmarkConfig(
            collection_name="benchmark_standard",
            num_requests=100,
            concurrent_requests=10,
            warmup_requests=10,
            test_documents=[
                "../data/multimodal/woods_frost.docx",
                "../data/multimodal/multimodal_test.pdf"
            ],
            output_dir="./benchmark_results/standard"
        ),
        "stress": BenchmarkConfig(
            collection_name="benchmark_stress",
            num_requests=500,
            concurrent_requests=20,
            warmup_requests=20,
            test_documents=[
                "../data/multimodal/woods_frost.docx",
                "../data/multimodal/multimodal_test.pdf",
                "../data/multimodal/embedded_table.pdf"
            ],
            output_dir="./benchmark_results/stress"
        ),
        "accuracy": BenchmarkConfig(
            collection_name="benchmark_accuracy",
            num_requests=50,
            concurrent_requests=5,
            warmup_requests=5,
            enable_reranker=True,
            enable_vlm=False,
            vdb_top_k=200,
            reranker_top_k=20,
            output_dir="./benchmark_results/accuracy"
        ),
        "performance": BenchmarkConfig(
            collection_name="benchmark_performance",
            num_requests=200,
            concurrent_requests=15,
            warmup_requests=10,
            enable_reranker=False,  # Disable for performance testing
            enable_vlm=False,
            vdb_top_k=50,
            output_dir="./benchmark_results/performance"
        )
    }
    return configs


async def main():
    parser = argparse.ArgumentParser(description="Run NVIDIA RAG Blueprint benchmarks")
    parser.add_argument(
        "--preset", 
        choices=["quick", "standard", "stress", "accuracy", "performance"],
        default="standard",
        help="Use preset configuration"
    )
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to custom configuration JSON file"
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save current configuration to JSON file"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available preset configurations"
    )
    
    args = parser.parse_args()
    
    if args.list_presets:
        print("Available preset configurations:")
        configs = create_preset_configs()
        for name, config in configs.items():
            print(f"  {name}: {config.num_requests} requests, {config.concurrent_requests} concurrent")
        return
    
    # Load configuration
    if args.config:
        config_dict = load_config(args.config)
        config = BenchmarkConfig(**config_dict)
    else:
        configs = create_preset_configs()
        config = configs[args.preset]
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, args.save_config)
        print(f"Configuration saved to {args.save_config}")
    
    # Run benchmark
    print(f"Running benchmark with preset: {args.preset}")
    await run_benchmark(config)


if __name__ == "__main__":
    asyncio.run(main()) 