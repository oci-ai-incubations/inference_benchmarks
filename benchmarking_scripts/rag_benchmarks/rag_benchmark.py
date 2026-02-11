#!/usr/bin/env python3
"""
NVIDIA RAG Blueprint Benchmarking Suite

This script provides comprehensive benchmarking capabilities for the NVIDIA RAG workflow,
measuring performance across multiple dimensions including latency, throughput, accuracy,
and resource utilization.
"""

import asyncio
import json
import logging
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from nvidia_rag import NvidiaRAG, NvidiaRAGIngestor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics"""
    test_name: str
    timestamp: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_rps: float
    avg_tokens_per_request: float
    total_tokens_processed: int
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    accuracy_score: Optional[float] = None
    retrieval_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None
    context_relevance_score: Optional[float] = None
    response_quality_score: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    detailed_latencies: List[float] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    collection_name: str = "benchmark_collection"
    num_requests: int = 100
    concurrent_requests: int = 10
    warmup_requests: int = 10
    timeout_seconds: int = 30
    test_queries: List[str] = field(default_factory=list)
    test_documents: List[str] = field(default_factory=list)
    enable_reranker: bool = True
    enable_vlm: bool = False
    vdb_top_k: int = 100
    reranker_top_k: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 150
    output_dir: str = "benchmark_results"
    save_detailed_logs: bool = True
    generate_plots: bool = True


class RAGBenchmarker:
    """Comprehensive RAG benchmarking suite"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.rag = NvidiaRAG()
        self.ingestor = NvidiaRAGIngestor()
        self.results: List[BenchmarkMetrics] = []
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Default test queries if none provided
        if not config.test_queries:
            self.config.test_queries = [
                "What is the main topic of this document?",
                "Can you summarize the key points?",
                "What are the main findings?",
                "How does this relate to the overall context?",
                "What are the implications of these results?",
                "Can you provide specific examples mentioned?",
                "What methodology was used?",
                "What are the limitations discussed?",
                "How does this compare to other approaches?",
                "What are the next steps recommended?"
            ]
    
    async def setup_test_environment(self) -> bool:
        """Setup the test environment with documents"""
        try:
            console.print("[bold blue]Setting up test environment...[/bold blue]")
            
            # Create collection
            response = self.ingestor.create_collection(
                collection_name=self.config.collection_name,
                vdb_endpoint="http://localhost:19530"
            )
            console.print(f"✓ Collection created: {response}")
            
            # Upload test documents if provided
            if self.config.test_documents:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console
                ) as progress:
                    task = progress.add_task("Uploading test documents...", total=len(self.config.test_documents))
                    
                    response = await self.ingestor.upload_documents(
                        collection_name=self.config.collection_name,
                        vdb_endpoint="http://localhost:19530",
                        blocking=True,
                        split_options={
                            "chunk_size": self.config.chunk_size,
                            "chunk_overlap": self.config.chunk_overlap
                        },
                        filepaths=self.config.test_documents,
                        generate_summary=False
                    )
                    progress.update(task, completed=len(self.config.test_documents))
                
                console.print(f"✓ Documents uploaded: {response}")
            
            # Check health
            health_status = await self.rag.health(check_dependencies=True)
            if health_status.get("status") == "healthy" or health_status.get("message") == "Service is up.":
                console.print("✓ All services healthy")
                return True
            else:
                console.print(f"✗ Health check failed: {health_status}")
                return False
                
        except Exception as e:
            console.print(f"✗ Setup failed: {str(e)}")
            return False
    
    async def warmup(self):
        """Perform warmup requests to stabilize the system"""
        console.print(f"[bold yellow]Performing {self.config.warmup_requests} warmup requests...[/bold yellow]")
        
        warmup_queries = self.config.test_queries[:min(3, len(self.config.test_queries))]
        
        for i in range(self.config.warmup_requests):
            query = warmup_queries[i % len(warmup_queries)]
            try:
                await self.rag.generate(
                    messages=[{"role": "user", "content": query}],
                    use_knowledge_base=True,
                    collection_names=[self.config.collection_name]
                )
            except Exception as e:
                logger.warning(f"Warmup request {i+1} failed: {e}")
    
    async def single_request_benchmark(self, query: str) -> Tuple[float, int, Optional[str]]:
        """Execute a single request and measure performance"""
        start_time = time.time()
        tokens_processed = 0
        error = None
        
        try:
            response_generator = self.rag.generate(
                messages=[{"role": "user", "content": query}],
                use_knowledge_base=True,
                collection_names=[self.config.collection_name],
                reranker_top_k=self.config.reranker_top_k if self.config.enable_reranker else None,
                vdb_top_k=self.config.vdb_top_k
            )
            
            # Process streaming response
            async for chunk in response_generator:
                if chunk.startswith("data: "):
                    chunk = chunk[len("data: "):].strip()
                if not chunk:
                    continue
                
                try:
                    data = json.loads(chunk)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "content" in delta:
                            tokens_processed += 1
                except json.JSONDecodeError:
                    continue
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            return latency, tokens_processed, None
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            error = str(e)
            return latency, tokens_processed, error
    
    async def concurrent_benchmark(self) -> BenchmarkMetrics:
        """Run concurrent benchmark test"""
        console.print(f"[bold green]Running concurrent benchmark with {self.config.num_requests} requests...[/bold green]")
        
        start_time = time.time()
        latencies = []
        token_counts = []
        errors = []
        successful_requests = 0
        
        # Get system resource usage before test
        initial_memory = psutil.virtual_memory().percent
        initial_cpu = psutil.cpu_percent(interval=1)
        
        # Prepare queries (cycle through test queries)
        queries = []
        for i in range(self.config.num_requests):
            query = self.config.test_queries[i % len(self.config.test_queries)]
            queries.append(query)
        
        # Execute concurrent requests
        semaphore = asyncio.Semaphore(self.config.concurrent_requests)
        
        async def execute_request(query: str):
            async with semaphore:
                return await self.single_request_benchmark(query)
        
        tasks = [execute_request(query) for query in queries]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Executing requests...", total=len(tasks))
            
            for coro in asyncio.as_completed(tasks):
                latency, tokens, error = await coro
                latencies.append(latency)
                token_counts.append(tokens)
                
                if error:
                    errors.append(error)
                else:
                    successful_requests += 1
                
                progress.update(task, advance=1)
        
        # Get system resource usage after test
        final_memory = psutil.virtual_memory().percent
        final_cpu = psutil.cpu_percent(interval=1)
        
        # Calculate metrics
        total_time = time.time() - start_time
        throughput = self.config.num_requests / total_time
        
        metrics = BenchmarkMetrics(
            test_name="concurrent_benchmark",
            timestamp=datetime.now(),
            total_requests=self.config.num_requests,
            successful_requests=successful_requests,
            failed_requests=len(errors),
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=np.percentile(latencies, 50),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            throughput_rps=throughput,
            avg_tokens_per_request=statistics.mean(token_counts) if token_counts else 0,
            total_tokens_processed=sum(token_counts),
            memory_usage_mb=(initial_memory + final_memory) / 2,
            cpu_usage_percent=(initial_cpu + final_cpu) / 2,
            errors=errors,
            detailed_latencies=latencies
        )
        
        return metrics
    
    async def latency_benchmark(self) -> BenchmarkMetrics:
        """Run latency-focused benchmark with different query types"""
        console.print("[bold green]Running latency benchmark...[/bold green]")
        
        query_types = {
            "simple": "What is this document about?",
            "complex": "Can you provide a detailed analysis of the main findings and their implications?",
            "specific": "What specific examples or data points are mentioned in the document?",
            "comparative": "How does this approach compare to alternative methods discussed?",
            "analytical": "What are the strengths and limitations of the methodology used?"
        }
        
        latencies_by_type = {}
        total_latencies = []
        total_tokens = 0
        errors = []
        successful_requests = 0
        
        for query_type, query in query_types.items():
            type_latencies = []
            for _ in range(self.config.num_requests // len(query_types)):
                latency, tokens, error = await self.single_request_benchmark(query)
                type_latencies.append(latency)
                total_latencies.append(latency)
                total_tokens += tokens
                
                if error:
                    errors.append(f"{query_type}: {error}")
                else:
                    successful_requests += 1
            
            latencies_by_type[query_type] = type_latencies
        
        metrics = BenchmarkMetrics(
            test_name="latency_benchmark",
            timestamp=datetime.now(),
            total_requests=self.config.num_requests,
            successful_requests=successful_requests,
            failed_requests=len(errors),
            avg_latency_ms=statistics.mean(total_latencies),
            p50_latency_ms=np.percentile(total_latencies, 50),
            p95_latency_ms=np.percentile(total_latencies, 95),
            p99_latency_ms=np.percentile(total_latencies, 99),
            min_latency_ms=min(total_latencies),
            max_latency_ms=max(total_latencies),
            throughput_rps=successful_requests / (sum(total_latencies) / 1000),
            avg_tokens_per_request=total_tokens / successful_requests if successful_requests > 0 else 0,
            total_tokens_processed=total_tokens,
            memory_usage_mb=psutil.virtual_memory().percent,
            cpu_usage_percent=psutil.cpu_percent(interval=1),
            errors=errors,
            detailed_latencies=total_latencies
        )
        
        # Store query type breakdown
        metrics.query_type_latencies = latencies_by_type
        
        return metrics
    
    async def accuracy_benchmark(self) -> BenchmarkMetrics:
        """Run accuracy-focused benchmark (requires ground truth)"""
        console.print("[bold green]Running accuracy benchmark...[/bold green]")
        
        # This is a placeholder for accuracy testing
        # In a real implementation, you would need ground truth data
        # and evaluation metrics like ROUGE, BLEU, or custom relevance scores
        
        # For now, we'll run a basic test and note that accuracy evaluation needs setup
        metrics = await self.concurrent_benchmark()
        metrics.test_name = "accuracy_benchmark"
        metrics.accuracy_score = None  # Would be calculated with ground truth
        metrics.retrieval_precision = None
        metrics.retrieval_recall = None
        metrics.context_relevance_score = None
        metrics.response_quality_score = None
        
        console.print("[yellow]Note: Accuracy evaluation requires ground truth data setup[/yellow]")
        
        return metrics
    
    def generate_report(self, metrics: BenchmarkMetrics):
        """Generate comprehensive benchmark report"""
        console.print(f"\n[bold blue]Benchmark Report: {metrics.test_name}[/bold blue]")
        
        # Summary table
        table = Table(title="Performance Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Requests", str(metrics.total_requests))
        table.add_row("Successful Requests", str(metrics.successful_requests))
        table.add_row("Failed Requests", str(metrics.failed_requests))
        table.add_row("Success Rate", f"{(metrics.successful_requests/metrics.total_requests)*100:.2f}%")
        table.add_row("Average Latency", f"{metrics.avg_latency_ms:.2f} ms")
        table.add_row("P95 Latency", f"{metrics.p95_latency_ms:.2f} ms")
        table.add_row("P99 Latency", f"{metrics.p99_latency_ms:.2f} ms")
        table.add_row("Throughput", f"{metrics.throughput_rps:.2f} RPS")
        table.add_row("Avg Tokens/Request", f"{metrics.avg_tokens_per_request:.1f}")
        table.add_row("Memory Usage", f"{metrics.memory_usage_mb:.1f}%")
        table.add_row("CPU Usage", f"{metrics.cpu_usage_percent:.1f}%")
        
        console.print(table)
        
        # Save detailed results
        if self.config.save_detailed_logs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.output_dir}/benchmark_{metrics.test_name}_{timestamp}.json"
            
            # Convert to dict for JSON serialization
            result_dict = {
                "test_name": metrics.test_name,
                "timestamp": metrics.timestamp.isoformat(),
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "avg_latency_ms": metrics.avg_latency_ms,
                "p50_latency_ms": metrics.p50_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
                "p99_latency_ms": metrics.p99_latency_ms,
                "min_latency_ms": metrics.min_latency_ms,
                "max_latency_ms": metrics.max_latency_ms,
                "throughput_rps": metrics.throughput_rps,
                "avg_tokens_per_request": metrics.avg_tokens_per_request,
                "total_tokens_processed": metrics.total_tokens_processed,
                "memory_usage_mb": metrics.memory_usage_mb,
                "cpu_usage_percent": metrics.cpu_usage_percent,
                "gpu_usage_percent": metrics.gpu_usage_percent,
                "accuracy_score": metrics.accuracy_score,
                "retrieval_precision": metrics.retrieval_precision,
                "retrieval_recall": metrics.retrieval_recall,
                "context_relevance_score": metrics.context_relevance_score,
                "response_quality_score": metrics.response_quality_score,
                "errors": metrics.errors,
                "detailed_latencies": metrics.detailed_latencies
            }
            
            with open(filename, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            console.print(f"✓ Detailed results saved to: {filename}")
        
        # Generate plots
        if self.config.generate_plots and metrics.detailed_latencies:
            self.generate_plots(metrics)
    
    def generate_plots(self, metrics: BenchmarkMetrics):
        """Generate visualization plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'RAG Benchmark Results: {metrics.test_name}', fontsize=16)
        
        # Latency distribution
        axes[0, 0].hist(metrics.detailed_latencies, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(metrics.avg_latency_ms, color='red', linestyle='--', label=f'Mean: {metrics.avg_latency_ms:.1f}ms')
        axes[0, 0].axvline(metrics.p95_latency_ms, color='orange', linestyle='--', label=f'P95: {metrics.p95_latency_ms:.1f}ms')
        axes[0, 0].set_xlabel('Latency (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Latency Distribution')
        axes[0, 0].legend()
        
        # Latency over time
        axes[0, 1].plot(metrics.detailed_latencies, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Request Number')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].set_title('Latency Over Time')
        
        # Box plot
        axes[1, 0].boxplot(metrics.detailed_latencies)
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].set_title('Latency Box Plot')
        
        # Performance summary
        summary_data = {
            'Metric': ['Avg', 'P50', 'P95', 'P99', 'Min', 'Max'],
            'Latency (ms)': [
                metrics.avg_latency_ms,
                metrics.p50_latency_ms,
                metrics.p95_latency_ms,
                metrics.p99_latency_ms,
                metrics.min_latency_ms,
                metrics.max_latency_ms
            ]
        }
        
        axes[1, 1].bar(summary_data['Metric'], summary_data['Latency (ms)'], color='lightcoral')
        axes[1, 1].set_ylabel('Latency (ms)')
        axes[1, 1].set_title('Latency Summary')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plot_filename = f"{self.config.output_dir}/benchmark_plots_{metrics.test_name}_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        console.print(f"✓ Plots saved to: {plot_filename}")
        plt.close()
    
    async def run_full_benchmark_suite(self):
        """Run the complete benchmark suite"""
        console.print(Panel.fit("[bold blue]NVIDIA RAG Blueprint Benchmark Suite[/bold blue]"))
        
        # Setup
        if not await self.setup_test_environment():
            console.print("[red]Failed to setup test environment. Exiting.[/red]")
            return
        
        # Warmup
        await self.warmup()
        
        # Run different benchmark types
        benchmark_types = [
            ("Concurrent Benchmark", self.concurrent_benchmark),
            ("Latency Benchmark", self.latency_benchmark),
            ("Accuracy Benchmark", self.accuracy_benchmark)
        ]
        
        for name, benchmark_func in benchmark_types:
            console.print(f"\n[bold cyan]Running {name}...[/bold cyan]")
            try:
                metrics = await benchmark_func()
                self.results.append(metrics)
                self.generate_report(metrics)
            except Exception as e:
                console.print(f"[red]Error in {name}: {str(e)}[/red]")
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate a summary report of all benchmark results"""
        if not self.results:
            return
        
        console.print("\n[bold blue]Summary Report[/bold blue]")
        
        summary_table = Table(title="Benchmark Summary")
        summary_table.add_column("Test", style="cyan")
        summary_table.add_column("Avg Latency (ms)", style="green")
        summary_table.add_column("P95 Latency (ms)", style="yellow")
        summary_table.add_column("Throughput (RPS)", style="magenta")
        summary_table.add_column("Success Rate (%)", style="blue")
        
        for result in self.results:
            success_rate = (result.successful_requests / result.total_requests) * 100
            summary_table.add_row(
                result.test_name,
                f"{result.avg_latency_ms:.1f}",
                f"{result.p95_latency_ms:.1f}",
                f"{result.throughput_rps:.2f}",
                f"{success_rate:.1f}"
            )
        
        console.print(summary_table)
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"{self.config.output_dir}/benchmark_summary_{timestamp}.json"
        
        summary_data = {
            "timestamp": timestamp,
            "total_tests": len(self.results),
            "results": [
                {
                    "test_name": r.test_name,
                    "avg_latency_ms": r.avg_latency_ms,
                    "p95_latency_ms": r.p95_latency_ms,
                    "throughput_rps": r.throughput_rps,
                    "success_rate": (r.successful_requests / r.total_requests) * 100
                }
                for r in self.results
            ]
        }
        
        with open(summary_filename, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        console.print(f"✓ Summary report saved to: {summary_filename}")


async def main():
    """Main function to run the benchmark suite"""
    # Configuration
    config = BenchmarkConfig(
        collection_name="benchmark_test",
        num_requests=50,  # Adjust based on your needs
        concurrent_requests=5,  # Adjust based on your system
        warmup_requests=5,
        timeout_seconds=30,
        test_documents=[
            "../data/multimodal/woods_frost.docx",
            "../data/multimodal/multimodal_test.pdf"
        ],
        enable_reranker=True,
        enable_vlm=False,
        vdb_top_k=100,
        reranker_top_k=10,
        output_dir="benchmark_results",
        save_detailed_logs=True,
        generate_plots=True
    )
    
    # Create and run benchmarker
    benchmarker = RAGBenchmarker(config)
    await benchmarker.run_full_benchmark_suite()


if __name__ == "__main__":
    asyncio.run(main()) 