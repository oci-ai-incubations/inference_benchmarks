#!/usr/bin/env python3
"""
Accuracy Evaluation for NVIDIA RAG Blueprint

This script provides accuracy evaluation capabilities for RAG systems using
ground truth data and various evaluation metrics.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from nvidia_rag import NvidiaRAG

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """Single evaluation sample with question and ground truth"""
    question: str
    ground_truth_answer: str
    expected_context: Optional[List[str]] = None
    expected_documents: Optional[List[str]] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None


@dataclass
class EvaluationResult:
    """Results for a single evaluation sample"""
    sample: EvaluationSample
    predicted_answer: str
    retrieved_context: List[str]
    retrieved_documents: List[str]
    answer_similarity: float
    context_relevance: float
    rouge_scores: Dict[str, float]
    bert_score: float
    retrieval_precision: float
    retrieval_recall: float
    latency_ms: float


@dataclass
class AccuracyMetrics:
    """Overall accuracy metrics"""
    total_samples: int
    avg_answer_similarity: float
    avg_context_relevance: float
    avg_rouge_f1: float
    avg_bert_score: float
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_latency_ms: float
    category_breakdown: Dict[str, Dict[str, float]]
    difficulty_breakdown: Dict[str, Dict[str, float]]
    detailed_results: List[EvaluationResult]


class AccuracyEvaluator:
    """Accuracy evaluation for RAG systems"""
    
    def __init__(self, collection_name: str = "benchmark_collection"):
        self.collection_name = collection_name
        self.rag = NvidiaRAG()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def create_sample_evaluation_data(self) -> List[EvaluationSample]:
        """Create sample evaluation data for testing"""
        # This is a placeholder - in practice, you would load real evaluation data
        samples = [
            EvaluationSample(
                question="What is the main topic of the document?",
                ground_truth_answer="The document discusses the implementation of RAG systems and their performance characteristics.",
                category="general",
                difficulty="easy"
            ),
            EvaluationSample(
                question="What are the key performance metrics mentioned?",
                ground_truth_answer="The key metrics include latency, throughput, accuracy, and resource utilization.",
                category="technical",
                difficulty="medium"
            ),
            EvaluationSample(
                question="How does the system handle concurrent requests?",
                ground_truth_answer="The system uses asynchronous processing and semaphores to manage concurrent requests efficiently.",
                category="technical",
                difficulty="hard"
            ),
            EvaluationSample(
                question="What are the limitations of the current approach?",
                ground_truth_answer="Limitations include dependency on GPU resources and potential latency issues with large document collections.",
                category="analysis",
                difficulty="medium"
            ),
            EvaluationSample(
                question="What recommendations are provided for optimization?",
                ground_truth_answer="Recommendations include tuning chunk sizes, adjusting concurrent request limits, and optimizing vector search parameters.",
                category="recommendations",
                difficulty="medium"
            )
        ]
        return samples
    
    def load_evaluation_data(self, filepath: str) -> List[EvaluationSample]:
        """Load evaluation data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = EvaluationSample(
                question=item['question'],
                ground_truth_answer=item['ground_truth_answer'],
                expected_context=item.get('expected_context'),
                expected_documents=item.get('expected_documents'),
                category=item.get('category'),
                difficulty=item.get('difficulty')
            )
            samples.append(sample)
        
        return samples
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            embeddings1 = self.embedding_model.encode([text1])
            embeddings2 = self.embedding_model.encode([text2])
            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def calculate_rouge_scores(self, predicted: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, predicted)
            return {
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_f1': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.warning(f"Error calculating ROUGE scores: {e}")
            return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}
    
    def calculate_retrieval_metrics(self, retrieved: List[str], expected: List[str]) -> Tuple[float, float]:
        """Calculate retrieval precision and recall"""
        if not expected:
            return 0.0, 0.0
        
        if not retrieved:
            return 0.0, 0.0
        
        # Simple word overlap for now - could be enhanced with semantic similarity
        retrieved_words = set(' '.join(retrieved).lower().split())
        expected_words = set(' '.join(expected).lower().split())
        
        if not expected_words:
            return 0.0, 0.0
        
        intersection = retrieved_words.intersection(expected_words)
        precision = len(intersection) / len(retrieved_words) if retrieved_words else 0.0
        recall = len(intersection) / len(expected_words)
        
        return precision, recall
    
    async def evaluate_single_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single sample"""
        import time
        
        start_time = time.time()
        
        try:
            # Get RAG response
            response_generator = self.rag.generate(
                messages=[{"role": "user", "content": sample.question}],
                use_knowledge_base=True,
                collection_names=[self.collection_name]
            )
            
            # Process streaming response
            predicted_answer = ""
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
                            predicted_answer += delta["content"]
                except json.JSONDecodeError:
                    continue
            
            latency = (time.time() - start_time) * 1000
            
            # Extract retrieved context (this would need to be enhanced based on your RAG implementation)
            retrieved_context = [predicted_answer]  # Placeholder
            retrieved_documents = []  # Placeholder
            
            # Calculate metrics
            answer_similarity = self.calculate_semantic_similarity(
                predicted_answer, sample.ground_truth_answer
            )
            
            context_relevance = 0.0
            if sample.expected_context:
                context_similarities = [
                    self.calculate_semantic_similarity(predicted_answer, ctx)
                    for ctx in sample.expected_context
                ]
                context_relevance = max(context_similarities) if context_similarities else 0.0
            
            rouge_scores = self.calculate_rouge_scores(predicted_answer, sample.ground_truth_answer)
            
            retrieval_precision, retrieval_recall = self.calculate_retrieval_metrics(
                retrieved_documents, sample.expected_documents or []
            )
            
            result = EvaluationResult(
                sample=sample,
                predicted_answer=predicted_answer,
                retrieved_context=retrieved_context,
                retrieved_documents=retrieved_documents,
                answer_similarity=answer_similarity,
                context_relevance=context_relevance,
                rouge_scores=rouge_scores,
                bert_score=answer_similarity,  # Using semantic similarity as BERT score
                retrieval_precision=retrieval_precision,
                retrieval_recall=retrieval_recall,
                latency_ms=latency
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating sample: {e}")
            # Return a failed result
            return EvaluationResult(
                sample=sample,
                predicted_answer="",
                retrieved_context=[],
                retrieved_documents=[],
                answer_similarity=0.0,
                context_relevance=0.0,
                rouge_scores={'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0},
                bert_score=0.0,
                retrieval_precision=0.0,
                retrieval_recall=0.0,
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def calculate_breakdown_metrics(self, results: List[EvaluationResult], 
                                  group_by: str) -> Dict[str, Dict[str, float]]:
        """Calculate metrics breakdown by category or difficulty"""
        breakdown = {}
        
        for result in results:
            group = getattr(result.sample, group_by)
            if group is None:
                continue
            
            if group not in breakdown:
                breakdown[group] = {
                    'count': 0,
                    'answer_similarity': [],
                    'context_relevance': [],
                    'rouge_f1': [],
                    'bert_score': [],
                    'retrieval_precision': [],
                    'retrieval_recall': [],
                    'latency_ms': []
                }
            
            breakdown[group]['count'] += 1
            breakdown[group]['answer_similarity'].append(result.answer_similarity)
            breakdown[group]['context_relevance'].append(result.context_relevance)
            breakdown[group]['rouge_f1'].append(result.rouge_scores['rougeL_f1'])
            breakdown[group]['bert_score'].append(result.bert_score)
            breakdown[group]['retrieval_precision'].append(result.retrieval_precision)
            breakdown[group]['retrieval_recall'].append(result.retrieval_recall)
            breakdown[group]['latency_ms'].append(result.latency_ms)
        
        # Calculate averages
        for group, metrics in breakdown.items():
            for metric, values in metrics.items():
                if metric != 'count' and values:
                    breakdown[group][f'avg_{metric}'] = np.mean(values)
                    breakdown[group][f'std_{metric}'] = np.std(values)
        
        return breakdown
    
    async def run_evaluation(self, samples: List[EvaluationSample]) -> AccuracyMetrics:
        """Run full accuracy evaluation"""
        console.print(f"[bold blue]Running accuracy evaluation with {len(samples)} samples...[/bold blue]")
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Evaluating samples...", total=len(samples))
            
            for sample in samples:
                result = await self.evaluate_single_sample(sample)
                results.append(result)
                progress.update(task, advance=1)
        
        # Calculate overall metrics
        avg_answer_similarity = np.mean([r.answer_similarity for r in results])
        avg_context_relevance = np.mean([r.context_relevance for r in results])
        avg_rouge_f1 = np.mean([r.rouge_scores['rougeL_f1'] for r in results])
        avg_bert_score = np.mean([r.bert_score for r in results])
        avg_retrieval_precision = np.mean([r.retrieval_precision for r in results])
        avg_retrieval_recall = np.mean([r.retrieval_recall for r in results])
        avg_latency_ms = np.mean([r.latency_ms for r in results])
        
        # Calculate breakdowns
        category_breakdown = self.calculate_breakdown_metrics(results, 'category')
        difficulty_breakdown = self.calculate_breakdown_metrics(results, 'difficulty')
        
        metrics = AccuracyMetrics(
            total_samples=len(samples),
            avg_answer_similarity=avg_answer_similarity,
            avg_context_relevance=avg_context_relevance,
            avg_rouge_f1=avg_rouge_f1,
            avg_bert_score=avg_bert_score,
            avg_retrieval_precision=avg_retrieval_precision,
            avg_retrieval_recall=avg_retrieval_recall,
            avg_latency_ms=avg_latency_ms,
            category_breakdown=category_breakdown,
            difficulty_breakdown=difficulty_breakdown,
            detailed_results=results
        )
        
        return metrics
    
    def generate_report(self, metrics: AccuracyMetrics, output_file: Optional[str] = None):
        """Generate accuracy evaluation report"""
        console.print("\n[bold blue]Accuracy Evaluation Report[/bold blue]")
        
        # Overall metrics table
        table = Table(title="Overall Accuracy Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Samples", str(metrics.total_samples))
        table.add_row("Avg Answer Similarity", f"{metrics.avg_answer_similarity:.3f}")
        table.add_row("Avg Context Relevance", f"{metrics.avg_context_relevance:.3f}")
        table.add_row("Avg ROUGE-L F1", f"{metrics.avg_rouge_f1:.3f}")
        table.add_row("Avg BERT Score", f"{metrics.avg_bert_score:.3f}")
        table.add_row("Avg Retrieval Precision", f"{metrics.avg_retrieval_precision:.3f}")
        table.add_row("Avg Retrieval Recall", f"{metrics.avg_retrieval_recall:.3f}")
        table.add_row("Avg Latency", f"{metrics.avg_latency_ms:.1f} ms")
        
        console.print(table)
        
        # Category breakdown
        if metrics.category_breakdown:
            console.print("\n[bold yellow]Category Breakdown[/bold yellow]")
            cat_table = Table()
            cat_table.add_column("Category", style="cyan")
            cat_table.add_column("Count", style="green")
            cat_table.add_column("Avg Similarity", style="yellow")
            cat_table.add_column("Avg ROUGE-L", style="magenta")
            cat_table.add_column("Avg Latency", style="blue")
            
            for category, data in metrics.category_breakdown.items():
                cat_table.add_row(
                    category,
                    str(data['count']),
                    f"{data.get('avg_answer_similarity', 0):.3f}",
                    f"{data.get('avg_rouge_f1', 0):.3f}",
                    f"{data.get('avg_latency_ms', 0):.1f} ms"
                )
            
            console.print(cat_table)
        
        # Save detailed results
        if output_file:
            result_dict = {
                "overall_metrics": {
                    "total_samples": metrics.total_samples,
                    "avg_answer_similarity": metrics.avg_answer_similarity,
                    "avg_context_relevance": metrics.avg_context_relevance,
                    "avg_rouge_f1": metrics.avg_rouge_f1,
                    "avg_bert_score": metrics.avg_bert_score,
                    "avg_retrieval_precision": metrics.avg_retrieval_precision,
                    "avg_retrieval_recall": metrics.avg_retrieval_recall,
                    "avg_latency_ms": metrics.avg_latency_ms
                },
                "category_breakdown": metrics.category_breakdown,
                "difficulty_breakdown": metrics.difficulty_breakdown,
                "detailed_results": [
                    {
                        "question": r.sample.question,
                        "ground_truth": r.sample.ground_truth_answer,
                        "predicted_answer": r.predicted_answer,
                        "answer_similarity": r.answer_similarity,
                        "rouge_scores": r.rouge_scores,
                        "latency_ms": r.latency_ms
                    }
                    for r in metrics.detailed_results
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            console.print(f"✓ Detailed results saved to: {output_file}")


async def main():
    """Main function to run accuracy evaluation"""
    evaluator = AccuracyEvaluator("benchmark_collection")
    
    # Create sample evaluation data
    samples = evaluator.create_sample_evaluation_data()
    
    # Run evaluation
    metrics = await evaluator.run_evaluation(samples)
    
    # Generate report
    evaluator.generate_report(metrics, "accuracy_evaluation_results.json")


if __name__ == "__main__":
    asyncio.run(main()) 