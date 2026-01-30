#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of B200 vs H200 training performance.
Generates visualizations and a detailed report for training decision-making.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
import os
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def load_data():
    """Load both datasets and add machine identifier."""
    h200_df = pd.read_csv('h200/matmul.csv')
    b200_df = pd.read_csv('b200/matmul.csv')
    
    h200_df['machine'] = 'H200'
    b200_df['machine'] = 'B200'
    
    # Combine datasets
    df = pd.concat([h200_df, b200_df], ignore_index=True)
    
    # Clean dtype column (remove 'torch.' prefix for cleaner labels)
    df['dtype_clean'] = df['dtype'].str.replace('torch.', '')
    
    # Create matrix size identifier
    df['matrix_size'] = df.apply(lambda x: f"{x['M']}x{x['K']}x{x['N']}", axis=1)
    df['matrix_product'] = df['M'] * df['K'] * df['N']
    
    return df

def create_dtype_comparison_plots(df, output_dir='.'):
    """Create visualizations comparing performance by dtype."""
    
    dtypes = sorted(df['dtype_clean'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mean TFLOP/s by dtype
    dtype_means = df.groupby(['machine', 'dtype_clean'])['mean(TFLOP/s)'].mean().reset_index()
    dtype_pivot = dtype_means.pivot(index='dtype_clean', columns='machine', values='mean(TFLOP/s)')
    
    dtype_pivot.plot(kind='bar', ax=axes[0, 0], color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[0, 0].set_title('Mean TFLOP/s by Data Type', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Data Type', fontweight='bold')
    axes[0, 0].set_ylabel('Mean TFLOP/s', fontweight='bold')
    axes[0, 0].legend(title='Machine')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # Add percentage difference annotations
    for i, dtype in enumerate(dtype_pivot.index):
        h200_val = dtype_pivot.loc[dtype, 'H200']
        b200_val = dtype_pivot.loc[dtype, 'B200']
        pct_diff = ((b200_val - h200_val) / h200_val) * 100
        axes[0, 0].text(i, max(h200_val, b200_val) * 1.05, 
                        f'{pct_diff:+.1f}%', 
                        ha='center', fontweight='bold', fontsize=9)
    
    # Plot 2: Median TFLOP/s by dtype
    dtype_medians = df.groupby(['machine', 'dtype_clean'])['mean(TFLOP/s)'].median().reset_index()
    dtype_med_pivot = dtype_medians.pivot(index='dtype_clean', columns='machine', values='mean(TFLOP/s)')
    
    dtype_med_pivot.plot(kind='bar', ax=axes[0, 1], color=['#2E86AB', '#A23B72'], alpha=0.8)
    axes[0, 1].set_title('Median TFLOP/s by Data Type', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Data Type', fontweight='bold')
    axes[0, 1].set_ylabel('Median TFLOP/s', fontweight='bold')
    axes[0, 1].legend(title='Machine')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # Plot 3 & 4: Box plots showing distribution by dtype (show first 2 dtypes)
    for i, dtype in enumerate(dtypes[:2]):
        row = 1
        col = i
        ax = axes[row, col]
        dtype_data = df[df['dtype_clean'] == dtype]
        sns.boxplot(data=dtype_data, x='machine', y='mean(TFLOP/s)', ax=ax, 
                   palette=['#2E86AB', '#A23B72'])
        ax.set_title(f'TFLOP/s Distribution: {dtype}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Machine', fontweight='bold')
        ax.set_ylabel('TFLOP/s', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Performance by Data Type: B200 vs H200', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dtype_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved dtype comparison plot: {output_dir}/dtype_comparison.png")
    plt.close()

def create_matrix_size_comparison_by_dtype(df, output_dir='.'):
    """Create visualizations comparing performance by matrix size, split by dtype."""
    
    dtypes = sorted(df['dtype_clean'].unique())
    
    # Create one plot per dtype
    for dtype in dtypes:
        dtype_data = df[df['dtype_clean'] == dtype]
        common_sizes = dtype_data.groupby(['M', 'K', 'N']).size().sort_values(ascending=False).head(20).index
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: Performance vs Matrix Product Size
        dtype_data['log_matrix_product'] = np.log10(dtype_data['matrix_product'])
        
        for machine in ['H200', 'B200']:
            machine_data = dtype_data[dtype_data['machine'] == machine]
            axes[0, 0].scatter(machine_data['log_matrix_product'], 
                              machine_data['mean(TFLOP/s)'],
                              alpha=0.5, label=machine, s=30)
        
        axes[0, 0].set_xlabel('Log10(Matrix Product: M×K×N)', fontweight='bold')
        axes[0, 0].set_ylabel('Mean TFLOP/s', fontweight='bold')
        axes[0, 0].set_title(f'Performance vs Matrix Size: {dtype}', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Performance by matrix size (top 15)
        comparison = []
        for (m, k, n) in common_sizes[:15]:
            h200_data = dtype_data[(dtype_data['M'] == m) & (dtype_data['K'] == k) & 
                                  (dtype_data['N'] == n) & (dtype_data['machine'] == 'H200')]
            b200_data = dtype_data[(dtype_data['M'] == m) & (dtype_data['K'] == k) & 
                                  (dtype_data['N'] == n) & (dtype_data['machine'] == 'B200')]
            
            if len(h200_data) > 0 and len(b200_data) > 0:
                h200_mean = h200_data['mean(TFLOP/s)'].mean()
                b200_mean = b200_data['mean(TFLOP/s)'].mean()
                pct_diff = ((b200_mean - h200_mean) / h200_mean) * 100
                comparison.append({
                    'Size': f"{m}×{k}×{n}",
                    'H200': h200_mean,
                    'B200': b200_mean,
                    'Pct_Diff': pct_diff
                })
        
        comp_df = pd.DataFrame(comparison).sort_values('Pct_Diff', ascending=False)
        
        x_pos = np.arange(len(comp_df))
        width = 0.35
        axes[0, 1].barh(x_pos - width/2, comp_df['H200'], width, label='H200', 
                        color='#2E86AB', alpha=0.8)
        axes[0, 1].barh(x_pos + width/2, comp_df['B200'], width, label='B200', 
                        color='#A23B72', alpha=0.8)
        axes[0, 1].set_yticks(x_pos)
        axes[0, 1].set_yticklabels(comp_df['Size'], fontsize=8)
        axes[0, 1].set_xlabel('Mean TFLOP/s', fontweight='bold')
        axes[0, 1].set_title(f'Performance by Matrix Size (Top 15): {dtype}', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # Plot 3: Performance by individual dimension (M)
        m_analysis = dtype_data.groupby(['machine', 'M'])['mean(TFLOP/s)'].mean().reset_index()
        m_pivot = m_analysis.pivot(index='M', columns='machine', values='mean(TFLOP/s)')
        m_pivot.plot(kind='line', ax=axes[1, 0], marker='o', linewidth=2, markersize=6,
                    color=['#2E86AB', '#A23B72'])
        axes[1, 0].set_title(f'Performance vs M Dimension: {dtype}', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('M Dimension', fontweight='bold')
        axes[1, 0].set_ylabel('Mean TFLOP/s', fontweight='bold')
        axes[1, 0].legend(title='Machine')
        axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Performance by K dimension
        k_analysis = dtype_data.groupby(['machine', 'K'])['mean(TFLOP/s)'].mean().reset_index()
        k_pivot = k_analysis.pivot(index='K', columns='machine', values='mean(TFLOP/s)')
        k_pivot.plot(kind='line', ax=axes[1, 1], marker='s', linewidth=2, markersize=6,
                    color=['#2E86AB', '#A23B72'])
        axes[1, 1].set_title(f'Performance vs K Dimension: {dtype}', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('K Dimension', fontweight='bold')
        axes[1, 1].set_ylabel('Mean TFLOP/s', fontweight='bold')
        axes[1, 1].legend(title='Machine')
        axes[1, 1].grid(alpha=0.3)
        
        plt.suptitle(f'Performance by Matrix Size: B200 vs H200 ({dtype})', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        safe_dtype = dtype.replace('/', '_')
        plt.savefig(f'{output_dir}/matrix_size_comparison_{safe_dtype}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved matrix size comparison plot: {output_dir}/matrix_size_comparison_{safe_dtype}.png")
        plt.close()

def create_gpu_scaling_plot(df, output_dir='.'):
    """Create visualization showing performance scaling by GPU count and dtype."""
    
    # Get unique GPU counts
    gpu_counts = sorted(df['num_gpus'].unique())
    dtypes = sorted(df['dtype_clean'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    # Create one subplot per dtype
    for idx, dtype in enumerate(dtypes):
        ax = axes[idx]
        dtype_data = df[df['dtype_clean'] == dtype]
        
        # Group by machine, GPU count, and calculate mean performance
        scaling_data = dtype_data.groupby(['machine', 'num_gpus'])['mean(TFLOP/s)'].mean().reset_index()
        
        for machine in ['H200', 'B200']:
            machine_scaling = scaling_data[scaling_data['machine'] == machine]
            ax.plot(machine_scaling['num_gpus'], machine_scaling['mean(TFLOP/s)'], 
                   marker='o', linewidth=2, markersize=8, label=machine,
                   color='#2E86AB' if machine == 'H200' else '#A23B72')
        
        ax.set_title(f'GPU Scaling: {dtype}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of GPUs', fontweight='bold')
        ax.set_ylabel('Mean TFLOP/s', fontweight='bold')
        ax.set_xticks(gpu_counts)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Performance Across All Matrix Sizes by Data Type & GPU Count', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gpu_scaling_by_dtype.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved GPU scaling plot: {output_dir}/gpu_scaling_by_dtype.png")
    plt.close()

def create_performance_ratio_plot(df, output_dir='.'):
    """Create visualization showing B200/H200 performance ratio by dtype."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ratio_by_dtype = []
    for dtype in sorted(df['dtype_clean'].unique()):
        h200_mean = df[(df['machine'] == 'H200') & (df['dtype_clean'] == dtype)]['mean(TFLOP/s)'].mean()
        b200_mean = df[(df['machine'] == 'B200') & (df['dtype_clean'] == dtype)]['mean(TFLOP/s)'].mean()
        ratio = b200_mean / h200_mean if h200_mean > 0 else 0
        ratio_by_dtype.append({'dtype': dtype, 'ratio': ratio})
    
    ratio_df = pd.DataFrame(ratio_by_dtype)
    bars = ax.bar(ratio_df['dtype'], ratio_df['ratio'], 
                   color=['#2E86AB' if x < 1 else '#A23B72' for x in ratio_df['ratio']], alpha=0.8)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Parity')
    ax.set_title('B200/H200 Performance Ratio by Data Type', fontsize=16, fontweight='bold')
    ax.set_xlabel('Data Type', fontweight='bold', fontsize=12)
    ax.set_ylabel('Performance Ratio (B200/H200)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_ratio_by_dtype.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved performance ratio plot: {output_dir}/performance_ratio_by_dtype.png")
    plt.close()

def create_comprehensive_comparison_by_dtype(df, output_dir='.'):
    """Create comprehensive comparisons split by dtype."""
    
    dtypes = sorted(df['dtype_clean'].unique())
    
    for dtype in dtypes:
        dtype_data = df[df['dtype_clean'] == dtype]
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Overall summary stats for this dtype
        ax1 = fig.add_subplot(gs[0, :])
        summary_stats = dtype_data.groupby('machine').agg({
            'mean(TFLOP/s)': ['mean', 'median', 'max']
        }).round(2)
        
        x = np.arange(len(summary_stats.index))
        width = 0.25
        
        ax1.bar(x - width, summary_stats[('mean(TFLOP/s)', 'mean')], width, 
               label='Mean TFLOP/s', color='#2E86AB', alpha=0.8)
        ax1.bar(x, summary_stats[('mean(TFLOP/s)', 'median')], width,
               label='Median TFLOP/s', color='#A23B72', alpha=0.8)
        ax1.bar(x + width, summary_stats[('mean(TFLOP/s)', 'max')], width,
               label='Max TFLOP/s', color='#F18F01', alpha=0.8)
        
        ax1.set_xlabel('Machine', fontweight='bold', fontsize=12)
        ax1.set_ylabel('TFLOP/s', fontweight='bold', fontsize=12)
        ax1.set_title(f'Overall Performance Summary: {dtype}', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary_stats.index)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # GPU count scaling for this dtype
        ax2 = fig.add_subplot(gs[1, 0])
        gpu_dist = dtype_data.groupby(['machine', 'num_gpus'])['mean(TFLOP/s)'].mean().reset_index()
        for machine in ['H200', 'B200']:
            machine_data = gpu_dist[gpu_dist['machine'] == machine]
            ax2.plot(machine_data['num_gpus'], machine_data['mean(TFLOP/s)'], 
                    marker='o', linewidth=2, markersize=8, label=machine,
                    color='#2E86AB' if machine == 'H200' else '#A23B72')
        ax2.set_title(f'Performance by GPU Count: {dtype}', fontweight='bold')
        ax2.set_xlabel('Number of GPUs', fontweight='bold')
        ax2.set_ylabel('Mean TFLOP/s', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Performance ratio for this dtype
        ax3 = fig.add_subplot(gs[1, 1])
        h200_mean = dtype_data[dtype_data['machine'] == 'H200']['mean(TFLOP/s)'].mean()
        b200_mean = dtype_data[dtype_data['machine'] == 'B200']['mean(TFLOP/s)'].mean()
        ratio = b200_mean / h200_mean if h200_mean > 0 else 0
        
        bars = ax3.bar([dtype], [ratio], 
                       color='#2E86AB' if ratio < 1 else '#A23B72', alpha=0.8)
        ax3.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Parity')
        ax3.set_title(f'B200/H200 Ratio: {dtype}', fontweight='bold')
        ax3.set_ylabel('Performance Ratio (B200/H200)', fontweight='bold')
        ax3.set_ylim([0, max(ratio * 1.2, 1.5)])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value label
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Box plot distribution
        ax4 = fig.add_subplot(gs[1, 2])
        sns.boxplot(data=dtype_data, x='machine', y='mean(TFLOP/s)', ax=ax4,
                   palette=['#2E86AB', '#A23B72'])
        ax4.set_title(f'Performance Distribution: {dtype}', fontweight='bold')
        ax4.set_xlabel('Machine', fontweight='bold')
        ax4.set_ylabel('TFLOP/s', fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # Matrix size performance (scatter)
        ax5 = fig.add_subplot(gs[2, :])
        for machine in ['H200', 'B200']:
            machine_data = dtype_data[dtype_data['machine'] == machine]
            ax5.scatter(machine_data['matrix_product'], machine_data['mean(TFLOP/s)'],
                       alpha=0.4, label=machine, s=20)
        
        ax5.set_xlabel('Matrix Product (M×K×N)', fontweight='bold', fontsize=11)
        ax5.set_ylabel('Mean TFLOP/s', fontweight='bold', fontsize=11)
        ax5.set_title(f'Performance Across All Matrix Sizes: {dtype}', fontsize=14, fontweight='bold')
        ax5.set_xscale('log')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        plt.suptitle(f'Comprehensive Training Performance Comparison: B200 vs H200 ({dtype})', 
                    fontsize=16, fontweight='bold', y=0.995)
        safe_dtype = dtype.replace('/', '_')
        plt.savefig(f'{output_dir}/comprehensive_comparison_{safe_dtype}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved comprehensive comparison plot: {output_dir}/comprehensive_comparison_{safe_dtype}.png")
        plt.close()

def generate_report(df, output_dir='.'):
    """Generate a comprehensive text report."""
    
    report = []
    report.append("=" * 80)
    report.append("B200 vs H200 Comprehensive Training Performance Analysis Report")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics by dtype
    report.append("OVERALL STATISTICS BY DATA TYPE")
    report.append("-" * 80)
    for dtype in sorted(df['dtype_clean'].unique()):
        dtype_data = df[df['dtype_clean'] == dtype]
        report.append(f"\n{dtype}:")
        overall_stats = dtype_data.groupby('machine').agg({
            'mean(TFLOP/s)': ['mean', 'median', 'std', 'min', 'max']
        }).round(2)
        report.append(overall_stats.to_string())
        report.append("")
    
    # Analysis by dtype
    report.append("PERFORMANCE BY DATA TYPE")
    report.append("-" * 80)
    dtype_stats = df.groupby(['machine', 'dtype_clean']).agg({
        'mean(TFLOP/s)': ['mean', 'median', 'max']
    }).round(2)
    report.append(dtype_stats.to_string())
    report.append("")
    
    # Winners by dtype
    report.append("WINNERS BY DATA TYPE")
    report.append("-" * 80)
    for dtype in df['dtype_clean'].unique():
        h200_mean = df[(df['machine'] == 'H200') & (df['dtype_clean'] == dtype)]['mean(TFLOP/s)'].mean()
        b200_mean = df[(df['machine'] == 'B200') & (df['dtype_clean'] == dtype)]['mean(TFLOP/s)'].mean()
        winner = 'B200' if b200_mean > h200_mean else 'H200'
        pct_diff = abs((b200_mean - h200_mean) / h200_mean) * 100
        report.append(f"{dtype:15s}: {winner:4s} wins ({pct_diff:.1f}% difference)")
        report.append(f"  H200: {h200_mean:8.2f} TFLOP/s | B200: {b200_mean:8.2f} TFLOP/s")
    report.append("")
    
    # GPU scaling analysis
    report.append("GPU SCALING ANALYSIS")
    report.append("-" * 80)
    for dtype in df['dtype_clean'].unique():
        report.append(f"{dtype}:")
        dtype_data = df[df['dtype_clean'] == dtype]
        for machine in ['H200', 'B200']:
            machine_data = dtype_data[dtype_data['machine'] == machine]
            scaling = machine_data.groupby('num_gpus')['mean(TFLOP/s)'].mean()
            report.append(f"  {machine}:")
            for gpus, perf in scaling.items():
                report.append(f"    {gpus} GPU(s): {perf:.2f} TFLOP/s")
        report.append("")
    
    # Matrix size analysis by dtype
    report.append("MATRIX SIZE ANALYSIS BY DATA TYPE")
    report.append("-" * 80)
    for dtype in sorted(df['dtype_clean'].unique()):
        dtype_data = df[df['dtype_clean'] == dtype]
        report.append(f"\n{dtype}:")
        report.append("Top 10 matrix sizes by average performance:")
        
        size_perf = dtype_data.groupby(['M', 'K', 'N'])['mean(TFLOP/s)'].mean().sort_values(ascending=False).head(10)
        for (m, k, n), perf in size_perf.items():
            h200_perf = dtype_data[(dtype_data['M'] == m) & (dtype_data['K'] == k) & 
                                   (dtype_data['N'] == n) & (dtype_data['machine'] == 'H200')]['mean(TFLOP/s)'].mean()
            b200_perf = dtype_data[(dtype_data['M'] == m) & (dtype_data['K'] == k) & 
                                   (dtype_data['N'] == n) & (dtype_data['machine'] == 'B200')]['mean(TFLOP/s)'].mean()
            winner = 'B200' if b200_perf > h200_perf else 'H200'
            report.append(f"  {m:5d}×{k:5d}×{n:5d}: {winner:4s} ({h200_perf:7.2f} vs {b200_perf:7.2f} TFLOP/s)")
        report.append("")
    
    # Training recommendations
    report.append("TRAINING RECOMMENDATIONS")
    report.append("-" * 80)
    
    # Analyze for different training scenarios
    scenarios = {
        'LLM Training (bfloat16)': 'bfloat16',
        'LLM Training (float16)': 'float16',
        'Scientific Computing (float64)': 'float64',
        'General ML (float32)': 'float32'
    }
    
    for scenario, dtype in scenarios.items():
        h200_perf = df[(df['machine'] == 'H200') & (df['dtype_clean'] == dtype)]['mean(TFLOP/s)'].mean()
        b200_perf = df[(df['machine'] == 'B200') & (df['dtype_clean'] == dtype)]['mean(TFLOP/s)'].mean()
        
        if h200_perf > 0 and b200_perf > 0:
            winner = 'B200' if b200_perf > h200_perf else 'H200'
            speedup = max(b200_perf, h200_perf) / min(b200_perf, h200_perf)
            report.append(f"{scenario}:")
            report.append(f"  Recommended: {winner}")
            report.append(f"  Performance: H200={h200_perf:.2f} TFLOP/s, B200={b200_perf:.2f} TFLOP/s")
            report.append(f"  Speedup: {speedup:.2f}x")
            report.append("")
    
    # Key insights
    report.append("KEY INSIGHTS")
    report.append("-" * 80)
    
    # Find best dtype for each machine
    h200_best_dtype = df[df['machine'] == 'H200'].groupby('dtype_clean')['mean(TFLOP/s)'].mean().idxmax()
    b200_best_dtype = df[df['machine'] == 'B200'].groupby('dtype_clean')['mean(TFLOP/s)'].mean().idxmax()
    
    report.append(f"• H200 performs best with: {h200_best_dtype}")
    report.append(f"• B200 performs best with: {b200_best_dtype}")
    
    # Overall winner by dtype
    for dtype in sorted(df['dtype_clean'].unique()):
        dtype_data = df[df['dtype_clean'] == dtype]
        h200_overall = dtype_data[dtype_data['machine'] == 'H200']['mean(TFLOP/s)'].mean()
        b200_overall = dtype_data[dtype_data['machine'] == 'B200']['mean(TFLOP/s)'].mean()
        overall_winner = 'B200' if b200_overall > h200_overall else 'H200'
        overall_speedup = max(b200_overall, h200_overall) / min(b200_overall, h200_overall)
        
        report.append(f"• {dtype}: {overall_winner} wins ({overall_speedup:.2f}x)")
    report.append("")
    
    report_text = "\n".join(report)
    
    # Save report
    with open(f'{output_dir}/training_performance_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"✓ Saved analysis report: {output_dir}/training_performance_report.txt")
    
    # Also print to console
    print("\n" + report_text)

def main():
    """Main execution function."""
    # Create output directory
    output_dir = 'training_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()
    
    print("Loading data...")
    df = load_data()
    print(f"Loaded {len(df)} total records")
    print(f"  H200: {len(df[df['machine'] == 'H200'])} records")
    print(f"  B200: {len(df[df['machine'] == 'B200'])} records")
    print()
    
    print("Creating visualizations...")
    create_dtype_comparison_plots(df, output_dir)
    create_matrix_size_comparison_by_dtype(df, output_dir)
    create_gpu_scaling_plot(df, output_dir)
    create_performance_ratio_plot(df, output_dir)
    create_comprehensive_comparison_by_dtype(df, output_dir)
    
    print("\nGenerating report...")
    generate_report(df, output_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete! Generated files in training_results/:")
    print("  - dtype_comparison.png")
    print("  - matrix_size_comparison_<dtype>.png (one per dtype)")
    print("  - gpu_scaling_by_dtype.png")
    print("  - performance_ratio_by_dtype.png")
    print("  - comprehensive_comparison_<dtype>.png (one per dtype)")
    print("  - training_performance_report.txt")
    print("=" * 80)

if __name__ == '__main__':
    main()