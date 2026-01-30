#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

# Read the tab-delimited file
# First, collect all MI355X and MI300X data separately
mi355x_data = {}
mi300x_data = {}

data_file = sys.argv[1]

with open(data_file, 'r') as f:
    lines = f.readlines()
    
    # Skip header (first line)
    for line in lines[1:]:
        row = line.strip().split('\t')
        if len(row) < 18:
            continue
        
        try:
            # Extract MI355X data (columns 0-7)
            mi355x_input = int(row[4])
            mi355x_output = int(row[5])
            mi355x_tokens = float(row[7])
            key = (mi355x_input, mi355x_output)
            mi355x_data[key] = mi355x_tokens
            
            # Extract MI300X data (columns 10-17)
            mi300x_input = int(row[14])  # Correct index
            mi300x_output = int(row[15])  # Correct index
            mi300x_tokens = float(row[17])  # Correct index
            key = (mi300x_input, mi300x_output)
            mi300x_data[key] = mi300x_tokens
        except (ValueError, IndexError) as e:
            continue

# Match configurations that exist in both datasets
data = []
for (input_len, output_len), mi355x_tokens in mi355x_data.items():
    key = (input_len, output_len)
    if key in mi300x_data:
        mi300x_tokens = mi300x_data[key]
        label = f"{input_len}/{output_len}"
        data.append({
            'label': label,
            'mi355x_tokens': mi355x_tokens,
            'mi300x_tokens': mi300x_tokens,
            'input': input_len,
            'output': output_len
        })

# Sort by input then output length
data.sort(key=lambda x: (x['input'], x['output']))

# Extract data for plotting
labels = [d['label'] for d in data]
mi355x_values = [d['mi355x_tokens'] for d in data]
mi300x_values = [d['mi300x_tokens'] for d in data]

# Calculate percent differences using the formula: 100 * ((x1 - x2)/((x1 + x2)/2))
percent_diffs = []
for d in data:
    x1 = d['mi355x_tokens']
    x2 = d['mi300x_tokens']
    # Formula: 100 * ((x1 - x2) / ((x1 + x2) / 2))
    pct_diff = 100 * ((x1 - x2) / ((x1 + x2) / 2))
    percent_diffs.append(pct_diff)

# Create the visualization
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(labels))
width = 0.35

# Create bars - ensure they're properly aligned
bars1 = ax.bar(x - width/2, mi355x_values, width, label='MI355X', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, mi300x_values, width, label='MI300X', color='#A23B72', alpha=0.8)

# Customize the plot
ax.set_xlabel('Input Length / Output Length', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Tokens per Second', fontsize=12, fontweight='bold')
# ax.set_title('MI355X vs MI300X: Throughput comparison for tp8 model amd/Llama-3.3-70B-Instruct-FP8-KV', fontsize=14, fontweight='bold')
ax.set_title('MI355X vs MI300X: Throughput comparison for tp8 model openai/gpt-oss-120b', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add percent difference labels on top of MI355X bars
for i, (bar, pct) in enumerate(zip(bars1, percent_diffs)):
    height = bar.get_height()
    # Position label closer to the bar
    label_y = height * 1.015
    ax.text(bar.get_x() + bar.get_width()/2., label_y,
            f'+{pct:.1f}%' if pct >= 0 else f'{pct:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2E86AB')

# Add value labels on top of MI300X bars only (to avoid clutter)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
            f'{height:.0f}',
            ha='center', va='bottom', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig('gpt_oss_comparison.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'gpt_oss_comparison.png'")
