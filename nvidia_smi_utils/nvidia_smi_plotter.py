#!/usr/bin/env python3

"""
Takes in the result of
```
nvidia-smi --query-gpu=timestamp,index,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,compute_mode,clocks.current.sm,clocks.current.memory,power.draw,power.limit,fan.speed,pcie.link.width.current,encoder.stats.sessionCount,encoder.stats.averageFps,encoder.stats.averageLatency,vbios_version,inforom.img,gpu_uuid,gpu_serial \
    --format=csv \
    -l 1 \
    -f gpu_metrics.csv
```
This polls all available nvidia smi metrics every second and writes them to csv.
"""

from io import StringIO
import os

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def date_parser(x):
    return datetime.strptime(x, '%Y/%m/%d %H:%M:%S.%f')

def plot_nvidia_smi_data(csv, output_file):
    """ Generates a plot of nvidia smi data from decorator."""
    if os.path.isfile(csv):
        df = pd.read_csv(csv,
                         delimiter=", ",
                         engine="python",
                         parse_dates=['timestamp'],
                         date_parser=date_parser)
    else:
        df = pd.read_csv(StringIO(csv),
                         delimiter=", ",
                         engine="python",
                         parse_dates=['timestamp'],
                         date_parser=date_parser)

    # Handle all unfriendliness from nvidia-smi.
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['clocks.current.sm [MHz]'] = df[f'clocks.current.sm [MHz]'].str.replace(' MHz', '').astype(int)
    df['power.draw [W]'] = df['power.draw [W]'].str.replace(' W', '').astype(float)
    df['power.limit [W]'] = df['power.limit [W]'].str.replace(' W', '').astype(float)
    df['memory.used [MiB]'] = df['memory.used [MiB]'].str.replace(' MiB', '').astype(int)
    df['utilization.gpu [%]'] = df['utilization.gpu [%]'].str.replace(' %', '').astype(int)
    df['power_usage_percent'] = (df['power.draw [W]'] / df['power.limit [W]']) * 100

    # Get the unique GPU indices
    gpu_indices = df['index'].unique()   


    # Create a figure with subplots (2x2 grid for the 4 plots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()  # Flatten to easily iterate over them

    # Plot 1: GPU Clock Speed
    for idx in gpu_indices:
        gpu_data = df[df['index'] == idx]
        axes[0].plot(gpu_data['timestamp'], gpu_data['clocks.current.sm [MHz]'], label=f'GPU {idx}')
    axes[0].set_ylabel('GPU Clock Speed (MHz)')
    axes[0].set_title('GPU Clock Speed (MHz)')

    # Plot 2: Power Draw as Percentage of Power Limit
    for idx in gpu_indices:
        gpu_data = df[df['index'] == idx]
        axes[1].plot(gpu_data['timestamp'], gpu_data['power_usage_percent'], label=f'GPU {idx}')
    axes[1].set_ylabel('Power Usage (%)')
    axes[1].set_title('Power Draw / Power Limit (%)')

    # Plot 3: Memory Used (MiB)
    for idx in gpu_indices:
        gpu_data = df[df['index'] == idx]
        axes[2].plot(gpu_data['timestamp'], gpu_data['memory.used [MiB]'], label=f'GPU {idx}')
    axes[2].set_ylabel('Memory Used (MiB)')
    axes[2].set_title('Memory Used (MiB)')

    # Plot 4: GPU Utilization
    for idx in gpu_indices:
        gpu_data = df[df['index'] == idx]
        axes[3].plot(gpu_data['timestamp'], gpu_data['utilization.gpu [%]'], label=f'GPU {idx}')
    axes[3].set_ylabel('GPU Utilization (%)')
    axes[3].set_title('GPU Utilization (%)')

    # Format the plots
    for ax in axes:
        ax.grid(True)
        ax.legend(loc='upper left')
        ax.tick_params(axis='x', rotation=45)

    # Get the CSV file name without extension for the title
    fig.suptitle(f"NVIDIA H100 80GB HBM3 metrics for VLLM run", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title

    # Save the plot to a file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()