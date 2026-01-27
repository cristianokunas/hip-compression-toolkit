#!/usr/bin/env python3
"""
Advanced Benchmark Visualization Script

Generates advanced comparative graphs including compression ratio analysis,
efficiency metrics, and trade-off visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def parse_csv_data(data_string):
    """Parse CSV data from string format."""
    lines = [line.strip() for line in data_string.strip().split('\n') if line.strip()]
    header = lines[0].split(',')
    rows = [line.split(',') for line in lines[1:]]
    df = pd.DataFrame(rows, columns=header)

    df['FileSize'] = df['FileSize'].astype(int)
    df['CompressionThroughput_GBps'] = df['CompressionThroughput_GBps'].astype(float)
    df['DecompressionThroughput_GBps'] = df['DecompressionThroughput_GBps'].astype(float)
    df['CompressionRatio'] = df['CompressionRatio'].astype(float)

    return df

def add_file_category(df):
    """Add file size and type categories."""
    def categorize_file(filename):
        if 'large' in filename:
            return 'Large (1GB)'
        elif 'medium' in filename:
            return 'Medium (100MB)'
        elif 'small' in filename:
            return 'Small (10MB)'
        return 'Unknown'

    def categorize_type(filename):
        if 'binary' in filename:
            return 'Binary'
        elif 'random' in filename:
            return 'Random'
        elif 'zeros' in filename:
            return 'Zeros'
        return 'Unknown'

    df['SizeCategory'] = df['TestFile'].apply(categorize_file)
    df['DataType'] = df['TestFile'].apply(categorize_type)
    return df

def plot_compression_ratio_comparison(df1, df2, output_dir):
    """Compare compression ratios between features."""
    algorithms = df1['Algorithm'].unique()

    fig, axes = plt.subplots(len(algorithms), 1, figsize=(16, 6 * len(algorithms)))
    if len(algorithms) == 1:
        axes = [axes]

    for idx, algo in enumerate(algorithms):
        df1_algo = df1[df1['Algorithm'] == algo].sort_values('TestFile')
        df2_algo = df2[df2['Algorithm'] == algo].sort_values('TestFile')

        x = np.arange(len(df1_algo))
        width = 0.35

        bars1 = axes[idx].bar(x - width/2, df1_algo['CompressionRatio'],
                             width, label='Feature 1 (Baseline)',
                             color='#3498db', alpha=0.8)
        bars2 = axes[idx].bar(x + width/2, df2_algo['CompressionRatio'],
                             width, label='Feature 2 (Wave64)',
                             color='#e74c3c', alpha=0.8)

        axes[idx].set_xlabel('Test File', fontweight='bold')
        axes[idx].set_ylabel('Compression Ratio', fontweight='bold')
        axes[idx].set_title(f'{algo.upper()} - Compression Ratio Comparison',
                           fontweight='bold', fontsize=14)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([f.replace('.bin', '').replace('_', '\n')
                                   for f in df1_algo['TestFile']],
                                  rotation=45, ha='right', fontsize=8)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[idx].annotate(f'{height:.2f}x',
                                  xy=(bar.get_x() + bar.get_width() / 2, height),
                                  xytext=(0, 3),
                                  textcoords="offset points",
                                  ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    output_file = output_dir / 'compression_ratio_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()

def plot_throughput_vs_ratio_scatter(df1, df2, output_dir):
    """Create scatter plot showing throughput vs compression ratio trade-off."""
    df1 = add_file_category(df1)
    df2 = add_file_category(df2)

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    algorithms = df1['Algorithm'].unique()
    colors = {'Binary': '#e74c3c', 'Random': '#3498db', 'Zeros': '#2ecc71'}
    markers_f1 = {'Binary': 'o', 'Random': 's', 'Zeros': '^'}
    markers_f2 = {'Binary': 'D', 'Random': 'P', 'Zeros': 'X'}

    # Compression throughput vs ratio
    ax1 = fig.add_subplot(gs[0, 0])
    for algo in algorithms:
        for data_type in ['Binary', 'Random', 'Zeros']:
            # Feature 1
            mask1 = (df1['Algorithm'] == algo) & (df1['DataType'] == data_type)
            ax1.scatter(df1[mask1]['CompressionRatio'],
                       df1[mask1]['CompressionThroughput_GBps'],
                       c=colors[data_type], marker=markers_f1[data_type],
                       s=150, alpha=0.6, edgecolors='black', linewidth=1,
                       label=f'{algo.upper()} {data_type} (F1)')

            # Feature 2
            mask2 = (df2['Algorithm'] == algo) & (df2['DataType'] == data_type)
            ax1.scatter(df2[mask2]['CompressionRatio'],
                       df2[mask2]['CompressionThroughput_GBps'],
                       c=colors[data_type], marker=markers_f2[data_type],
                       s=150, alpha=0.9, edgecolors='black', linewidth=1.5,
                       label=f'{algo.upper()} {data_type} (F2)')

    ax1.set_xlabel('Compression Ratio', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Compression Throughput (GB/s)', fontweight='bold', fontsize=12)
    ax1.set_title('Compression: Throughput vs Ratio Trade-off', fontweight='bold', fontsize=14)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Decompression throughput vs ratio
    ax2 = fig.add_subplot(gs[0, 1])
    for algo in algorithms:
        for data_type in ['Binary', 'Random', 'Zeros']:
            mask1 = (df1['Algorithm'] == algo) & (df1['DataType'] == data_type)
            ax2.scatter(df1[mask1]['CompressionRatio'],
                       df1[mask1]['DecompressionThroughput_GBps'],
                       c=colors[data_type], marker=markers_f1[data_type],
                       s=150, alpha=0.6, edgecolors='black', linewidth=1)

            mask2 = (df2['Algorithm'] == algo) & (df2['DataType'] == data_type)
            ax2.scatter(df2[mask2]['CompressionRatio'],
                       df2[mask2]['DecompressionThroughput_GBps'],
                       c=colors[data_type], marker=markers_f2[data_type],
                       s=150, alpha=0.9, edgecolors='black', linewidth=1.5)

    ax2.set_xlabel('Compression Ratio', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Decompression Throughput (GB/s)', fontweight='bold', fontsize=12)
    ax2.set_title('Decompression: Throughput vs Ratio Trade-off', fontweight='bold', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')

    # Legend for markers
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, label='Feature 1', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                   markersize=10, label='Feature 2', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
                   markersize=10, label='Binary', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498db',
                   markersize=10, label='Random', markeredgecolor='black'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ecc71',
                   markersize=10, label='Zeros', markeredgecolor='black'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Efficiency score (throughput * ratio) - Compression
    ax3 = fig.add_subplot(gs[1, 0])
    df1['CompEfficiency'] = df1['CompressionThroughput_GBps'] * df1['CompressionRatio']
    df2['CompEfficiency'] = df2['CompressionThroughput_GBps'] * df2['CompressionRatio']

    for algo in algorithms:
        algo_data1 = df1[df1['Algorithm'] == algo].sort_values('TestFile')
        algo_data2 = df2[df2['Algorithm'] == algo].sort_values('TestFile')

        x = np.arange(len(algo_data1))
        width = 0.35

        ax3.bar(x - width/2, algo_data1['CompEfficiency'], width,
               label=f'{algo.upper()} F1', alpha=0.7)
        ax3.bar(x + width/2, algo_data2['CompEfficiency'], width,
               label=f'{algo.upper()} F2', alpha=0.9)

    ax3.set_xlabel('Test File Index', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Efficiency Score (Throughput × Ratio)', fontweight='bold', fontsize=12)
    ax3.set_title('Compression Efficiency Score', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Efficiency score - Decompression
    ax4 = fig.add_subplot(gs[1, 1])
    df1['DecompEfficiency'] = df1['DecompressionThroughput_GBps'] * df1['CompressionRatio']
    df2['DecompEfficiency'] = df2['DecompressionThroughput_GBps'] * df2['CompressionRatio']

    for algo in algorithms:
        algo_data1 = df1[df1['Algorithm'] == algo].sort_values('TestFile')
        algo_data2 = df2[df2['Algorithm'] == algo].sort_values('TestFile')

        x = np.arange(len(algo_data1))
        width = 0.35

        ax4.bar(x - width/2, algo_data1['DecompEfficiency'], width,
               label=f'{algo.upper()} F1', alpha=0.7)
        ax4.bar(x + width/2, algo_data2['DecompEfficiency'], width,
               label=f'{algo.upper()} F2', alpha=0.9)

    ax4.set_xlabel('Test File Index', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Efficiency Score (Throughput × Ratio)', fontweight='bold', fontsize=12)
    ax4.set_title('Decompression Efficiency Score', fontweight='bold', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    output_file = output_dir / 'throughput_vs_ratio_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()

def plot_performance_by_data_type(df1, df2, output_dir):
    """Create heatmap-style visualization grouped by data type."""
    df1 = add_file_category(df1)
    df2 = add_file_category(df2)

    algorithms = df1['Algorithm'].unique()
    data_types = ['Binary', 'Random', 'Zeros']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Compression throughput by data type
    for i, feature_name in enumerate(['Feature 1', 'Feature 2']):
        df = df1 if i == 0 else df2

        comp_data = []
        decomp_data = []

        for algo in algorithms:
            comp_row = []
            decomp_row = []
            for dtype in data_types:
                mask = (df['Algorithm'] == algo) & (df['DataType'] == dtype)
                comp_row.append(df[mask]['CompressionThroughput_GBps'].mean())
                decomp_row.append(df[mask]['DecompressionThroughput_GBps'].mean())
            comp_data.append(comp_row)
            decomp_data.append(decomp_row)

        # Compression heatmap
        im1 = axes[0, i].imshow(comp_data, cmap='YlOrRd', aspect='auto')
        axes[0, i].set_xticks(np.arange(len(data_types)))
        axes[0, i].set_yticks(np.arange(len(algorithms)))
        axes[0, i].set_xticklabels(data_types)
        axes[0, i].set_yticklabels([a.upper() for a in algorithms])
        axes[0, i].set_title(f'{feature_name} - Compression Throughput (GB/s)',
                            fontweight='bold', fontsize=12)

        # Add text annotations
        for y in range(len(algorithms)):
            for x in range(len(data_types)):
                text = axes[0, i].text(x, y, f'{comp_data[y][x]:.1f}',
                                      ha="center", va="center",
                                      color="black", fontweight='bold')

        fig.colorbar(im1, ax=axes[0, i])

        # Decompression heatmap
        im2 = axes[1, i].imshow(decomp_data, cmap='YlGnBu', aspect='auto')
        axes[1, i].set_xticks(np.arange(len(data_types)))
        axes[1, i].set_yticks(np.arange(len(algorithms)))
        axes[1, i].set_xticklabels(data_types)
        axes[1, i].set_yticklabels([a.upper() for a in algorithms])
        axes[1, i].set_title(f'{feature_name} - Decompression Throughput (GB/s)',
                            fontweight='bold', fontsize=12)

        # Add text annotations
        for y in range(len(algorithms)):
            for x in range(len(data_types)):
                text = axes[1, i].text(x, y, f'{decomp_data[y][x]:.1f}',
                                      ha="center", va="center",
                                      color="black", fontweight='bold')

        fig.colorbar(im2, ax=axes[1, i])

    plt.tight_layout()
    output_file = output_dir / 'performance_by_datatype_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()

def plot_algorithm_radar_chart(df1, df2, output_dir):
    """Create radar chart comparing algorithm characteristics."""
    df1 = add_file_category(df1)
    df2 = add_file_category(df2)

    algorithms = df1['Algorithm'].unique()

    fig, axes = plt.subplots(1, len(algorithms), figsize=(8 * len(algorithms), 8),
                            subplot_kw=dict(projection='polar'))

    if len(algorithms) == 1:
        axes = [axes]

    categories = ['Comp\nSpeed', 'Decomp\nSpeed', 'Comp\nRatio',
                  'Binary\nPerf', 'Random\nPerf', 'Zeros\nPerf']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for idx, algo in enumerate(algorithms):
        # Calculate metrics for Feature 1
        algo_df1 = df1[df1['Algorithm'] == algo]
        values_f1 = [
            algo_df1['CompressionThroughput_GBps'].mean(),
            algo_df1['DecompressionThroughput_GBps'].mean(),
            algo_df1['CompressionRatio'].mean() * 20,  # Scale for visibility
            algo_df1[algo_df1['DataType'] == 'Binary']['CompressionThroughput_GBps'].mean(),
            algo_df1[algo_df1['DataType'] == 'Random']['CompressionThroughput_GBps'].mean(),
            algo_df1[algo_df1['DataType'] == 'Zeros']['CompressionThroughput_GBps'].mean(),
        ]
        values_f1 += values_f1[:1]

        # Calculate metrics for Feature 2
        algo_df2 = df2[df2['Algorithm'] == algo]
        values_f2 = [
            algo_df2['CompressionThroughput_GBps'].mean(),
            algo_df2['DecompressionThroughput_GBps'].mean(),
            algo_df2['CompressionRatio'].mean() * 20,
            algo_df2[algo_df2['DataType'] == 'Binary']['CompressionThroughput_GBps'].mean(),
            algo_df2[algo_df2['DataType'] == 'Random']['CompressionThroughput_GBps'].mean(),
            algo_df2[algo_df2['DataType'] == 'Zeros']['CompressionThroughput_GBps'].mean(),
        ]
        values_f2 += values_f2[:1]

        # Plot
        axes[idx].plot(angles, values_f1, 'o-', linewidth=2, label='Feature 1',
                      color='#3498db', alpha=0.6)
        axes[idx].fill(angles, values_f1, alpha=0.15, color='#3498db')

        axes[idx].plot(angles, values_f2, 'o-', linewidth=2, label='Feature 2',
                      color='#e74c3c', alpha=0.8)
        axes[idx].fill(angles, values_f2, alpha=0.25, color='#e74c3c')

        axes[idx].set_xticks(angles[:-1])
        axes[idx].set_xticklabels(categories, size=10)
        axes[idx].set_ylim(0, max(max(values_f1), max(values_f2)) * 1.1)
        axes[idx].set_title(f'{algo.upper()} Performance Profile',
                           fontweight='bold', fontsize=14, pad=20)
        axes[idx].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        axes[idx].grid(True)

    plt.tight_layout()
    output_file = output_dir / 'algorithm_radar_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()

def plot_ratio_improvement_analysis(df1, df2, output_dir):
    """Analyze compression ratio changes between features."""
    merged = df1.merge(df2, on=['Algorithm', 'TestFile'], suffixes=('_f1', '_f2'))
    merged = add_file_category(merged)

    merged['RatioDiff'] = merged['CompressionRatio_f2'] - merged['CompressionRatio_f1']
    merged['RatioChange%'] = (merged['RatioDiff'] / merged['CompressionRatio_f1']) * 100

    algorithms = merged['Algorithm'].unique()

    fig, axes = plt.subplots(len(algorithms), 2, figsize=(18, 6 * len(algorithms)))
    if len(algorithms) == 1:
        axes = axes.reshape(1, -1)

    for idx, algo in enumerate(algorithms):
        algo_data = merged[merged['Algorithm'] == algo].sort_values('TestFile')

        # Absolute ratio difference
        colors = ['#27ae60' if x >= 0 else '#e74c3c' for x in algo_data['RatioDiff']]
        bars1 = axes[idx, 0].barh(range(len(algo_data)), algo_data['RatioDiff'],
                                   color=colors, alpha=0.8)

        axes[idx, 0].set_yticks(range(len(algo_data)))
        axes[idx, 0].set_yticklabels([f.replace('.bin', '').replace('_', ' ')
                                      for f in algo_data['TestFile']], fontsize=9)
        axes[idx, 0].set_xlabel('Ratio Difference (F2 - F1)', fontweight='bold')
        axes[idx, 0].set_title(f'{algo.upper()} - Compression Ratio Change',
                              fontweight='bold', fontsize=12)
        axes[idx, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[idx, 0].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, algo_data['RatioDiff'])):
            axes[idx, 0].text(val, i, f' {val:+.3f}',
                            va='center', ha='left' if val > 0 else 'right',
                            fontweight='bold', fontsize=8)

        # Percentage change
        colors2 = ['#27ae60' if x >= 0 else '#e74c3c' for x in algo_data['RatioChange%']]
        bars2 = axes[idx, 1].barh(range(len(algo_data)), algo_data['RatioChange%'],
                                   color=colors2, alpha=0.8)

        axes[idx, 1].set_yticks(range(len(algo_data)))
        axes[idx, 1].set_yticklabels([f.replace('.bin', '').replace('_', ' ')
                                      for f in algo_data['TestFile']], fontsize=9)
        axes[idx, 1].set_xlabel('Ratio Change (%)', fontweight='bold')
        axes[idx, 1].set_title(f'{algo.upper()} - Compression Ratio % Change',
                              fontweight='bold', fontsize=12)
        axes[idx, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[idx, 1].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, algo_data['RatioChange%'])):
            axes[idx, 1].text(val, i, f' {val:+.2f}%',
                            va='center', ha='left' if val > 0 else 'right',
                            fontweight='bold', fontsize=8)

    plt.tight_layout()
    output_file = output_dir / 'compression_ratio_changes.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()

def plot_overall_performance_summary(df1, df2, output_dir):
    """Create comprehensive performance summary dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    algorithms = df1['Algorithm'].unique()
    colors_algo = {'lz4': '#3498db', 'snappy': '#e74c3c', 'cascaded': '#2ecc71'}

    # 1. Average throughput by algorithm
    ax1 = fig.add_subplot(gs[0, 0])
    comp_f1 = [df1[df1['Algorithm'] == a]['CompressionThroughput_GBps'].mean() for a in algorithms]
    comp_f2 = [df2[df2['Algorithm'] == a]['CompressionThroughput_GBps'].mean() for a in algorithms]

    x = np.arange(len(algorithms))
    width = 0.35
    ax1.bar(x - width/2, comp_f1, width, label='Feature 1', color='#95a5a6', alpha=0.7)
    ax1.bar(x + width/2, comp_f2, width, label='Feature 2', color='#85929E', alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper() for a in algorithms])
    ax1.set_ylabel('Avg Throughput (GB/s)', fontweight='bold')
    ax1.set_title('Average Compression Throughput', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Average decompression throughput
    ax2 = fig.add_subplot(gs[0, 1])
    decomp_f1 = [df1[df1['Algorithm'] == a]['DecompressionThroughput_GBps'].mean() for a in algorithms]
    decomp_f2 = [df2[df2['Algorithm'] == a]['DecompressionThroughput_GBps'].mean() for a in algorithms]

    ax2.bar(x - width/2, decomp_f1, width, label='Feature 1', color='#95a5a6', alpha=0.7)
    ax2.bar(x + width/2, decomp_f2, width, label='Feature 2', color='#85929E', alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.upper() for a in algorithms])
    ax2.set_ylabel('Avg Throughput (GB/s)', fontweight='bold')
    ax2.set_title('Average Decompression Throughput', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Average compression ratio
    ax3 = fig.add_subplot(gs[0, 2])
    ratio_f1 = [df1[df1['Algorithm'] == a]['CompressionRatio'].mean() for a in algorithms]
    ratio_f2 = [df2[df2['Algorithm'] == a]['CompressionRatio'].mean() for a in algorithms]

    ax3.bar(x - width/2, ratio_f1, width, label='Feature 1', color='#95a5a6', alpha=0.7)
    ax3.bar(x + width/2, ratio_f2, width, label='Feature 2', color='#85929E', alpha=0.9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([a.upper() for a in algorithms])
    ax3.set_ylabel('Avg Compression Ratio', fontweight='bold')
    ax3.set_title('Average Compression Ratio', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # 4-6. Improvement percentages
    ax4 = fig.add_subplot(gs[1, 0])
    comp_improve = [(c2 - c1) / c1 * 100 for c1, c2 in zip(comp_f1, comp_f2)]
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in comp_improve]
    bars = ax4.bar(x, comp_improve, color=colors, alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([a.upper() for a in algorithms])
    ax4.set_ylabel('Improvement (%)', fontweight='bold')
    ax4.set_title('Compression Throughput Improvement', fontweight='bold')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, comp_improve):
        ax4.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.2f}%',
                ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

    ax5 = fig.add_subplot(gs[1, 1])
    decomp_improve = [(d2 - d1) / d1 * 100 for d1, d2 in zip(decomp_f1, decomp_f2)]
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in decomp_improve]
    bars = ax5.bar(x, decomp_improve, color=colors, alpha=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels([a.upper() for a in algorithms])
    ax5.set_ylabel('Improvement (%)', fontweight='bold')
    ax5.set_title('Decompression Throughput Improvement', fontweight='bold')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, decomp_improve):
        ax5.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.2f}%',
                ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

    ax6 = fig.add_subplot(gs[1, 2])
    ratio_improve = [(r2 - r1) / r1 * 100 for r1, r2 in zip(ratio_f1, ratio_f2)]
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in ratio_improve]
    bars = ax6.bar(x, ratio_improve, color=colors, alpha=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels([a.upper() for a in algorithms])
    ax6.set_ylabel('Improvement (%)', fontweight='bold')
    ax6.set_title('Compression Ratio Improvement', fontweight='bold')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax6.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, ratio_improve):
        ax6.text(bar.get_x() + bar.get_width()/2, val, f'{val:+.2f}%',
                ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

    # 7-9. Distribution plots
    df1_cat = add_file_category(df1)
    df2_cat = add_file_category(df2)

    ax7 = fig.add_subplot(gs[2, 0])
    for algo in algorithms:
        data_f1 = df1_cat[df1_cat['Algorithm'] == algo]['CompressionThroughput_GBps']
        data_f2 = df2_cat[df2_cat['Algorithm'] == algo]['CompressionThroughput_GBps']
        ax7.violinplot([data_f1, data_f2], positions=[x[list(algorithms).index(algo)]*2,
                       x[list(algorithms).index(algo)]*2+0.8], widths=0.7)
    ax7.set_ylabel('Compression Throughput (GB/s)', fontweight='bold')
    ax7.set_title('Throughput Distribution (Compression)', fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    ax8 = fig.add_subplot(gs[2, 1])
    for algo in algorithms:
        data_f1 = df1_cat[df1_cat['Algorithm'] == algo]['DecompressionThroughput_GBps']
        data_f2 = df2_cat[df2_cat['Algorithm'] == algo]['DecompressionThroughput_GBps']
        ax8.violinplot([data_f1, data_f2], positions=[x[list(algorithms).index(algo)]*2,
                       x[list(algorithms).index(algo)]*2+0.8], widths=0.7)
    ax8.set_ylabel('Decompression Throughput (GB/s)', fontweight='bold')
    ax8.set_title('Throughput Distribution (Decompression)', fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

    ax9 = fig.add_subplot(gs[2, 2])
    for algo in algorithms:
        data_f1 = df1_cat[df1_cat['Algorithm'] == algo]['CompressionRatio']
        data_f2 = df2_cat[df2_cat['Algorithm'] == algo]['CompressionRatio']
        ax9.violinplot([data_f1, data_f2], positions=[x[list(algorithms).index(algo)]*2,
                       x[list(algorithms).index(algo)]*2+0.8], widths=0.7)
    ax9.set_ylabel('Compression Ratio', fontweight='bold')
    ax9.set_title('Ratio Distribution', fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')

    plt.suptitle('hipCOMP Performance Dashboard: Feature 1 vs Feature 2',
                 fontsize=16, fontweight='bold', y=0.995)

    output_file = output_dir / 'performance_dashboard.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()

def main():
    # Feature 1 data (baseline)
    feature1_data = """Algorithm,TestFile,FileSize,CompressionThroughput_GBps,DecompressionThroughput_GBps,CompressionRatio
lz4,large_binary_1gb.bin,1073741824,22.44,221.81,3.29
lz4,large_random_1gb.bin,1073741824,7.30,177.64,1.00
lz4,large_zeros_1gb.bin,1073741824,162.47,242.97,245.45
lz4,medium_binary_100mb.bin,104857600,16.37,226.35,3.29
lz4,medium_random_100mb.bin,104857600,5.54,171.63,1.00
lz4,meduim_zeros_100mb.bin,104857600,103.57,406.79,245.45
lz4,small_binary_10mb.bin,10485760,2.14,39.95,3.29
lz4,small_random_10mb.bin,10485760,1.89,42.89,1.00
lz4,small_zeros_10mb.bin,10485760,24.60,63.71,245.45
snappy,large_binary_1gb.bin,1073741824,28.10,64.17,2.98
snappy,large_random_1gb.bin,1073741824,26.96,75.55,0.99
snappy,large_zeros_1gb.bin,1073741824,28.73,61.06,21.30
snappy,medium_binary_100mb.bin,104857600,25.22,94.81,2.98
snappy,medium_random_100mb.bin,104857600,24.74,120.18,0.99
snappy,meduim_zeros_100mb.bin,104857600,25.72,94.33,21.30
snappy,small_binary_10mb.bin,10485760,9.27,23.60,2.98
snappy,small_random_10mb.bin,10485760,8.35,34.40,0.99
snappy,small_zeros_10mb.bin,10485760,9.13,24.41,21.30
cascaded,large_binary_1gb.bin,1073741824,111.51,50.27,3.25
cascaded,large_random_1gb.bin,1073741824,57.48,278.89,1.00
cascaded,large_zeros_1gb.bin,1073741824,177.10,41.66,92.04
cascaded,medium_binary_100mb.bin,104857600,89.04,79.35,3.25
cascaded,medium_random_100mb.bin,104857600,53.73,328.52,1.00
cascaded,meduim_zeros_100mb.bin,104857600,151.15,51.33,92.04
cascaded,small_binary_10mb.bin,10485760,28.81,24.41,3.25
cascaded,small_random_10mb.bin,10485760,20.32,165.41,1.00
cascaded,small_zeros_10mb.bin,10485760,64.07,20.23,92.04"""

    # Feature 2 data (wave64 optimizations)
    feature2_data = """Algorithm,TestFile,FileSize,CompressionThroughput_GBps,DecompressionThroughput_GBps,CompressionRatio
lz4,large_binary_1gb.bin,1073741824,22.46,224.87,3.29
lz4,large_random_1gb.bin,1073741824,7.30,183.95,1.00
lz4,large_zeros_1gb.bin,1073741824,162.38,243.42,245.45
lz4,medium_binary_100mb.bin,104857600,16.34,229.61,3.29
lz4,medium_random_100mb.bin,104857600,5.54,172.18,1.00
lz4,meduim_zeros_100mb.bin,104857600,103.03,407.51,245.45
lz4,small_binary_10mb.bin,10485760,2.13,40.10,3.29
lz4,small_random_10mb.bin,10485760,1.89,43.11,1.00
lz4,small_zeros_10mb.bin,10485760,22.90,59.09,245.45
snappy,large_binary_1gb.bin,1073741824,28.08,64.82,2.98
snappy,large_random_1gb.bin,1073741824,26.92,75.21,0.99
snappy,large_zeros_1gb.bin,1073741824,28.70,61.20,21.30
snappy,medium_binary_100mb.bin,104857600,25.39,95.29,2.98
snappy,medium_random_100mb.bin,104857600,24.81,120.63,0.99
snappy,meduim_zeros_100mb.bin,104857600,25.68,94.37,21.30
snappy,small_binary_10mb.bin,10485760,9.17,24.44,2.98
snappy,small_random_10mb.bin,10485760,8.53,34.53,0.99
snappy,small_zeros_10mb.bin,10485760,9.91,25.03,21.30
cascaded,large_binary_1gb.bin,1073741824,111.63,49.80,3.25
cascaded,large_random_1gb.bin,1073741824,57.33,279.43,1.00
cascaded,large_zeros_1gb.bin,1073741824,177.55,41.34,92.04
cascaded,medium_binary_100mb.bin,104857600,89.14,79.63,3.25
cascaded,medium_random_100mb.bin,104857600,53.57,330.25,1.00
cascaded,meduim_zeros_100mb.bin,104857600,151.28,51.47,92.04
cascaded,small_binary_10mb.bin,10485760,26.49,22.43,3.25
cascaded,small_random_10mb.bin,10485760,20.59,186.91,1.00
cascaded,small_zeros_10mb.bin,10485760,64.46,20.33,92.04"""

    # Parse data
    print("Parsing benchmark data...")
    df_feature1 = parse_csv_data(feature1_data)
    df_feature2 = parse_csv_data(feature2_data)

    # Create output directory
    output_dir = Path('visualization_output')
    output_dir.mkdir(exist_ok=True)

    # Generate advanced visualizations
    print("\nGenerating advanced visualizations...")
    print("=" * 60)

    plot_compression_ratio_comparison(df_feature1, df_feature2, output_dir)
    plot_throughput_vs_ratio_scatter(df_feature1, df_feature2, output_dir)
    plot_performance_by_data_type(df_feature1, df_feature2, output_dir)
    plot_algorithm_radar_chart(df_feature1, df_feature2, output_dir)
    plot_ratio_improvement_analysis(df_feature1, df_feature2, output_dir)
    plot_overall_performance_summary(df_feature1, df_feature2, output_dir)

    print("=" * 60)
    print(f"\n✅ All advanced visualizations saved to: {output_dir.absolute()}/")
    print("\nNew files generated:")
    new_files = [
        'compression_ratio_comparison.png',
        'throughput_vs_ratio_analysis.png',
        'performance_by_datatype_heatmap.png',
        'algorithm_radar_comparison.png',
        'compression_ratio_changes.png',
        'performance_dashboard.png'
    ]
    for f in new_files:
        print(f"  - {f}")

if __name__ == '__main__':
    main()
