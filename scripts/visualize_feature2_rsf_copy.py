#!/usr/bin/env python3
"""
Feature 2 Wave64 Benchmark Visualization with RSF Data

Generates visualizations for Feature 2 benchmarks including RSF/TTI seismic data
alongside synthetic test data (binary, random, zeros).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_csv_data(csv_path):
    """Load benchmark results from CSV file."""
    df = pd.read_csv(csv_path)
    return df

def add_file_metadata(df):
    """Add metadata columns for categorization."""
    def get_size_category(filename):
        if 'xlarge' in filename.lower():
            return 'XLarge (4GB)'
        elif 'large' in filename.lower():
            return 'Large (1GB)'
        elif 'medium' in filename.lower():
            return 'Medium (100MB)'
        elif 'small' in filename.lower():
            return 'Small (10MB)'
        return 'Unknown'
    
    def get_data_type(filename):
        if 'TTI' in filename or 'tti' in filename.lower():
            return 'TTI (Seismic)'
        elif 'binary' in filename.lower():
            return 'Binary (Mixed)'
        elif 'random' in filename.lower():
            return 'Random (Incompressible)'
        elif 'zero' in filename.lower():
            return 'Zeros (Highly Compressible)'
        return 'Unknown'
    
    def get_size_mb(row):
        return row['FileSize'] / (1024 * 1024)
    
    df['SizeCategory'] = df['TestFile'].apply(get_size_category)
    df['DataType'] = df['TestFile'].apply(get_data_type)
    df['SizeMB'] = df.apply(get_size_mb, axis=1)
    
    return df

def plot_algorithm_comparison_by_size(df, output_dir):
    """Compare algorithms across different file sizes."""
    algorithms = df['Algorithm'].unique()
    size_categories = ['Small (10MB)', 'Medium (100MB)', 'Large (1GB)', 'XLarge (4GB)']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    x = np.arange(len(size_categories))
    width = 0.25
    
    for idx, algo in enumerate(algorithms):
        comp_means = []
        decomp_means = []
        
        for size_cat in size_categories:
            subset = df[(df['Algorithm'] == algo) & (df['SizeCategory'] == size_cat)]
            if len(subset) > 0:
                comp_means.append(subset['CompressionThroughput_GBps'].mean())
                decomp_means.append(subset['DecompressionThroughput_GBps'].mean())
            else:
                comp_means.append(0)
                decomp_means.append(0)
        
        offset = (idx - len(algorithms)/2 + 0.5) * width
        bars1 = ax1.bar(x + offset, comp_means, width, label=algo.upper(), alpha=0.8)
        bars2 = ax2.bar(x + offset, decomp_means, width, label=algo.upper(), alpha=0.8)
        
        # Add value labels
        for bars, values in [(bars1, comp_means), (bars2, decomp_means)]:
            for bar, val in zip(bars, values):
                if val > 0:
                    height = bar.get_height()
                    ax = bar.axes
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}',
                           ha='center', va='bottom', fontsize=8, rotation=0)
    
    ax1.set_xlabel('File Size Category', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Throughput (GB/s)', fontweight='bold', fontsize=12)
    ax1.set_title('Compression Throughput by File Size', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(size_categories, rotation=15, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.set_xlabel('File Size Category', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Throughput (GB/s)', fontweight='bold', fontsize=12)
    ax2.set_title('Decompression Throughput by File Size', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(size_categories, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = output_dir / 'algorithm_comparison_by_size.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_file.name}")
    plt.close()

def plot_data_type_comparison(df, output_dir):
    """Compare performance across different data types."""
    algorithms = df['Algorithm'].unique()
    data_types = df['DataType'].unique()
    
    fig, axes = plt.subplots(len(algorithms), 2, figsize=(18, 6 * len(algorithms)))
    if len(algorithms) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, algo in enumerate(algorithms):
        algo_data = df[df['Algorithm'] == algo]
        
        comp_data = []
        decomp_data = []
        labels = []
        
        for dtype in sorted(data_types):
            subset = algo_data[algo_data['DataType'] == dtype]
            if len(subset) > 0:
                comp_data.append(subset['CompressionThroughput_GBps'].mean())
                decomp_data.append(subset['DecompressionThroughput_GBps'].mean())
                labels.append(dtype.replace(' (', '\n('))
        
        x = np.arange(len(labels))
        
        # Compression
        bars1 = axes[idx, 0].bar(x, comp_data, alpha=0.8, color='#3498db')
        axes[idx, 0].set_ylabel('Throughput (GB/s)', fontweight='bold')
        axes[idx, 0].set_title(f'{algo.upper()} - Compression by Data Type', 
                               fontweight='bold', fontsize=12)
        axes[idx, 0].set_xticks(x)
        axes[idx, 0].set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
        axes[idx, 0].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, comp_data):
            height = bar.get_height()
            axes[idx, 0].text(bar.get_x() + bar.get_width()/2., height,
                             f'{val:.1f}',
                             ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Decompression
        bars2 = axes[idx, 1].bar(x, decomp_data, alpha=0.8, color='#e74c3c')
        axes[idx, 1].set_ylabel('Throughput (GB/s)', fontweight='bold')
        axes[idx, 1].set_title(f'{algo.upper()} - Decompression by Data Type',
                               fontweight='bold', fontsize=12)
        axes[idx, 1].set_xticks(x)
        axes[idx, 1].set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
        axes[idx, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, decomp_data):
            height = bar.get_height()
            axes[idx, 1].text(bar.get_x() + bar.get_width()/2., height,
                             f'{val:.1f}',
                             ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'data_type_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_file.name}")
    plt.close()

def plot_tti_performance_focus(df, output_dir):
    """Focus specifically on TTI seismic data performance."""
    tti_data = df[df['DataType'] == 'TTI (Seismic)'].copy()
    
    if len(tti_data) == 0:
        print("⚠ No TTI data found, skipping TTI-specific plot")
        return
    
    # Sort by file size
    tti_data = tti_data.sort_values('FileSize')
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    algorithms = tti_data['Algorithm'].unique()
    colors = {'lz4': '#3498db', 'snappy': '#e74c3c', 'cascaded': '#2ecc71'}
    
    # 1. Compression throughput
    ax1 = axes[0, 0]
    for algo in algorithms:
        algo_tti = tti_data[tti_data['Algorithm'] == algo].sort_values('FileSize')
        x = range(len(algo_tti))
        ax1.plot(x, algo_tti['CompressionThroughput_GBps'], 
                marker='o', linewidth=2, markersize=8, 
                label=algo.upper(), color=colors.get(algo, '#85929E'))
    
    ax1.set_xlabel('TTI File (by size)', fontweight='bold')
    ax1.set_ylabel('Throughput (GB/s)', fontweight='bold')
    ax1.set_title('TTI Seismic Data - Compression Throughput', fontweight='bold', fontsize=12)
    ax1.set_xticks(range(len(tti_data['TestFile'].unique())))
    ax1.set_xticklabels([f.replace('_', '\n') for f in sorted(tti_data['TestFile'].unique())],
                        rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Decompression throughput
    ax2 = axes[0, 1]
    for algo in algorithms:
        algo_tti = tti_data[tti_data['Algorithm'] == algo].sort_values('FileSize')
        x = range(len(algo_tti))
        ax2.plot(x, algo_tti['DecompressionThroughput_GBps'],
                marker='s', linewidth=2, markersize=8,
                label=algo.upper(), color=colors.get(algo, '#85929E'))
    
    ax2.set_xlabel('TTI File (by size)', fontweight='bold')
    ax2.set_ylabel('Throughput (GB/s)', fontweight='bold')
    ax2.set_title('TTI Seismic Data - Decompression Throughput', fontweight='bold', fontsize=12)
    ax2.set_xticks(range(len(tti_data['TestFile'].unique())))
    ax2.set_xticklabels([f.replace('_', '\n') for f in sorted(tti_data['TestFile'].unique())],
                        rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Compression ratio
    ax3 = axes[1, 0]
    width = 0.25
    tti_files = sorted(tti_data['TestFile'].unique())
    x = np.arange(len(tti_files))
    
    for idx, algo in enumerate(algorithms):
        ratios = []
        for tfile in tti_files:
            subset = tti_data[(tti_data['Algorithm'] == algo) & (tti_data['TestFile'] == tfile)]
            if len(subset) > 0:
                ratios.append(subset['CompressionRatio'].iloc[0])
            else:
                ratios.append(0)
        
        offset = (idx - len(algorithms)/2 + 0.5) * width
        bars = ax3.bar(x + offset, ratios, width, label=algo.upper(),
                      alpha=0.8, color=colors.get(algo, '#85929E'))
        
        for bar, val in zip(bars, ratios):
            if val > 0:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('TTI File', fontweight='bold')
    ax3.set_ylabel('Compression Ratio', fontweight='bold')
    ax3.set_title('TTI Seismic Data - Compression Ratio', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f.replace('_', '\n') for f in tti_files], 
                        rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Scalability (throughput vs file size)
    ax4 = axes[1, 1]
    for algo in algorithms:
        algo_tti = tti_data[tti_data['Algorithm'] == algo].sort_values('SizeMB')
        ax4.scatter(algo_tti['SizeMB'], algo_tti['CompressionThroughput_GBps'],
                   s=100, alpha=0.6, label=f'{algo.upper()} Comp',
                   marker='o', color=colors.get(algo, '#85929E'))
        ax4.scatter(algo_tti['SizeMB'], algo_tti['DecompressionThroughput_GBps'],
                   s=100, alpha=0.6, label=f'{algo.upper()} Decomp',
                   marker='s', color=colors.get(algo, '#85929E'), edgecolors='black')
    
    ax4.set_xlabel('File Size (MB)', fontweight='bold')
    ax4.set_ylabel('Throughput (GB/s)', fontweight='bold')
    ax4.set_title('TTI Scalability: Throughput vs File Size', fontweight='bold', fontsize=12)
    ax4.set_xscale('log')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Feature 2 Wave64 - TTI Seismic Data Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / 'tti_seismic_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_file.name}")
    plt.close()

def plot_compression_ratio_analysis(df, output_dir):
    """Analyze compression ratios across all data types and algorithms."""
    algorithms = df['Algorithm'].unique()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Group by data type
    data_types = sorted(df['DataType'].unique())
    
    for algo in algorithms:
        ratios = []
        for dtype in data_types:
            subset = df[(df['Algorithm'] == algo) & (df['DataType'] == dtype)]
            if len(subset) > 0:
                ratios.append(subset['CompressionRatio'].mean())
            else:
                ratios.append(0)
        
        ax1.plot(range(len(data_types)), ratios, marker='o', linewidth=2, 
                markersize=8, label=algo.upper())
    
    ax1.set_xlabel('Data Type', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Average Compression Ratio', fontweight='bold', fontsize=12)
    ax1.set_title('Compression Ratio by Data Type', fontweight='bold', fontsize=14)
    ax1.set_xticks(range(len(data_types)))
    ax1.set_xticklabels([d.replace(' (', '\n(') for d in data_types], 
                        rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Throughput vs Ratio scatter
    colors_algo = {'lz4': '#3498db', 'snappy': '#e74c3c', 'cascaded': '#2ecc71'}
    markers_type = {
        'TTI (Seismic)': 'o',
        'Binary (Mixed)': 's',
        'Random (Incompressible)': '^',
        'Zeros (Highly Compressible)': 'D'
    }
    
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        for dtype in data_types:
            subset = algo_data[algo_data['DataType'] == dtype]
            if len(subset) > 0:
                ax2.scatter(subset['CompressionRatio'], 
                           subset['CompressionThroughput_GBps'],
                           s=100, alpha=0.7,
                           color=colors_algo.get(algo, '#85929E'),
                           marker=markers_type.get(dtype, 'o'),
                           label=f'{algo.upper()} - {dtype}')
    
    ax2.set_xlabel('Compression Ratio', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Compression Throughput (GB/s)', fontweight='bold', fontsize=12)
    ax2.set_title('Throughput vs Compression Ratio Trade-off', fontweight='bold', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / 'compression_ratio_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_file.name}")
    plt.close()

def plot_overall_summary(df, output_dir):
    """Create comprehensive summary dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    algorithms = df['Algorithm'].unique()
    
    # 1. Average throughput by algorithm
    ax1 = axes[0, 0]
    comp_avg = [df[df['Algorithm'] == a]['CompressionThroughput_GBps'].mean() for a in algorithms]
    decomp_avg = [df[df['Algorithm'] == a]['DecompressionThroughput_GBps'].mean() for a in algorithms]
    
    x = np.arange(len(algorithms))
    width = 0.35
    ax1.bar(x - width/2, comp_avg, width, label='Compression', alpha=0.8, color='#3498db')
    ax1.bar(x + width/2, decomp_avg, width, label='Decompression', alpha=0.8, color='#e74c3c')
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper() for a in algorithms])
    ax1.set_ylabel('Throughput (GB/s)', fontweight='bold')
    ax1.set_title('Average Throughput by Algorithm', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bars in [ax1.patches[:len(algorithms)], ax1.patches[len(algorithms):]]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Average compression ratio
    ax2 = axes[0, 1]
    ratio_avg = [df[df['Algorithm'] == a]['CompressionRatio'].mean() for a in algorithms]
    bars = ax2.bar(algorithms, ratio_avg, alpha=0.8, color='#2ecc71')
    ax2.set_ylabel('Compression Ratio', fontweight='bold')
    ax2.set_title('Average Compression Ratio', fontweight='bold')
    ax2.set_xticklabels([a.upper() for a in algorithms])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, ratio_avg):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. File count by type
    ax3 = axes[0, 2]
    type_counts = df.groupby('DataType').size()
    ax3.pie(type_counts.values, labels=[t.replace(' (', '\n(') for t in type_counts.index],
           autopct='%1.1f%%', startangle=90)
    ax3.set_title('Test Data Distribution', fontweight='bold')
    
    # 4. Throughput heatmap (algorithms x data types)
    ax4 = axes[1, 0]
    data_types = sorted(df['DataType'].unique())
    heatmap_data = []
    for algo in algorithms:
        row = []
        for dtype in data_types:
            subset = df[(df['Algorithm'] == algo) & (df['DataType'] == dtype)]
            if len(subset) > 0:
                row.append(subset['CompressionThroughput_GBps'].mean())
            else:
                row.append(0)
        heatmap_data.append(row)
    
    im = ax4.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(data_types)))
    ax4.set_yticks(range(len(algorithms)))
    ax4.set_xticklabels([d.replace(' ', '\n') for d in data_types], rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels([a.upper() for a in algorithms])
    ax4.set_title('Compression Throughput Heatmap', fontweight='bold')
    
    for i in range(len(algorithms)):
        for j in range(len(data_types)):
            text = ax4.text(j, i, f'{heatmap_data[i][j]:.1f}',
                           ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax4, label='GB/s')
    
    # 5. Best algorithm per data type (compression)
    ax5 = axes[1, 1]
    best_comp = {}
    for dtype in data_types:
        best_throughput = 0
        best_algo = None
        for algo in algorithms:
            subset = df[(df['Algorithm'] == algo) & (df['DataType'] == dtype)]
            if len(subset) > 0:
                avg_tp = subset['CompressionThroughput_GBps'].mean()
                if avg_tp > best_throughput:
                    best_throughput = avg_tp
                    best_algo = algo
        if best_algo:
            best_comp[dtype] = (best_algo, best_throughput)
    
    x_pos = range(len(best_comp))
    bars = ax5.bar(x_pos, [v[1] for v in best_comp.values()], alpha=0.8)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels([k.replace(' (', '\n(') for k in best_comp.keys()], 
                        rotation=45, ha='right', fontsize=9)
    ax5.set_ylabel('Best Throughput (GB/s)', fontweight='bold')
    ax5.set_title('Best Algorithm per Data Type (Compression)', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, (dtype, (algo, val)) in zip(bars, best_comp.items()):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{algo.upper()}\n{val:.1f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 6. File size distribution
    ax6 = axes[1, 2]
    size_counts = df.groupby('SizeCategory').size()
    bars = ax6.barh(range(len(size_counts)), size_counts.values, alpha=0.8, color='#9b59b6')
    ax6.set_yticks(range(len(size_counts)))
    ax6.set_yticklabels(size_counts.index)
    ax6.set_xlabel('Number of Tests', fontweight='bold')
    ax6.set_title('Tests by File Size', fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, size_counts.values):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f' {val}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Feature 2 Wave64 - Benchmark Summary Dashboard',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = output_dir / 'summary_dashboard.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Created: {output_file.name}")
    plt.close()

def generate_report(df, output_dir):
    """Generate text report with key statistics."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("FEATURE 2 WAVE64 - BENCHMARK REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total tests: {len(df)}")
    report_lines.append(f"Algorithms: {', '.join([a.upper() for a in df['Algorithm'].unique()])}")
    report_lines.append(f"Data types: {len(df['DataType'].unique())}")
    report_lines.append(f"File sizes: {df['SizeMB'].min():.1f} MB - {df['SizeMB'].max():.1f} MB")
    report_lines.append("")
    
    # Per-algorithm statistics
    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo]
        report_lines.append(f"{algo.upper()} PERFORMANCE")
        report_lines.append("-" * 80)
        report_lines.append(f"  Compression:")
        report_lines.append(f"    Average: {algo_data['CompressionThroughput_GBps'].mean():.2f} GB/s")
        report_lines.append(f"    Min:     {algo_data['CompressionThroughput_GBps'].min():.2f} GB/s")
        report_lines.append(f"    Max:     {algo_data['CompressionThroughput_GBps'].max():.2f} GB/s")
        report_lines.append(f"  Decompression:")
        report_lines.append(f"    Average: {algo_data['DecompressionThroughput_GBps'].mean():.2f} GB/s")
        report_lines.append(f"    Min:     {algo_data['DecompressionThroughput_GBps'].min():.2f} GB/s")
        report_lines.append(f"    Max:     {algo_data['DecompressionThroughput_GBps'].max():.2f} GB/s")
        report_lines.append(f"  Compression Ratio:")
        report_lines.append(f"    Average: {algo_data['CompressionRatio'].mean():.2f}x")
        report_lines.append(f"    Min:     {algo_data['CompressionRatio'].min():.2f}x")
        report_lines.append(f"    Max:     {algo_data['CompressionRatio'].max():.2f}x")
        report_lines.append("")
    
    # TTI-specific statistics
    tti_data = df[df['DataType'] == 'TTI (Seismic)']
    if len(tti_data) > 0:
        report_lines.append("TTI SEISMIC DATA PERFORMANCE")
        report_lines.append("-" * 80)
        for algo in tti_data['Algorithm'].unique():
            algo_tti = tti_data[tti_data['Algorithm'] == algo]
            report_lines.append(f"  {algo.upper()}:")
            report_lines.append(f"    Compression:   {algo_tti['CompressionThroughput_GBps'].mean():.2f} GB/s")
            report_lines.append(f"    Decompression: {algo_tti['DecompressionThroughput_GBps'].mean():.2f} GB/s")
            report_lines.append(f"    Ratio:         {algo_tti['CompressionRatio'].mean():.2f}x")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save to file
    report_file = output_dir / 'benchmark_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"✓ Created: {report_file.name}")
    print("\n" + report_text)

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_feature2_rsf.py <path_to_results.csv>")
        print("\nExample:")
        print("  python visualize_feature2_rsf.py results/feature2_wave64/20251201_170939_87a26a0/results.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"Feature 2 Wave64 Benchmark Visualization")
    print(f"{'='*80}")
    print(f"\nLoading data from: {csv_path}")
    
    # Load and process data
    df = load_csv_data(csv_path)
    df = add_file_metadata(df)
    
    print(f"  Total tests: {len(df)}")
    print(f"  Algorithms: {', '.join(df['Algorithm'].unique())}")
    print(f"  Data types: {', '.join(df['DataType'].unique())}")
    
    # Create output directory
    output_dir = csv_path.parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating visualizations in: {output_dir}")
    print(f"{'-'*80}")
    
    # Generate all plots
    plot_algorithm_comparison_by_size(df, output_dir)
    plot_data_type_comparison(df, output_dir)
    plot_tti_performance_focus(df, output_dir)
    plot_compression_ratio_analysis(df, output_dir)
    plot_overall_summary(df, output_dir)
    
    # Generate report
    print(f"{'-'*80}")
    generate_report(df, output_dir)
    
    print(f"\n{'='*80}")
    print(f"✅ Visualization complete!")
    print(f"\nOutput files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  📊 {f.name}")
    print(f"  📄 benchmark_report.txt")
    print(f"\nLocation: {output_dir.absolute()}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
