#!/usr/bin/env python3
"""
Benchmark Results Visualization Script

Generates comparative graphs between different feature implementations
for hipCOMP-core performance analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

def parse_csv_data(data_string):
    """Parse CSV data from string format."""
    lines = [line.strip() for line in data_string.strip().split('\n') if line.strip()]

    # Parse header
    header = lines[0].split(',')

    # Parse data rows
    rows = []
    for line in lines[1:]:
        rows.append(line.split(','))

    df = pd.DataFrame(rows, columns=header)

    # Convert numeric columns
    df['FileSize'] = df['FileSize'].astype(int)
    df['CompressionThroughput_GBps'] = df['CompressionThroughput_GBps'].astype(float)
    df['DecompressionThroughput_GBps'] = df['DecompressionThroughput_GBps'].astype(float)
    df['CompressionRatio'] = df['CompressionRatio'].astype(float)

    return df

def add_file_category(df):
    """Add file size category for grouping."""
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

def calculate_improvements(df1, df2):
    """Calculate percentage improvements from feature1 to feature2."""
    merged = df1.merge(df2, on=['Algorithm', 'TestFile'], suffixes=('_f1', '_f2'))

    merged['CompThroughput_Improvement'] = (
        (merged['CompressionThroughput_GBps_f2'] - merged['CompressionThroughput_GBps_f1']) /
        merged['CompressionThroughput_GBps_f1'] * 100
    )

    merged['DecompThroughput_Improvement'] = (
        (merged['DecompressionThroughput_GBps_f2'] - merged['DecompressionThroughput_GBps_f1']) /
        merged['DecompressionThroughput_GBps_f1'] * 100
    )

    return merged

def plot_throughput_comparison(df1, df2, output_dir):
    """Create side-by-side throughput comparison charts."""
    algorithms = df1['Algorithm'].unique()

    for algo in algorithms:
        df1_algo = df1[df1['Algorithm'] == algo].sort_values('TestFile')
        df2_algo = df2[df2['Algorithm'] == algo].sort_values('TestFile')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        x = np.arange(len(df1_algo))
        width = 0.35

        # Compression throughput
        bars1 = ax1.bar(x - width/2, df1_algo['CompressionThroughput_GBps'],
                        width, label='Feature 1 (Baseline)', color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, df2_algo['CompressionThroughput_GBps'],
                        width, label='Feature 2 (Wave64)', color='#e74c3c', alpha=0.8)

        ax1.set_xlabel('Test File', fontweight='bold')
        ax1.set_ylabel('Throughput (GB/s)', fontweight='bold')
        ax1.set_title(f'{algo.upper()} - Compression Throughput Comparison', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f.replace('.bin', '').replace('_', '\n')
                             for f in df1_algo['TestFile']], rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

        # Decompression throughput
        bars3 = ax2.bar(x - width/2, df1_algo['DecompressionThroughput_GBps'],
                        width, label='Feature 1 (Baseline)', color='#3498db', alpha=0.8)
        bars4 = ax2.bar(x + width/2, df2_algo['DecompressionThroughput_GBps'],
                        width, label='Feature 2 (Wave64)', color='#e74c3c', alpha=0.8)

        ax2.set_xlabel('Test File', fontweight='bold')
        ax2.set_ylabel('Throughput (GB/s)', fontweight='bold')
        ax2.set_title(f'{algo.upper()} - Decompression Throughput Comparison', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f.replace('.bin', '').replace('_', '\n')
                             for f in df1_algo['TestFile']], rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        output_file = output_dir / f'throughput_comparison_{algo}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Created: {output_file}")
        plt.close()

def plot_improvement_heatmap(improvements, output_dir):
    """Create heatmap showing percentage improvements."""
    algorithms = improvements['Algorithm'].unique()

    fig, axes = plt.subplots(len(algorithms), 2, figsize=(16, 6 * len(algorithms)))

    if len(algorithms) == 1:
        axes = axes.reshape(1, -1)

    for idx, algo in enumerate(algorithms):
        imp_algo = improvements[improvements['Algorithm'] == algo].sort_values('TestFile')

        test_files = [f.replace('.bin', '').replace('_', '\n') for f in imp_algo['TestFile']]
        comp_improvements = imp_algo['CompThroughput_Improvement'].values
        decomp_improvements = imp_algo['DecompThroughput_Improvement'].values

        # Compression improvements
        colors_comp = ['#27ae60' if x > 0 else '#e74c3c' for x in comp_improvements]
        bars1 = axes[idx, 0].barh(test_files, comp_improvements, color=colors_comp, alpha=0.8)
        axes[idx, 0].set_xlabel('Improvement (%)', fontweight='bold')
        axes[idx, 0].set_title(f'{algo.upper()} - Compression Throughput Improvement',
                               fontweight='bold', fontsize=12)
        axes[idx, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[idx, 0].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, comp_improvements)):
            axes[idx, 0].text(val, i, f' {val:+.2f}%',
                            va='center', ha='left' if val > 0 else 'right',
                            fontweight='bold', fontsize=9)

        # Decompression improvements
        colors_decomp = ['#27ae60' if x > 0 else '#e74c3c' for x in decomp_improvements]
        bars2 = axes[idx, 1].barh(test_files, decomp_improvements, color=colors_decomp, alpha=0.8)
        axes[idx, 1].set_xlabel('Improvement (%)', fontweight='bold')
        axes[idx, 1].set_title(f'{algo.upper()} - Decompression Throughput Improvement',
                               fontweight='bold', fontsize=12)
        axes[idx, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[idx, 1].grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, decomp_improvements)):
            axes[idx, 1].text(val, i, f' {val:+.2f}%',
                            va='center', ha='left' if val > 0 else 'right',
                            fontweight='bold', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / 'improvement_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()

def plot_summary_statistics(improvements, output_dir):
    """Create summary statistics chart."""
    algorithms = improvements['Algorithm'].unique()

    summary_data = []
    for algo in algorithms:
        imp_algo = improvements[improvements['Algorithm'] == algo]
        summary_data.append({
            'Algorithm': algo.upper(),
            'Avg Comp Improvement': imp_algo['CompThroughput_Improvement'].mean(),
            'Avg Decomp Improvement': imp_algo['DecompThroughput_Improvement'].mean(),
            'Max Comp Improvement': imp_algo['CompThroughput_Improvement'].max(),
            'Max Decomp Improvement': imp_algo['DecompThroughput_Improvement'].max()
        })

    summary_df = pd.DataFrame(summary_data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(algorithms))
    width = 0.35

    # Average improvements
    bars1 = ax1.bar(x - width/2, summary_df['Avg Comp Improvement'],
                    width, label='Compression', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, summary_df['Avg Decomp Improvement'],
                    width, label='Decompression', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('Algorithm', fontweight='bold')
    ax1.set_ylabel('Average Improvement (%)', fontweight='bold')
    ax1.set_title('Average Throughput Improvement by Algorithm', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_df['Algorithm'])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:+.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontweight='bold', fontsize=10)

    # Maximum improvements
    bars3 = ax2.bar(x - width/2, summary_df['Max Comp Improvement'],
                    width, label='Compression', color='#3498db', alpha=0.8)
    bars4 = ax2.bar(x + width/2, summary_df['Max Decomp Improvement'],
                    width, label='Decompression', color='#e74c3c', alpha=0.8)

    ax2.set_xlabel('Algorithm', fontweight='bold')
    ax2.set_ylabel('Maximum Improvement (%)', fontweight='bold')
    ax2.set_title('Maximum Throughput Improvement by Algorithm', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df['Algorithm'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:+.2f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height > 0 else -15),
                       textcoords="offset points",
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontweight='bold', fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'summary_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Created: {output_file}")
    plt.close()

    return summary_df

def plot_by_file_size(df1, df2, output_dir):
    """Create comparison grouped by file size."""
    df1 = add_file_category(df1)
    df2 = add_file_category(df2)

    algorithms = df1['Algorithm'].unique()
    size_categories = ['Small (10MB)', 'Medium (100MB)', 'Large (1GB)']

    for algo in algorithms:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        comp_f1 = []
        comp_f2 = []
        decomp_f1 = []
        decomp_f2 = []

        for size_cat in size_categories:
            df1_subset = df1[(df1['Algorithm'] == algo) & (df1['SizeCategory'] == size_cat)]
            df2_subset = df2[(df2['Algorithm'] == algo) & (df2['SizeCategory'] == size_cat)]

            comp_f1.append(df1_subset['CompressionThroughput_GBps'].mean())
            comp_f2.append(df2_subset['CompressionThroughput_GBps'].mean())
            decomp_f1.append(df1_subset['DecompressionThroughput_GBps'].mean())
            decomp_f2.append(df2_subset['DecompressionThroughput_GBps'].mean())

        x = np.arange(len(size_categories))
        width = 0.35

        # Compression by size
        bars1 = ax1.bar(x - width/2, comp_f1, width, label='Feature 1 (Baseline)',
                       color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, comp_f2, width, label='Feature 2 (Wave64)',
                       color='#e74c3c', alpha=0.8)

        ax1.set_xlabel('File Size', fontweight='bold')
        ax1.set_ylabel('Average Throughput (GB/s)', fontweight='bold')
        ax1.set_title(f'{algo.upper()} - Compression by File Size', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(size_categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        # Decompression by size
        bars3 = ax2.bar(x - width/2, decomp_f1, width, label='Feature 1 (Baseline)',
                       color='#3498db', alpha=0.8)
        bars4 = ax2.bar(x + width/2, decomp_f2, width, label='Feature 2 (Wave64)',
                       color='#e74c3c', alpha=0.8)

        ax2.set_xlabel('File Size', fontweight='bold')
        ax2.set_ylabel('Average Throughput (GB/s)', fontweight='bold')
        ax2.set_title(f'{algo.upper()} - Decompression by File Size', fontweight='bold', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(size_categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        output_file = output_dir / f'filesize_comparison_{algo}.png'
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

    # Calculate improvements
    print("Calculating improvements...")
    improvements = calculate_improvements(df_feature1, df_feature2)

    # Create output directory
    output_dir = Path('visualization_output')
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    print("=" * 60)

    plot_throughput_comparison(df_feature1, df_feature2, output_dir)
    plot_improvement_heatmap(improvements, output_dir)
    summary_df = plot_summary_statistics(improvements, output_dir)
    plot_by_file_size(df_feature1, df_feature2, output_dir)

    print("=" * 60)
    print("\n📊 Summary Statistics:")
    print(summary_df.to_string(index=False))

    print(f"\n✅ All visualizations saved to: {output_dir.absolute()}/")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"  - {f.name}")

if __name__ == '__main__':
    main()
