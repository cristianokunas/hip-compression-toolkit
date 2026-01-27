#!/usr/bin/env python3
"""
Feature 2 Wave64 - Faceted Comparison Charts

Generates faceted bar charts for compression, decompression, and ratio
organized by data type with algorithms compared side-by-side.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys

def load_csv_data(csv_path):
    """Load benchmark results from CSV file."""
    df = pd.read_csv(csv_path)
    return df

def add_file_metadata(df):
    """Add metadata columns for categorization."""
    def get_size_category(filename):
        if 'xlarge' in filename.lower():
            return 'XLarge'
        elif 'large' in filename.lower():
            return 'Large'
        elif 'medium' in filename.lower() or 'meduim' in filename.lower():
            return 'Medium'
        elif 'small' in filename.lower():
            return 'Small'
        return 'Unknown'
    
    def get_data_type(filename):
        if 'TTI' in filename or 'tti' in filename.lower():
            return 'TTI Seismic'
        elif 'binary' in filename.lower():
            return 'Binary Mixed'
        elif 'random' in filename.lower():
            return 'Random'
        elif 'zero' in filename.lower():
            return 'Zeros'
        return 'Unknown'
    
    def get_size_order(cat):
        order = {'Small': 1, 'Medium': 2, 'Large': 3, 'XLarge': 4}
        return order.get(cat, 0)
    
    df['SizeCategory'] = df['TestFile'].apply(get_size_category)
    df['DataType'] = df['TestFile'].apply(get_data_type)
    df['SizeMB'] = df['FileSize'] / (1024 * 1024)
    df['SizeOrder'] = df['SizeCategory'].apply(get_size_order)
    
    return df

def plot_faceted_compression(df, output_dir):
    """
    Faceted chart: Compression throughput by data type.
    Each facet is a data type, bars are algorithms, grouped by file size.
    """
    data_types = sorted(df['DataType'].unique())
    n_types = len(data_types)
    
    # Create subplots - one per data type
    fig = make_subplots(
        rows=1, cols=n_types,
        subplot_titles=data_types,
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    colors = {'lz4': '#3498db', 'snappy': '#e74c3c', 'cascaded': '#2ecc71'}
    algorithms = sorted(df['Algorithm'].unique())
    sizes = sorted(df['SizeCategory'].unique(), 
                  key=lambda x: df[df['SizeCategory']==x]['SizeOrder'].iloc[0])
    
    for col_idx, dtype in enumerate(data_types, 1):
        dtype_data = df[df['DataType'] == dtype]
        
        for algo in algorithms:
            algo_data = dtype_data[dtype_data['Algorithm'] == algo]
            
            # Aggregate by size category
            values = []
            labels = []
            for size in sizes:
                size_data = algo_data[algo_data['SizeCategory'] == size]
                if len(size_data) > 0:
                    values.append(size_data['CompressionThroughput_GBps'].mean())
                    labels.append(size)
                else:
                    values.append(0)
                    labels.append(size)
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    name=algo.upper(),
                    marker_color=colors.get(algo, '#85929E'),
                    text=[f'{v:.1f}' if v > 0 else '' for v in values],
                    textposition='outside',
                    textfont=dict(size=9),
                    legendgroup=algo,
                    showlegend=(col_idx == 1)  # Show legend only in first subplot
                ),
                row=1, col=col_idx
            )
        
        # Update x-axis
        fig.update_xaxes(title_text="File Size", row=1, col=col_idx)
    
    # Update y-axis (only leftmost)
    fig.update_yaxes(title_text="Compression Throughput (GB/s)", row=1, col=1)
    
    fig.update_layout(
        height=600,
        width=400 * n_types,
        title_text="Compression Performance by Data Type and Algorithm",
        title_font_size=22,
        template='plotly_white',
        font=dict(size=11),
        barmode='group',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    output_file = output_dir / 'faceted_compression.png'
    fig.write_image(str(output_file), width=400 * n_types, height=720, scale=2)
    print(f"✓ Created: {output_file.name}")

def plot_faceted_decompression(df, output_dir):
    """
    Faceted chart: Decompression throughput by data type.
    """
    data_types = sorted(df['DataType'].unique())
    n_types = len(data_types)
    
    fig = make_subplots(
        rows=1, cols=n_types,
        subplot_titles=data_types,
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    colors = {'lz4': '#3498db', 'snappy': '#e74c3c', 'cascaded': '#2ecc71'}
    algorithms = sorted(df['Algorithm'].unique())
    sizes = sorted(df['SizeCategory'].unique(), 
                  key=lambda x: df[df['SizeCategory']==x]['SizeOrder'].iloc[0])
    
    for col_idx, dtype in enumerate(data_types, 1):
        dtype_data = df[df['DataType'] == dtype]
        
        for algo in algorithms:
            algo_data = dtype_data[dtype_data['Algorithm'] == algo]
            
            values = []
            labels = []
            for size in sizes:
                size_data = algo_data[algo_data['SizeCategory'] == size]
                if len(size_data) > 0:
                    values.append(size_data['DecompressionThroughput_GBps'].mean())
                    labels.append(size)
                else:
                    values.append(0)
                    labels.append(size)
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    name=algo.upper(),
                    marker_color=colors.get(algo, '#85929E'),
                    text=[f'{v:.1f}' if v > 0 else '' for v in values],
                    textposition='outside',
                    textfont=dict(size=9),
                    legendgroup=algo,
                    showlegend=(col_idx == 1)
                ),
                row=1, col=col_idx
            )
        
        fig.update_xaxes(title_text="File Size", row=1, col=col_idx)
    
    fig.update_yaxes(title_text="Decompression Throughput (GB/s)", row=1, col=1)
    
    fig.update_layout(
        height=600,
        width=400 * n_types,
        title_text="Decompression Performance by Data Type and Algorithm",
        title_font_size=22,
        template='plotly_white',
        font=dict(size=11),
        barmode='group',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    output_file = output_dir / 'faceted_decompression.png'
    fig.write_image(str(output_file), width=400 * n_types, height=720, scale=2)
    print(f"✓ Created: {output_file.name}")

def plot_faceted_ratio(df, output_dir):
    """
    Faceted chart: Compression ratio by data type.
    """
    data_types = sorted(df['DataType'].unique())
    n_types = len(data_types)
    
    fig = make_subplots(
        rows=1, cols=n_types,
        subplot_titles=data_types,
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    colors = {'lz4': '#3498db', 'snappy': '#e74c3c', 'cascaded': '#2ecc71'}
    algorithms = sorted(df['Algorithm'].unique())
    sizes = sorted(df['SizeCategory'].unique(), 
                  key=lambda x: df[df['SizeCategory']==x]['SizeOrder'].iloc[0])
    
    for col_idx, dtype in enumerate(data_types, 1):
        dtype_data = df[df['DataType'] == dtype]
        
        for algo in algorithms:
            algo_data = dtype_data[dtype_data['Algorithm'] == algo]
            
            values = []
            labels = []
            for size in sizes:
                size_data = algo_data[algo_data['SizeCategory'] == size]
                if len(size_data) > 0:
                    values.append(size_data['CompressionRatio'].mean())
                    labels.append(size)
                else:
                    values.append(0)
                    labels.append(size)
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    name=algo.upper(),
                    marker_color=colors.get(algo, '#85929E'),
                    text=[f'{v:.1f}x' if v > 0 else '' for v in values],
                    textposition='outside',
                    textfont=dict(size=9),
                    legendgroup=algo,
                    showlegend=(col_idx == 1)
                ),
                row=1, col=col_idx
            )
        
        fig.update_xaxes(title_text="File Size", row=1, col=col_idx)
    
    fig.update_yaxes(title_text="Compression Ratio (x)", type="log", row=1, col=1)
    
    fig.update_layout(
        height=600,
        width=400 * n_types,
        title_text="Compression Ratio by Data Type and Algorithm",
        title_font_size=22,
        template='plotly_white',
        font=dict(size=11),
        barmode='group',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    output_file = output_dir / 'faceted_ratio.png'
    fig.write_image(str(output_file), width=400 * n_types, height=720, scale=2)
    print(f"✓ Created: {output_file.name}")

def plot_combined_faceted_metrics(df, output_dir):
    """
    Combined view: All three metrics in a single figure.
    Rows = metrics (compression, decompression, ratio)
    Cols = data types
    """
    data_types = sorted(df['DataType'].unique())
    n_types = len(data_types)
    
    fig = make_subplots(
        rows=3, cols=n_types,
        subplot_titles=[f"{dtype}" for dtype in data_types] + 
                       [f"{dtype}" for dtype in data_types] +
                       [f"{dtype}" for dtype in data_types],
        shared_yaxes='rows',
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        specs=[[{"type": "bar"} for _ in range(n_types)],
               [{"type": "bar"} for _ in range(n_types)],
               [{"type": "bar"} for _ in range(n_types)]]
    )
    
    colors = {'lz4': '#5DADE2', 'snappy': '#EC7063', 'cascaded': '#58D68D'}  # Light modern colors
    algorithms = sorted(df['Algorithm'].unique())
    sizes = sorted(df['SizeCategory'].unique(), 
                  key=lambda x: df[df['SizeCategory']==x]['SizeOrder'].iloc[0])
    
    # Row 1: Compression
    for col_idx, dtype in enumerate(data_types, 1):
        dtype_data = df[df['DataType'] == dtype]
        
        for algo in algorithms:
            algo_data = dtype_data[dtype_data['Algorithm'] == algo]
            
            values = []
            labels = []
            for size in sizes:
                size_data = algo_data[algo_data['SizeCategory'] == size]
                if len(size_data) > 0:
                    values.append(size_data['CompressionThroughput_GBps'].mean())
                    labels.append(size)
                else:
                    values.append(0)
                    labels.append(size)
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    name=algo.upper(),
                    marker_color=colors.get(algo, '#85929E'),
                    text=[f'{v:.1f}' if v > 0 else '' for v in values],
                    textposition='outside',
                    textfont=dict(size=8),
                    legendgroup=algo,
                    showlegend=(col_idx == 1)
                ),
                row=1, col=col_idx
            )
    
    # Row 2: Decompression
    for col_idx, dtype in enumerate(data_types, 1):
        dtype_data = df[df['DataType'] == dtype]
        
        for algo in algorithms:
            algo_data = dtype_data[dtype_data['Algorithm'] == algo]
            
            values = []
            labels = []
            for size in sizes:
                size_data = algo_data[algo_data['SizeCategory'] == size]
                if len(size_data) > 0:
                    values.append(size_data['DecompressionThroughput_GBps'].mean())
                    labels.append(size)
                else:
                    values.append(0)
                    labels.append(size)
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    name=algo.upper(),
                    marker_color=colors.get(algo, '#85929E'),
                    text=[f'{v:.1f}' if v > 0 else '' for v in values],
                    textposition='outside',
                    textfont=dict(size=8),
                    legendgroup=algo,
                    showlegend=False
                ),
                row=2, col=col_idx
            )
    
    # Row 3: Ratio
    for col_idx, dtype in enumerate(data_types, 1):
        dtype_data = df[df['DataType'] == dtype]
        
        for algo in algorithms:
            algo_data = dtype_data[dtype_data['Algorithm'] == algo]
            
            values = []
            labels = []
            for size in sizes:
                size_data = algo_data[algo_data['SizeCategory'] == size]
                if len(size_data) > 0:
                    values.append(size_data['CompressionRatio'].mean())
                    labels.append(size)
                else:
                    values.append(0)
                    labels.append(size)
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    name=algo.upper(),
                    marker_color=colors.get(algo, '#85929E'),
                    text=[f'{v:.1f}x' if v > 0 else '' for v in values],
                    textposition='outside',
                    textfont=dict(size=8),
                    legendgroup=algo,
                    showlegend=False
                ),
                row=3, col=col_idx
            )
        
        # Update x-axis for bottom row only
        fig.update_xaxes(title_text="File Size", row=3, col=col_idx)
    
    # Update y-axes
    fig.update_yaxes(title_text="Compression (GB/s)", row=1, col=1)
    fig.update_yaxes(title_text="Decompression (GB/s)", row=2, col=1)
    fig.update_yaxes(title_text="Ratio (x)", type="log", row=3, col=1)
    
    # Add row labels on the right side
    fig.add_annotation(
        text="<b>Compression</b>",
        xref="paper", yref="paper",
        x=1.02, y=0.85,
        showarrow=False,
        textangle=-90,
        font=dict(size=14)
    )
    fig.add_annotation(
        text="<b>Decompression</b>",
        xref="paper", yref="paper",
        x=1.02, y=0.50,
        showarrow=False,
        textangle=-90,
        font=dict(size=14)
    )
    fig.add_annotation(
        text="<b>Ratio</b>",
        xref="paper", yref="paper",
        x=1.02, y=0.15,
        showarrow=False,
        textangle=-90,
        font=dict(size=14)
    )
    
    fig.update_layout(
        height=1400,
        width=400 * n_types,
        title_text="Complete Performance Comparison: All Metrics by Data Type and Algorithm",
        title_font_size=24,
        template='plotly_white',
        font=dict(size=10),
        barmode='group',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="center",
            x=0.5
        )
    )
    
    output_file = output_dir / 'faceted_all_metrics.png'
    fig.write_image(str(output_file), width=400 * n_types, height=1680, scale=2)
    print(f"✓ Created: {output_file.name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python faceted_charts.py <path_to_results.csv>")
        print("\nExample:")
        print("  python faceted_charts.py results/feature2_wave64/20251201_170939_87a26a0/results.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"\n{'='*90}")
    print(f"Feature 2 Wave64 - Faceted Charts Generation")
    print(f"{'='*90}")
    print(f"\nLoading data from: {csv_path}")
    
    # Load and process data
    df = load_csv_data(csv_path)
    df = add_file_metadata(df)
    
    print(f"  Total tests: {len(df)}")
    print(f"  Algorithms: {', '.join(sorted(df['Algorithm'].unique()))}")
    print(f"  Data types: {', '.join(sorted(df['DataType'].unique()))}")
    
    # Create output directory
    output_dir = csv_path.parent / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nGenerating faceted visualizations in: {output_dir}")
    print(f"{'-'*90}")
    
    # Generate faceted plots
    plot_faceted_compression(df, output_dir)
    plot_faceted_decompression(df, output_dir)
    plot_faceted_ratio(df, output_dir)
    plot_combined_faceted_metrics(df, output_dir)
    
    print(f"\n{'='*90}")
    print(f"✅ Faceted visualization complete!")
    print(f"\nGenerated charts:")
    print(f"  📊 faceted_compression.png - Compression throughput by data type")
    print(f"  📊 faceted_decompression.png - Decompression throughput by data type")
    print(f"  📊 faceted_ratio.png - Compression ratio by data type")
    print(f"  📊 faceted_all_metrics.png - Combined view (all metrics)")
    print(f"\nLocation: {output_dir.absolute()}")
    print(f"{'='*90}\n")

if __name__ == '__main__':
    main()
