#!/usr/bin/env python3
"""
Convert RSF (Madagascar Seismic File Format) to raw binary format for benchmarks.

RSF format consists of:
- .rsf file: ASCII header with metadata
- .rsf@ file: binary data

This script extracts the binary data and optionally validates dimensions.
No external dependencies required (pure Python).
"""

import argparse
import os
import sys
import struct


def parse_rsf_header(rsf_file):
    """Parse RSF header file and extract metadata."""
    metadata = {}
    
    with open(rsf_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse key=value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"')
                metadata[key] = value
    
    return metadata


def get_data_info(metadata):
    """Extract data format and dimensions from metadata."""
    info = {
        'data_format': metadata.get('data_format', 'native_float'),
        'esize': int(metadata.get('esize', 4)),
        'dimensions': [],
        'spacing': [],
        'total_elements': 1
    }
    
    # Extract dimensions (n1, n2, n3, ...)
    i = 1
    while f'n{i}' in metadata:
        dim = int(metadata[f'n{i}'])
        info['dimensions'].append(dim)
        info['total_elements'] *= dim
        i += 1
    
    # Extract spacing (d1, d2, d3, ...)
    i = 1
    while f'd{i}' in metadata:
        spacing = float(metadata[f'd{i}'])
        info['spacing'].append(spacing)
        i += 1
    
    return info


def determine_dtype(data_format, esize):
    """Determine data type info from RSF data format."""
    format_map = {
        'native_float': ('f', 4, 'float32'),
        'native_double': ('d', 8, 'float64'),
        'native_int': ('i', 4, 'int32'),
        'native_short': ('h', 2, 'int16'),
        'native_char': ('b', 1, 'int8'),
        'native_uchar': ('B', 1, 'uint8'),
    }
    
    if data_format in format_map:
        dtype_char, expected_size, dtype_name = format_map[data_format]
        if esize == expected_size:
            return {'char': dtype_char, 'size': esize, 'name': dtype_name}
    
    # Fallback based on esize
    if esize == 4:
        return {'char': 'f', 'size': 4, 'name': 'float32'}
    elif esize == 8:
        return {'char': 'd', 'size': 8, 'name': 'float64'}
    elif esize == 2:
        return {'char': 'h', 'size': 2, 'name': 'int16'}
    elif esize == 1:
        return {'char': 'b', 'size': 1, 'name': 'int8'}
    else:
        raise ValueError(f"Unsupported data format: {data_format} with esize {esize}")


def compute_statistics(binary_path, dtype_info, num_elements, max_samples=10000):
    """Compute basic statistics from binary data without loading entire file."""
    
    # Determine how many elements to sample
    sample_size = min(num_elements, max_samples)
    
    values = []
    with open(binary_path, 'rb') as f:
        # Sample uniformly across the file
        if num_elements <= max_samples:
            # Read all
            for i in range(num_elements):
                raw = f.read(dtype_info['size'])
                if len(raw) < dtype_info['size']:
                    break
                val = struct.unpack(dtype_info['char'], raw)[0]
                values.append(val)
        else:
            # Sample at regular intervals
            step = num_elements // max_samples
            for i in range(0, num_elements, step):
                f.seek(i * dtype_info['size'])
                raw = f.read(dtype_info['size'])
                if len(raw) < dtype_info['size']:
                    break
                val = struct.unpack(dtype_info['char'], raw)[0]
                values.append(val)
    
    if not values:
        return None
    
    # Compute statistics
    min_val = min(values)
    max_val = max(values)
    mean_val = sum(values) / len(values)
    
    # Compute standard deviation
    variance = sum((x - mean_val) ** 2 for x in values) / len(values)
    std_val = variance ** 0.5
    
    return {
        'min': min_val,
        'max': max_val,
        'mean': mean_val,
        'std': std_val,
        'samples': len(values)
    }


def convert_rsf_to_binary(rsf_header_file, output_file, validate=True, verbose=True):
    """Convert RSF format to raw binary."""
    
    # Parse header
    if verbose:
        print(f"Reading RSF header: {rsf_header_file}")
    
    metadata = parse_rsf_header(rsf_header_file)
    info = get_data_info(metadata)
    
    if verbose:
        print(f"  Data format: {info['data_format']}")
        print(f"  Element size: {info['esize']} bytes")
        print(f"  Dimensions: {' × '.join(map(str, info['dimensions']))}")
        print(f"  Total elements: {info['total_elements']:,}")
        print(f"  Expected file size: {info['total_elements'] * info['esize']:,} bytes")
    
    # Determine binary data file
    binary_file = metadata.get('in', '').strip('"')
    if binary_file.startswith('./'):
        binary_file = binary_file[2:]
    
    # Resolve path relative to header file
    header_dir = os.path.dirname(os.path.abspath(rsf_header_file))
    binary_path = os.path.join(header_dir, binary_file)
    
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"Binary data file not found: {binary_path}")
    
    # Check file size
    actual_size = os.path.getsize(binary_path)
    expected_size = info['total_elements'] * info['esize']
    
    if verbose:
        print(f"\nReading binary data: {binary_path}")
        print(f"  Actual file size: {actual_size:,} bytes")
    
    if validate and actual_size != expected_size:
        print(f"  WARNING: File size mismatch!")
        print(f"  Expected: {expected_size:,} bytes")
        print(f"  Actual: {actual_size:,} bytes")
        
        if actual_size < expected_size:
            raise ValueError("Binary file is smaller than expected!")
    
    # Read and optionally validate data
    if validate:
        dtype_info = determine_dtype(info['data_format'], info['esize'])
        if verbose:
            print(f"  Data type: {dtype_info['name']}")
        
        # Compute statistics on sample
        if verbose:
            print(f"\nComputing data statistics (sampling)...")
        
        stats = compute_statistics(binary_path, dtype_info, info['total_elements'])
        
        if stats and verbose:
            print(f"  Samples analyzed: {stats['samples']:,} / {info['total_elements']:,}")
            print(f"  Min value: {stats['min']}")
            print(f"  Max value: {stats['max']}")
            print(f"  Mean value: {stats['mean']:.6f}")
            print(f"  Std dev: {stats['std']:.6f}")
    
    # Simple binary copy (works for both validate and non-validate modes)
    if verbose:
        print(f"\nCopying binary data to: {output_file}")
    
    with open(binary_path, 'rb') as src, open(output_file, 'wb') as dst:
        # Copy in chunks for large files
        chunk_size = 64 * 1024 * 1024  # 64 MB chunks
        copied = 0
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
            if verbose and copied % (512 * 1024 * 1024) == 0:  # Print every 512 MB
                print(f"  Copied: {copied / (1024**3):.2f} GB...")
    
    output_size = os.path.getsize(output_file)
    
    if verbose:
        print(f"  Output size: {output_size:,} bytes ({output_size / (1024**2):.2f} MB)")
        print(f"\n✓ Conversion complete!")
    
    return {
        'metadata': metadata,
        'info': info,
        'output_file': output_file,
        'output_size': output_size
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert RSF format to raw binary for hipCOMP benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert RSF to binary
  %(prog)s TTI.rsf TTI_float32.bin
  
  # Convert without validation (faster for large files)
  %(prog)s TTI.rsf TTI_float32.bin --no-validate
  
  # Quiet mode
  %(prog)s TTI.rsf TTI_float32.bin -q
"""
    )
    
    parser.add_argument('rsf_file', 
                        help='Input RSF header file (.rsf)')
    parser.add_argument('output_file', 
                        help='Output raw binary file (.bin)')
    parser.add_argument('--no-validate', 
                        action='store_true',
                        help='Skip validation (faster, just copy binary data)')
    parser.add_argument('-q', '--quiet', 
                        action='store_true',
                        help='Quiet mode (minimal output)')
    
    args = parser.parse_args()
    
    try:
        result = convert_rsf_to_binary(
            args.rsf_file,
            args.output_file,
            validate=not args.no_validate,
            verbose=not args.quiet
        )
        
        if args.quiet:
            print(f"{result['output_file']}: {result['output_size']} bytes")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
