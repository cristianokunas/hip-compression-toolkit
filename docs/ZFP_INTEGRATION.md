# ZFP Integration for hipCOMP

## Overview

This document describes the integration of ZFP (Zstandard Floating Point) compression library with hipCOMP for GPU-accelerated floating-point data compression.

## Features

- **GPU-native ZFP compression** via HIP backend
- **FIXED_RATE mode** for predictable compression ratios
- Separate `zfp_hip_impl` library to avoid CMake flag contamination
- Compatible with existing hipCOMP API patterns

## GPU Limitations

**Important**: The ZFP GPU backend (CUDA/HIP) only supports **FIXED_RATE mode**.

| Mode | GPU Support | CPU Support | Description |
|------|-------------|-------------|-------------|
| FIXED_RATE | ✅ | ✅ | Fixed bits per value |
| FIXED_PRECISION | ❌ | ✅ | Fixed significant bits |
| FIXED_ACCURACY | ❌ | ✅ | Maximum error tolerance |
| REVERSIBLE | ❌ | ✅ | Lossless compression |

Reference: [ZFP CUDA Limitations](https://zfp.readthedocs.io/en/release1.0.1/execution.html#cuda-limitations)

## API Usage

### Include Header
```c
#include <hipcomp/zfp.h>
```

### Initialize Options
```c
// Create options for 3D float array with rate=16 bits/value
hipcompZfpOpts opts = hipcompZfpDefaultOpts(nx, ny, nz, 16.0);
opts.type = HIPCOMP_ZFP_TYPE_FLOAT;
opts.dims = 3;
opts.mode = HIPCOMP_ZFP_MODE_FIXED_RATE;
```

### Compress
```c
size_t compressed_size;
hipcompZfpCompressAsync(
    d_input,           // Device pointer to input data
    &opts,             // Compression options
    NULL, 0,           // Temp buffer (managed internally)
    d_output,          // Device pointer to output buffer
    &compressed_size,  // Output: compressed size
    stream);           // HIP stream
```

### Decompress
```c
hipcompZfpDecompressAsync(
    d_compressed,      // Device pointer to compressed data
    compressed_size,   // Size of compressed data
    &opts,             // Same options used for compression
    NULL, 0,           // Temp buffer (managed internally)
    d_output,          // Device pointer to output buffer
    stream);           // HIP stream
```

## Rate vs Compression Ratio

| ZFP Rate | Bits/Value | Compression | Typical Error |
|----------|------------|-------------|---------------|
| 32.0 | 32 bits | ~1x | ~10⁻⁷ |
| 24.0 | 24 bits | ~1.33x | ~10⁻⁵ |
| 16.0 | 16 bits | ~2x | ~10⁻³ |
| 8.0 | 8 bits | ~4x | ~10⁻¹ |

## Building

ZFP support is enabled by default when building hipCOMP:

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
make -j$(nproc)
make install
```

This produces two libraries:
- `libhipcomp.so` - Main hipCOMP library
- `libzfp_hip_impl.so` - ZFP HIP implementation

## Linking

```bash
-L/path/to/hipcomp/lib -lhipcomp -lzfp_hip_impl -Wl,-rpath,/path/to/hipcomp/lib
```

## Architecture Notes

### Separate Library Design

The ZFP HIP implementation is built as a separate library (`zfp_hip_impl`) to avoid CMake flag contamination issues. HIP compilation requires specific flags (`--offload-arch=gfxXXXX`) that conflict with standard C/C++ compilation.

### File Extensions

- `.hip` files: Compiled with hipcc and HIP-specific flags
- `.cpp` files: Compiled with standard C++ compiler

## Error Validation

For scientific applications, error metrics can be computed:

```c
// Enable validation (adds overhead)
// Set ZFP_VALIDATE=1 environment variable before running

// Metrics available after compression:
// - Max Absolute Error
// - Max Relative Error  
// - RMSE (Root Mean Square Error)
// - PSNR (Peak Signal-to-Noise Ratio in dB)
```

## Performance Considerations

1. **GPU Memory**: ZFP operates on device memory; no host-device copies needed
2. **Fixed-Rate Predictability**: Output size is predictable: `input_size * (rate/32)`
3. **Batch Processing**: For multiple arrays, consider batching for better GPU utilization

## References

- [ZFP Documentation](https://zfp.readthedocs.io/)
- [ZFP GitHub](https://github.com/LLNL/zfp)
- [hipCOMP Documentation](../README.md)
