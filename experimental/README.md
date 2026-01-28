# Experimental Algorithms

This directory contains compression algorithms that are **not yet fully supported** in hipcomp.

## Status

| Algorithm | Status | Notes |
|-----------|--------|-------|
| ANS | 🚧 Experimental | Asymmetric Numeral Systems - requires further testing |
| Bitcomp | 🚧 Experimental | Bit-level compression - not validated on AMD GPUs |
| GDeflate | 🚧 Experimental | GPU-accelerated DEFLATE - incomplete port |

## Why Experimental?

These algorithms were ported from nvCOMP 2.2 but have not been:
- Fully tested on AMD GPUs
- Validated for correctness
- Optimized for AMD architectures (RDNA/CDNA)

## Supported Algorithms

The following algorithms are **production-ready** in the main library:

- **LZ4** - Fast lossless compression ✅
- **Snappy** - High-speed compression ✅
- **Cascaded** - Multi-stage compression (Delta + RLE + Bit-packing) ✅

## Contributing

If you'd like to help complete the port of these algorithms, contributions are welcome!
Please open an issue or pull request on the main repository.
