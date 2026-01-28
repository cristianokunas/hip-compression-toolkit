/*
 * Copyright (c) 2024, hipCOMP contributors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ZFP compression wrapper for hipCOMP
 * ZFP is a library for compressed numerical arrays that support high throughput
 * read and write random access. This wrapper provides GPU-accelerated compression
 * using HIP for AMD GPUs.
 *
 * ZFP Original License: BSD-3-Clause (Lawrence Livermore National Security)
 */

#ifndef HIPCOMP_ZFP_H
#define HIPCOMP_ZFP_H

#include "hipcomp.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief ZFP compression mode
 */
typedef enum {
    HIPCOMP_ZFP_MODE_FIXED_RATE = 0,      /**< Fixed bits per value */
    HIPCOMP_ZFP_MODE_FIXED_PRECISION = 1, /**< Fixed precision (bit planes) */
    HIPCOMP_ZFP_MODE_FIXED_ACCURACY = 2,  /**< Fixed accuracy (error tolerance) */
    HIPCOMP_ZFP_MODE_REVERSIBLE = 3       /**< Lossless compression */
} hipcompZfpMode;

/**
 * @brief ZFP data type
 */
typedef enum {
    HIPCOMP_ZFP_TYPE_INT32 = 0,   /**< 32-bit signed integer */
    HIPCOMP_ZFP_TYPE_INT64 = 1,   /**< 64-bit signed integer */
    HIPCOMP_ZFP_TYPE_FLOAT = 2,   /**< 32-bit floating point */
    HIPCOMP_ZFP_TYPE_DOUBLE = 3   /**< 64-bit floating point */
} hipcompZfpType;

/**
 * @brief ZFP compression options
 */
typedef struct {
    hipcompZfpMode mode;     /**< Compression mode */
    hipcompZfpType type;     /**< Data type */
    int dims;                /**< Dimensionality (1, 2, or 3) */
    size_t nx;               /**< Size in x dimension */
    size_t ny;               /**< Size in y dimension (ignored for 1D) */
    size_t nz;               /**< Size in z dimension (ignored for 1D/2D) */
    
    /* Mode-specific parameters */
    double rate;             /**< Bits per value (fixed-rate mode) */
    uint32_t precision;      /**< Bit planes (fixed-precision mode) */
    double tolerance;        /**< Error tolerance (fixed-accuracy mode) */
} hipcompZfpOpts;

/**
 * @brief Get default ZFP options for 3D float array with fixed-rate compression
 * 
 * @param nx Size in x dimension
 * @param ny Size in y dimension
 * @param nz Size in z dimension
 * @param rate Bits per value (e.g., 8.0 for 8 bits per float = 4x compression)
 * @return Default options structure
 */
hipcompZfpOpts hipcompZfpDefaultOpts(size_t nx, size_t ny, size_t nz, double rate);

/**
 * @brief Initialize ZFP compression on GPU
 * 
 * @return hipcompSuccess on success
 */
hipcompStatus_t hipcompZfpInit(void);

/**
 * @brief Finalize ZFP compression (cleanup resources)
 */
void hipcompZfpFinalize(void);

/**
 * @brief Get temporary buffer size for ZFP compression
 * 
 * @param opts ZFP options
 * @param temp_bytes Output: required temporary buffer size
 * @return hipcompSuccess on success
 */
hipcompStatus_t hipcompZfpCompressGetTempSize(
    const hipcompZfpOpts* opts,
    size_t* temp_bytes
);

/**
 * @brief Get maximum compressed output size
 * 
 * @param opts ZFP options
 * @param max_compressed_bytes Output: maximum compressed size
 * @return hipcompSuccess on success
 */
hipcompStatus_t hipcompZfpCompressGetMaxOutputSize(
    const hipcompZfpOpts* opts,
    size_t* max_compressed_bytes
);

/**
 * @brief Compress data using ZFP on GPU
 * 
 * @param d_in Device pointer to input data
 * @param opts ZFP options (includes dimensions and type)
 * @param d_temp Device pointer to temporary buffer
 * @param temp_bytes Size of temporary buffer
 * @param d_out Device pointer to output buffer
 * @param d_out_bytes Device pointer to output size (will be written)
 * @param stream HIP stream
 * @return hipcompSuccess on success
 */
hipcompStatus_t hipcompZfpCompressAsync(
    const void* d_in,
    const hipcompZfpOpts* opts,
    void* d_temp,
    size_t temp_bytes,
    void* d_out,
    size_t* d_out_bytes,
    hipStream_t stream
);

/**
 * @brief Decompress ZFP-compressed data on GPU
 * 
 * @param d_in Device pointer to compressed data
 * @param in_bytes Size of compressed data
 * @param opts ZFP options (must match compression options)
 * @param d_temp Device pointer to temporary buffer
 * @param temp_bytes Size of temporary buffer
 * @param d_out Device pointer to output buffer
 * @param stream HIP stream
 * @return hipcompSuccess on success
 */
hipcompStatus_t hipcompZfpDecompressAsync(
    const void* d_in,
    size_t in_bytes,
    const hipcompZfpOpts* opts,
    void* d_temp,
    size_t temp_bytes,
    void* d_out,
    hipStream_t stream
);

/**
 * @brief Get decompression temporary buffer size
 * 
 * @param opts ZFP options
 * @param temp_bytes Output: required temporary buffer size
 * @return hipcompSuccess on success
 */
hipcompStatus_t hipcompZfpDecompressGetTempSize(
    const hipcompZfpOpts* opts,
    size_t* temp_bytes
);

#ifdef __cplusplus
}
#endif

#endif /* HIPCOMP_ZFP_H */
