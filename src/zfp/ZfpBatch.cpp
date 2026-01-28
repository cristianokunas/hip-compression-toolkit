/*
 * Copyright (c) 2024, hipCOMP contributors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ZFP compression wrapper implementation for hipCOMP
 */

#include "hipcomp/zfp.h"
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <string.h>

/* Include ZFP headers */
#include "../../external/zfp/include/zfp.h"

/* Static state for ZFP */
static int zfp_initialized = 0;

hipcompZfpOpts hipcompZfpDefaultOpts(size_t nx, size_t ny, size_t nz, double rate)
{
    hipcompZfpOpts opts;
    memset(&opts, 0, sizeof(opts));
    
    opts.mode = HIPCOMP_ZFP_MODE_FIXED_RATE;
    opts.type = HIPCOMP_ZFP_TYPE_FLOAT;
    opts.dims = 3;
    opts.nx = nx;
    opts.ny = ny;
    opts.nz = nz;
    opts.rate = rate;
    opts.precision = 0;
    opts.tolerance = 0.0;
    
    return opts;
}

hipcompStatus_t hipcompZfpInit(void)
{
    if (zfp_initialized) {
        return hipcompSuccess;
    }
    
    /* Verify HIP is available */
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        return hipcompErrorNotSupported;
    }
    
    zfp_initialized = 1;
    return hipcompSuccess;
}

void hipcompZfpFinalize(void)
{
    zfp_initialized = 0;
}

/* Helper to convert hipcomp type to zfp type */
static zfp_type get_zfp_type(hipcompZfpType type)
{
    switch (type) {
        case HIPCOMP_ZFP_TYPE_INT32:  return zfp_type_int32;
        case HIPCOMP_ZFP_TYPE_INT64:  return zfp_type_int64;
        case HIPCOMP_ZFP_TYPE_FLOAT:  return zfp_type_float;
        case HIPCOMP_ZFP_TYPE_DOUBLE: return zfp_type_double;
        default: return zfp_type_float;
    }
}

/* Helper to get element size */
static size_t get_element_size(hipcompZfpType type)
{
    switch (type) {
        case HIPCOMP_ZFP_TYPE_INT32:  return sizeof(int32_t);
        case HIPCOMP_ZFP_TYPE_INT64:  return sizeof(int64_t);
        case HIPCOMP_ZFP_TYPE_FLOAT:  return sizeof(float);
        case HIPCOMP_ZFP_TYPE_DOUBLE: return sizeof(double);
        default: return sizeof(float);
    }
}

hipcompStatus_t hipcompZfpCompressGetTempSize(
    const hipcompZfpOpts* opts,
    size_t* temp_bytes)
{
    if (!opts || !temp_bytes) {
        return hipcompErrorInvalidValue;
    }
    
    /* ZFP HIP backend needs minimal temp space for stream management */
    *temp_bytes = 1024 * 1024; /* 1MB should be sufficient */
    return hipcompSuccess;
}

hipcompStatus_t hipcompZfpCompressGetMaxOutputSize(
    const hipcompZfpOpts* opts,
    size_t* max_compressed_bytes)
{
    if (!opts || !max_compressed_bytes) {
        return hipcompErrorInvalidValue;
    }
    
    /* Calculate total elements */
    size_t num_elements = opts->nx;
    if (opts->dims >= 2) num_elements *= opts->ny;
    if (opts->dims >= 3) num_elements *= opts->nz;
    
    size_t element_size = get_element_size(opts->type);
    size_t uncompressed_size = num_elements * element_size;
    
    /* ZFP worst case is slightly larger than input + header overhead */
    /* Using rate, the compressed size is predictable */
    if (opts->mode == HIPCOMP_ZFP_MODE_FIXED_RATE && opts->rate > 0) {
        /* Fixed rate: bits_per_element * num_elements / 8 + header */
        *max_compressed_bytes = (size_t)(opts->rate * num_elements / 8.0) + 4096;
    } else {
        /* Other modes: assume worst case is 1.1x input + header */
        *max_compressed_bytes = (size_t)(uncompressed_size * 1.1) + 4096;
    }
    
    return hipcompSuccess;
}

hipcompStatus_t hipcompZfpCompressAsync(
    const void* d_in,
    const hipcompZfpOpts* opts,
    void* d_temp,
    size_t temp_bytes,
    void* d_out,
    size_t* d_out_bytes,
    hipStream_t stream)
{
    if (!d_in || !opts || !d_out || !d_out_bytes) {
        return hipcompErrorInvalidValue;
    }
    
    if (!zfp_initialized) {
        hipcompStatus_t status = hipcompZfpInit();
        if (status != hipcompSuccess) {
            return status;
        }
    }
    
    /* Calculate sizes */
    size_t num_elements = opts->nx;
    if (opts->dims >= 2) num_elements *= opts->ny;
    if (opts->dims >= 3) num_elements *= opts->nz;
    
    size_t element_size = get_element_size(opts->type);
    size_t uncompressed_size = num_elements * element_size;
    
    /* Get max output size for buffer allocation */
    size_t max_out_size;
    hipcompZfpCompressGetMaxOutputSize(opts, &max_out_size);
    
    /* Allocate host buffers for ZFP (ZFP HIP backend handles device memory internally) */
    void* h_in = malloc(uncompressed_size);
    void* h_out = malloc(max_out_size);
    if (!h_in || !h_out) {
        free(h_in);
        free(h_out);
        return hipcompErrorNotSupported;
    }
    
    /* Copy input from device to host */
    hipError_t err = hipMemcpyAsync(h_in, d_in, uncompressed_size, hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    hipStreamSynchronize(stream);
    
    /* Setup ZFP field */
    zfp_type type = get_zfp_type(opts->type);
    zfp_field* field = NULL;
    
    switch (opts->dims) {
        case 1:
            field = zfp_field_1d(h_in, type, opts->nx);
            break;
        case 2:
            field = zfp_field_2d(h_in, type, opts->nx, opts->ny);
            break;
        case 3:
            field = zfp_field_3d(h_in, type, opts->nx, opts->ny, opts->nz);
            break;
        default:
            free(h_in);
            free(h_out);
            return hipcompErrorInvalidValue;
    }
    
    if (!field) {
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    /* Setup ZFP stream */
    zfp_stream* zfp = zfp_stream_open(NULL);
    if (!zfp) {
        zfp_field_free(field);
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    /* Set compression mode */
    switch (opts->mode) {
        case HIPCOMP_ZFP_MODE_FIXED_RATE:
            zfp_stream_set_rate(zfp, opts->rate, type, opts->dims, zfp_false);
            break;
        case HIPCOMP_ZFP_MODE_FIXED_PRECISION:
            zfp_stream_set_precision(zfp, opts->precision);
            break;
        case HIPCOMP_ZFP_MODE_FIXED_ACCURACY:
            zfp_stream_set_accuracy(zfp, opts->tolerance);
            break;
        case HIPCOMP_ZFP_MODE_REVERSIBLE:
            zfp_stream_set_reversible(zfp);
            break;
    }
    
    /* Set execution policy to HIP if available */
#ifdef ZFP_WITH_HIP
    if (!zfp_stream_set_execution(zfp, zfp_exec_hip)) {
        /* Fall back to serial if HIP execution fails */
        zfp_stream_set_execution(zfp, zfp_exec_serial);
    }
#endif
    
    /* Allocate bit stream */
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    bitstream* bs = stream_open(h_out, bufsize);
    if (!bs) {
        zfp_stream_close(zfp);
        zfp_field_free(field);
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    zfp_stream_set_bit_stream(zfp, bs);
    zfp_stream_rewind(zfp);
    
    /* Compress */
    size_t compressed_size = zfp_compress(zfp, field);
    
    /* Cleanup ZFP */
    stream_close(bs);
    zfp_stream_close(zfp);
    zfp_field_free(field);
    
    if (compressed_size == 0) {
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    /* Copy output to device */
    err = hipMemcpyAsync(d_out, h_out, compressed_size, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    /* Store output size */
    *d_out_bytes = compressed_size;
    
    free(h_in);
    free(h_out);
    
    return hipcompSuccess;
}

hipcompStatus_t hipcompZfpDecompressGetTempSize(
    const hipcompZfpOpts* opts,
    size_t* temp_bytes)
{
    if (!opts || !temp_bytes) {
        return hipcompErrorInvalidValue;
    }
    
    *temp_bytes = 1024 * 1024; /* 1MB */
    return hipcompSuccess;
}

hipcompStatus_t hipcompZfpDecompressAsync(
    const void* d_in,
    size_t in_bytes,
    const hipcompZfpOpts* opts,
    void* d_temp,
    size_t temp_bytes,
    void* d_out,
    hipStream_t stream)
{
    if (!d_in || !opts || !d_out || in_bytes == 0) {
        return hipcompErrorInvalidValue;
    }
    
    if (!zfp_initialized) {
        hipcompStatus_t status = hipcompZfpInit();
        if (status != hipcompSuccess) {
            return status;
        }
    }
    
    /* Calculate output size */
    size_t num_elements = opts->nx;
    if (opts->dims >= 2) num_elements *= opts->ny;
    if (opts->dims >= 3) num_elements *= opts->nz;
    
    size_t element_size = get_element_size(opts->type);
    size_t uncompressed_size = num_elements * element_size;
    
    /* Allocate host buffers */
    void* h_in = malloc(in_bytes);
    void* h_out = malloc(uncompressed_size);
    if (!h_in || !h_out) {
        free(h_in);
        free(h_out);
        return hipcompErrorNotSupported;
    }
    
    /* Copy compressed data from device to host */
    hipError_t err = hipMemcpyAsync(h_in, d_in, in_bytes, hipMemcpyDeviceToHost, stream);
    if (err != hipSuccess) {
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    hipStreamSynchronize(stream);
    
    /* Setup ZFP field */
    zfp_type type = get_zfp_type(opts->type);
    zfp_field* field = NULL;
    
    switch (opts->dims) {
        case 1:
            field = zfp_field_1d(h_out, type, opts->nx);
            break;
        case 2:
            field = zfp_field_2d(h_out, type, opts->nx, opts->ny);
            break;
        case 3:
            field = zfp_field_3d(h_out, type, opts->nx, opts->ny, opts->nz);
            break;
        default:
            free(h_in);
            free(h_out);
            return hipcompErrorInvalidValue;
    }
    
    if (!field) {
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    /* Setup ZFP stream */
    zfp_stream* zfp = zfp_stream_open(NULL);
    if (!zfp) {
        zfp_field_free(field);
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    /* Set compression mode (must match compression) */
    switch (opts->mode) {
        case HIPCOMP_ZFP_MODE_FIXED_RATE:
            zfp_stream_set_rate(zfp, opts->rate, type, opts->dims, zfp_false);
            break;
        case HIPCOMP_ZFP_MODE_FIXED_PRECISION:
            zfp_stream_set_precision(zfp, opts->precision);
            break;
        case HIPCOMP_ZFP_MODE_FIXED_ACCURACY:
            zfp_stream_set_accuracy(zfp, opts->tolerance);
            break;
        case HIPCOMP_ZFP_MODE_REVERSIBLE:
            zfp_stream_set_reversible(zfp);
            break;
    }
    
    /* Set execution policy to HIP if available */
#ifdef ZFP_WITH_HIP
    if (!zfp_stream_set_execution(zfp, zfp_exec_hip)) {
        zfp_stream_set_execution(zfp, zfp_exec_serial);
    }
#endif
    
    /* Allocate bit stream */
    bitstream* bs = stream_open(h_in, in_bytes);
    if (!bs) {
        zfp_stream_close(zfp);
        zfp_field_free(field);
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    zfp_stream_set_bit_stream(zfp, bs);
    zfp_stream_rewind(zfp);
    
    /* Decompress */
    size_t decompressed_size = zfp_decompress(zfp, field);
    
    /* Cleanup ZFP */
    stream_close(bs);
    zfp_stream_close(zfp);
    zfp_field_free(field);
    
    if (decompressed_size == 0) {
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    /* Copy output to device */
    err = hipMemcpyAsync(d_out, h_out, uncompressed_size, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        free(h_in);
        free(h_out);
        return hipcompErrorInternal;
    }
    
    free(h_in);
    free(h_out);
    
    return hipcompSuccess;
}
