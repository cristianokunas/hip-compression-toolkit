/*
 * Copyright (c) 2024, hipCOMP contributors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ZFP GPU-native compression wrapper for hipCOMP
 * Uses ZFP's native HIP backend for compression directly on GPU memory
 */

#include "hipcomp/zfp.h"
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Include ZFP headers */
#include "zfp.h"

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
    *temp_bytes = 0; /* ZFP HIP backend manages its own temp memory */
    return hipcompSuccess;
}

hipcompStatus_t hipcompZfpCompressGetMaxOutputSize(
    const hipcompZfpOpts* opts,
    size_t* max_compressed_bytes)
{
    if (!opts || !max_compressed_bytes) {
        return hipcompErrorInvalidValue;
    }
    
    size_t num_elements = opts->nx;
    if (opts->dims >= 2) num_elements *= opts->ny;
    if (opts->dims >= 3) num_elements *= opts->nz;
    
    size_t element_size = get_element_size(opts->type);
    size_t uncompressed_size = num_elements * element_size;
    
    if (opts->mode == HIPCOMP_ZFP_MODE_FIXED_RATE && opts->rate > 0) {
        *max_compressed_bytes = (size_t)(opts->rate * num_elements / 8.0) + 4096;
    } else {
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
    size_t* compressed_bytes,
    hipStream_t stream)
{
    (void)d_temp;
    (void)temp_bytes;
    
    if (!d_in || !opts || !d_out || !compressed_bytes) {
        return hipcompErrorInvalidValue;
    }
    
    if (!zfp_initialized) {
        hipcompStatus_t status = hipcompZfpInit();
        if (status != hipcompSuccess) return status;
    }
    
    /* Synchronize stream before ZFP operations */
    hipStreamSynchronize(stream);
    
    /* Calculate sizes */
    size_t num_elements = opts->nx;
    if (opts->dims >= 2) num_elements *= opts->ny;
    if (opts->dims >= 3) num_elements *= opts->nz;
    
    zfp_type type = get_zfp_type(opts->type);
    
    /* Create ZFP field pointing to device memory */
    zfp_field* field = NULL;
    switch (opts->dims) {
        case 1:
            field = zfp_field_1d((void*)d_in, type, opts->nx);
            break;
        case 2:
            field = zfp_field_2d((void*)d_in, type, opts->nx, opts->ny);
            break;
        case 3:
            field = zfp_field_3d((void*)d_in, type, opts->nx, opts->ny, opts->nz);
            break;
        default:
            return hipcompErrorInvalidValue;
    }
    
    if (!field) return hipcompErrorInternal;
    
    /* Create ZFP stream */
    zfp_stream* zfp = zfp_stream_open(NULL);
    if (!zfp) {
        zfp_field_free(field);
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
    
    /* Get maximum buffer size */
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    
    /* Try to use HIP execution (GPU-native compression) */
    zfp_bool use_gpu = zfp_false;
#ifdef ZFP_WITH_HIP
    use_gpu = zfp_stream_set_execution(zfp, zfp_exec_hip);
    if (use_gpu) {
        /* For HIP execution, d_out must be device memory - ZFP handles this */
        printf("[ZFP] Using GPU-native HIP compression\n");
    }
#endif
    
    if (!use_gpu) {
        /* Fallback: copy to host, compress, copy back */
        printf("[ZFP] Falling back to CPU compression (ZFP_WITH_HIP not enabled)\n");
        
        size_t element_size = get_element_size(opts->type);
        size_t data_size = num_elements * element_size;
        
        void* h_in = malloc(data_size);
        void* h_out = malloc(bufsize);
        if (!h_in || !h_out) {
            free(h_in);
            free(h_out);
            zfp_stream_close(zfp);
            zfp_field_free(field);
            return hipcompErrorInternal;
        }
        
        hipMemcpy(h_in, d_in, data_size, hipMemcpyDeviceToHost);
        
        /* Update field to point to host memory */
        zfp_field_set_pointer(field, h_in);
        
        /* Setup bitstream on host */
        bitstream* bs = stream_open(h_out, bufsize);
        zfp_stream_set_bit_stream(zfp, bs);
        zfp_stream_rewind(zfp);
        
        /* Compress */
        size_t zfpsize = zfp_compress(zfp, field);
        
        if (zfpsize == 0) {
            stream_close(bs);
            zfp_stream_close(zfp);
            zfp_field_free(field);
            free(h_in);
            free(h_out);
            return hipcompErrorInternal;
        }
        
        /* Copy compressed data back to device */
        hipMemcpy(d_out, h_out, zfpsize, hipMemcpyHostToDevice);
        *compressed_bytes = zfpsize;
        
        stream_close(bs);
        free(h_in);
        free(h_out);
    } else {
        /* GPU-native path: ZFP HIP backend handles device memory directly */
        /* Allocate device buffer for compressed output if needed */
        bitstream* bs = stream_open(d_out, bufsize);
        zfp_stream_set_bit_stream(zfp, bs);
        zfp_stream_rewind(zfp);
        
        size_t zfpsize = zfp_compress(zfp, field);
        
        if (zfpsize == 0) {
            stream_close(bs);
            zfp_stream_close(zfp);
            zfp_field_free(field);
            return hipcompErrorInternal;
        }
        
        *compressed_bytes = zfpsize;
        stream_close(bs);
    }
    
    zfp_stream_close(zfp);
    zfp_field_free(field);
    
    return hipcompSuccess;
}

hipcompStatus_t hipcompZfpDecompressAsync(
    const void* d_in,
    size_t compressed_bytes,
    const hipcompZfpOpts* opts,
    void* d_temp,
    size_t temp_bytes,
    void* d_out,
    hipStream_t stream)
{
    (void)d_temp;
    (void)temp_bytes;
    
    if (!d_in || !opts || !d_out) {
        return hipcompErrorInvalidValue;
    }
    
    if (!zfp_initialized) {
        hipcompStatus_t status = hipcompZfpInit();
        if (status != hipcompSuccess) return status;
    }
    
    hipStreamSynchronize(stream);
    
    size_t num_elements = opts->nx;
    if (opts->dims >= 2) num_elements *= opts->ny;
    if (opts->dims >= 3) num_elements *= opts->nz;
    
    zfp_type type = get_zfp_type(opts->type);
    
    /* Create ZFP field */
    zfp_field* field = NULL;
    switch (opts->dims) {
        case 1:
            field = zfp_field_1d(d_out, type, opts->nx);
            break;
        case 2:
            field = zfp_field_2d(d_out, type, opts->nx, opts->ny);
            break;
        case 3:
            field = zfp_field_3d(d_out, type, opts->nx, opts->ny, opts->nz);
            break;
        default:
            return hipcompErrorInvalidValue;
    }
    
    if (!field) return hipcompErrorInternal;
    
    /* Create ZFP stream */
    zfp_stream* zfp = zfp_stream_open(NULL);
    if (!zfp) {
        zfp_field_free(field);
        return hipcompErrorInternal;
    }
    
    /* Set compression mode (must match compression settings) */
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
    
    /* Try GPU execution */
    zfp_bool use_gpu = zfp_false;
#ifdef ZFP_WITH_HIP
    use_gpu = zfp_stream_set_execution(zfp, zfp_exec_hip);
    if (use_gpu) {
        printf("[ZFP] Using GPU-native HIP decompression\n");
    }
#endif
    
    if (!use_gpu) {
        /* Fallback: copy to host, decompress, copy back */
        printf("[ZFP] Falling back to CPU decompression\n");
        
        size_t element_size = get_element_size(opts->type);
        size_t data_size = num_elements * element_size;
        
        void* h_in = malloc(compressed_bytes);
        void* h_out = malloc(data_size);
        if (!h_in || !h_out) {
            free(h_in);
            free(h_out);
            zfp_stream_close(zfp);
            zfp_field_free(field);
            return hipcompErrorInternal;
        }
        
        hipMemcpy(h_in, d_in, compressed_bytes, hipMemcpyDeviceToHost);
        
        /* Update field to point to host memory */
        zfp_field_set_pointer(field, h_out);
        
        /* Setup bitstream */
        bitstream* bs = stream_open(h_in, compressed_bytes);
        zfp_stream_set_bit_stream(zfp, bs);
        zfp_stream_rewind(zfp);
        
        /* Decompress */
        size_t result = zfp_decompress(zfp, field);
        
        stream_close(bs);
        
        if (result == 0) {
            zfp_stream_close(zfp);
            zfp_field_free(field);
            free(h_in);
            free(h_out);
            return hipcompErrorInternal;
        }
        
        /* Copy decompressed data to device */
        hipMemcpy(d_out, h_out, data_size, hipMemcpyHostToDevice);
        
        free(h_in);
        free(h_out);
    } else {
        /* GPU-native decompression */
        bitstream* bs = stream_open((void*)d_in, compressed_bytes);
        zfp_stream_set_bit_stream(zfp, bs);
        zfp_stream_rewind(zfp);
        
        size_t result = zfp_decompress(zfp, field);
        stream_close(bs);
        
        if (result == 0) {
            zfp_stream_close(zfp);
            zfp_field_free(field);
            return hipcompErrorInternal;
        }
    }
    
    zfp_stream_close(zfp);
    zfp_field_free(field);
    
    return hipcompSuccess;
}

hipcompStatus_t hipcompZfpDecompressGetTempSize(
    const hipcompZfpOpts* opts,
    size_t* temp_bytes)
{
    if (!opts || !temp_bytes) {
        return hipcompErrorInvalidValue;
    }
    *temp_bytes = 0;
    return hipcompSuccess;
}

hipcompStatus_t hipcompZfpGetDecompressedSize(
    const void* compressed_data,
    size_t compressed_bytes,
    const hipcompZfpOpts* opts,
    size_t* uncompressed_bytes)
{
    if (!opts || !uncompressed_bytes) {
        return hipcompErrorInvalidValue;
    }
    
    (void)compressed_data;
    (void)compressed_bytes;
    
    size_t num_elements = opts->nx;
    if (opts->dims >= 2) num_elements *= opts->ny;
    if (opts->dims >= 3) num_elements *= opts->nz;
    
    *uncompressed_bytes = num_elements * get_element_size(opts->type);
    return hipcompSuccess;
}
