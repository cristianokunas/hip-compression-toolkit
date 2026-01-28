/*
 * Copyright (c) 2024, hipCOMP contributors
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * ZFP compression benchmark for hipCOMP
 * 
 * ZFP is designed for floating-point arrays and uses a different API
 * than the batched compressors (LZ4, Snappy, etc.)
 */

#include "benchmark_common.h"

#ifdef __HIP_PLATFORM_AMD__
#include "hipcomp/zfp.h"
#include <hip/hip_runtime.h>
#else
#error "ZFP benchmark currently only supports HIP/AMD backend"
#endif

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <random>
#include <iomanip>

// Default parameters
static size_t DEFAULT_NX = 256;
static size_t DEFAULT_NY = 256;
static size_t DEFAULT_NZ = 256;
static double DEFAULT_RATE = 16.0;  // bits per value
static int DEFAULT_WARMUP = 3;
static int DEFAULT_ITERATIONS = 10;

struct BenchmarkConfig {
    size_t nx, ny, nz;
    double rate;
    int warmup_iterations;
    int benchmark_iterations;
    std::string input_file;
    bool generate_data;
    int data_pattern;  // 0=random, 1=smooth, 2=zeros
};

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -n <nx> <ny> <nz>  Array dimensions (default: 256 256 256)" << std::endl;
    std::cout << "  -r <rate>          Bits per value (default: 16.0)" << std::endl;
    std::cout << "  -w <warmup>        Warmup iterations (default: 3)" << std::endl;
    std::cout << "  -i <iterations>    Benchmark iterations (default: 10)" << std::endl;
    std::cout << "  -f <file>          Input file (binary float data)" << std::endl;
    std::cout << "  -p <pattern>       Data pattern: 0=random, 1=smooth, 2=zeros (default: 1)" << std::endl;
    std::cout << "  -h                 Show this help" << std::endl;
    std::cout << std::endl;
    std::cout << "Rate reference:" << std::endl;
    std::cout << "  32.0 = ~1x compression (minimal loss)" << std::endl;
    std::cout << "  24.0 = ~1.33x compression" << std::endl;
    std::cout << "  16.0 = ~2x compression" << std::endl;
    std::cout << "  8.0  = ~4x compression" << std::endl;
    std::cout << "  4.0  = ~8x compression" << std::endl;
}

BenchmarkConfig parse_args(int argc, char** argv) {
    BenchmarkConfig config;
    config.nx = DEFAULT_NX;
    config.ny = DEFAULT_NY;
    config.nz = DEFAULT_NZ;
    config.rate = DEFAULT_RATE;
    config.warmup_iterations = DEFAULT_WARMUP;
    config.benchmark_iterations = DEFAULT_ITERATIONS;
    config.generate_data = true;
    config.data_pattern = 1;  // smooth by default

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 3 < argc) {
            config.nx = std::stoull(argv[++i]);
            config.ny = std::stoull(argv[++i]);
            config.nz = std::stoull(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            config.rate = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            config.warmup_iterations = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            config.benchmark_iterations = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
            config.input_file = argv[++i];
            config.generate_data = false;
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            config.data_pattern = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }

    return config;
}

void generate_test_data(std::vector<float>& data, size_t nx, size_t ny, size_t nz, int pattern) {
    size_t total = nx * ny * nz;
    data.resize(total);

    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

    switch (pattern) {
        case 0:  // Random noise
            std::cout << "Generating random noise data..." << std::endl;
            for (size_t i = 0; i < total; i++) {
                data[i] = uniform(rng);
            }
            break;

        case 1:  // Smooth (sine wave) - typical scientific data
            std::cout << "Generating smooth sinusoidal data..." << std::endl;
            for (size_t k = 0; k < nz; k++) {
                for (size_t j = 0; j < ny; j++) {
                    for (size_t i = 0; i < nx; i++) {
                        size_t idx = k * ny * nx + j * nx + i;
                        float x = (float)i / nx;
                        float y = (float)j / ny;
                        float z = (float)k / nz;
                        data[idx] = std::sin(2.0f * M_PI * x) * 
                                   std::cos(2.0f * M_PI * y) * 
                                   std::sin(4.0f * M_PI * z) +
                                   0.1f * uniform(rng);  // Small noise
                    }
                }
            }
            break;

        case 2:  // Zeros (best compression)
            std::cout << "Generating zero data..." << std::endl;
            std::fill(data.begin(), data.end(), 0.0f);
            break;

        default:
            std::cerr << "Unknown pattern: " << pattern << std::endl;
            exit(1);
    }
}

void load_data_from_file(std::vector<float>& data, const std::string& filename, size_t expected_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (file_size < expected_size * sizeof(float)) {
        std::cerr << "Error: File size (" << file_size << " bytes) is smaller than expected ("
                  << expected_size * sizeof(float) << " bytes)" << std::endl;
        exit(1);
    }

    data.resize(expected_size);
    file.read(reinterpret_cast<char*>(data.data()), expected_size * sizeof(float));
    file.close();

    std::cout << "Loaded " << expected_size << " floats from " << filename << std::endl;
}

struct BenchmarkResults {
    double compression_time_ms;
    double decompression_time_ms;
    double compression_throughput_gbs;
    double decompression_throughput_gbs;
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    double max_abs_error;
    double max_rel_error;
    double rmse;
    double psnr;
};

BenchmarkResults run_benchmark(const std::vector<float>& h_data, const BenchmarkConfig& config) {
    BenchmarkResults results = {0};
    
    size_t num_elements = config.nx * config.ny * config.nz;
    size_t data_size = num_elements * sizeof(float);
    results.original_size = data_size;

    // Allocate device memory
    float* d_input = nullptr;
    float* d_output = nullptr;
    void* d_compressed = nullptr;
    
    GPU_CHECK(hipMalloc(&d_input, data_size));
    GPU_CHECK(hipMalloc(&d_output, data_size));
    
    // Copy input to device
    GPU_CHECK(hipMemcpy(d_input, h_data.data(), data_size, hipMemcpyHostToDevice));

    // Setup ZFP options
    hipcompZfpOpts opts = hipcompZfpDefaultOpts(config.nx, config.ny, config.nz, config.rate);
    opts.type = HIPCOMP_ZFP_TYPE_FLOAT;
    opts.dims = 3;
    opts.mode = HIPCOMP_ZFP_MODE_FIXED_RATE;

    // Get max compressed size
    size_t max_compressed_size;
    hipcompZfpCompressGetMaxOutputSize(&opts, &max_compressed_size);
    GPU_CHECK(hipMalloc(&d_compressed, max_compressed_size));

    // Create stream and events
    hipStream_t stream;
    hipEvent_t start, stop;
    GPU_CHECK(hipStreamCreate(&stream));
    GPU_CHECK(hipEventCreate(&start));
    GPU_CHECK(hipEventCreate(&stop));

    // Warmup
    std::cout << "Running " << config.warmup_iterations << " warmup iterations..." << std::endl;
    for (int i = 0; i < config.warmup_iterations; i++) {
        size_t compressed_size = 0;
        hipcompZfpCompressAsync(d_input, &opts, nullptr, 0, d_compressed, &compressed_size, stream);
        GPU_CHECK(hipStreamSynchronize(stream));
    }

    // Benchmark compression
    std::cout << "Running " << config.benchmark_iterations << " compression iterations..." << std::endl;
    size_t compressed_size = 0;
    GPU_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < config.benchmark_iterations; i++) {
        hipcompZfpCompressAsync(d_input, &opts, nullptr, 0, d_compressed, &compressed_size, stream);
    }
    GPU_CHECK(hipEventRecord(stop, stream));
    GPU_CHECK(hipStreamSynchronize(stream));

    float compression_ms;
    GPU_CHECK(hipEventElapsedTime(&compression_ms, start, stop));
    results.compression_time_ms = compression_ms / config.benchmark_iterations;
    results.compressed_size = compressed_size;
    results.compression_ratio = (double)data_size / compressed_size;
    results.compression_throughput_gbs = (data_size / 1e9) / (results.compression_time_ms / 1000.0);

    // Benchmark decompression
    std::cout << "Running " << config.benchmark_iterations << " decompression iterations..." << std::endl;
    GPU_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < config.benchmark_iterations; i++) {
        hipcompZfpDecompressAsync(d_compressed, compressed_size, &opts, nullptr, 0, d_output, stream);
    }
    GPU_CHECK(hipEventRecord(stop, stream));
    GPU_CHECK(hipStreamSynchronize(stream));

    float decompression_ms;
    GPU_CHECK(hipEventElapsedTime(&decompression_ms, start, stop));
    results.decompression_time_ms = decompression_ms / config.benchmark_iterations;
    results.decompression_throughput_gbs = (data_size / 1e9) / (results.decompression_time_ms / 1000.0);

    // Verify correctness and compute error statistics
    std::vector<float> h_output(num_elements);
    GPU_CHECK(hipMemcpy(h_output.data(), d_output, data_size, hipMemcpyDeviceToHost));

    double sum_sq_error = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double max_val = h_data[0];
    double min_val = h_data[0];

    for (size_t i = 0; i < num_elements; i++) {
        double orig = h_data[i];
        double decomp = h_output[i];
        double error = std::fabs(orig - decomp);
        double rel_error = (std::fabs(orig) > 1e-10) ? error / std::fabs(orig) : 0.0;

        sum_sq_error += error * error;
        if (error > max_abs) max_abs = error;
        if (rel_error > max_rel) max_rel = rel_error;
        if (orig > max_val) max_val = orig;
        if (orig < min_val) min_val = orig;
    }

    double range = max_val - min_val;
    results.max_abs_error = max_abs;
    results.max_rel_error = max_rel;
    results.rmse = std::sqrt(sum_sq_error / num_elements);
    results.psnr = (range > 1e-10) ? 20.0 * std::log10(range / results.rmse) : 0.0;

    // Cleanup
    GPU_CHECK(hipEventDestroy(start));
    GPU_CHECK(hipEventDestroy(stop));
    GPU_CHECK(hipStreamDestroy(stream));
    GPU_CHECK(hipFree(d_input));
    GPU_CHECK(hipFree(d_output));
    GPU_CHECK(hipFree(d_compressed));

    return results;
}

void print_results(const BenchmarkResults& results, const BenchmarkConfig& config) {
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "ZFP Compression Benchmark Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Dimensions:      " << config.nx << " x " << config.ny << " x " << config.nz << std::endl;
    std::cout << "  Total elements:  " << config.nx * config.ny * config.nz << std::endl;
    std::cout << "  Rate:            " << std::fixed << std::setprecision(1) << config.rate << " bits/value" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Size:" << std::endl;
    std::cout << "  Original:        " << std::fixed << std::setprecision(2) 
              << results.original_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Compressed:      " << results.compressed_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Ratio:           " << results.compression_ratio << "x" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Performance:" << std::endl;
    std::cout << "  Compression:     " << std::fixed << std::setprecision(3) 
              << results.compression_time_ms << " ms (" 
              << std::setprecision(2) << results.compression_throughput_gbs << " GB/s)" << std::endl;
    std::cout << "  Decompression:   " << std::setprecision(3) 
              << results.decompression_time_ms << " ms ("
              << std::setprecision(2) << results.decompression_throughput_gbs << " GB/s)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Quality (lossy compression error):" << std::endl;
    std::cout << "  Max Abs Error:   " << std::scientific << std::setprecision(6) << results.max_abs_error << std::endl;
    std::cout << "  Max Rel Error:   " << results.max_rel_error << std::endl;
    std::cout << "  RMSE:            " << results.rmse << std::endl;
    std::cout << "  PSNR:            " << std::fixed << std::setprecision(2) << results.psnr << " dB" << std::endl;
    std::cout << std::endl;
    
    // CSV output
    std::cout << "CSV: nx,ny,nz,rate,original_MB,compressed_MB,ratio,comp_ms,decomp_ms,comp_GBs,decomp_GBs,max_abs,max_rel,rmse,psnr" << std::endl;
    std::cout << "CSV: " << config.nx << "," << config.ny << "," << config.nz << ","
              << std::fixed << std::setprecision(1) << config.rate << ","
              << std::setprecision(2) << results.original_size / (1024.0 * 1024.0) << ","
              << results.compressed_size / (1024.0 * 1024.0) << ","
              << results.compression_ratio << ","
              << std::setprecision(3) << results.compression_time_ms << ","
              << results.decompression_time_ms << ","
              << std::setprecision(2) << results.compression_throughput_gbs << ","
              << results.decompression_throughput_gbs << ","
              << std::scientific << std::setprecision(6) << results.max_abs_error << ","
              << results.max_rel_error << ","
              << results.rmse << ","
              << std::fixed << std::setprecision(2) << results.psnr << std::endl;
}

int main(int argc, char** argv) {
    BenchmarkConfig config = parse_args(argc, argv);

    // Print GPU info
    int device = 0;
    hipDeviceProp_t props;
    GPU_CHECK(hipSetDevice(device));
    GPU_CHECK(hipGetDeviceProperties(&props, device));
    std::cout << "Using GPU: " << props.name << std::endl;
    std::cout << std::endl;

    // Prepare data
    std::vector<float> h_data;
    size_t total_elements = config.nx * config.ny * config.nz;

    if (config.generate_data) {
        generate_test_data(h_data, config.nx, config.ny, config.nz, config.data_pattern);
    } else {
        load_data_from_file(h_data, config.input_file, total_elements);
    }

    // Run benchmark
    BenchmarkResults results = run_benchmark(h_data, config);

    // Print results
    print_results(results, config);

    return 0;
}
