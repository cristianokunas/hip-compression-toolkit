#!/bin/bash
# Run compression benchmarks and collect results
# Usage: ./scripts/run_experiments.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Default parameters
TESTDATA_DIR="$PROJECT_ROOT/testdata"
RESULTS_DIR="$PROJECT_ROOT/results"
ALGORITHMS="lz4 snappy cascaded gdeflate bitcomp ans"
ITERATIONS=10
WARMUP=2
GPU_DEVICE=0

# Parse arguments
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run compression benchmark experiments

OPTIONS:
    -d, --testdata DIR      Test data directory (default: testdata)
    -o, --output DIR        Results output directory (default: results)
    -a, --algorithms ALGOS  Space-separated algorithms (default: all)
                           Available: lz4, snappy, cascaded, gdeflate, bitcomp, ans
    -i, --iterations N      Number of iterations per test (default: 10)
    -w, --warmup N         Number of warmup iterations (default: 2)
    -g, --gpu N            GPU device ID (default: 0)
    -f, --feature COMMIT   Test specific feature/commit (default: current)
    -c, --compare          Compare multiple features
    -h, --help             Show this help message

EXAMPLES:
    # Run all benchmarks with default settings
    $0

    # Run only LZ4 and Snappy on custom test data
    $0 -d /data/tests -a "lz4 snappy"

    # Run with more iterations for accuracy
    $0 -i 50 -w 5

    # Compare baseline vs optimized
    $0 --compare

EOF
}

# Parse command line
COMPARE_MODE=false
FEATURE_COMMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--testdata)
            TESTDATA_DIR="$2"
            shift 2
            ;;
        -o|--output)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -a|--algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_DEVICE="$2"
            shift 2
            ;;
        -f|--feature)
            FEATURE_COMMIT="$2"
            shift 2
            ;;
        -c|--compare)
            COMPARE_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate test data directory
if [ ! -d "$TESTDATA_DIR" ]; then
    print_error "Test data directory not found: $TESTDATA_DIR"
    print_info "Run: ./scripts/generate_testdata.sh $TESTDATA_DIR"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Get current git info
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
CURRENT_COMMIT=$(git rev-parse --short HEAD)
CURRENT_COMMIT_MSG=$(git log -1 --pretty=%B | head -1)

# Get GPU info
GPU_NAME="Unknown"
if command -v rocm-smi &> /dev/null; then
    GPU_NAME=$(rocm-smi --showproductname --device $GPU_DEVICE 2>/dev/null | grep "GPU" | awk -F: '{print $2}' | xargs || echo "Unknown")
fi

# Create experiment metadata
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_ID="${TIMESTAMP}_${CURRENT_COMMIT}"
EXPERIMENT_DIR="$RESULTS_DIR/$EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR"

# Save metadata
cat > "$EXPERIMENT_DIR/metadata.txt" << EOF
Experiment ID: $EXPERIMENT_ID
Timestamp: $(date)
Git Branch: $CURRENT_BRANCH
Git Commit: $CURRENT_COMMIT
Commit Message: $CURRENT_COMMIT_MSG
GPU: $GPU_NAME
GPU Device: $GPU_DEVICE
Test Data: $TESTDATA_DIR
Iterations: $ITERATIONS
Warmup: $WARMUP
Algorithms: $ALGORITHMS
EOF

print_header "Experiment Setup"
cat "$EXPERIMENT_DIR/metadata.txt"
echo ""

# Find build directory
if [ -L "$PROJECT_ROOT/build_latest" ]; then
    BUILD_DIR="$PROJECT_ROOT/build_latest"
elif [ -d "$PROJECT_ROOT/build" ]; then
    BUILD_DIR="$PROJECT_ROOT/build"
else
    print_error "No build directory found. Run ./scripts/build_for_arch.sh first"
    exit 1
fi

# Check for benchmarks in both possible locations
if [ -d "$BUILD_DIR/bin" ]; then
    BENCHMARK_DIR="$BUILD_DIR/bin"
elif [ -d "$BUILD_DIR/benchmarks" ]; then
    BENCHMARK_DIR="$BUILD_DIR/benchmarks"
else
    print_error "Benchmarks not found in $BUILD_DIR/bin or $BUILD_DIR/benchmarks"
    exit 1
fi

print_info "Using benchmarks from: $BENCHMARK_DIR"
echo ""

# Function to run a single benchmark
run_benchmark() {
    local algo="$1"
    local testfile="$2"
    local output_csv="$3"

    print_info "  -> run_benchmark called: algo=$algo, file=$(basename "$testfile")"

    local benchmark_exe="$BENCHMARK_DIR/benchmark_${algo}_chunked"
    print_info "  -> Looking for: $benchmark_exe"

    if [ ! -f "$benchmark_exe" ]; then
        print_error "  -> Benchmark executable NOT FOUND: $benchmark_exe"
        echo "$algo,$(basename "$testfile"),ERROR,NOTFOUND,NOTFOUND,NOTFOUND" >> "$output_csv"
        return 1
    fi

    print_info "  -> Found benchmark executable, running..."

    local testfile_name=$(basename "$testfile")
    local testfile_size=$(stat -c%s "$testfile" 2>/dev/null || echo "0")

    print_info "Running: $algo on $testfile_name ($(numfmt --to=iec-i --suffix=B $testfile_size))"

    # Run benchmark and capture output
    local temp_output=$(mktemp)

    if $benchmark_exe -f "$testfile" -i $ITERATIONS -w $WARMUP -g $GPU_DEVICE > "$temp_output" 2>&1; then
        # Parse results (this is a simplified parser, adjust based on actual output format)
        # Expected format: lines with throughput information

        # Extract key metrics and save to CSV
        local comp_throughput=$(grep -i "compression.*throughput\|comp.*GB/s" "$temp_output" | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "N/A")
        local decomp_throughput=$(grep -i "decompression.*throughput\|decomp.*GB/s" "$temp_output" | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "N/A")
        local comp_ratio=$(grep -i "ratio" "$temp_output" | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "N/A")

        # Append to CSV
        echo "$algo,$testfile_name,$testfile_size,$comp_throughput,$decomp_throughput,$comp_ratio" >> "$output_csv"

        print_info "  Compression: ${comp_throughput} GB/s, Decompression: ${decomp_throughput} GB/s, Ratio: ${comp_ratio}x"
    else
        print_warn "  Benchmark failed, see: $temp_output"
        echo "$algo,$testfile_name,$testfile_size,ERROR,ERROR,ERROR" >> "$output_csv"
    fi

    # Save full output for reference
    local log_file="$EXPERIMENT_DIR/${algo}_${testfile_name}.log"
    mv "$temp_output" "$log_file"
}

# Main experiment execution
print_header "Running Benchmarks"

# Create CSV header
RESULTS_CSV="$EXPERIMENT_DIR/results.csv"
echo "Algorithm,TestFile,FileSize,CompressionThroughput_GBps,DecompressionThroughput_GBps,CompressionRatio" > "$RESULTS_CSV"

# Get all test files
print_info "Searching for test files in: $TESTDATA_DIR"
mapfile -t TEST_FILES < <(find "$TESTDATA_DIR" -name "*.bin" -type f | sort)

if [ ${#TEST_FILES[@]} -eq 0 ]; then
    print_error "No test files (*.bin) found in $TESTDATA_DIR"
    print_info "Run: ./scripts/generate_testdata.sh $TESTDATA_DIR"
    exit 1
fi

# Show found files
num_files=${#TEST_FILES[@]}
print_info "Found $num_files test file(s)"
for i in {0..4}; do
    if [ $i -lt $num_files ]; then
        print_info "  - $(basename "${TEST_FILES[$i]}")"
    fi
done
if [ $num_files -gt 5 ]; then
    print_info "  ... and $((num_files - 5)) more"
fi
echo ""

# Run benchmarks
print_info "Counting tests..."
total_tests=0
for algo in $ALGORITHMS; do
    total_tests=$((total_tests + num_files))
done
print_info "Total tests to run: $total_tests"
echo ""

current_test=0
for algo in $ALGORITHMS; do
    print_header "Algorithm: ${algo^^}"
    print_info "Starting tests for $algo (${#TEST_FILES[@]} files)..."
    print_info "DEBUG: num_files=$num_files"
    print_info "DEBUG: About to enter loop from 0 to $((num_files - 1))"

    # Use C-style for loop (pure bash, no external commands)
    for ((i=0; i<num_files; i++)); do
        print_info "DEBUG: Loop iteration i=$i"
        testfile="${TEST_FILES[$i]}"
        print_info "DEBUG: testfile=$testfile"

        print_info "DEBUG: About to increment current_test (was $current_test)"
        current_test=$((current_test + 1))
        print_info "DEBUG: Incremented current_test to $current_test"

        print_info "DEBUG: About to get basename"
        testfile_basename=$(basename "$testfile")
        print_info "DEBUG: basename=$testfile_basename"

        print_info "Test [$current_test/$total_tests]: $testfile_basename"

        print_info "DEBUG: About to call run_benchmark"
        run_benchmark "$algo" "$testfile" "$RESULTS_CSV"
        print_info "DEBUG: run_benchmark returned"
    done
    echo ""
done

print_header "Experiment Complete"
print_info "Results saved to: $EXPERIMENT_DIR"
print_info "CSV: $RESULTS_CSV"
print_info "Logs: $EXPERIMENT_DIR/*.log"
echo ""

# Generate quick summary
print_header "Quick Summary"
if command -v column &> /dev/null; then
    head -20 "$RESULTS_CSV" | column -t -s ','
else
    head -20 "$RESULTS_CSV"
fi
echo ""

print_info "For detailed analysis, use: ./scripts/compare_results.sh"
