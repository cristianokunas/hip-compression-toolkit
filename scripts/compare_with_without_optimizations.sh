#!/bin/bash
# Compare Feature 2 optimizations ON vs OFF for working algorithms
# This avoids needing to checkout old commits

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Working algorithms only
WORKING_ALGOS="lz4 snappy"

cd "$PROJECT_ROOT"

print_header "Comparison: Optimizations ON vs OFF"
echo ""

# Step 1: Run with optimizations ON (current state)
print_info "Step 1: Testing WITH optimizations (current Feature 2)..."
print_info "Current commit: $(git log -1 --oneline)"
echo ""

cd build
./run_experiments.sh -a "$WORKING_ALGOS" -o ../results/feature2_optimized

# Step 2: Disable optimizations and rebuild
print_info "Step 2: Disabling AMD optimizations..."
cd "$PROJECT_ROOT"

# Temporarily disable optimizations by commenting out the defines
# This is a hack but avoids checking out old commits
print_info "Adding -DDISABLE_AMD_OPTIMIZATIONS to build..."

cd build
rm -rf CMakeCache.txt CMakeFiles/
cmake .. \
  -D BUILD_BENCHMARKS=ON \
  -D CMAKE_HIP_ARCHITECTURES="gfx1100" \
  -D USE_WARPSIZE_32=ON \
  -D CMAKE_CXX_FLAGS="-DDISABLE_AMD_OPTIMIZATIONS" \
  -D CMAKE_HIP_FLAGS="-DDISABLE_AMD_OPTIMIZATIONS"

make clean
make -j$(nproc)

print_info "Step 3: Testing WITHOUT optimizations..."
./run_experiments.sh -a "$WORKING_ALGOS" -o ../results/feature2_no_optimizations

# Step 4: Restore optimizations
print_info "Step 4: Restoring optimizations..."
rm -rf CMakeCache.txt CMakeFiles/
cmake .. \
  -D BUILD_BENCHMARKS=ON \
  -D CMAKE_HIP_ARCHITECTURES="gfx1100" \
  -D USE_WARPSIZE_32=ON

make -j$(nproc)

# Step 5: Compare
cd "$PROJECT_ROOT"
print_header "Comparing Results"
./scripts/compare_results.sh results/feature2_no_optimizations results/feature2_optimized

print_info "Comparison complete!"
print_info "Results:"
print_info "  - No optimizations: results/feature2_no_optimizations"
print_info "  - With optimizations: results/feature2_optimized"
