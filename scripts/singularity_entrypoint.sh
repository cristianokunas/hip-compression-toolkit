#!/bin/bash
# =============================================================================
# Singularity Entrypoint for hipCOMP Benchmark Automation
# 
# This script is the default %runscript for the Singularity container.
# It detects the GPU, prepares test data, and runs benchmarks automatically.
#
# Usage (from host):
#   singularity run --bind /path/to/rsf:/rsf --bind /path/to/output:/output hipcomp.sif [OPTIONS]
#
# Or manually:
#   singularity exec hipcomp.sif /opt/hip-compression-toolkit/scripts/run_benchmarks_auto.sh [OPTIONS]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${HIPCOMP_ROOT:-/opt/hip-compression-toolkit}"

echo "=============================================="
echo "  hipCOMP Benchmark Container"
echo "=============================================="
echo ""

# Check GPU access
if command -v rocm-smi &> /dev/null; then
    echo "[INFO] ROCm detected"
    rocm-smi --showproductname 2>/dev/null | head -5 || echo "[WARN] Could not query GPU"
elif command -v nvidia-smi &> /dev/null; then
    echo "[INFO] NVIDIA GPU detected"
    nvidia-smi -L 2>/dev/null | head -5 || echo "[WARN] Could not query GPU"
else
    echo "[ERROR] No GPU runtime detected (need ROCm or CUDA)"
    exit 1
fi
echo ""

# Forward all arguments to the benchmark runner
exec "$PROJECT_ROOT/scripts/run_benchmarks_auto.sh" "$@"
