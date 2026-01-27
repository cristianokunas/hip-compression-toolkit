#!/bin/bash
#
# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

set -e

usage() {
  echo "Usage: $0 <algorithm> <data_directory> [<gpu_id>]"
  echo ""
  echo "Algorithms: lz4, snappy, cascaded, gdeflate, bitcomp, ans"
  echo ""
  echo "Example:"
  echo "  $0 lz4 /path/to/test/data 0"
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
fi

ALGORITHM=$1
DATA_DIR=$2
GPU_ID=${3:-0}

# Validate algorithm
case $ALGORITHM in
  lz4|snappy|cascaded|gdeflate|bitcomp|ans)
    ;;
  *)
    echo "ERROR: Unknown algorithm '$ALGORITHM'"
    usage
    ;;
esac

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: Data directory '$DATA_DIR' does not exist"
  exit 1
fi

# Find benchmark binary
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN_DIR="${SCRIPT_DIR}/../build/benchmarks"

if [ ! -d "$BIN_DIR" ]; then
  BIN_DIR="${SCRIPT_DIR}/../bin/benchmarks"
fi

BENCHMARK_BIN="${BIN_DIR}/benchmark_${ALGORITHM}_chunked"

if [ ! -f "$BENCHMARK_BIN" ]; then
  echo "ERROR: Benchmark binary not found: $BENCHMARK_BIN"
  echo "Please build the project with benchmarks enabled"
  exit 1
fi

# Create temporary directory for logs
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

echo "Running $ALGORITHM benchmark on GPU $GPU_ID"
echo "Data directory: $DATA_DIR"
echo "----------------------------------------------------------------------"

# Run benchmark header
run_benchmark() {
  local file=$1
  local log_file="$TMP_DIR/bench.log"

  # Run benchmark
  $BENCHMARK_BIN -g $GPU_ID -f "$file" -i 10 -w 2 -c true -t false > "$log_file" 2>&1

  # Parse results
  if [ -f "$log_file" ]; then
    # Get the data line (second line, after header)
    tail -n 1 "$log_file" | awk -F',' '{
      printf "%s,%s,%s,%.2f,%.2f\n",
        FILENAME,
        $7,      # uncompressed size
        $9,      # compression ratio
        $10,     # compression throughput
        $11      # decompression throughput
    }' FILENAME="$file"
  fi
}

# Print CSV header
echo "dataset,uncompressed bytes,compression ratio,compression throughput (GB/s),decompression throughput (GB/s)"

# Process all files in data directory
for file in "$DATA_DIR"/*; do
  if [ -f "$file" ]; then
    run_benchmark "$file"
  fi
done

echo "----------------------------------------------------------------------"
echo "Benchmark complete!"
