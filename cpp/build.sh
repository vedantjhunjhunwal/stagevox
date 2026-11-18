#!/usr/bin/env bash
# Build the C++ streaming harness against the LibTorch that ships inside the pip torch wheel.
# (CMake's TorchConfig refuses CUDA wheels on CPU-only machines, so this calls g++ directly.
#  With a CPU LibTorch download you can use CMakeLists.txt instead.)
set -euo pipefail
cd "$(dirname "$0")"
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
ABI=$(python3 -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
g++ -O2 -std=c++20 -D_GLIBCXX_USE_CXX11_ABI=$ABI stream_infer.cpp \
  -I"$T/include" -I"$T/include/torch/csrc/api/include" -L"$T/lib" -Wl,-rpath,"$T/lib" \
  -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 -o stream_infer
echo "built cpp/stream_infer"
