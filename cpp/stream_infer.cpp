// Real-time streaming inference in C++ (LibTorch): feeds audio one hop at a time
// through the exported TorchScript StreamStep and reports per-hop compute time.
//
// usage: stream_infer <stream_step.pt> <input.f32> <output.f32> [hop]
//   input/output are raw little-endian float32 mono at 16 kHz.
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: stream_infer <stream_step.pt> <in.f32> <out.f32> [hop]\n";
    return 1;
  }
  torch::set_num_threads(1);
  torch::NoGradGuard no_grad;
  auto step = torch::jit::load(argv[1]);
  step.eval();
  const int64_t hop = argc > 4 ? std::stoi(argv[4]) : 64;

  std::ifstream fin(argv[2], std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
  std::vector<float> in(bytes.size() / sizeof(float));
  std::memcpy(in.data(), bytes.data(), in.size() * sizeof(float));
  const int64_t n_hops = static_cast<int64_t>(in.size()) / hop;

  // state = (inbuf, ola, hist_re, hist_im, conv_buf, gru_h)
  std::vector<torch::jit::IValue> st = step.get_method("initial_state")({}).toTuple()->elements().vec();
  std::vector<float> out;
  out.reserve(n_hops * hop);
  std::vector<double> ms;
  ms.reserve(n_hops);

  for (int64_t k = 0; k < n_hops; ++k) {
    auto chunk = torch::from_blob(in.data() + k * hop, {hop}, torch::kFloat32).clone();
    std::vector<torch::jit::IValue> args{chunk};
    for (auto& s : st) args.push_back(s);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<torch::jit::IValue> res = step.forward(args).toTuple()->elements().vec();
    auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    auto y = res[0].toTensor().contiguous();
    out.insert(out.end(), y.data_ptr<float>(), y.data_ptr<float>() + hop);
    st.assign(res.begin() + 1, res.end());
  }
  std::ofstream(argv[3], std::ios::binary)
      .write(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));

  // skip warm-up hops for timing
  std::vector<double> t(ms.begin() + std::min<int64_t>(50, n_hops / 2), ms.end());
  std::sort(t.begin(), t.end());
  const double hop_ms = 1000.0 * hop / 16000.0;
  double mean = 0;
  for (double v : t) mean += v;
  mean /= t.size();
  int64_t miss = std::count_if(t.begin(), t.end(), [&](double v) { return v > hop_ms; });
  std::printf("hops=%zu hop=%.1f ms  p50=%.3f ms  p99=%.3f ms  max=%.3f ms  RTF=%.3f  deadline misses=%lld\n",
              t.size(), hop_ms, t[t.size() / 2], t[static_cast<size_t>(t.size() * 0.99)], t.back(),
              mean / hop_ms, static_cast<long long>(miss));
  return 0;
}
