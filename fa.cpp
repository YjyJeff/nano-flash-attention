#include <torch/extension.h>

torch::Tensor attention(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor attention_kv_parallel(torch::Tensor Q, torch::Tensor K,
                                    torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention", &attention, "attention");
  m.def("attention_kv_parallel", &attention_kv_parallel,
        "attention_kv_parallel");
}
