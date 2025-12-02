#include <torch/extension.h>
torch::Tensor launch(torch::Tensor input, bool inplace);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("launch", torch::wrap_pybind_function(launch), "launch");
}