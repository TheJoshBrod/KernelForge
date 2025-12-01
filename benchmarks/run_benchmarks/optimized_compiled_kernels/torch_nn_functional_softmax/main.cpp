#include <torch/extension.h>
torch::Tensor launch(torch::Tensor input, int64_t dim, int64_t _stacklevel, c10::optional<at::ScalarType> dtype);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("launch", torch::wrap_pybind_function(launch), "launch");
}