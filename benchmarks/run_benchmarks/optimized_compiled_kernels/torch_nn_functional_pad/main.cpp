#include <torch/extension.h>
torch::Tensor launch(
    torch::Tensor input, 
    std::vector<int64_t> pad,
    std::string mode,
    c10::optional<double> value
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("launch", torch::wrap_pybind_function(launch), "launch");
}