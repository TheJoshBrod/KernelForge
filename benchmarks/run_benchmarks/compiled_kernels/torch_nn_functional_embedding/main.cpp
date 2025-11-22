#include <torch/extension.h>
torch::Tensor launch(
    torch::Tensor arg0,
    torch::Tensor arg1,
    c10::optional<int64_t> arg2,
    c10::optional<double> arg3,
    double arg4,
    bool arg5,
    bool arg6
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("launch", torch::wrap_pybind_function(launch), "launch");
}