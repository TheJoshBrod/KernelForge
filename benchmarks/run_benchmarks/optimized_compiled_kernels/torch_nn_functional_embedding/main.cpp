#include <torch/extension.h>
torch::Tensor launch(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<int64_t> padding_idx,
    c10::optional<double> max_norm,
    double norm_type,
    bool scale_grad_by_freq,
    bool sparse
);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("launch", torch::wrap_pybind_function(launch), "launch");
}