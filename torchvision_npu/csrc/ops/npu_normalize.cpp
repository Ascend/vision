#include "npu_normalize.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor npu_normalize(
    const at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    at::ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_normalize", "")
      .typed<decltype(npu_normalize)>();
  return op.call(self, mean, variance, dtype);
}

at::Tensor& npu_normalize_(
    at::Tensor &self,
    c10::optional<c10::ArrayRef<double>> mean,
    c10::optional<c10::ArrayRef<double>> variance,
    at::ScalarType dtype) {
  static auto op = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torchvision::npu_normalize_", "")
      .typed<decltype(npu_normalize_)>();
  return op.call(self, mean, variance, dtype);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_normalize(Tensor self, \
      float[]? mean, float[]? variance, ScalarType dtype=float) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::npu_normalize_(Tensor(a!) self, \
      float[]? mean, float[]? variance, ScalarType dtype=float) -> Tensor(a!)"));
}

} // namespace ops
} // namespace vision
