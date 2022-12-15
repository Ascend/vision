#include "add.h"

#include <torch/types.h>

namespace vision {
namespace ops {

at::Tensor add(
    const at::Tensor& a,
    const at::Tensor& b) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::add", "")
                       .typed<decltype(add)>();
  return op.call(a, b);
}

TORCH_LIBRARY_FRAGMENT(torchvision, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "torchvision::add(Tensor a, Tensor b) -> Tensor"));
}

} // namespace ops
} // namespace vision
