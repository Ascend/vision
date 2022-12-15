#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

template <typename scalar_t>
at::Tensor add_kernel_impl(
    const at::Tensor& a,
    const at::Tensor& b) {

    return a;
}

at::Tensor add_kernel(
    const at::Tensor& a,
    const at::Tensor& b) {

  auto result = at::empty({0}, a.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "add_kernel", [&] {
    result = add_kernel_impl<scalar_t>(a, b);
  });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::add"), TORCH_FN(add_kernel));
}

} // namespace ops
} // namespace vision
