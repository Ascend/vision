#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

template <typename scalar_t>
at::Tensor nms_kernel_impl(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {

  auto ndets = dets.size(0);
  at::Tensor keep_t = at::zeros({ndets}, dets.options().dtype(at::kLong));
  return keep_t.narrow(/*dim=*/0, /*start=*/0, /*length=*/ndets);

}

at::Tensor nms_kernel(
    const at::Tensor& dets,
    const at::Tensor& scores,
    double iou_threshold) {


  auto result = at::empty({0}, dets.options());

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "nms_kernel", [&] {
    result = nms_kernel_impl<scalar_t>(dets, scores, iou_threshold);
  });
  return result;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, XLA, m) {
  m.impl(TORCH_SELECTIVE_NAME("torchvision::nms"), TORCH_FN(nms_kernel));
}

} // namespace ops
} // namespace vision
