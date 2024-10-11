import torch
import torch_npu
import torchvision
from torch import Tensor


def roi_pool(input_tensor, boxes, output_size, spatial_scale: float = 1.0) -> Tensor:
    """There are some differences between the native implementation of TorchVision and the implementation
    provided by the NPU operator when calc roi_pool. This can lead to inaccurate calculations.

    This function calculates the boxes coordinate value first,
    and then passes it into the operator with a spatial scale of 1 to ensure
    that the accuracy is consistent with the CPU.

    Ref to torchvision/csrc/ops/cuda/roi_pool_kernel.cu, CUDA calc box roi_width and roi_height by:
        roi_width = round(boxes[:,3] * spatial_scale) - round(boxes[:,1] * spatial_scale) + 1
    NPU calc roi_pool according to implementation of MMCV.
    Ref to mmcv/ops/csrc/common/cuda/roi_pool_cuda_kernel.cuh, NPU calc box roi_width and roi_height by:
        roi_width = (boxes[:,3] + 1) * spatial_scale - boxes[:,1] * spatial_scale

    to meet the diff, we do round operation before ahead and construct spatial_scale=1
    
    Args:
    input (Tensor[N, C, H, W]): input tensor
    boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
        format where the regions will be taken from.
        The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        If a single Tensor is passed,
        then the first column should contain the batch index. If a list of Tensors
        is passed, then each Tensor will correspond to the boxes for an element i
        in a batch
    output_size (int or Tuple[int, int]): the size of the output after the cropping
        is performed, as (height, width)
    spatial_scale (float): a scaling factor that maps the input coordinates to
        the box coordinates. Default: 1.0

    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """
    if input_tensor.device.type == "npu":
        boxes[:, 1:] = torch.round(boxes[:, 1:] * spatial_scale)
        spatial_scale = 1.0
    return torchvision.ops.tv_roi_pool(input_tensor, boxes, output_size, spatial_scale)


def patch_roi_pool():
    setattr(torchvision.ops, "tv_roi_pool", torchvision.ops.roi_pool)
    torchvision.ops.roi_pool = roi_pool