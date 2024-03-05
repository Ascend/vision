import math
from typing import Optional, Tuple

import torch
import torch_npu
import torchvision
from torch import nn, Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torchvision.extension import _assert_has_ops


def deform_conv2d(
    inputs: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:

    _assert_has_ops()
    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch.zeros((inputs.shape[0], 0), device=inputs.device, dtype=inputs.dtype)

    if bias is None:
        bias = torch.zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, in_h, in_w = inputs.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    if n_offset_grps == 0:
        raise RuntimeError(
            "the shape of the offset tensor at dimension 1 is not valid. It should "
            "be a multiple of 2 * weight.size[2] * weight.size[3].\n"
            "Got offset.shape[1]={}, while 2 * weight.size[2] * weight.size[3]={}".format(
                offset.shape[1], 2 * weights_h * weights_w))

    if not inputs.is_npu:
        return torch.ops.torchvision.deform_conv2d(
            inputs,
            weight,
            offset,
            mask,
            bias,
            stride_h, stride_w,
            pad_h, pad_w,
            dil_h, dil_w,
            n_weight_grps,
            n_offset_grps,
            use_mask,)
    else:
        return npu_deform_conv2d(
            inputs,
            offset,
            weight,
            bias,
            (stride_h, stride_w),
            (pad_h, pad_w),
            (dil_h, dil_w),
            mask if use_mask else None,
            n_weight_grps,
            n_offset_grps
        )


def npu_deform_conv2d(
        inputs: Tensor,
        offset: Tensor,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
        mask: Optional[Tensor] = None,
        groups: Optional[int] = 1,
        deform_groups: Optional[int] = 1):
    _, _, kernel_h, kernel_w = weight.shape
    conv2d_bias = bias
    sort_index_fp, sort_index_bp = _calculate_sort_index(
            kernel_h, kernel_w, deform_groups)
    select_offset = offset.index_select(1, sort_index_fp)
    if mask is None:
        mask_shape, _ = torch.chunk(offset, 2, dim=1)
        mask = torch.ones_like(mask_shape).to(inputs.device)
    offset_all = torch.cat([select_offset, mask], dim=1)
    output, offset_out = torch.npu_deformable_conv2d(
        inputs,
        weight,
        offset_all,
        conv2d_bias,
        kernel_size=[kernel_h, kernel_w],
        stride=[1, 1, stride[0], stride[1]],
        padding=[padding[0], padding[0], padding[1], padding[1]],
        dilation=[1, 1, dilation[0], dilation[1]],
        groups=groups,
        deformable_groups=deform_groups,
        modulated=True)
    return output


def _calculate_sort_index(kernel_h, kernel_w, deformable_group):
    split_num = deformable_group * 2 * kernel_h * kernel_w
    sort_index = list(range(split_num))
    sort_index_fp = (sort_index[1::2] + sort_index[::2])
    sort_index_bp_dict = {i: idx for idx, i in enumerate(sort_index_fp)}
    sort_index_bp = [sort_index_bp_dict[i] for i in sort_index]
    sort_index_fp = torch.IntTensor(sort_index_fp)
    sort_index_bp = torch.IntTensor(sort_index_bp)
    sort_index_fp = sort_index_fp.npu()
    sort_index_bp = sort_index_bp.npu()
    return sort_index_fp, sort_index_bp


def patch_deform_conv():
    torchvision.ops.deform_conv2d = deform_conv2d
    torchvision.ops.deform_conv.deform_conv2d = deform_conv2d
