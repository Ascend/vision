# import warnings
# from typing import Tuple, List, Optional
# from collections.abc import Sequence
# import math
import torch
from torch import Tensor
import torch_npu
import torchvision
from torchvision.transforms import functional as F
# from torchvision.transforms.transforms import _setup_size
# from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int
from torchvision.transforms import transforms as TRANS




def add_transform_methods():
    torchvision.__name__ = 'torchvision_npu'
    torchvision.transforms.ToTensor.__call__ = ToTensor.__call__
    torchvision.transforms.RandomResizedCrop.forward = RandomResizedCrop.forward
    torchvision.transforms.RandomHorizontalFlip.forward = RandomHorizontalFlip.forward
    torchvision.transforms.Normalize.forward = Normalize.forward


class ToTensor:
    @classmethod
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            if pic.device.type == 'npu':
                return pic.to(dtype=torch.get_default_dtype()).div(255)
        return F.to_tensor(pic)


class Normalize(torch.nn.Module):
    # def __init__(self, mean, std, inplace=False):
    #     super().__init__()
    #     self.mean = mean
    #     self.std = std
    #     self.inplace = inplace

    # def __repr__(self):
    #     return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.device.type == 'npu':
            if self.inplace == True:
                return torch_npu.normalize_(tensor, torch.as_tensor(self.mean).npu(), torch.as_tensor(self.std).npu())
            return torch_npu.normalize(tensor, torch.as_tensor(self.mean).npu(), torch.as_tensor(self.std).npu())
        return F.normalize(tensor, self.mean, self.std, self.inplace)


class RandomHorizontalFlip(torch.nn.Module):
    # def __init__(self, p=0.5):
    #     super().__init__()
    #     self.p = p
    #
    # def __repr__(self):
    #     return self.__class__.__name__ + '(p={})'.format(self.p)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        if torch.rand(1) < self.p:
            if isinstance(img, torch.Tensor):
                if img.device.type == 'npu':
                    return torch_npu.reverse(img, axis = [2])
                else:
                    return F.hflip(img)
        return img



class RandomResizedCrop(torch.nn.Module):
    # def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
    #     super().__init__()
    #     self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
    #
    #     if not isinstance(scale, Sequence):
    #         raise TypeError("Scale should be a sequence")
    #     if not isinstance(ratio, Sequence):
    #         raise TypeError("Ratio should be a sequence")
    #     if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
    #         warnings.warn("Scale and ratio should be of kind (min, max)")
    #
    #     # Backward compatibility with integer value
    #     if isinstance(interpolation, int):
    #         warnings.warn(
    #             "Argument interpolation should be of type InterpolationMode instead of int. "
    #             "Please, use InterpolationMode enum."
    #         )
    #         interpolation = _interpolation_modes_from_int(interpolation)
    #
    #     self.interpolation = interpolation
    #     self.scale = scale
    #     self.ratio = ratio
    #
    # def __repr__(self):
    #     interpolate_str = self.interpolation.value
    #     format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
    #     format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
    #     format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
    #     format_string += ', interpolation={0})'.format(interpolate_str)
    #     return format_string

    # @staticmethod
    # def get_params(
    #         img: Tensor, scale: List[float], ratio: List[float]
    # ) -> Tuple[int, int, int, int]:
    #     """Get parameters for ``crop`` for a random sized crop.
    #
    #     Args:
    #         img (PIL Image or Tensor): Input image.
    #         scale (list): range of scale of the origin size cropped
    #         ratio (list): range of aspect ratio of the origin aspect ratio cropped
    #
    #     Returns:
    #         tuple: params (i, j, h, w) to be passed to ``crop`` for a random
    #             sized crop.
    #     """
    #     width, height = F._get_image_size(img)
    #     area = height * width
    #
    #     log_ratio = torch.log(torch.tensor(ratio))
    #     for _ in range(10):
    #         target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
    #         aspect_ratio = torch.exp(
    #             torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
    #         ).item()
    #
    #         w = int(round(math.sqrt(target_area * aspect_ratio)))
    #         h = int(round(math.sqrt(target_area / aspect_ratio)))
    #
    #         if 0 < w <= width and 0 < h <= height:
    #             i = torch.randint(0, height - h + 1, size=(1,)).item()
    #             j = torch.randint(0, width - w + 1, size=(1,)).item()
    #             return i, j, h, w
    #
    #     # Fallback to central crop
    #     in_ratio = float(width) / float(height)
    #     if in_ratio < min(ratio):
    #         w = width
    #         h = int(round(w / min(ratio)))
    #     elif in_ratio > max(ratio):
    #         h = height
    #         w = int(round(h * max(ratio)))
    #     else:  # whole image
    #         w = width
    #         h = height
    #     i = (height - h) // 2
    #     j = (width - w) // 2
    #     return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = TRANS.RandomResizedCrop.get_params(img, self.scale, self.ratio)#self.get_params(img, self.scale, self.ratio)
        if isinstance(img, torch.Tensor):
            if img.device.type == 'npu':
                height, width = F._get_image_size(img)
                boxes = torch.as_tensor([[i /(height - 1), j / (width - 1), (i + h)/(height - 1), (j + w)/(width - 1)]]).npu()
                box_index = torch.as_tensor([0]).npu()
                crop_size = self.size()
                return torch_npu.crop_and_resize(img, boxes, box_index, crop_size)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)



