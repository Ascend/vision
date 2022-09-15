import torch
from torch import Tensor
import torch_npu
import torchvision
from torchvision.transforms import functional as F
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
    def forward(self, tensor: Tensor) -> Tensor:
        if tensor.device.type == 'npu':
            if self.inplace == True:
                return torch_npu.normalize_(tensor, torch.as_tensor(self.mean).npu(), torch.as_tensor(self.std).npu())
            return torch_npu.normalize(tensor, torch.as_tensor(self.mean).npu(), torch.as_tensor(self.std).npu())
        return F.normalize(tensor, self.mean, self.std, self.inplace)


class RandomHorizontalFlip(torch.nn.Module):
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
                    return torch_npu.reverse(img, axis=[3])
                else:
                    return F.hflip(img)
        return img


class RandomResizedCrop(torch.nn.Module):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = TRANS.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        if isinstance(img, torch.Tensor):
            if img.device.type == 'npu':
                img = img.squeeze(0).reshape(img.shape[2], img.shape[3], img.shape[1])
                img = img.permute((2, 0, 1)).unsqueeze(0)
                width, height = F._get_image_size(img)
                boxes = torch.as_tensor([[i / (height - 1), j / (width - 1), (i + h) / (height - 1), \
                                          (j + w) / (width - 1)]], dtype=torch.float32).npu()
                box_index = torch.as_tensor([0], dtype=torch.int32).npu()
                crop_size = self.size
                return torch_npu.crop_and_resize(img, boxes, box_index, crop_size)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)



