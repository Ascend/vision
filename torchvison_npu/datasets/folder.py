from typing import Any, Callable, Optional, Tuple
import numpy as np
import torch
import torch_npu
import torchvision
from torchvision.datasets import folder as fold
from torchvision_npu.datasets.decode_jpeg import extract_jpeg_shape, pack


def add_dataset_imagefolder():
    torchvision.__name__ = 'torchvision_npu'
    torchvision.datasets.ImageFolder = ImageFolder
    torchvision.datasets.DatasetFolder.__getitem__ = DatasetFolder.__getitem__


def npu_loader(path:str) -> Any:
    with open(path, "rb") as f:
        f.seek(0)
        prefix = f.read(16)
        if prefix[:3] == b"\xff\xd8\xff" :
            f.seek(0)
            image_shape = extract_jpeg_shape(f)

            f.seek(0)
            bytes_1 = f.read()
            arr = np.frombuffer(bytes_1, dtype=np.uint8)
            addr = 16
            length = len(bytes_1)
            addr_arr = list(map(int, pack('<Q', addr)))
            len_arr = list(map(int, pack('<Q', length)))
            arr = np.hstack((addr_arr, len_arr, arr, [0]))
            arr = np.array(arr, dtype=np.uint8)
            uint8_tensor = torch.as_tensor(arr.copy()).npu()
            return torch_npu.decode_jpeg(uint8_tensor, image_shape = image_shape, channels = 3)
        else:
            image = fold.pil_loader(path)
            return torch.from_numpy(np.array(image)).npu()


class DatasetFolder(fold.VisionDataset):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            if sample.is_npu:
                sample = sample.cpu()
                sample = sample.squeeze(0)
                sample = sample.transpose(0, 2).transpose(1, 2)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ImageFolder(fold.DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = fold.default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, fold.IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.accelerate_enable = False
        self.device = torch.device("cpu")

    def accelerate(self):
        if torch_npu.npu.is_available():
            self.accelerate_enable = True
            self.loader = npu_loader
            self.device = torch_npu.npu.current_device()
