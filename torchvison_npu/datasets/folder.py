import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
import numpy as np
import torch
import torch_npu
import torchvision
from torchvision.datasets.folder import VisionDataset,DatasetFolder, pil_loader, make_dataset, default_loader, IMG_EXTENSIONS
from torchvision_npu.datasets.decode_jpeg import extract_jpeg_shape, pack


def add_dataset_imagefolder():
    torchvision.__name__ = 'torchvision_npu'
    torchvision.datasets.ImageFolder = ImageFolder


def npu_loader(path:str) -> Any:
    with open(path, "rb") as f:
        f.seek(0)
        prefix = f.read(16)
        if prefix[:3] == b"\xFF\xDB\xFF" :
            f.seek(0)
            image_shape = extract_jpeg_shape(f)

            f.seek(0)
            bytes_1 = f.read()
            arr = np.frombuffer(bytes_1, dtype = np.uint8)
            addr = 16
            length = len(bytes_1)
            addr_arr = list(map(int, pack('<Q', addr)))
            len_arr = list(map(int, pack('<Q', length)))
            arr = np.hstack((addr_arr, len_arr, arr, [0]))
            arr = np.array(arr, dtype=np.uint8)
            uint8_tensor = torch.as_tensor(arr.copy().npu())
            return torch_npu.decode_jpeg(uint8_tensor, image_shape = image_shape, channels = 3)
        else:
            image = pil_loader(path)
            return torch.from_numpy(np.array(image)).npu()


class ImageFolder(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
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
