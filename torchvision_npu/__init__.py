from torchvision_npu.datasets import add_dataset_imagefolder, npu_loader
from torchvision_npu.transforms import add_transform_methods
from torchvision_npu.runner import *


def apply_class_patches():
    add_dataset_imagefolder()
    add_transform_methods()

apply_class_patches()