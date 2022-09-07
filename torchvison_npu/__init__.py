from torchvision_npu.datasets import add_dataset_imagefolder
from torchvision_npu.transforms import add_transform_methods


def apply_class_patches():
    add_dataset_imagefolder()
    add_transform_methods()

apply_class_patches()