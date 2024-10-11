from torchvision_npu.io._image import patch_io_image
from torchvision_npu.io._video import patch_io_video


def patch_io_methods():
    patch_io_image()
    patch_io_video()
