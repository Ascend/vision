from torchvision_npu.io.image import patch_io_image
from torchvision_npu.io.video import patch_io_video


def patch_io_methods():
    patch_io_image()
    patch_io_video()
