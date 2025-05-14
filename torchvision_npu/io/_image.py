import torch
import torchvision
from torchvision.utils import _log_api_usage_once
from torchvision.io.image import ImageReadMode


def patch_io_image():
    setattr(torchvision.io.image, "encode_jpeg_ori", torchvision.io.image.encode_jpeg)
    torchvision.io.image.encode_jpeg = encode_jpeg
    torchvision.io.encode_jpeg = encode_jpeg
    setattr(torchvision.io.image, "decode_jpeg_ori", torchvision.io.image.decode_jpeg)
    torchvision.io.image.decode_jpeg = _decode_jpeg
    torchvision.io.decode_jpeg = _decode_jpeg


def encode_jpeg(img: torch.Tensor, quality: int = 75) -> torch.Tensor:
    """
    Takes an input tensor in CHW layout and returns a buffer with the contents
    of its corresponding JPEG file.

    Args:
        img (Tensor[channels, image_height, image_width])): int8 image tensor of
            ``c`` channels, where ``c`` must be 1 or 3. 
            For npu backend, img format should be NCHW(N=1, C=1or3).
        quality (int): Quality of the resulting JPEG file, it must be a number between
            1 and 100. Default: 75

    Returns:
        output (Tensor[1]): A one dimensional int8 tensor that contains the raw bytes of the
            JPEG file.
    """
    if img.device.type == 'npu':
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            _log_api_usage_once(encode_jpeg)
        if quality < 1 or quality > 100:
            raise ValueError("Image quality should be a positive number between 1 and 100")
        img = img.unsqueeze(0)
        return torch.ops.torchvision._encode_jpeg_aclnn(img, quality)

    return torchvision.io.image.encode_jpeg_ori(img, quality)


def _decode_jpeg_npu(
    input: torch.Tensor,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
) -> torch.Tensor:
    shape = torch.ops.torchvision._get_jpeg_image_shape(input)
    uint8_tensor = input.npu(non_blocking=True)
    channels = shape[2]
    if mode == ImageReadMode.GRAY:
        channels = 1
    elif mode == ImageReadMode.RGB:
        channels = 3

    return torch.ops.torchvision._decode_jpeg_aclnn(
        uint8_tensor, image_shape=shape, channels=channels).squeeze(0)


def _decode_jpeg(
    input: torch.Tensor,
    mode: ImageReadMode = ImageReadMode.UNCHANGED,
    device: str = "cpu",
) -> torch.Tensor:
    """Decode JPEG image(s) into 3D RGB or grayscale Tensor(s), on CPU、CUDA or NPU.

    The values of the output tensor are uint8 between 0 and 255.
    """
    device = torch.device(device)
    if device.type != 'npu':
        return torchvision.io.image.decode_jpeg_ori(input, mode, device)

    return _decode_jpeg_npu(input, mode)
