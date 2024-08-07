import io
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision
import torchvision_npu
from torchvision_npu.datasets.decode_jpeg import extract_jpeg_shape


class TestEncodeJpeg(TestCase):
    def test_encode_jpeg(self):
        path = "../Data/cat/cat.0001.jpg"
        cpu_input = torchvision.datasets.folder.pil_loader(path)
        cpu_input = torch.tensor(np.array(cpu_input)).permute(2, 0, 1)
        npu_input = cpu_input.npu(non_blocking=True)
        quality = 50

        torch.npu.set_compile_mode(jit_compile=False)
        torch.ops.torchvision._dvpp_init()

        npu_output = torchvision.io.image.encode_jpeg(npu_input, quality)
        self.assertEqual(npu_output.device.type, 'npu')

        f = npu_output.cpu().numpy().tobytes()
        f = io.BytesIO(f)
        f.seek(0)
        prefix = f.read(16)
        if prefix[:3] == b"\xff\xd8\xff":
            f.seek(0)
            image_shape = extract_jpeg_shape(f)
            output = torch.ops.torchvision._decode_jpeg_aclnn(
                npu_output, image_shape=image_shape, channels=3).squeeze(0)
            self.assertEqual(npu_input.shape, output.shape)


if __name__ == '__main__':
    run_tests()
