import numpy as np
import torch
import torch_npu
from torchvision_npu.extensions import _HAS_OPS

import torch_npu.npu.utils as utils
from torch_npu.testing.testcase import TestCase, run_tests


class TestImgToTensor(TestCase):
    def test_npu_img_to_tensor_numpy(self):
        cpu_input = np.random.uniform(0, 255, (1, 3, 224, 224)).astype(np.uint8)
        npu_input = torch.from_numpy(cpu_input).npu()

        cpu_output = cpu_input.astype(np.float32) / 255
        npu_output = torch.ops.torchvision.npu_img_to_tensor(npu_input).cpu().numpy()

        self.assertRtolEqual(cpu_output, npu_output)

    def test_npu_img_to_tensor_torch(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 255, (1, 3, 224, 224)).astype(np.uint8))
        npu_input = cpu_input.npu()

        cpu_output = cpu_input.to(torch.get_default_dtype()).div(255)
        npu_output = torch.ops.torchvision.npu_img_to_tensor(npu_input).cpu()

        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    torch_npu.npu.set_device(0)
    if utils.get_soc_version() in range(220, 224):
        run_tests()
