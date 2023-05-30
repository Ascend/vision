import numpy as np

import torch
import torch_npu
import torchvision
import torchvision_npu
from torchvision.transforms.functional import InterpolationMode

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuResize(TestCase):
    @staticmethod
    def cpu_op_exec(input1, size, mode):
        output = torchvision.transforms.Resize(size, mode)(input1)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec(input1, size, mode):
        mode = mode.value[2:] if mode.value[:2] == "bi" else mode.value
        output = torch.ops.torchvision.npu_resize(input1, size, mode=mode)
        output = output.cpu().squeeze(0)
        output = output.numpy()
        return output

    def setUp(self):
        torch.npu.set_compile_mode(jit_compile=True)

    def tearDown(self):
        torch.npu.set_compile_mode(jit_compile=False)
    
    def test_npu_resize(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 255, (3, 375, 500)).astype(np.float16))
        npu_input = cpu_input.unsqueeze(0).npu()
        size = [224, 224]
        mode = InterpolationMode.BILINEAR

        cpu_output = self.cpu_op_exec(cpu_input, size, mode)
        npu_output = self.npu_op_exec(npu_input, size, mode)

        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
