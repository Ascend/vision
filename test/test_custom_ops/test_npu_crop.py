import numpy as np

import torch
import torch_npu
import torchvision
import torchvision_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuCrop(TestCase):
    @staticmethod
    def cpu_op_exec(input1, top, left, height, width):
        output = torchvision.transforms.functional.crop(input1, top, left, height, width)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec(input1, top, left, height, width):
        size = [1, 3, height, width]
        axis = 2
        offsets = [top, left]
        output = torch.ops.torchvision.npu_crop(input1, size, axis=axis, offsets=offsets)
        output = output.cpu().squeeze(0)
        output = output.numpy()
        return output

    def test_npu_crop(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 255, (3, 375, 500)).astype(np.uint8))
        npu_input = cpu_input.unsqueeze(0).npu()
        top, left, height, width = 50, 50, 100, 100
        
        cpu_output = self.cpu_op_exec(cpu_input, top, left, height, width)
        npu_output = self.npu_op_exec(npu_input, top, left, height, width)

        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
