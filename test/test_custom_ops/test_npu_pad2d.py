import numpy as np

import torch
import torch_npu
import torchvision
import torchvision_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuPad2d(TestCase):
    @staticmethod
    def cpu_op_exec(input1, pad, fill, mode):
        output = torchvision.transforms.Pad(padding=[pad[0], pad[2], pad[1], pad[3]],
                                            fill=fill, padding_mode=mode)(input1)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec(input1, pad, fill, mode):
        output = torch.ops.torchvision.npu_pad2d(input1, pad=pad, constant_values=fill, mode=mode)
        output = output.cpu().squeeze(0)
        output = output.numpy()
        return output

    def test_npu_pad2d(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 255, (3, 375, 500)).astype(np.uint8))
        npu_input = cpu_input.unsqueeze(0).npu()
        pad = 10, 20, 30, 40
        fill = 0
        padding_mode = "constant"

        cpu_output = self.cpu_op_exec(cpu_input, pad, fill, padding_mode)
        npu_output = self.npu_op_exec(npu_input, pad, fill, padding_mode)

        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()
