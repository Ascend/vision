import numpy as np
import torch
import torch_npu
from torchvision_npu.extensions import _HAS_OPS

from torch_npu.testing.testcase import TestCase, run_tests


class TestReverse(TestCase):
    @staticmethod
    def cpu_op_exec(input1, axis):
        output = np.flip(input1, axis)
        return output

    @staticmethod
    def npu_op_exec(input1, axis):
        output = torch.ops.torchvision.npu_reverse(input1, [axis])
        output = output.cpu().numpy()
        return output

    def test_npu_reverse(self):
        cpu_input = np.random.uniform(0, 255, (1, 3, 224, 224)).astype(np.uint8)
        npu_input = torch.from_numpy(cpu_input).npu()

        for axis in range(4):
            cpu_output = self.cpu_op_exec(cpu_input, axis)
            npu_output = self.npu_op_exec(npu_input, axis)

            self.assertRtolEqual(cpu_output, npu_output)


if __name__ == "__main__":
    run_tests()