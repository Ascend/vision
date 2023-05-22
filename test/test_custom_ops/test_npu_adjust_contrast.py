import numpy as np

import torch
import torch_npu
import torchvision
import torchvision_npu

from torch_npu.testing.testcase import TestCase, run_tests


class TestNpuAdjustContrast(TestCase):
    @staticmethod
    def cpu_op_exec(input1, factor):
        output = torchvision.transforms.functional.adjust_contrast(input1, factor)
        return output

    @staticmethod
    def npu_op_exec(input1, factor):
        output = torch.ops.torchvision.npu_adjust_contrast(input1.float(), factor)
        output = output.cpu().to(torch.uint8).squeeze(0)
        return output

    def result_error(self, npu_img, cpu_img):
        if npu_img.shape != cpu_img.shape:
            self.fail("shape error")
        if npu_img.dtype != cpu_img.dtype:
            self.fail("dtype error")
        result = np.abs(npu_img.to(torch.int16) - cpu_img.to(torch.int16))
        if result.max() > 2:
            self.fail("result error")

    def test_npu_adjust_contrast(self):
        cpu_input = torch.from_numpy(np.random.uniform(0, 255, (3, 375, 500)).astype(np.uint8))
        npu_input = cpu_input.unsqueeze(0).npu()
        factor = np.random.uniform(0, 1)

        cpu_output = self.cpu_op_exec(cpu_input, factor)
        npu_output = self.npu_op_exec(npu_input, factor)

        self.result_error(npu_output, cpu_output)


if __name__ == "__main__":
    run_tests()
