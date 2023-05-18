import unittest

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision_npu
import torchvision.transforms as transforms

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestPad(TestCase):
    @staticmethod
    def cpu_op_exec(input1, padding, mode):
        output = transforms.Pad(padding, padding_mode=mode)(input1)
        output = output.numpy()
        return output

    @staticmethod
    def npu_op_exec(input1, padding, mode):
        output = transforms.Pad(padding, padding_mode=mode)(input1)
        output = output.cpu().squeeze(0)
        output = output.numpy()
        return output

    def test_pad_constant_edge(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        padding = (50, 100, 20, 70)
        for mode in ["constant", "edge"]:
            cpu_output = self.cpu_op_exec(cpu_input, padding, mode)
            npu_output = self.npu_op_exec(npu_input, padding, mode)
            self.assertEqual(cpu_output, npu_output)
    
    @unittest.expectedFailure
    def test_pad_reflect_symmetric(self):
        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        padding = (50, 100, 20, 70)
        for mode in ["reflect", "symmetric"]:
            cpu_output = self.cpu_op_exec(cpu_input, padding, mode)
            npu_output = self.npu_op_exec(npu_input, padding, mode)
            self.assertEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
