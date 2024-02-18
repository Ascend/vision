import torch
import torch_npu
from torch_npu.testing.testcase import run_tests
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import torchvision_npu
from torchvision_npu.testing.test_deviation_case import TestCase

torch_npu.npu.current_stream().set_data_preprocess_stream(True)


class TestResize(TestCase):
    def test_resize_vision(self):
        torch.ops.torchvision._dvpp_init()

        path = "../Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder._npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)

        for interpolation in [InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC]:
            cpu_output = transforms.Resize((224, 224), interpolation)(cpu_input)

            torch.npu.set_compile_mode(jit_compile=False)
            npu_output = transforms.Resize((224, 224), interpolation)(npu_input).cpu().squeeze(0)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)

            torch.npu.set_compile_mode(jit_compile=True)
            npu_output = transforms.Resize((224, 224), interpolation)(npu_input).cpu().squeeze(0)
            self.assert_acceptable_deviation(npu_output, cpu_output, 2)


if __name__ == '__main__':
    run_tests()
