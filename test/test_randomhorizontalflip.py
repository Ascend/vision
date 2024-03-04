# Copyright (c) 2022, Huawei Technologies.All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torchvision.transforms as transforms
from torch_npu.testing.testcase import TestCase, run_tests
import torchvision_npu


class TestRandomHorizontalFlip(TestCase):
    def test_randomhorizontalflip(self):
        path = "./Data/dog/dog.0001.jpg"
        npu_input = torchvision_npu.datasets.folder.npu_loader(path)
        cpu_input = npu_input.cpu().squeeze(0)
        cpu_output = transforms.RandomHorizontalFlip(p=1)(cpu_input)
        npu_output = transforms.RandomHorizontalFlip(p=1)(npu_input).cpu().squeeze(0)
        self.assertRtolEqual(cpu_output, npu_output)


if __name__ == '__main__':
    run_tests()
