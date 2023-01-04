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
import unittest
import torch
import torchvision

import torchvision_npu
import torchvision.transforms as transforms


class TestToTensor(unittest.TestCase):
    def test_totensor(self):
        path = "./Data/dog/dog.0001.jpg"
        img = torchvision_npu.datasets.folder.npu_loader(path)
        output = transforms.ToTensor()(img)
        cpuout = output.cpu()
        self.assertEqual(output.dtype, torch.float32)
        self.assertGreaterEqual(torch.tensor(1), torch.max(cpuout))
        self.assertGreaterEqual(torch.max(cpuout), torch.tensor(0))

    def test_pil_totensor(self):
        path = "./Data/dog/dog.0001.jpg"
        img = torchvision.datasets.folder.pil_loader(path)
        output = transforms.ToTensor()(img)
        cpuout = output.cpu()
        self.assertEqual(output.dtype, torch.float32)
        self.assertGreaterEqual(torch.tensor(1), torch.max(cpuout))
        self.assertGreaterEqual(torch.max(cpuout), torch.tensor(0))


if __name__ == '__main__':
    unittest.main()
