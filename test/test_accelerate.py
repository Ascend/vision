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
import torchvision.datasets as datasets
import torchvision_npu


class TestAccelerate(unittest.TestCase):
    def test_accelerate(self):
        path = "./Data/"
        train_datasets = datasets.ImageFolder(path)
        train_datasets.accelerate()
        self.assertTrue(train_datasets.accelerate_enable)
        self.assertNotEqual(train_datasets.device, torch.device('cpu'))

    def test_nonaccelerate(self):
        path = "./Data/"
        train_datasets = datasets.ImageFolder(path)
        self.assertFalse(train_datasets.accelerate_enable)
        self.assertEqual(train_datasets.device, torch.device('cpu'))

if __name__ == '__main__':
    unittest.main()
