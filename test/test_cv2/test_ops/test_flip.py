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

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms as trans
import torchvision_npu


@pytest.mark.parametrize(
    ["img_path", "p"],
    [
        ("./test/Data/fish/fish_11.jpg", 0),
        ("./test/Data/fish/fish_22.jpg", 0.1),
        ("./test/Data/fish/fish_33.jpg", 0.5),
        ("./test/Data/fish/fish_44.jpg", 0.7),
        ("./test/Data/fish/fish_55.jpg", 1),
    ],
)
def test_vfilp(img_path, p):
    pil_img = Image.open(img_path)

    # using pil vflip
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_vflip = trans.RandomVerticalFlip(p=p)(pil_img)

    # using cv2 vflip
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_vflip = trans.RandomVerticalFlip(p=p)(cv2_img)

    assert isinstance(pil_vflip, Image.Image) and isinstance(cv2_vflip, np.ndarray)
    assert pil_vflip.size == cv2_vflip.shape[:2][::-1]
    assert (np.array(pil_vflip) == cv2_vflip).all()
    
    pil_img.close()


@pytest.mark.parametrize(
    ["img_path", "p"],
    [
        ("./test/Data/fish/fish_11.jpg", 0),
        ("./test/Data/fish/fish_22.jpg", 0.1),
        ("./test/Data/fish/fish_33.jpg", 0.5),
        ("./test/Data/fish/fish_44.jpg", 0.7),
        ("./test/Data/fish/fish_55.jpg", 1),
    ],
)
def test_hfilp(img_path, p):
    pil_img = Image.open(img_path)

    # using pil hflip
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_hflip = trans.RandomHorizontalFlip(p=p)(pil_img)

    # using cv2 hflip
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_hflip = trans.RandomHorizontalFlip(p=p)(cv2_img)

    assert isinstance(pil_hflip, Image.Image) and isinstance(cv2_hflip, np.ndarray)
    assert pil_hflip.size == cv2_hflip.shape[:2][::-1]
    assert (np.array(pil_hflip) == cv2_hflip).all()
    
    pil_img.close()
