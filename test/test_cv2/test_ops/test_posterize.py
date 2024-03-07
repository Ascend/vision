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
    ["img_path", "bit", "p"],
    [
        ("./test/Data/fish/fish_11.jpg", 0, 0.1),
        ("./test/Data/fish/fish_22.jpg", 1, 0.3),
        ("./test/Data/fish/fish_33.jpg", 3, 0.5),
        ("./test/Data/fish/fish_44.jpg", 5, 0.7),
        ("./test/Data/fish/fish_55.jpg", 7, 1),
    ],
)
def test_posterize(img_path, bit, p):
    pil_img = Image.open(img_path)

    # using pil posterize
    torchvision.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_posterize = trans.RandomPosterize(bits=bit, p=p)(pil_img)

    # using cv2 posterize
    torchvision.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_posterize = trans.RandomPosterize(bits=bit, p=p)(cv2_img)

    assert isinstance(pil_posterize, Image.Image) and isinstance(cv2_posterize, np.ndarray)
    assert pil_posterize.size == cv2_posterize.shape[:2][::-1]
    assert (np.array(pil_posterize) == cv2_posterize).all()
    
    pil_img.close()
