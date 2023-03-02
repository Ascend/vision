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


@pytest.mark.parametrize(
    ["img_path", "threshold", "p"],
    [
        ("./test/Data/fish/fish_11.jpg", 100, 0.3),
        ("./test/Data/fish/fish_22.jpg", 10, 0.7),
        ("./test/Data/fish/fish_22.jpg", 20, 0.1),
        ("./test/Data/fish/fish_22.jpg", 50, 0.5),
        ("./test/Data/fish/fish_22.jpg", 200, 1),
    ],
)
def test_solarize(img_path, threshold, p):
    pil_img = Image.open(img_path)

    # using pil solarize
    torch.manual_seed(10)
    pil_solarize = trans.RandomSolarize(threshold=threshold, p=p)(pil_img)

    # using cv2+convert solarize
    import torchvision_npu
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_solarize = trans.RandomSolarize(threshold=threshold, p=p)(pil_img)

    assert type(pil_solarize) == type(cv2_solarize)
    assert pil_solarize.size == cv2_solarize.size
    assert (np.array(pil_solarize) == np.array(cv2_solarize)).all()
