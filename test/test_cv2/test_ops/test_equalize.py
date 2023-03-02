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
    ["img_path", "p"],
    [
        ("./test/Data/fish/fish_11.jpg", 0),
        ("./test/Data/fish/fish_22.jpg", 0.1),
        ("./test/Data/fish/fish_33.jpg", 0.5),
        ("./test/Data/fish/fish_44.jpg", 0.7),
        ("./test/Data/fish/fish_55.jpg", 1),
    ],
)
def test_equalize(img_path, p):
    pil_img = Image.open(img_path)

    # using pil equalize
    torch.manual_seed(10)
    pil_equalize = trans.RandomEqualize(p=p)(pil_img)

    # using cv2+convert equalize
    import torchvision_npu
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_equalize = trans.RandomEqualize(p=p)(pil_img)

    assert type(pil_equalize) == type(cv2_equalize)
    assert pil_equalize.size == cv2_equalize.size
    assert (np.array(pil_equalize) == np.array(cv2_equalize)).all()
