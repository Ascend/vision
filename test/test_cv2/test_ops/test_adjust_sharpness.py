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
from test_cv2_utils import image_similarity_vectors_via_cos


@pytest.mark.parametrize(
    ["img_path", "sharpness_factor", "p"],
    [
        ("./test/Data/fish/fish_11.jpg", 0, 0.3),
        ("./test/Data/fish/fish_22.jpg", 1, 0.7),
        ("./test/Data/fish/fish_33.jpg", 0, 1),
        ("./test/Data/fish/fish_44.jpg", 0.1, 0),
        ("./test/Data/fish/fish_55.jpg", 0.9, 0.1),
    ],
)
def test_adjustSharpness(img_path, sharpness_factor, p):
    pil_img = Image.open(img_path)

    # using pil adjustSharpness
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_adjustSharpness = trans.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=p)(pil_img)

    # using cv2 adjustSharpness
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    torchvision_npu.set_image_backend("cv2")
    cv2_adjustSharpness = trans.RandomAdjustSharpness(sharpness_factor=sharpness_factor, p=p)(cv2_img)

    assert isinstance(pil_adjustSharpness, Image.Image) and isinstance(cv2_adjustSharpness, np.ndarray)
    assert pil_adjustSharpness.size == cv2_adjustSharpness.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_adjustSharpness, Image.fromarray(cv2_adjustSharpness))
