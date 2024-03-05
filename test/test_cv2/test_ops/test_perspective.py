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
from test_cv2_utils import image_similarity_vectors_via_cos
import torchvision_npu


@pytest.mark.parametrize(
    ["img_path", "distortion_scale", "p", "interpolation", "fill"],
    [
        ("./test/Data/fish/fish_11.jpg", 0.6, 0.5, 2, 0),
        ("./test/Data/fish/fish_22.jpg", 0.1, 0.3, 2, 0),
        ("./test/Data/fish/fish_33.jpg", 1, 1, 0, (255, 0, 0)),
        ("./test/Data/fish/fish_44.jpg", 0, 0.3, 1, (255, 0, 0)),
        ("./test/Data/fish/fish_55.jpg", 1, 0.5, 2, (255, 0, 0)),
    ],
)
def test_perspective(img_path, distortion_scale, p, interpolation, fill):
    pil_img = Image.open(img_path)

    # using pil perspective
    torchvision_npu.set_image_backend("PIL")
    torch.manual_seed(10)
    pil_perspective = trans.RandomPerspective(distortion_scale=distortion_scale, p=p, interpolation=interpolation,
                                              fill=fill)(pil_img)

    # using cv2+convert perspective
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_perspective = trans.RandomPerspective(distortion_scale=distortion_scale, p=p, interpolation=interpolation,
                                              fill=fill)(cv2_img)

    assert isinstance(pil_perspective, Image.Image) and isinstance(cv2_perspective, np.ndarray)
    assert pil_perspective.size == cv2_perspective.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_perspective, Image.fromarray(cv2_perspective))
    
    pil_img.close()
