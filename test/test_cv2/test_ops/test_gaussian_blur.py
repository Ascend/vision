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

from PIL import Image
import pytest
from torchvision import transforms as trans
from test_cv2_utils import image_similarity_vectors_via_cos
import numpy as np
import torchvision_npu


@pytest.mark.parametrize(
    ["img_path", "kernel_size", "sigma"],
    [
        ("./test/Data/fish/fish_11.jpg", 1, 10),
        ("./test/Data/fish/fish_22.jpg", 11, 100),
        ("./test/Data/fish/fish_33.jpg", 3, (0.1, 2.0)),
        ("./test/Data/fish/fish_44.jpg", 5, 2),
        ("./test/Data/fish/fish_55.jpg", 11, 100),
    ],
)
def test_gaussian_blur(img_path, kernel_size, sigma):
    pil_img = Image.open(img_path)

    # using pil gaussian_blur
    torchvision_npu.set_image_backend("PIL")
    pil_gaussian_blur = trans.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(pil_img)

    # using cv2 gaussian_blur
    torchvision_npu.set_image_backend("cv2")
    cv2_img = np.asarray(pil_img)
    cv2_gaussian_blur = trans.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(cv2_img)

    assert isinstance(pil_gaussian_blur, Image.Image) and isinstance(cv2_gaussian_blur, np.ndarray)
    assert pil_gaussian_blur.size == cv2_gaussian_blur.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_gaussian_blur, Image.fromarray(cv2_gaussian_blur))
