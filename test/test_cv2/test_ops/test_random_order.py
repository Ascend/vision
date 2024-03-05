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

import random
import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms as trans
from test_cv2_utils import image_similarity_vectors_via_cos
import torchvision_npu


@pytest.mark.parametrize(
    ["img_path", "transforms"],
    [
        ("./test/Data/fish/fish_11.jpg",
         [trans.RandomCrop(224), trans.GaussianBlur(1, 10)]),
        ("./test/Data/fish/fish_22.jpg",
         [trans.RandomResizedCrop(224), trans.RandomHorizontalFlip()]),
        ("./test/Data/fish/fish_33.jpg",
         [trans.Resize(224), trans.CenterCrop(224)]),
        ("./test/Data/fish/fish_44.jpg",
         [trans.RandomRotation(60), trans.GaussianBlur(1, 10)]),
        ("./test/Data/fish/fish_55.jpg",
         [trans.RandomAdjustSharpness(1, 0.7), trans.RandomPerspective(0.5)]),
    ],
)
def test_compose(img_path, transforms):
    pil_img = Image.open(img_path)

    # using pil order
    torch.manual_seed(10)
    random.seed(10)
    torchvision_npu.set_image_backend("PIL")
    pil_order = trans.RandomOrder(transforms=transforms)(
        pil_img)

    # using cv2 order
    torchvision_npu.set_image_backend("cv2")
    torch.manual_seed(10)
    random.seed(10)
    cv2_img = np.asarray(pil_img)
    cv2_order = trans.RandomOrder(transforms=transforms)(cv2_img)

    assert isinstance(pil_order, Image.Image) and isinstance(cv2_order, np.ndarray)
    assert pil_order.size == cv2_order.shape[:2][::-1]
    assert image_similarity_vectors_via_cos(pil_order, Image.fromarray(cv2_order))
    
    pil_img.close()
