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
import torchvision
from torchvision import transforms as trans
import torchvision_npu


@pytest.mark.parametrize(
    "img_path",
    [
        "./test/Data/fish/fish_11.jpg",
        "./test/Data/fish/fish_22.jpg",
        "./test/Data/fish/fish_33.jpg",
        "./test/Data/fish/fish_44.jpg",
        "./test/Data/fish/fish_55.jpg",
    ],
)
def test_pil2tensor(img_path):
    pil_img = Image.open(img_path)

    # using pil totensor
    torchvision.set_image_backend("PIL")
    pil_totensor = trans.PILToTensor()(pil_img)

    # using cv2 totensor
    torchvision.set_image_backend("cv2")
    cv2_img = np.asarray(pil_img)
    cv2_totensor = trans.PILToTensor()(cv2_img)

    assert type(pil_totensor) == type(cv2_totensor)
    assert pil_totensor.shape == cv2_totensor.shape
    assert (np.array(pil_totensor) == np.array(cv2_totensor)).all()
    
    pil_img.close()
