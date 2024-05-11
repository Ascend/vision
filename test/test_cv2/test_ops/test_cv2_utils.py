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
from numpy import average, linalg, dot


def image_similarity_vectors_via_cos(img1, img2, threshold=0.99):
    imgs = [img1, img2]
    vectors = []
    norms = []
    for img in imgs:
        vector = []
        for pixel_tuple in img.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    img1, img2 = vectors
    img1_norm, img2_norm = norms
    cos_res = dot(img1 / img1_norm, img2 / img2_norm)
    print(cos_res)
    if cos_res > threshold:
        return True
    else:
        return False
