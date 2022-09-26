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

from struct import pack, unpack_from
from PIL.ImageFile import _safe_read


def i8(c):
    return c if c.__class__ is int else c[0]


def i16(c, o=0):
    return unpack_from(">H", c, o)[0]


def skip(fp):
    n = i16(fp.read(2)) - 2
    _safe_read(fp, n)


def sof(fp):
    n = i16(fp.read(2)) - 2
    s = _safe_read(fp, n)
    h, w, c = i16(s, 1), i16(s, 3), i8(s[5])
    return h, w, c


marker_sof = {
    0xFFC0, 0xFFC1, 0xFFC2, 0xFFC3, 0xFFC5, 0xFFC6, 0xFFC7, 0xFFC9, 0xFFCA, 0xFFCB, 0xFFCD,
    0xFFCE, 0xFFCF, 0xFFDE
}
marker_skip = {
    0xFFC4, 0xFFC8, 0xFFCC, 0xFFD0, 0xFFD1, 0xFFD2, 0xFFD3, 0xFFD4, 0xFFD5, 0xFFD6, 0xFFD7,
    0xFFD8, 0xFFD9, 0xFFDA, 0xFFDB, 0xFFDC, 0xFFDD, 0xFFDF, 0xFFE0, 0xFFE1, 0xFFE2, 0xFFE3,
    0xFFE4, 0xFFE5, 0xFFE6, 0xFFE7, 0xFFE8, 0xFFE9, 0xFFEA, 0xFFEB, 0xFFEC, 0xFFED, 0xFFEE,
    0xFFEF, 0xFFF0, 0xFFF1, 0xFFF2, 0xFFF3, 0xFFF4, 0xFFF5, 0xFFF6, 0xFFF7, 0xFFF8, 0xFFF9,
    0xFFFA, 0xFFFB, 0xFFFC, 0xFFFD, 0xFFFE
}


def extract_jpeg_shape(fp):
    s = fp.read(3)
    s = b"\xFF"

    while True:
        i = s[0]
        if i == 0xFF:
            s = s + fp.read(1)
            i = i16(s)
        else:
            s = fp.read(1)
            continue

        if i in marker_sof:
            h, w, c = sof(fp)
            break
        elif i in marker_skip:
            skip(fp)
            s = fp.read(1)
        elif i == 0 or i == 0xFFFF:
            s = b"\xff"
        elif i == 0xFF00:
            s = fp.read(1)
        else:
            raise SyntaxError("no marker found")

    return h, w, c