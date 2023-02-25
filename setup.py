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

import os
import glob

import torch
import torch_npu

from setuptools import find_packages, setup
from torch_npu.utils.cpp_extension import NpuExtension
from torch.utils.cpp_extension import BuildExtension, CppExtension

__vision__ = '0.12.0'

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'torchvision_npu', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, 'ops', 'npu', '*.cpp'))

    # sources = main_file + source_cpu
    sources = main_file
    extension = NpuExtension


    define_macros = []

    extra_compile_args = {"cxx": []}

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir, '*.hpp']

    ext_modules = [
        extension(
            'torchvision_npu._C',
            sorted(sources),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

package_name = os.environ.get('TORCHVISION_NPU_PACKAGE_NAME', 'torchvision_npu')
setup(name=package_name,
      version=__vision__,
      description='NPU bridge for Torchvision',
      url='https://gitee.com/ascend/vision',
      packages=find_packages(),
      package_data={package_name: ['lib/*.so', '*.so']},
      ext_modules=get_extensions(),
      cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
      )