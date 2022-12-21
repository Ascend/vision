import distutils
import os
import glob
import shutil

import torch
import torch_npu

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup
from torch_npu.utils.cpp_extension import NpuExtension
from torch.utils.cpp_extension import BuildExtension, CppExtension

__vision__ = '0.9.1'

# ext_modules = [
#       # Pybind11Extension(
#       #   "torchvision_npu.runner",
#       #   sources=glob.glob(r'torchvision_npu/csrc/*.cpp'),
#       #   # [r'torchvision_npu/csrc/pybind.cpp'],
#       #   defind_macros=[("VISION_INFO", __vision__)]
#       # ),
#     NpuExtension(
#         name="torchvision_npu.ops",
#         sources=glob.glob(r'torchvision_npu/csrc/InitNpuBinding.cpp') ,
#     )
# ]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'torchvision_npu', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp')) \
                + glob.glob(os.path.join(extensions_dir, 'ops', '*.cpp')) \
                + glob.glob(os.path.join(extensions_dir, 'ops', 'npu', '*.cpp'))

    # sources = main_file + source_cpu
    sources = main_file
    extension = NpuExtension


    define_macros = []

    extra_compile_args = {"cxx": []}

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

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