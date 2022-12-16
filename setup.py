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

ext_modules = [
      # Pybind11Extension(
      #   "torchvision_npu.runner",
      #   sources=glob.glob(r'torchvision_npu/csrc/*.cpp'),
      #   # [r'torchvision_npu/csrc/pybind.cpp'],
      #   defind_macros=[("VISION_INFO", __vision__)]
      # ),
    NpuExtension(
        name="torchvision_npu.ops",
        sources=glob.glob(r'torchvision_npu/csrc/InitNpuBinding.cpp') ,
    )
]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'torchvision_npu', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp')) \
                + glob.glob(os.path.join(extensions_dir, 'ops', '*.cpp')) \
                + glob.glob(os.path.join(extensions_dir, 'ops', 'npu', '*.cpp'))

    # sources = main_file + source_cpu
    sources = main_file
    extension = CppExtension

    compile_cpp_tests = os.getenv('WITH_CPP_MODELS_TEST', '0') == '1'
    if compile_cpp_tests:
        test_dir = os.path.join(this_dir, 'test')
        models_dir = os.path.join(this_dir, 'torchvision', 'csrc', 'models')
        test_file = glob.glob(os.path.join(test_dir, '*.cpp'))
        source_models = glob.glob(os.path.join(models_dir, '*.cpp'))

        test_file = [os.path.join(test_dir, s) for s in test_file]
        source_models = [os.path.join(models_dir, s) for s in source_models]
        tests = test_file + source_models
        tests_include_dirs = [test_dir, models_dir]

    define_macros = []

    extra_compile_args = {}

    debug_mode = os.getenv('DEBUG', '0') == '1'
    if debug_mode:
        print("Compile in debug mode")
        extra_compile_args['cxx'].append("-g")
        extra_compile_args['cxx'].append("-O0")
        if "nvcc" in extra_compile_args:
            # we have to remove "-OX" and "-g" flag if exists and append
            nvcc_flags = extra_compile_args["nvcc"]
            extra_compile_args["nvcc"] = [
                f for f in nvcc_flags if not ("-O" in f or "-g" in f)
            ]
            extra_compile_args["nvcc"].append("-O0")
            extra_compile_args["nvcc"].append("-g")

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
    if compile_cpp_tests:
        ext_modules.append(
            extension(
                'torchvision_npu._C_tests',
                tests,
                include_dirs=tests_include_dirs,
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
            )
        )
    return ext_modules

package_name = os.environ.get('TORCHVISION_NPU_PACKAGE_NAME', 'torchvision_npu')
setup(name=package_name,
      version=__vision__,
      description='NPU bridge for Torchvision',
      url='https://gitee.com/ascend/vision',
      packages=find_packages(),
      package_data={package_name: ['lib/*.so', '*.so']},
      ext_modules=ext_modules,
      cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
      )