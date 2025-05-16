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
import importlib
import subprocess
import stat
from pathlib import Path
from typing import Union

import torch
import torch_npu

from setuptools import find_packages, setup
from torch_npu.utils.cpp_extension import NpuExtension
from torch.utils.cpp_extension import BuildExtension, CppExtension


VERSION = '0.21.0'
UNKNOWN = "Unknown"


def get_sha(vision_root: Union[str, Path]) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vision_root)  # Compliant
            .decode("ascii")
            .strip()
        )
    except Exception:
        return UNKNOWN


def generate_torchvision_npu_version():
    torchvision_npu_root = Path(__file__).resolve().parent
    version_path = torchvision_npu_root / "torchvision_npu" / "version.py"
    if version_path.exists():
        version_path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    modes = stat.S_IWUSR | stat.S_IRUSR
    sha = get_sha(torchvision_npu_root)
    global VERSION
    VERSION += "+git" + sha[:7]
    with os.fdopen(os.open(version_path, flags, modes), 'w') as f:
        f.write("__version__ = '{version}'\n".format(version=VERSION))
    os.chmod(version_path, mode=stat.S_IRUSR | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)


generate_torchvision_npu_version()


def is_neon_supported():
    cpu_info = "/proc/cpuinfo"
    neon_support = False
    if os.path.exists(cpu_info):
        with open(cpu_info, "r") as f:
            for line in f:
                if "neon" in line.lower() or "asimd" in line.lower():
                    neon_support = True
                    break
    return neon_support


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'torchvision_npu', 'csrc')

    main_file = glob.glob(os.path.join(extensions_dir, 'ops', 'cpu', '*.cpp')) + \
                glob.glob(os.path.join(extensions_dir, 'ops', 'npu', '*.cpp')) + \
                glob.glob(os.path.join(extensions_dir, 'ops', 'autocast', '*.cpp')) + \
                glob.glob(os.path.join(extensions_dir, 'ops', '*.cpp')) + \
                glob.glob(os.path.join(extensions_dir, '*.cpp'))
    if is_neon_supported():
        main_file += glob.glob(os.path.join(extensions_dir, 'ops', 'kp_cpu', '*.cpp'))

    sources = main_file
    extension = NpuExtension

    define_macros = []

    extra_compile_args = [
        '-Wno-sign-compare',
        '-Wno-deprecated-declarations',
        '-Wno-return-type',
        '-fstack-protector-all',
        '-fPIE',
        '-fPIC',
        '-pie',
        '-fvisibility=hidden'
    ]

    extra_link_args = [
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now"
    ]

    DEBUG = (os.getenv('DEBUG', default='').upper() in ['ON', '1', 'YES', 'TRUE', 'Y'])
    if DEBUG:
        extra_compile_args += ['-O0', '-g']
        extra_link_args += ['-O0', '-g']
    else:
        extra_compile_args += ['-O3']
        extra_compile_args += ['-fopenmp']
        extra_link_args += ['-s']

    try:
        extra_compile_args += [
            '-D__FILENAME__=\"$$(notdir $$(abspath $$<))\"'
        ]
        torch_npu_path = importlib.util.find_spec('torch_npu').submodule_search_locations[0]
        extra_compile_args += [
            '-I' + os.path.join(torch_npu_path, 'include', 'third_party', 'acl', 'inc')
        ]
    except Exception as e:
        raise ImportError('can not find any torch_npu') from e

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir, '*.hpp']
    extra_objects = glob.glob(os.path.join(extensions_dir, 'ops', 'kp_cpu', '*.s')) if is_neon_supported() else []

    ext_modules = [
        extension(
            'torchvision_npu._C',
            sorted(sources),
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            extra_objects=extra_objects,
        )
    ]

    return ext_modules


package_name = os.environ.get('TORCHVISION_NPU_PACKAGE_NAME', 'torchvision_npu')
setup(name=package_name,
      version=VERSION,
      description='NPU bridge for Torchvision',
      packages=find_packages(),
      package_data={package_name: ['lib/*.so', '*.so']},
      ext_modules=get_extensions(),
      cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
      )
