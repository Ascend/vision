#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export BUILD_TYPE=wheel
setup_env 0.12.0
setup_wheel_python
pip_install numpy pyyaml future ninja
python setup.py clean

# Install auditwheel to get some inspection utilities
pip_install auditwheel

# Point to custom libraries
export LD_LIBRARY_PATH=$(pwd)/ext_libraries/lib:$LD_LIBRARY_PATH
export TORCHVISION_INCLUDE=$(pwd)/ext_libraries/include
export TORCHVISION_LIBRARY=$(pwd)/ext_libraries/lib

IS_WHEEL=1 python setup.py bdist_wheel

