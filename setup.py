import os
from setuptools import find_packages, setup


setup(name=os.environ.get('TORCHVISION_NPU_PACKAGE_NAME', 'torchvision_npu'),
      version='0.9.1',
      description='NPU bridge for Torchvision',
      packages=find_packages()
      )
