import os
from setuptools import find_packages, setup

os.chdir(os.path.dirname(os.path.abspath(__file__)))

setup(
    name='pondconcise',
    packages=find_packages(),
    version='0.0.1',
    description='Implementation of a podcasts automatic summary package.',
    author='Pierre Massey',
    licenses_files=('LICENSE.txt',),
    url="https://github.com/PierreMsy/PodConcise.git",
    include_package_data=True
)