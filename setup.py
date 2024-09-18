# setup.py

from setuptools import setup, find_packages

setup(
    name='TorchDevice',
    version='1.0.0',
    description='Intercepts PyTorch calls to enable transparent code portability between CUDA and MPS hardware.',
    author='unixwzrd',
    author_email='unixwzrd@unixwzrd.ai',
    url='https://github.com/unixwzrd/TorchDevice',
    packages=find_packages(),
    install_requires=[
        'torch>=2.2',
        'numpy',
        'psutil',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)