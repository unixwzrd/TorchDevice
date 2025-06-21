# setup.py

from setuptools import setup, find_packages

setup(
    name='TorchDevice',

    description='Intercepts PyTorch calls to enable transparent code portability between CUDA and MPS hardware.',
    author='unixwzrd',
    author_email='unixwzrd@unixwzrd.ai',
    url='https://github.com/unixwzrd/TorchDevice',
    packages=find_packages(),
    install_requires=[
        'torch>=2.2',
        'numpy',
        'psutil',
        'transformers>=4.30.0',  # For BERT and transformer model support
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)
