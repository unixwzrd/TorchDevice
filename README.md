# TorchDevice.py Project

A simple module to abstract a Torch device based on what types of processors are available for handling PyTorch GPU/TPU processing.

The idea here is simple allow code written specifically for CUDA devices to run on MPS devices with no change. This is done by capturing all CUDA calls and replacing them with MPS calls.

When a CUDA device is defined, it will instead return an MPS device if it exists, otherwise it will return a CUDA device.

This should all be transparent to the user or developer, except for messages passed to STDERR indicating the one device was switched out for another.

Additionally whenever a replacement call or override call is made, a traceback will be produced at that point so the developer knows where changes are required in their code to handle multiple device types.

This is not for long-term use, though I'm sure some will use it as such, but more to allow developers an easy way to get their software running on Apple Silicon quickly using the existing CUDA codebase of applications and brining them to macOS quickly.
