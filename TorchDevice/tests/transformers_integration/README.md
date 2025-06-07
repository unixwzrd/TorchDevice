# Transformers Integration Tests for TorchDevice

This directory contains copies of selected tests from the Hugging Face Transformers library,
modified to include `import TorchDevice` at the beginning.

These tests are used to:
1. Validate TorchDevice's compatibility with real-world PyTorch usage in the Transformers library.
2. Drive the identification of PyTorch functionalities that require patching or improved handling
   by TorchDevice for seamless CUDA-to-MPS (or CPU) redirection.
3. Serve as integration tests for TorchDevice.

Original Transformers version for these tests: v4.51.3
Corresponding PyTorch version: Nightly 2.8.0 (commit b182d84228cd3fcff23cbc5e945af307fd762f5b)
