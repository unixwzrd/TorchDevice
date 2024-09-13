import os
import traceback
import torch
from unittest.mock import patch


class TorchDevice:
    """
    This class represents a compute device used for Torch operations.
    It provides methods to check device availability, allocate device index, and clear GPU cache.
    """
    # Class variables needed by __init__
    DEVICE_TYPES = {
        "cuda": {"check_availability": torch.cuda.is_available, "fallback": "cpu"},
        "mps": {"check_availability": torch.backends.mps.is_available, "fallback": "cuda"},
    }
    devices = []
    gpu_available = None
    gpu_device = None

    def __init__(self, device_type=None, explicit_index=None):
        if TorchDevice.has_gpu is None:
            TorchDevice.initialize()

        if device_type and ":" in device_type:
            self.device_type, self.device_index = device_type.split(":")
            self.device_type = self.check_device_availability(self.device_type)
        else:
            self.device_type = TorchDevice.gpu_device
            self.device_index = self.get_device_index()

        print(f"Using {self.device_type} device with index {self.device_index}")
        self.device_index = explicit_index if explicit_index is not None else self.allocate_index()
        try:
            self.device = torch.device(f"{self.device_type}:{self.device_index}")
        except RuntimeError:
            self.device_type = "cpu"
            self.device = torch.device(f"{self.device_type}:{self.device_index}")
        TorchDevice.devices.append(self)

    @classmethod
    def initialize(cls):
        for device_type, device_info in cls.DEVICE_TYPES.items():
            if device_info["check_availability"]():
                cls.gpu_device = device_type
                cls.gpu_available = True  # update class variable
                break
        else:
            cls.gpu_device = "cpu"
            cls.gpu_available = False

    @classmethod
    def allocate_index(cls):
        """
        Allocates an index for the compute device.

        Returns:
            int: The allocated index for the compute device.
        """
        if len(cls.devices) > 0:
            return len(cls.devices)
        return 0

    @classmethod
    def has_gpu(cls):
        """
        Checks if GPU acceleration is available.

        Returns:
            bool: True if GPU acceleration is available, False otherwise.
        """
        if cls.gpu_available is None:
            cls.initialize()
        return cls.gpu_available  # return class variable

    def check_device_availability(self, device_type):
        """
        Checks if the specified device type is available.

        Returns:
            bool: True if the device type is available, False otherwise.
        """
        return device_type in TorchDevice.DEVICE_TYPES and TorchDevice.DEVICE_TYPES[device_type]["check_availability"]()
        
    @classmethod
    def default_device(cls):
        """
        Returns the default device type.

        Returns:
            str: The default device type.
        """
        if cls.gpu_device is None:
            cls.initialize()
        return cls.gpu_device

    def gpu_dev(self):
        """
        Returns the name of the GPU device available.

        Returns:
            str: The name of the GPU device.
        """
        return self.gpu_device


    @staticmethod
    def get_device_index():
        """
        Returns the index of the compute device.

        Returns:
            int: The index of the compute device.
        """
        device_index = int(os.getenv("LOCAL_RANK", "0"))
        return device_index

    def get_device_info(self):
        """
        Returns the device name and index for this TorchDevice instance.

        Returns:
            str: The device name and index in the format "devname:index".
        """
        return f"{self.device_type}:{self.device_index}"

    def clear_gpu_cache(self):
        """
        Removes the device from the devices list and clears the GPU cache.
        """
        self.remove()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def remove(self):
        """
        Removes the device from the devices list.
        """
        TorchDevice.devices.remove(self)

    # Mock device methods for migration of CUDA torch calls to MPS calls..
    @staticmethod
    def mock_torch_device(device_type):
        """
        Replacement for torch.device. if MPS is available, we will prefer that over CUDA.
        If CUDA is not available, we will default to MPS. Otherwise, it will work just as it would normally.
        """
        # Print the call stack
        traceback.print_stack()
        if device_type.startswith("cuda") and not torch.cuda.is_available() and torch.backends.mps.is_available():
            print("CUDA device is not available. Switching to MPS device.")
            device_type = "mps"
        elif device_type == "cpu" and torch.backends.mps.is_available():
            print("CPU device specified. Switching to MPS device.")
            device_type = "mps"
        # Call the original torch.device function
        return torch.device(device_type)

    @staticmethod
    def mock_is_cuda_available():
        """
        Replacement for torch.backends.cuda.is_available.
        If MPS is available, return True. Otherwise, we will pass things through to the original
        torch.backends.cuda.is_available function.
        """
        # Print the call stack
        traceback.print_stack()
        # If MPS is available, return True
        if torch.backends.mps.is_available():
            return True
        # Call the original torch.backends.cuda.is_available function
        return torch.cuda.is_available()

    @staticmethod
    def mock_device_count():
        """
        Replacement for torch.cuda.device_count.
        If MPS is available, return 1. Otherwise, return the number of available CUDA devices.
        """
        # Print the call stack
        traceback.print_stack()
        if torch.backends.mps.is_available():
            return 1
        else:
            return torch.cuda.device_count()

    @staticmethod
    def mock_is_cuda_built():
        """
        Replacement for torch.backends.cuda.is_built. If CUDA support is built into the PyTorch kernel,
        return True. Otherwise, return False.
        """
        # Print the call stack
        traceback.print_stack()
        if torch.backends.mps.is_available():
            return torch.backends.mps.is_built()
        return torch.backends.cuda.is_built()

    @staticmethod
    def mock_empty_cache(self):
        # Print the call stack
        traceback.print_stack()
        return self.clear_gpu_cache()

    @staticmethod
    def mock_get_device_properties(device):
        """
        Replacement for torch.cuda.get_device_properties.
        If MPS is available, return the total system memory.
        Otherwise, call the original torch.cuda.get_device_properties function.
        """
        # Print the call stack
        traceback.print_stack()
        # If MPS is available, return the total system memory
        if torch.backends.mps.is_available():
            # Get the total system memory in bytes
            total_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            # Return a dictionary with the total_memory key
            return {"total_memory": total_memory}
        # Call the original torch.cuda.get_device_properties function
        return torch.cuda.get_device_properties(device)

    def apply_patches(self):
        """
        Applies patches to replace torch.device and torch.backends.cuda.is_available with the mock functions.
        """
        with (
            patch("torch.device", new=self.mock_torch_device),
            patch("torch.cuda.is_available", new=self.mock_is_cuda_available),
            patch("torch.backends.cuda.is_built", new=self.mock_is_cuda_built),
        ):
            # Intercept instance methods
            torch.cuda.device_count = self.mock_device_count
            torch.cuda.empty_cache = self.mock_empty_cache
            torch.cuda.get_device_properties = self.mock_get_device_properties

