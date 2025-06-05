"""
Test to verify tensor movement operations after fixing CPU override inconsistency.
"""

import os
import sys
import unittest
import torch
import logging

# Add the project directory to sys.path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import TorchDevice
from TorchDevice.core.device import DeviceManager


class TestFixedTensorMove(unittest.TestCase):
    """Test case to verify fixed tensor movement operations."""

    def test_device_manager_state(self):
        """Verify the DeviceManager state and CPU override flag."""
        logger.info("Default device type: %s", DeviceManager._default_device_type)
        logger.info("CPU override flag: %s", DeviceManager._cpu_override)
        logger.info("Torch device replacement result for 'mps': %s", 
                    DeviceManager.torch_device_replacement('mps'))
        
        # Verify CPU override is not active
        self.assertFalse(DeviceManager._cpu_override)
        
        # Verify torch_device_replacement works correctly for MPS
        if torch.backends.mps.is_available():
            device = DeviceManager.torch_device_replacement('mps')
            self.assertEqual(device.type, 'mps')

    def test_tensor_movement(self):
        """Test tensor movement operations with various devices."""
        # Create a tensor on CPU
        x = torch.tensor([1, 2, 3])
        logger.info("Original tensor device: %s", x.device)
        
        # Try moving to MPS if available
        if torch.backends.mps.is_available():
            try:
                # Direct movement to MPS
                x_mps = x.to('mps')
                logger.info("Tensor moved to MPS device: %s", x_mps.device)
                self.assertEqual(x_mps.device.type, 'mps')
                
                # Move back to CPU
                x_cpu = x_mps.to('cpu')
                logger.info("Tensor moved back to CPU: %s", x_cpu.device)
                self.assertEqual(x_cpu.device.type, 'cpu')
                
                # Try the mps() method if available
                if hasattr(x, 'mps'):
                    x_mps2 = x.mps()
                    logger.info("Tensor moved to MPS using mps() method: %s", x_mps2.device)
                    self.assertEqual(x_mps2.device.type, 'mps')
            except Exception as e:
                logger.error("Error during MPS tensor movement: %s", str(e))
                raise

        # Try moving to CUDA if available
        if torch.cuda.is_available():
            try:
                # Direct movement to CUDA
                x_cuda = x.to('cuda')
                logger.info("Tensor moved to CUDA device: %s", x_cuda.device)
                self.assertEqual(x_cuda.device.type, 'cuda')
                
                # Move back to CPU
                x_cpu = x_cuda.to('cpu')
                logger.info("Tensor moved back to CPU: %s", x_cpu.device)
                self.assertEqual(x_cpu.device.type, 'cpu')
            except Exception as e:
                logger.error("Error during CUDA tensor movement: %s", str(e))
                raise


if __name__ == '__main__':
    unittest.main()
