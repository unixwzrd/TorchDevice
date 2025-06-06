"""
Test to verify tensor movement operations after fixing CPU override inconsistency.
"""

import os
import sys
import unittest
import torch
import logging
import random
import numpy as np

# Add the project directory to sys.path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Configure module-level logger (can be overridden by test runner)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# General logger for the module, not used directly in tests if self.test_logger is defined
# logger = logging.getLogger(__name__)

import TorchDevice # Ensures TorchDevice patches are applied

class TestFixedTensorMove(unittest.TestCase):
    """Test case to verify fixed tensor movement operations and CPU override."""

    def setUp(self):
        """Set up test environment before each test."""
        # Per-test logger
        self.test_logger = logging.getLogger(f"{__name__}.{self.id().split('.')[-1]}")
        self.test_logger.info("--- Test Setup ---")

        # Seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        if torch.cuda.is_available(): # Patched by TorchDevice
            torch.cuda.manual_seed_all(42)
            self.test_logger.info("CUDA (or MPS via redirect) is available. Seeded with torch.cuda.manual_seed_all(42).")
        else:
            self.test_logger.info("CUDA (and MPS via redirect) not available. Not calling torch.cuda.manual_seed_all.")

        from TorchDevice.core.device import DeviceManager # For fixture state management
        self._initial_cpu_override_state_fixture = DeviceManager.cpu_override()
        self.test_logger.info(f"Initial CPU override state (from DeviceManager): {self._initial_cpu_override_state_fixture}")

        if self._initial_cpu_override_state_fixture:
            self.test_logger.info("CPU override was ON. Toggling OFF for test.")
            torch.device("cpu:-1")
            self.test_logger.info(f"CPU override state after toggle attempt (from DeviceManager): {DeviceManager.cpu_override()}")

        self.test_logger.info(f"Default device at test start: {torch.get_default_device()}")
        if torch.backends.mps.is_available() or (hasattr(torch.backends, 'cuda') and torch.backends.cuda.is_built()): # True hardware check
             self.assertFalse(DeviceManager.cpu_override(), "CPU override should be OFF at test start if MPS/CUDA hardware exists.")
        self.test_logger.info("--- End Test Setup ---")

    def tearDown(self):
        """Clean up after each test."""
        self.test_logger.info("--- Test Teardown ---")
        from TorchDevice.core.device import DeviceManager # For fixture state management
        current_cpu_override_state = DeviceManager.cpu_override()
        if current_cpu_override_state != self._initial_cpu_override_state_fixture:
            self.test_logger.info(f"CPU override state changed (current: {current_cpu_override_state}, initial: {self._initial_cpu_override_state_fixture}). Restoring.")
            torch.device("cpu:-1")
            self.test_logger.info(f"CPU override state after restoration (from DeviceManager): {DeviceManager.cpu_override()}")
        self.test_logger.info(f"Default device at test end: {torch.get_default_device()}")
        self.test_logger.info("--- End Test Teardown ---")

    def test_cpu_override_public_api_toggle(self):
        """Verify the CPU override toggle behavior via public torch.device API."""
        self.test_logger.info("Starting test_cpu_override_public_api_toggle")
        device_with_override_off = torch.get_default_device()
        self.test_logger.info(f"Default device (CPU override intended OFF via setUp): {device_with_override_off}")
        if torch.backends.mps.is_available(): # Actual MPS hardware
            self.assertEqual(device_with_override_off.type, 'mps', "With CPU override OFF, default should be MPS if available.")

        self.test_logger.info("Toggling CPU override ON using torch.device('cpu:-1')")
        torch.device("cpu:-1")
        device_after_on_toggle = torch.get_default_device()
        self.test_logger.info(f"Device after toggling override ON: {device_after_on_toggle}")
        self.assertEqual(device_after_on_toggle.type, 'cpu', "After toggling override ON, default device should be CPU.")

        self.test_logger.info("Toggling CPU override OFF again using torch.device('cpu:-1')")
        torch.device("cpu:-1")
        device_after_off_toggle = torch.get_default_device()
        self.test_logger.info(f"Device after toggling override OFF again: {device_after_off_toggle}")
        self.assertEqual(device_after_off_toggle.type, device_with_override_off.type,
                         f"Device type should revert to '{device_with_override_off.type}'.")
        self.assertTrue(device_after_off_toggle.index == device_with_override_off.index or \
                        (device_with_override_off.index is None and device_after_off_toggle.index == 0) or \
                        (device_after_off_toggle.index is None and device_with_override_off.index == 0),
                        f"Device index should revert. Got {device_after_off_toggle}, expected {device_with_override_off}")
        self.test_logger.info("Finished test_cpu_override_public_api_toggle")

    def test_tensor_movement(self):
        """Test tensor movement operations with various devices (CPU override OFF)."""
        self.test_logger.info("Starting test_tensor_movement (CPU override is OFF via setUp)")
        x_initial = torch.tensor([1, 2, 3]) # Default device (e.g., MPS on user's Mac)
        self.test_logger.info(f"Initial tensor device (default): {x_initial.device}")

        expected_initial_type = 'cpu' # Fallback if no GPU
        if torch.backends.mps.is_available(): # Actual MPS hardware
            expected_initial_type = 'mps'
        elif hasattr(torch.backends, 'cuda') and torch.backends.cuda.is_built(): # Actual CUDA hardware
             expected_initial_type = 'cuda'
        self.assertEqual(x_initial.device.type, expected_initial_type, f"Initial tensor should be on {expected_initial_type}")

        if torch.backends.mps.is_available(): # Actual MPS hardware
            self.test_logger.info("Moving initial tensor to 'mps'")
            x_mps = x_initial.to('mps')
            self.test_logger.info(f"Tensor after .to('mps'): {x_mps.device}")
            self.assertEqual(x_mps.device.type, 'mps', "Tensor should be on MPS after .to('mps')")

            self.test_logger.info("Moving MPS tensor to 'cpu'")
            x_cpu_from_mps = x_mps.to('cpu')
            self.test_logger.info(f"Tensor after .to('cpu') from MPS: {x_cpu_from_mps.device}")
            self.assertEqual(x_cpu_from_mps.device.type, expected_initial_type, f"Tensor should be on '{expected_initial_type}' after .to('cpu') from MPS when CPU override is OFF.")
        
        if torch.cuda.is_available(): # Patched by TorchDevice (True if MPS available)
            self.test_logger.info("Moving initial tensor to 'cuda' (expecting redirection or native)")
            x_to_cuda = x_initial.to('cuda')
            self.test_logger.info(f"Tensor after .to('cuda'): {x_to_cuda.device}")
            
            # When CPU override is OFF (as in this test setup), torch_device_replacement('cuda')
            # will return the default device type (e.g., 'mps' on this system).
            # expected_initial_type already stores this default device type.
            final_expected_type_for_cuda_call = expected_initial_type
            
            self.assertEqual(x_to_cuda.device.type, final_expected_type_for_cuda_call, 
                             f"Tensor .to('cuda') should be on '{final_expected_type_for_cuda_call}' when CPU override is OFF, due to default device redirection.")

            self.test_logger.info(f"Moving '{final_expected_type_for_cuda_call}' tensor to 'cpu'")
            x_cpu_from_cuda = x_to_cuda.to('cpu')
            self.test_logger.info(f"Tensor after .to('cpu') from '{final_expected_type_for_cuda_call}': {x_cpu_from_cuda.device}")
            self.assertEqual(x_cpu_from_cuda.device.type, expected_initial_type, f"Tensor should be on '{expected_initial_type}' after .to('cpu') from '{final_expected_type_for_cuda_call}' when CPU override is OFF.")
        self.test_logger.info("Finished test_tensor_movement")

if __name__ == '__main__':
    unittest.main(argv=sys.argv[:1])
