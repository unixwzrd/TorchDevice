#!/usr/bin/env python
"""
Refactored test for tensor movement and DeviceManager interactions,
focusing on CPU override behavior using public APIs.
"""
import unittest
import torch
import logging
import sys
import TorchDevice
import os

# Configure a specific logger for this test file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent double logging if a handler is already attached by run_tests_and_install.py
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class TestRefactoredTensorMovement(unittest.TestCase):
    """Test tensor movement with CPU override managed via public APIs."""

    _default_non_cpu_device_type = None
    _original_cpu_override_state_was_on = None # Store initial state

    @classmethod
    def setUpClass(cls):
        """Determine default non-CPU device and store initial CPU override state."""
        cls.test_logger = logger
        cls.test_logger.info("--- TestRefactoredTensorMovement: setUpClass ---")

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            cls._default_non_cpu_device_type = "mps"
        elif hasattr(torch.backends, 'cuda') and torch.backends.cuda.is_available():
            # This case is unlikely on the user's Mac MPS setup but included for completeness
            cls._default_non_cpu_device_type = "cuda" 
        else:
            # Fallback if TorchDevice doesn't make MPS appear available through torch.backends
            # Forcing 'mps' as per typical test environment for this project on Mac
            cls._default_non_cpu_device_type = "mps"
            cls.test_logger.warning(
                f"Neither MPS nor CUDA explicitly available via torch.backends. "
                f"Assuming '{cls._default_non_cpu_device_type}' due to TorchDevice patching for tests."
            )
        
        cls.test_logger.info(f"Determined default non-CPU device type for tests: {cls._default_non_cpu_device_type}")

        # Record if CPU override was initially ON, then ensure it's OFF for the test suite.
        # We use the public API torch.device("cpu:-1") to toggle.
        initial_default_device = torch.get_default_device()
        cls._original_cpu_override_state_was_on = (initial_default_device.type == 'cpu')
        
        if cls._original_cpu_override_state_was_on:
            cls.test_logger.info(f"CPU override is ON (default device: {initial_default_device.type}). Toggling OFF for test suite.")
            torch.device("cpu:-1") # Toggle CPU override OFF
        else:
            cls.test_logger.info(f"CPU override is OFF (default device: {initial_default_device.type}). State is as desired for tests.")
        
        cls.test_logger.info(f"Default device after setUpClass adjustment: {torch.get_default_device()}")
        # Sanity check
        if torch.get_default_device().type == 'cpu':
             cls.test_logger.error("CRITICAL: CPU override FAILED to turn OFF in setUpClass!")

        cls.test_logger.info("--- End setUpClass ---")

    @classmethod
    def tearDownClass(cls):
        """Restore initial CPU override state if it was changed."""
        cls.test_logger.info("--- TestRefactoredTensorMovement: tearDownClass ---")
        current_default_is_cpu = (torch.get_default_device().type == 'cpu')
        
        if cls._original_cpu_override_state_was_on and not current_default_is_cpu:
            cls.test_logger.info("Original CPU override was ON. Restoring CPU override to ON.")
            torch.device("cpu:-1") # Toggle ON
        elif not cls._original_cpu_override_state_was_on and current_default_is_cpu:
            # This case means a test left CPU override ON when it shouldn't have,
            # or the initial state was OFF and somehow it got turned ON.
            cls.test_logger.warning("Original CPU override was OFF, but it's ON now. Restoring to OFF.")
            torch.device("cpu:-1") # Toggle OFF
            
        cls.test_logger.info(f"Final default device after tearDownClass: {torch.get_default_device()}")
        cls.test_logger.info("--- End tearDownClass ---")

    def setUp(self):
        """Ensure CPU override is OFF before each test method by default."""
        self.test_logger.info(f"--- Test: {self.id()} setUp ---")
        # Most tests expect CPU override to be OFF.
        # setUpClass should have already set it OFF. This is a safeguard.
        if torch.get_default_device().type == 'cpu':
            self.test_logger.warning(f"CPU override is ON at start of '{self.id()}'. Toggling OFF.")
            torch.device("cpu:-1") 
        self.test_logger.info(f"Default device at start of test '{self.id()}': {torch.get_default_device()}")

    def tearDown(self):
        """Ensure CPU override is reset to OFF after each test (unless a test specifically requires it ON)."""
        self.test_logger.info(f"--- Test: {self.id()} tearDown ---")
        # This ensures that if a test method (like test_tensor_movement_cpu_override_on)
        # turns CPU override ON, it's turned OFF here for the next test.
        if torch.get_default_device().type == 'cpu':
            self.test_logger.info(f"CPU override is ON at end of '{self.id()}'. Toggling OFF.")
            torch.device("cpu:-1")
        self.test_logger.info(f"Default device at end of test '{self.id()}': {torch.get_default_device()}")

    def test_tensor_movement_cpu_override_off(self):
        """Test tensor movements when CPU override is OFF."""
        self.test_logger.info("Starting test_tensor_movement_cpu_override_off (CPU override should be OFF)")
        
        current_default_device = torch.get_default_device()
        self.assertEqual(current_default_device.type, self._default_non_cpu_device_type,
                         f"CPU override should be OFF. Expected default device '{self._default_non_cpu_device_type}', got '{current_default_device.type}'.")

        x_cpu = torch.randn(2, 3, device='cpu') # Explicitly create on CPU first
        self.test_logger.info(f"Initial tensor created on CPU: {x_cpu.device}")

        self.test_logger.info(f"Moving CPU tensor to '{self._default_non_cpu_device_type}'")
        x_non_cpu = x_cpu.to(self._default_non_cpu_device_type)
        self.test_logger.info(f"Tensor after .to('{self._default_non_cpu_device_type}'): {x_non_cpu.device}")
        self.assertEqual(x_non_cpu.device.type, self._default_non_cpu_device_type,
                         f"Tensor should be on '{self._default_non_cpu_device_type}'.")

        self.test_logger.info(f"Moving '{self._default_non_cpu_device_type}' tensor to 'cpu'")
        x_cpu_from_non_cpu = x_non_cpu.to('cpu')
        self.test_logger.info(f"Tensor after .to('cpu') from '{self._default_non_cpu_device_type}': {x_cpu_from_non_cpu.device}")
        self.assertEqual(x_cpu_from_non_cpu.device.type, self._default_non_cpu_device_type,
                         f"Tensor should remain on '{self._default_non_cpu_device_type}' after .to('cpu') when CPU override is OFF.")

        if self._default_non_cpu_device_type == "mps":
            self.test_logger.info("Moving initial CPU tensor to 'cuda' (expecting MPS redirection)")
            x_cuda_redirect = x_cpu.to('cuda')
            self.test_logger.info(f"Tensor after .to('cuda'): {x_cuda_redirect.device}")
            self.assertEqual(x_cuda_redirect.device.type, "mps",
                             "Tensor .to('cuda') should be redirected to 'mps' when default is MPS and CPU override is OFF.")
        self.test_logger.info("Finished test_tensor_movement_cpu_override_off")

    def test_tensor_movement_cpu_override_on(self):
        """Test tensor movements when CPU override is ON."""
        self.test_logger.info("Starting test_tensor_movement_cpu_override_on")

        # Turn CPU override ON for this specific test
        if not torch.get_default_device().type == 'cpu':
            self.test_logger.info("CPU override is OFF. Toggling ON for this test.")
            torch.device("cpu:-1") 
        
        current_default_device = torch.get_default_device()
        self.assertEqual(current_default_device.type, "cpu",
                         f"CPU override should be ON. Expected default 'cpu', got '{current_default_device.type}'.")

        x_initial = torch.randn(2, 3) # Should default to CPU
        self.test_logger.info(f"Initial tensor (default device with override ON): {x_initial.device}")
        self.assertEqual(x_initial.device.type, "cpu", "Tensor should be on CPU by default when override is ON.")

        self.test_logger.info(f"Moving tensor to '{self._default_non_cpu_device_type}' with CPU override ON")
        x_to_non_cpu = x_initial.to(self._default_non_cpu_device_type)
        self.test_logger.info(f"Tensor after .to('{self._default_non_cpu_device_type}'): {x_to_non_cpu.device}")
        self.assertEqual(x_to_non_cpu.device.type, "cpu",
                         f"Tensor should remain on 'cpu' when moved to '{self._default_non_cpu_device_type}' with CPU override ON.")

        self.test_logger.info("Moving tensor to 'cuda' with CPU override ON")
        x_to_cuda = x_initial.to('cuda')
        self.test_logger.info(f"Tensor after .to('cuda'): {x_to_cuda.device}")
        self.assertEqual(x_to_cuda.device.type, "cpu",
                         "Tensor should remain on 'cpu' when moved to 'cuda' with CPU override ON.")
        
        self.test_logger.info("Finished test_tensor_movement_cpu_override_on")

if __name__ == '__main__':
    # This setup for sys.path might be needed if TorchDevice is not installed
    # and this script is run directly.
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.abspath(os.path.join(current_dir, '..')) # Assuming tests/ is one level down
    # if project_root not in sys.path:
    #     sys.path.insert(0, project_root)
    
    unittest.main(argv=sys.argv[:1], verbosity=2)