#!/usr/bin/env python
"""
Quick verification tests for TorchDevice patching and device logic.

This module verifies:
1. torch.device patching preserves type information for isinstance checks
2. Deterministic seed setting (if available) does not cause recursion errors
3. Tensor creation with various device specifications works as expected
"""
import unittest
import torch
import TorchDevice

try:
    from common.test_utils import PrefixedTestCase, set_deterministic_seed, devices_equivalent
    HAS_DETERMINISTIC = True
except ImportError:
    PrefixedTestCase = unittest.TestCase
    HAS_DETERMINISTIC = False

SEED = 42

class QuickVerificationTest(PrefixedTestCase):
    """Quick verification tests for TorchDevice patching and device logic."""

    def setUp(self):
        super().setUp()
        if HAS_DETERMINISTIC:
            set_deterministic_seed(SEED)
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.has_cuda = torch.cuda.is_available()
        if self.has_mps:
            self.expected_accel = 'mps'
        elif self.has_cuda:
            self.expected_accel = 'cuda'
        else:
            self.expected_accel = 'cpu'
        self.default_device = torch.get_default_device()

    def test_device_type_isinstance(self):
        """Test that torch.device patching preserves type information for isinstance checks."""
        original_device_type = torch.device('cpu').__class__
        cpu_device = torch.device('cpu')
        default_device = torch.device('cpu')
        self.assertIsInstance(cpu_device, original_device_type)
        self.assertIsInstance(default_device, original_device_type)
        # Simulate external code check
        def external_code_check(param):
            if isinstance(param, torch.device):
                return True
            return False
        self.assertTrue(external_code_check(default_device))
        self.info(f"isinstance(cpu_device, {original_device_type.__name__}) = {isinstance(cpu_device, original_device_type)}")

    @unittest.skipUnless(HAS_DETERMINISTIC, "set_deterministic_seed not available")
    def test_deterministic_seed(self):
        """Test that setting deterministic seed does not cause recursion error."""
        set_deterministic_seed(SEED)
        t1 = torch.randn(3, 3)
        set_deterministic_seed(SEED)
        t2 = torch.randn(3, 3)
        self.assertTrue(torch.all(torch.eq(t1, t2)))
        self.info("Seed consistency check: PASSED")

    def test_tensor_creation_devices(self):
        """Test tensor creation with various device specifications."""
        tensor_default = torch.randn(2, 2)
        tensor_cpu = torch.randn(2, 2, device='cpu')
        # Try to create a CUDA tensor if available, else MPS, else CPU
        if torch.cuda.is_available():
            tensor_cuda = torch.randn(2, 2, device='cuda')
            self.assertEqual(tensor_cuda.device.type, torch.get_default_device().type)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            tensor_mps = torch.randn(2, 2, device='mps')
            self.assertEqual(tensor_mps.device.type, 'mps')
        else:
            tensor_cpu2 = torch.randn(2, 2, device='cpu')
            self.assertEqual(tensor_cpu2.device.type, 'cpu')
        self.assertTrue(devices_equivalent(tensor_default.device, tensor_cpu.device))
        self.info(f"Default tensor device: {tensor_default.device}")
        self.info(f"CPU tensor device: {tensor_cpu.device}")

    def test_patch_and_redirection(self):
        t = torch.randn(1)
        self.assertTrue(devices_equivalent(t.device, self.default_device))

    def test_cpu_override(self):
        t_cpu = torch.randn(1, device='cpu:-1')
        self.assertEqual(t_cpu.device.type, 'cpu')
        t_cpu2 = torch.randn(1, device='cpu')
        self.assertEqual(t_cpu2.device.type, 'cpu')
        torch.device('cpu:-1')
        t_accel = torch.randn(1, device='cpu')
        self.assertTrue(devices_equivalent(t_accel.device, self.default_device))

    def test_explicit_accelerator(self):
        t = torch.randn(1, device='cuda')
        self.assertTrue(devices_equivalent(t.device, self.default_device))
        t2 = torch.randn(1, device='mps')
        self.assertTrue(devices_equivalent(t2.device, self.default_device))

    def test_to_and_cpu_methods(self):
        t = torch.randn(1)
        t_accel = t.to('cpu')
        self.info(f".to('cpu') result device: {t_accel.device}")
        self.assertTrue(devices_equivalent(t_accel.device, self.default_device))
        # .cpu() should redirect to accelerator
        t_accel2 = t.cpu()
        self.info(f".cpu() result device: {t_accel2.device}")
        self.assertTrue(devices_equivalent(t_accel2.device, self.expected_accel))
        # .to('cpu:-1') should always go to CPU
        t_cpu = t.to('cpu:-1')
        self.assertEqual(t_cpu.device.type, 'cpu')
        # torch.device('cpu:-1')
        t_cpu2 = t.to('cpu')
        self.assertEqual(t_cpu2.device.type, 'cpu')
        torch.device('cpu:-1')

class TorchDeviceBehaviorTest(PrefixedTestCase):
    """Tests for TorchDevice behavior as documented in TorchDevice_Behavior.md."""

    def setUp(self):
        super().setUp()
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.has_cuda = torch.cuda.is_available()
        if self.has_mps:
            self.expected_accel = 'mps'
        elif self.has_cuda:
            self.expected_accel = 'cuda'
        else:
            self.expected_accel = 'cpu'

    def test_implicit_tensor_creation(self):
        """Implicit tensor creation should go to accelerator if available, else CPU."""
        t = torch.randn(2, 2)
        self.info(f"Implicit tensor device: {t.device}")
        self.assertEqual(t.device.type, self.expected_accel)

    def test_explicit_cpu_redirect(self):
        """Explicit 'cpu' device should be redirected to accelerator unless override is active."""
        t = torch.randn(2, 2, device='cpu')
        self.info(f"Explicit 'cpu' tensor device: {t.device}")
        self.assertEqual(t.device.type, self.expected_accel)

    def test_explicit_cpu_override(self):
        """'cpu:-1' should create a CPU tensor, and override should persist until toggled off."""
        # Activate override
        t_cpu = torch.randn(2, 2, device='cpu:-1')
        self.info(f"Override ON: tensor device: {t_cpu.device}")
        self.assertEqual(t_cpu.device.type, 'cpu')
        # Now, explicit 'cpu' should also yield CPU
        t_cpu2 = torch.randn(2, 2, device='cpu')
        self.info(f"Override ON: explicit 'cpu' tensor device: {t_cpu2.device}")
        self.assertEqual(t_cpu2.device.type, 'cpu')
        # Toggle override OFF
        torch.device('cpu:-1')
        # Now, explicit 'cpu' should be redirected again
        t_accel = torch.randn(2, 2, device='cpu')
        self.info(f"Override OFF: explicit 'cpu' tensor device: {t_accel.device}")
        self.assertEqual(t_accel.device.type, self.expected_accel)

    def test_explicit_accelerator_requests(self):
        """Explicit accelerator requests should be honored if available, else fallback as per policy."""
        if self.has_cuda:
            t_cuda = torch.randn(2, 2, device='cuda')
            self.info(f"Explicit 'cuda' tensor device: {t_cuda.device}")
            self.assertEqual(t_cuda.device.type, torch.get_default_device().type)
        elif self.has_mps:
            t_mps = torch.randn(2, 2, device='mps')
            self.info(f"Explicit 'mps' tensor device: {t_mps.device}")
            self.assertEqual(t_mps.device.type, 'mps')
        else:
            t_cpu = torch.randn(2, 2, device='cpu')
            self.info(f"No accelerator: explicit 'cpu' tensor device: {t_cpu.device}")
            self.assertEqual(t_cpu.device.type, 'cpu')

    def test_to_and_cpu_methods(self):
        """.to('cpu') and .cpu() should redirect to accelerator unless override is active; .to('cpu:-1') should always go to CPU."""
        t = torch.randn(2, 2)
        # .to('cpu') should redirect to accelerator
        t_accel = t.to('cpu')
        self.info(f".to('cpu') result device: {t_accel.device}")
        self.assertTrue(devices_equivalent(t_accel.device, self.expected_accel))
        # .cpu() should redirect to accelerator
        t_accel2 = t.cpu()
        self.info(f".cpu() result device: {t_accel2.device}")
        self.assertTrue(devices_equivalent(t_accel2.device, self.expected_accel))
        # .to('cpu:-1') should always go to CPU
        t_cpu = t.to('cpu:-1')
        self.info(f".to('cpu') result device: {t_cpu.device}")
        self.assertEqual(t_cpu.device.type, 'cpu')
        # After override, .to('cpu') should yield CPU
        torch.device('cpu')
        t_cpu2 = t.to('cpu')
        self.info(f"Override ON: .to('cpu') result device: {t_cpu2.device}")
        self.assertEqual(t_cpu2.device.type, 'cpu')
        # Toggle override OFF
        torch.device('cpu:-1')

if __name__ == '__main__':
    unittest.main() 