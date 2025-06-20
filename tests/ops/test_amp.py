"""
Unit tests for TorchDevice Automatic Mixed Precision (AMP) functionalities.
Ensures that torch.cuda.amp.autocast and torch.cuda.amp.GradScaler
behave correctly when patched by TorchDevice, across different effective devices.
"""

import unittest
import torch
import TorchDevice # This will trigger patches
import torch.backends.mps # For checking MPS availability
from tests.common.test_utils import PrefixedTestCase # set_deterministic_seed is called by PrefixedTestCase.setUp

# Helper model for testing AMP
class SimpleModel(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = torch.nn.Linear(input_features, output_features)

    def forward(self, x):
        return self.linear(x)

class TestAutomaticMixedPrecision(PrefixedTestCase):

    def setUp(self):
        super().setUp()
        # TorchDevice patches are applied by the module-level import.
        # PrefixedTestCase.setUp calls set_deterministic_seed()

    def tearDown(self):
        # No specific TorchDevice state to restore as we are not manipulating internals.
        super().tearDown()

    def _run_autocast_test(self, intended_pytorch_device_str):
        self.info(f"Running autocast test, intended PyTorch device: '{intended_pytorch_device_str}'")
        
        actual_torch_device = torch.device(intended_pytorch_device_str) # Let TorchDevice resolve it
        self.info(f"Actual resolved torch.device: {actual_torch_device} (type: {actual_torch_device.type})")

        # Determine expected behavior based on actual_torch_device.type
        expected_op_dtype_inside_autocast = None
        if actual_torch_device.type == 'cuda':
            expected_op_dtype_inside_autocast = torch.float16
        elif actual_torch_device.type == 'mps':
            expected_op_dtype_inside_autocast = torch.float16
        elif actual_torch_device.type == 'cpu':
            if hasattr(torch, 'bfloat16'):
                try:
                    # Check if bfloat16 is usable for autocast on CPU
                    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                        # Create a tensor on CPU to ensure context is right for this check
                        if SimpleModel(1,1).to('cpu')(torch.randn(1,1).to('cpu')).dtype == torch.bfloat16:
                            expected_op_dtype_inside_autocast = torch.bfloat16
                        else:
                            expected_op_dtype_inside_autocast = torch.float32
                except RuntimeError: # bfloat16 not supported or other issue
                    expected_op_dtype_inside_autocast = torch.float32
            else:
                expected_op_dtype_inside_autocast = torch.float32
        
        model = SimpleModel(10, 5).to(actual_torch_device)
        input_tensor = torch.randn(4, 10, device=actual_torch_device)
        
        self.info(f"Before autocast block: model.linear.weight.dtype: {model.linear.weight.dtype}, input_tensor.dtype: {input_tensor.dtype}")
        self.assertEqual(model.linear.weight.dtype, torch.float32)
        self.assertEqual(input_tensor.dtype, torch.float32)

        output_tensor_after_autocast_block = None # Define to ensure it's available for logging if exception occurs early

        try:
            # torch.cuda.amp.autocast() is patched by TorchDevice to call
            # torch.autocast(device_type=DeviceManager.get_default_device().type, ...)
            # The DeviceManager.get_default_device().type should reflect actual_torch_device.type
            with torch.cuda.amp.autocast(): 
                self.info(f"Inside autocast, BEFORE model call: model.linear.weight.dtype: {model.linear.weight.dtype}, input_tensor.dtype: {input_tensor.dtype}")
                output_tensor_inside_autocast = model(input_tensor)
                self.info(f"Inside autocast, AFTER model call: model.linear.weight.dtype: {model.linear.weight.dtype}")
                self.info(f"Inside autocast, output_tensor_inside_autocast.dtype: {output_tensor_inside_autocast.dtype}")
                if expected_op_dtype_inside_autocast:
                    self.assertEqual(output_tensor_inside_autocast.dtype, expected_op_dtype_inside_autocast)
                output_tensor_after_autocast_block = output_tensor_inside_autocast # Assign here for use outside

            self.info(f"After autocast block: model.linear.weight.dtype: {model.linear.weight.dtype}, input_tensor.dtype: {input_tensor.dtype}")
            if output_tensor_after_autocast_block is not None:
                self.info(f"After autocast block, output_tensor_after_autocast_block.dtype: {output_tensor_after_autocast_block.dtype}")
                # The output tensor will retain the dtype it had when created inside the autocast context.
                self.assertEqual(output_tensor_after_autocast_block.dtype, expected_op_dtype_inside_autocast, "Output tensor should retain its dtype from within the autocast context")
            else:
                self.fail("output_tensor_after_autocast_block was not assigned, error likely occurred inside autocast.")

            self.info(f"Autocast test successful for intended device '{intended_pytorch_device_str}' -> actual '{actual_torch_device.type}'")
        except Exception as e:
            self.fail(f"Autocast test for intended device '{intended_pytorch_device_str}' (actual '{actual_torch_device.type}') failed: {e}")

    def test_autocast_when_targeting_cuda(self):
        # This test will run on actual CUDA if available, or be redirected by TorchDevice to MPS/CPU.
        # The expected behavior inside _run_autocast_test will adapt.
        self._run_autocast_test('cuda')

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS backend not available")
    def test_autocast_when_targeting_mps(self):
        # This test specifically targets MPS. It should only run if MPS is truly available.
        self._run_autocast_test('mps')

    def test_autocast_when_targeting_cpu(self):
        self._run_autocast_test('cpu')

    # --- GradScaler Tests ---
    def _run_grad_scaler_test(self, intended_pytorch_device_str):
        self.info(f"Running GradScaler test, intended PyTorch device: '{intended_pytorch_device_str}'")
        actual_torch_device = torch.device(intended_pytorch_device_str) # Let TorchDevice resolve
        self.info(f"Actual resolved torch.device: {actual_torch_device} (type: {actual_torch_device.type})")

        # GradScalerReplacement in TorchDevice/ops/amp.py sets enabled=False if not CUDA,
        # and also sets device='cpu' for the GradScaler's state if not CUDA.
        scaler_should_be_enabled = (actual_torch_device.type == 'cuda')

        try:
            scaler = torch.cuda.amp.GradScaler() # Patched by TorchDevice
            self.info(f"GradScaler instantiated for actual device type {actual_torch_device.type}. Reported enabled: {scaler.is_enabled()}")
            self.assertEqual(scaler.is_enabled(), scaler_should_be_enabled)

            # The dummy tensor should be on the actual device type for the model
            dummy_tensor = torch.randn(2, 2, device=actual_torch_device)
            
            scaled_tensor = scaler.scale(dummy_tensor)
            if scaler_should_be_enabled:
                self.assertEqual(scaled_tensor.device.type, dummy_tensor.device.type)
                self.assertEqual(scaled_tensor.dtype, dummy_tensor.dtype)
                # If scaler is 1.0, output might be same object. If not, it's a new tensor.
                # Not asserting is(scaled_tensor, dummy_tensor) here.
            else:
                self.assertIs(scaled_tensor, dummy_tensor) # Disabled scaler.scale is a no-op

            model_for_optim = SimpleModel(2,2).to(actual_torch_device)
            optimizer = torch.optim.SGD(model_for_optim.parameters(), lr=0.01)
            
            scaler.step(optimizer)
            scaler.update()

            self.info(f"GradScaler test successful for intended device '{intended_pytorch_device_str}' -> actual '{actual_torch_device.type}'")
        except Exception as e:
            self.fail(f"GradScaler test for intended device '{intended_pytorch_device_str}' (actual '{actual_torch_device.type}') failed: {e}")

    def test_grad_scaler_when_targeting_cuda(self):
        self._run_grad_scaler_test('cuda')

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS backend not available")
    def test_grad_scaler_when_targeting_mps(self):
        self._run_grad_scaler_test('mps')

    def test_grad_scaler_when_targeting_cpu(self):
        self._run_grad_scaler_test('cpu')

if __name__ == '__main__':
    unittest.main()
