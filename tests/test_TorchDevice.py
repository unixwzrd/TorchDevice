#!/usr/bin/env python
import unittest
import torch
import numpy as np
import torchdevice  # Ensure this module is imported to apply patches

class TestTorchDevice(unittest.TestCase):

    def setUp(self):
        # Determine the available hardware
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = torch.backends.mps.is_available()
        self.device = torch.device('cuda' if self.has_cuda else 'mps' if self.has_mps else 'cpu')

    def test_device_instantiation(self):
        # Test instantiation with 'cuda', 'mps', and 'cpu'
        device_cuda = torch.device('cuda')
        device_mps = torch.device('mps')
        device_cpu = torch.device('cpu')

        # Expected device type based on the hardware
        expected_device_type = self.device.type

        self.assertEqual(device_cuda.type, expected_device_type)
        self.assertEqual(device_mps.type, expected_device_type)
        self.assertEqual(device_cpu.type, 'cpu')

    def test_submodule_call(self):
        # Import the sub-module
        from test_submodule import ModelTrainer

        # Create an instance of ModelTrainer
        trainer = ModelTrainer()
        trainer.start_training()

        # Verify that the device is as expected
        expected_device_type = self.device.type
        device = torch.device('cuda')  # This will be redirected
        self.assertEqual(device.type, expected_device_type)

        # Since the device in start_training() should match the expected device
        # We can also capture the printed output if needed
            # But for this test, we are focusing on ensuring no exceptions occur

    def test_nested_function_call(self):
        from test_submodule import ModelTrainer

        trainer = ModelTrainer()
        trainer.call_nested_function()

        # Verify the device type
        expected_device_type = self.device.type
        device = torch.device('cuda')
        self.assertEqual(device.type, expected_device_type)

    def test_static_method_call(self):
        from test_submodule import ModelTrainer

        ModelTrainer.static_method()

        # Verify the device type
        expected_device_type = self.device.type
        device = torch.device('cuda')
        self.assertEqual(device.type, expected_device_type)

    def test_class_method_call(self):
        from test_submodule import ModelTrainer

        ModelTrainer.class_method()

        # Verify the device type
        expected_device_type = self.device.type
        device = torch.device('cuda')
        self.assertEqual(device.type, expected_device_type)


    def test_cuda_functions_on_mps(self):
        # Test CUDA functions on MPS hardware
        is_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()

        self.assertTrue(is_available)
        if self.has_cuda:
            self.assertGreaterEqual(device_count, 1)
            self.assertGreaterEqual(current_device, 0)
        elif self.has_mps:
            self.assertEqual(device_count, 1)
            self.assertEqual(current_device, 0)
        else:
            self.assertEqual(device_count, 0)
            self.assertEqual(current_device, -1)

    def test_tensor_operations(self):
        # Create a NumPy array and convert it to a PyTorch tensor with float32
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensor = torch.from_numpy(np_array).to(self.device)

        # Perform a simple operation
        result = tensor * 2

        # Verify the result
        expected = np_array * 2
        np.testing.assert_array_almost_equal(result.cpu().numpy(), expected)

    def test_tensor_device_movement(self):
        # Create a tensor on the CPU
        tensor_cpu = torch.tensor([1.0, 2.0, 3.0])

        # Move tensor to the appropriate device
        tensor_device = tensor_cpu.to(self.device)

        # Verify that the tensor is on the correct device
        if self.has_cuda or self.has_mps:
            self.assertNotEqual(tensor_device.device.type, 'cpu')
            self.assertEqual(tensor_device.device.type, self.device.type)
        else:
            self.assertEqual(tensor_device.device.type, 'cpu')

    def test_memory_functions(self):
        # Test memory-related CUDA functions
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        self.assertIsInstance(allocated, int)
        self.assertIsInstance(reserved, int)

    def test_unsupported_cuda_functions(self):
        # Test an unsupported CUDA function
        try:
            torch.cuda.ipc_collect()
            # If no exception, pass the test
            self.assertTrue(True)
        except Exception as e:
            # If an exception occurs, the test should fail
            self.fail(f"torch.cuda.ipc_collect() raised an exception: {e}")

    def test_get_device_properties(self):
        # Test getting device properties
        try:
            props = torch.cuda.get_device_properties(0)
            self.assertIsNotNone(props)
            self.assertTrue(hasattr(props, 'name'))
            self.assertTrue(hasattr(props, 'total_memory'))
        except RuntimeError as e:
            if not self.has_cuda and not self.has_mps:
                self.assertIn("No GPU device available", str(e))
            else:
                self.fail(f"Unexpected RuntimeError: {e}")

    def test_device_name_and_capability(self):
        # Test get_device_name and get_device_capability
        name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)

        if self.has_cuda:
            self.assertIsInstance(name, str)
            self.assertIsInstance(capability, tuple)
        elif self.has_mps:
            self.assertEqual(name, 'Apple MPS')
            self.assertEqual(capability, (0, 0))
        else:
            self.assertEqual(name, 'CPU')
            self.assertEqual(capability, (0, 0))

    def test_memory_summary(self):
        # Test memory summary function
        summary = torch.cuda.memory_summary()
        self.assertIsInstance(summary, str)
        self.assertIn("Memory Allocated", summary)
        self.assertIn("Memory Reserved", summary)

    def test_stream_functions(self):
        # Test unsupported stream functions
        try:
            torch.cuda.stream()
            torch.cuda.synchronize()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Stream functions raised an exception: {e}")

    def test_device_context_manager(self):
        # Test using torch.cuda.device as a context manager
        try:
            with torch.cuda.device(0):
                tensor = torch.tensor([1.0, 2.0]).to(self.device.type)
                self.assertEqual(tensor.device.type, self.device.type)
        except RuntimeError as e:
            self.fail(f"Unexpected exception: {e}")

    def test_is_initialized(self):
        # Test if CUDA is initialized
        initialized = torch.cuda.is_initialized()
        self.assertTrue(initialized)

    def test_is_built(self):
        # Test if CUDA backend is built
        is_built = torch.backends.cuda.is_built()
        self.assertTrue(is_built)

    def test_empty_cache(self):
        # Test empty_cache function
        try:
            torch.cuda.empty_cache()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"empty_cache raised an exception: {e}")

    def test_synchronize(self):
        # Test synchronize function
        try:
            torch.cuda.synchronize()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"synchronize raised an exception: {e}")

    def test_memory_stats(self):
        # Test memory_stats function
        stats = torch.cuda.memory_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn('active.all.current', stats)
        self.assertIn('reserved_bytes.all.current', stats)

    def test_memory_snapshot(self):
        # Test memory_snapshot function
        snapshot = torch.cuda.memory_snapshot()
        self.assertIsInstance(snapshot, list)

    def test_get_arch_list(self):
        # Test get_arch_list function
        arch_list = torch.cuda.get_arch_list()
        if self.has_cuda:
            self.assertIsInstance(arch_list, list)
            self.assertGreater(len(arch_list), 0)
        elif self.has_mps:
            self.assertEqual(arch_list, ['mps'])
        else:
            self.assertEqual(arch_list, [])

    def test_mock_function_stub(self):
        # Test calling a mocked function that is unsupported
        try:
            torch.cuda.reset_peak_memory_stats()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"reset_peak_memory_stats raised an exception: {e}")

    def test_device_type_inference(self):
        # Test device type inference without specifying device
        tensor = torch.tensor([1, 2, 3]).to(self.device)
        self.assertEqual(tensor.device.type, self.device.type)

    def test_tensor_creation_on_device(self):
        # Test tensor creation directly on the device
        tensor = torch.ones(5, device=self.device)
        self.assertEqual(tensor.device.type, self.device.type)
        self.assertTrue(torch.all(tensor == 1))

    def test_module_import_order(self):
        # Test that TorchDevice works regardless of import order
        import torchdevice
        import torch

        device = torch.device('cuda')
        self.assertEqual(device.type, self.device.type)

    def test_tensor_operations_between_devices(self):
        # Test operations between tensors on different devices
        tensor_cpu = torch.tensor([1.0, 2.0, 3.0])
        tensor_device = tensor_cpu.to(self.device)

        # Attempt to add tensors from different devices
        with self.assertRaises(RuntimeError) as context:
            result = tensor_cpu + tensor_device

        self.assertIn("Expected all tensors to be on the same device", str(context.exception))

    def test_multiple_device_indices(self):
        # Test setting device index
        if self.has_cuda and torch.cuda.device_count() > 1:
            torch.cuda.set_device(1)
            self.assertEqual(torch.cuda.current_device(), 1)
        elif self.has_mps:
            # MPS does not support multiple devices
            torch.cuda.set_device(0)
            self.assertEqual(torch.cuda.current_device(), 0)
        else:
            self.assertEqual(torch.cuda.current_device(), -1)

if __name__ == '__main__':
    unittest.main()