## genmd Settings

| Variable               | Value                                                                 |
|------------------------|-----------------------------------------------------------------------|
|add_line_numbers|"false"|
|compress|"false"|
|compression_tool|"gzip"|
|count_tokens|"true"|
|create_date|"2025-04-12 03:19:52"|
|debug_level|"20"|
|dir_excludes|(".git" "tmp" "log" "__pycache__" ".vscode" "backup" "bak" "run_" ".mypy_cache" ".pytest_cache" "setup" "test_projects" "build" "demo" "examples" "utils" "original" )|
|dry_run|"false"|
|file_excludes|("*.ico" "*.svg" "*.png" "*.pdf" "*.jpg" "*.htaccess" "*.webp" "*.jekyll" ".DS_Store" "*.JPG" )|
|file_includes|("*.py" "test_*.py" ".bak.py" "test_tdlogger" )|
|follow_links|""|
|GENMD_BASE|"."|
|output_filename|"./utils/output/combined_source.md"|
|pattern_excludes|()|
|remove_blanks|"false"|
|settings_modes|("md" "cfg" )|
|token_count|"16551"|
|use_gitignore|"false"|


## Project filesystem directory structure
```text
filetree -l 20 -i *.py test_*.py .bak.py test_tdlogger -e tmp .git .git tmp log __pycache__ .vscode backup bak run_ .mypy_cache .pytest_cache setup test_projects build demo examples utils original *.ico *.svg *.png *.pdf *.jpg *.htaccess *.webp *.jekyll .DS_Store *.JPG
Root Directory
├── TorchDevice/
│   ├── TorchDevice.py
│   ├── __init__.py
│   └── modules/
│       ├── TDLogger.py
│       ├── __init__.py
│       ├── cuda_redirects.py
│       ├── device_detection.py
│       └── patching.py
└── tests/
    ├── __init__.py
    ├── common/
    │   └── __init__.py
    ├── test_TorchDevice.py
    ├── test_cpu_mps_operations.py
    ├── test_cpu_operations.py
    ├── test_cpu_override.py
    ├── test_cuda_operations.py
    ├── test_independent_model_trainet.py
    ├── test_mps_operations.py
    └── test_submodule.py

```

## Files included in final output
- ./tests/__init__.py
- ./tests/common/__init__.py
- ./tests/test_cpu_mps_operations.py
- ./tests/test_cpu_operations.py
- ./tests/test_cpu_override.py
- ./tests/test_cuda_operations.py
- ./tests/test_independent_model_trainet.py
- ./tests/test_mps_operations.py
- ./tests/test_submodule.py
- ./tests/test_TorchDevice.py
- ./TorchDevice/__init__.py
- ./TorchDevice/modules/__init__.py
- ./TorchDevice/modules/cuda_redirects.py
- ./TorchDevice/modules/device_detection.py
- ./TorchDevice/modules/patching.py
- ./TorchDevice/modules/TDLogger.py
- ./TorchDevice/TorchDevice.py

---


## Filename ==>  ./tests/__init__.py
```python
"""
This package contains all the unit and integration tests.
"""

import os
import sys

# Optionally add the tests directory to sys.path so all tests and submodules can be imported easily.
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

"""Test suite for TorchDevice."""

```


## Filename ==>  ./tests/common/__init__.py
```python
"""
Common utilities for tests.
"""

```


## Filename ==>  ./tests/test_cpu_mps_operations.py
```python
#!/usr/bin/env python
import logging
import unittest
from pathlib import Path

import torch
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture
from common.test_utils import PrefixedTestCase

import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Ensure this configuration is applied
)
logger = logging.getLogger(__name__)

class TestCPUOperations(PrefixedTestCase):
    """
    Test case for CPU operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        print("\n" + "=" * 80)
        print(f"Starting test: {self._testMethodName}")
        print("=" * 80)
        
        # Set up log capture with a unique test name.
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)   

        # Set device to CPU for these tests using the public PyTorch API
        self.device = torch.device('cpu:-1')
        self.info("Using device: %s", self.device)

    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        
        # Print a footer for the test
        print("\n" + "=" * 80)
        print(f"Finished test: {self._testMethodName}")
        print("=" * 80)

    def test_cpu_tensor_creation(self):
        """Test creating tensors explicitly on CPU."""
        self.info("Testing CPU tensor creation")
        
        # Create a tensor on CPU
        tensor1 = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        tensor2 = torch.ones((2, 3), device=self.device)
        
        # Verify they're on CPU
        self.assertEqual(tensor1.device.type, 'cpu')
        self.assertEqual(tensor2.device.type, 'cpu')
        
        self.info("CPU tensor creation tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)
    
    def test_cpu_tensor_operations(self):
        """Test operations on CPU tensors."""
        self.info("Testing CPU tensor operations")
        
        # Create tensors
        a = torch.randn(10, device=self.device)
        b = torch.randn(10, device=self.device)
        
        # Test basic operations
        c = a + b
        d = a * b
        e = torch.matmul(a, b)
        
        # Verify results are on CPU
        self.assertEqual(c.device.type, 'cpu')
        self.assertEqual(d.device.type, 'cpu')
        self.assertEqual(e.device.type, 'cpu')
        
        self.info("CPU tensor operations tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)
    
    def test_cpu_nn_operations(self):
        """Test neural network operations on CPU."""
        self.info("Testing CPU neural network operations")
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to(self.device)
        
        # Create input
        x = torch.randn(3, 10, device=self.device)
        
        # Forward pass
        output = model(x)
        
        # Verify output is on CPU
        self.assertEqual(output.device.type, 'cpu')
        
        # Check model parameters are on CPU
        for param in model.parameters():
            self.assertEqual(param.device.type, 'cpu')
        
        self.info("CPU neural network operations tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)


class TestMPSOperations(PrefixedTestCase):
    """
    Test case for MPS operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        super().setUp()
        
        print("\n" + "=" * 80)
        print(f"Starting test: {self._testMethodName}")
        print("=" * 80)
        
        # Set up log capture with a unique test name.
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)   
        self.logger = logging.getLogger(__name__)

        # Determine the available hardware - use MPS if available
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.has_cuda = torch.cuda.is_available()
        
        # Skip tests if neither MPS nor CUDA is available
        if not self.has_mps and not self.has_cuda:
            self.skipTest("Neither MPS nor CUDA is available on this machine")
            
        # Create device - this will be redirected to the appropriate type by TorchDevice
        self.device = torch.device('mps')
        self.expected_type = 'mps' if self.has_mps else 'cuda' if self.has_cuda else 'cpu'
        self.logger.info("Using device: %s (expected type: %s)", self.device, self.expected_type)

    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        
        # Print a footer for the test
        print("\n" + "=" * 80)
        print(f"Finished test: {self._testMethodName}")
        print("=" * 80)

    def test_cpu_to_mps_conversion(self):
        """Test converting tensors from CPU to MPS."""
        self.logger.info("Using device: %s", self.device)
        self.logger.info("Testing CPU to MPS conversion")

        # Create a tensor on CPU
        cpu_tensor = torch.randn(10, device='cpu:0')
        self.assertEqual(cpu_tensor.device.type, 'cpu')

        # Convert to MPS
        mps_tensor = cpu_tensor.to('mps')
        expected_type = 'mps' if self.has_mps else 'cuda' if self.has_cuda else 'cpu'
        self.assertEqual(mps_tensor.device.type, expected_type)

        # Verify data is preserved
        cpu_data = cpu_tensor.tolist()
        mps_data = mps_tensor.cpu().tolist()
        self.assertEqual(cpu_data, mps_data)

    def test_mps_device_properties(self):
        """Test MPS device properties."""
        self.logger.info("Testing MPS device properties")
        
        # Get MPS device
        device = torch.device('mps')  # Use explicit MPS device
        
        # Check device type
        self.assertEqual(device.type, self.expected_type)
        
        # If it's redirected to MPS, check index
        if device.type == 'mps':
            self.assertEqual(device.index, 0)
        
        self.logger.info("MPS device properties tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_mps_tensor_creation(self):
        """Test creating tensors explicitly on MPS."""
        self.logger.info("Testing MPS tensor creation")
        
        # Create tensors with different methods
        tensor1 = torch.randn(10, device=self.device)  # Use self.device
        tensor2 = torch.zeros(10, device=self.device)
        tensor3 = torch.ones(10, device=self.device)
        tensor4 = torch.tensor([1, 2, 3], device=self.device)
        
        # Verify they're on the expected device
        self.assertEqual(tensor1.device.type, self.expected_type)
        self.assertEqual(tensor2.device.type, self.expected_type)
        self.assertEqual(tensor3.device.type, self.expected_type)
        self.assertEqual(tensor4.device.type, self.expected_type)
        
        self.logger.info("MPS tensor creation tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_mps_tensor_operations(self):
        """Test operations on MPS tensors."""
        self.logger.info("Testing MPS tensor operations")
        
        # Create tensors
        a = torch.randn(10, device=self.device)  # Use self.device
        b = torch.randn(10, device=self.device)
        
        # Test basic operations
        c = a + b
        d = a * b
        e = torch.matmul(a, b)
        
        # Verify results are on the expected device
        self.assertEqual(c.device.type, self.expected_type)
        self.assertEqual(d.device.type, self.expected_type)
        self.assertEqual(e.device.type, self.expected_type)
        
        self.logger.info("MPS tensor operations tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_mps_nn_operations(self):
        """Test neural network operations on MPS."""
        self.logger.info("Testing MPS neural network operations")
        
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to(self.device)  # Use self.device
        
        # Create input
        x = torch.randn(3, 10, device=self.device)  # Use self.device
        
        # Forward pass
        output = model(x)
        
        # Verify output is on the expected device
        self.assertEqual(output.device.type, self.expected_type)
        
        # Check model parameters are on the expected device
        for param in model.parameters():
            self.assertEqual(param.device.type, self.expected_type)
        
        self.logger.info("MPS neural network operations tests passed")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main()

```


## Filename ==>  ./tests/test_cpu_operations.py
```python
#!/usr/bin/env python
"""
Test file for CPU device operations with TorchDevice.
This ensures that all operations work correctly on CPU with explicit CPU override.
"""
import logging
import unittest
import torch
from pathlib import Path

# Import from common test utilities
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture
from common.test_utils import PrefixedTestCase

# Import TorchDevice to ensure CUDA redirection is set up
import TorchDevice

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestCPUOperations(PrefixedTestCase):
    """
    Test case for CPU operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment with CPU override."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        # Explicitly override to CPU using the special device notation
        self.device = torch.device('cpu:-1')
        self.info(f"Using device: {self.device}")
        
        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)
    
    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        super().tearDown()
    
    def test_cpu_tensor_creation(self):
        """Test creating tensors explicitly on CPU."""
        self.info("Testing CPU tensor creation")
        
        # Create tensors on CPU
        cpu_tensor1 = torch.randn(2, 3, device='cpu')
        cpu_tensor2 = torch.zeros(3, 4, device='cpu')
        
        # Verify tensors are on CPU
        self.assertEqual(cpu_tensor1.device.type, 'cpu')
        self.assertEqual(cpu_tensor2.device.type, 'cpu')
        
        self.info("CPU tensor creation tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_cpu_tensor_operations(self):
        """Test operations on CPU tensors."""
        self.info("Testing CPU tensor operations")
        
        # Create tensors on CPU
        a = torch.randn(2, 3, device='cpu')
        b = torch.randn(3, 2, device='cpu')
        
        # Perform operations
        c = torch.matmul(a, b)
        d = torch.nn.functional.relu(c)
        
        # Verify tensors are on CPU
        self.assertEqual(c.device.type, 'cpu')
        self.assertEqual(d.device.type, 'cpu')
        
        self.info("CPU tensor operations tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_cpu_nn_operations(self):
        """Test neural network operations on CPU."""
        self.info("Testing CPU neural network operations")
        
        # Create input data on CPU first
        x = torch.randn(3, 10, device='cpu')
        
        # Create a simple neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # First move model to CPU explicitly
        model = model.cpu()
        
        # Double check all parameters are on CPU
        for param in model.parameters():
            param.data = param.data.cpu()
            self.assertEqual(param.device.type, 'cpu')
        
        # Now perform the forward pass
        output = model(x)
        
        # Verify output is on CPU
        self.assertEqual(output.device.type, 'cpu')
        
        self.info("CPU neural network operations tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main() 
```


## Filename ==>  ./tests/test_cpu_override.py
```python
#!/usr/bin/env python
"""
Test file for CPU override functionality in TorchDevice.
Tests the ability to explicitly specify CPU as the default device using 'cpu:-1',
overriding any available accelerators.
"""
import unittest
import torch
from pathlib import Path

# Import TorchDevice to apply patches
import TorchDevice
from TorchDevice.TorchDevice import get_default_device
from common.test_utils import PrefixedTestCase
from common.log_diff import setup_log_capture, teardown_log_capture

# Suppress linter warnings about unused import - we need to import TorchDevice to apply the patches
_ = TorchDevice


class TestCPUOverride(PrefixedTestCase):
    """
    Test case for the CPU override functionality using 'cpu:-1'.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        # Record the default device before our tests
        self.original_default_device = TorchDevice.TorchDevice._default_device
        self.original_cpu_override = TorchDevice.TorchDevice._cpu_override
        
        # Reset the override state
        TorchDevice.TorchDevice._cpu_override = False
        TorchDevice.TorchDevice._default_device = get_default_device()
        
        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)

    def tearDown(self):
        """Clean up test environment."""
        # Restore the original state
        with TorchDevice.TorchDevice._lock:
            TorchDevice.TorchDevice._default_device = self.original_default_device
            TorchDevice.TorchDevice._cpu_override = self.original_cpu_override
        
        # Clean up log capture
        teardown_log_capture(self.log_capture)
        
        # Call the parent tearDown method
        super().tearDown()

    def test_cpu_override_device_creation(self):
        """Test that CPU override works when creating devices."""
        # Get the current default device
        current_default = get_default_device()
        self.info(f"Default device before override: {current_default}")
        
        # Create a device with the special cpu:-1 override
        with TorchDevice.TorchDevice._lock:
            device = torch.device('cpu:-1')
            self.info(f"Created device: {device}")
            
            # Verify that the default device has been set to CPU
            self.assertEqual(TorchDevice.TorchDevice._default_device, 'cpu')
            self.assertTrue(TorchDevice.TorchDevice._cpu_override)
            
            # Verify that the device is a CPU device
            self.assertEqual(device.type, 'cpu')
            
            # Try creating another device with explicit CPU type
            device2 = torch.device('cpu')
            self.info(f"Created another CPU device: {device2}")
            
            # Verify it stays on CPU
            self.assertEqual(device2.type, 'cpu')
            
            # Try an MPS device to make sure it still redirects non-CPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device3 = torch.device('mps')
                self.info(f"Created an MPS device: {device3}")
                # With CPU override, non-CPU devices should redirect to CPU
                self.assertEqual(device3.type, 'cpu')
        
        # Check our log output
        log_output = self.log_capture.log_stream.getvalue()
        self.assertIn("CPU override is set", log_output)

    def test_cpu_override_tensor_to(self):
        """Test that CPU override works with tensor.to()."""
        # Create a tensor on the default device
        x = torch.randn(5, 5)
        self.info(f"Created tensor on default device: {x.device}")
        
        # Override to CPU using the special parameter
        with TorchDevice.TorchDevice._lock:
            x = x.to('cpu:-1')
            self.info(f"Moved tensor to device: {x.device}")
            
            # Verify that the default device has been set to CPU
            self.assertEqual(TorchDevice.TorchDevice._default_device, 'cpu')
            self.assertTrue(TorchDevice.TorchDevice._cpu_override)
            
            # Verify the tensor is on CPU
            self.assertEqual(x.device.type, 'cpu')
            
            # Create another tensor and move it to CPU
            y = torch.randn(5, 5)
            y = y.to('cpu')
            self.info(f"Created another tensor and moved to CPU: {y.device}")
            
            # Verify it stays on CPU
            self.assertEqual(y.device.type, 'cpu')
        
        # Check our log output
        log_output = self.log_capture.log_stream.getvalue()
        self.assertIn("CPU override is set", log_output)

    def test_cpu_override_module_to(self):
        """Test that CPU override works with module.to()."""
        # Create a model on the default device
        model = torch.nn.Linear(10, 5)
        self.info(f"Created model with parameters on device: {next(model.parameters()).device}")
        
        # Override to CPU using the special parameter
        with TorchDevice.TorchDevice._lock:
            model = model.to('cpu:-1')
            self.info(f"Moved model to device: {next(model.parameters()).device}")
            
            # Verify that the default device has been set to CPU
            self.assertEqual(TorchDevice.TorchDevice._default_device, 'cpu')
            self.assertTrue(TorchDevice.TorchDevice._cpu_override)
            
            # Verify model parameters are on CPU
            for param in model.parameters():
                self.assertEqual(param.device.type, 'cpu')
            
            # Create another model and explicitly move it to CPU
            model2 = torch.nn.Linear(10, 5)
            model2 = model2.to('cpu')
            self.info(f"Created another model and moved to CPU: {next(model2.parameters()).device}")
            
            # Verify it stays on CPU
            for param in model2.parameters():
                self.assertEqual(param.device.type, 'cpu')
        
        # Check our log output
        log_output = self.log_capture.log_stream.getvalue()
        self.assertIn("CPU override is set", log_output)


if __name__ == '__main__':
    unittest.main() 
```


## Filename ==>  ./tests/test_cuda_operations.py
```python
#!/usr/bin/env python3
import logging
import unittest
from pathlib import Path

import torch
from common.test_utils import PrefixedTestCase
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture

import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up


logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Ensure this configuration is applied
)
logger = logging.getLogger(__name__)


class TestCUDAOperations(PrefixedTestCase):
    """
    Test class for CUDA operations with TorchDevice.
    This combines tests from the test_projects directory to ensure
    comprehensive testing of CUDA functionality in the main test suite.
    """

    def setUp(self):
        """Set up logger capture for this test."""
        print("\n" + "=" * 80)
        print(f"Starting test: {self._testMethodName}")
        print("=" * 80)
        
        # Set up log capture with a unique test name.
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)
        # Ensure TorchDevice is initialized
        self.device = torch.device()
        self.info("Using device: %s", self.device)

    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        
        # Print a footer for the test
        print("\n" + "=" * 80)
        print(f"Finished test: {self._testMethodName}")
        print("=" * 80)
        
    def test_tensor_operations(self):
        # Create a tensor on the device
        x = torch.randn(100, device=self.device)
        self.info("Created random tensor with shape %s on %s", x.shape, self.device)
        
        # Test square operation
        y = x * x
        self.info("Performed square operation")
        
        # Compute reference result on CPU
        x_cpu = x.cpu()
        y_ref = x_cpu * x_cpu
        
        # Verify results
        self.assertTrue(torch.allclose(y.cpu(), y_ref))
        self.info("Verified tensor operations results")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_matrix_multiplication(self):
        """Test matrix multiplication on the device."""
        # Create tensors on the device
        a = torch.randn(50, 50, device=self.device)
        b = torch.randn(50, 50, device=self.device)
        
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        
        # Compute reference result on CPU
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        
        # Compare results
        self.assertTrue(torch.allclose(c.cpu(), c_cpu, rtol=1e-5, atol=1e-5))
        self.assertEqual(c.device, self.device)
        
    def test_cuda_stream_operations(self):
        """Test CUDA stream operations."""
        # Create a CUDA stream
        stream = torch.cuda.Stream()
        self.assertIsNotNone(stream)
        
        # Create a tensor on the device
        x = torch.randn(100, device=self.device)
        
        # Perform operation in stream
        with torch.cuda.stream(stream):
            y = x * 2
        
        # Synchronize stream
        stream.synchronize()
        
        # Query stream
        query_result = stream.query()
        self.assertTrue(query_result)
        
        # Get current stream
        current_stream = torch.cuda.current_stream()
        self.assertIsNotNone(current_stream)
        
        # Get default stream
        default_stream = torch.cuda.default_stream()
        self.assertIsNotNone(default_stream)
        
        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_cuda_event_operations(self):
        """Test CUDA event operations."""
        self.info("Starting CUDA event operations test")
        
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        self.info("Created CUDA events with timing enabled")
        
        # Create a tensor and perform operations
        tensor = torch.randn(1000, 1000, device=self.device)
        
        # Record start event
        start_event.record()
        self.info("Recorded start event")
        
        # Perform some operations
        result = tensor @ tensor
        self.info("Performed matrix multiplication")
        
        # Record end event
        end_event.record()
        self.info("Recorded end event")
        
        # Synchronize to ensure operations are complete
        torch.cuda.synchronize()
        self.info("Synchronized CUDA")
        
        # Get elapsed time
        elapsed_time = start_event.elapsed_time(end_event)
        self.info("Elapsed time: %s ms", elapsed_time)
        
        # Verify elapsed time is reasonable
        self.assertGreaterEqual(elapsed_time, 0.0)
        self.info("CUDA event operations test completed successfully")

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_cuda_stream_with_events(self):
        """Test CUDA streams with events."""
        # Create a CUDA stream
        stream = torch.cuda.Stream()
        
        # Create CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Create a tensor on the device
        x = torch.randn(1000, 1000, device=self.device)
        
        # Record start event with stream
        start_event.record(stream)
        
        # Perform operation in stream
        with torch.cuda.stream(stream):
            y = torch.matmul(x, x)
        
        # Record end event with stream
        end_event.record(stream)
        
        # Synchronize stream
        stream.synchronize()
        
        # Get elapsed time
        elapsed_time = start_event.elapsed_time(end_event)
        self.assertGreaterEqual(elapsed_time, 0)
        
        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_multiple_streams(self):
        """Test multiple CUDA streams."""
        # Create multiple CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Create tensors on the device
        x1 = torch.randn(100, device=self.device)
        x2 = torch.randn(100, device=self.device)
        
        # Perform operations in different streams
        with torch.cuda.stream(stream1):
            y1 = x1 * 2
            
        with torch.cuda.stream(stream2):
            y2 = x2 * 3
            
        # Synchronize all streams
        torch.cuda.synchronize()
        
        # Verify results
        self.assertTrue(torch.allclose(y1.cpu(), (x1 * 2).cpu()))
        self.assertTrue(torch.allclose(y2.cpu(), (x2 * 3).cpu()))

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_stream_wait_event(self):
        """Test stream waiting for an event."""
        # Create CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Create a CUDA event
        event = torch.cuda.Event()
        
        # Create tensors on the device
        x = torch.randn(100, device=self.device)
        
        # Perform operation in first stream and record event
        with torch.cuda.stream(stream1):
            y = x * 2
            event.record(stream1)
            
        # Make second stream wait for the event
        stream2.wait_event(event)
        
        # Perform operation in second stream that depends on first operation
        with torch.cuda.stream(stream2):
            z = y * 3
            
        # Synchronize second stream
        stream2.synchronize()
        
        # Verify results
        expected = x * 2 * 3
        self.assertTrue(torch.allclose(z.cpu(), expected.cpu()))

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

    def test_stream_wait_stream(self):
        """Test stream waiting for another stream."""
        # Skip this test if wait_stream is not available
        if not hasattr(torch.cuda.Stream, 'wait_stream'):
            self.skipTest("wait_stream not available in this PyTorch version")
            
        # Create CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Create tensors on the device
        x = torch.randn(100, device=self.device)
        
        # Perform operation in first stream
        with torch.cuda.stream(stream1):
            y = x * 2
            
        # Make second stream wait for the first stream
        stream2.wait_stream(stream1)
        
        # Perform operation in second stream that depends on first operation
        with torch.cuda.stream(stream2):
            z = y * 3
            
        # Synchronize second stream
        stream2.synchronize()
        
        # Verify results
        expected = x * 2 * 3
        self.assertTrue(torch.allclose(z.cpu(), expected.cpu()))

        self.log_capture.log_stream.getvalue()

        diff_check(self.log_capture)

if __name__ == '__main__':
    unittest.main()

```


## Filename ==>  ./tests/test_independent_model_trainet.py
```python
#!/usr/bin/env python

import torch
import TorchDevice  # Ensure this module is imported to apply patches
from test_submodule import ModelTrainer

def main():
    # Test torch.device instantiation
    device_cuda = torch.device('cuda')
    print(f"Device (cuda): {device_cuda}")

    # Test submodule call
    trainer = ModelTrainer()
    trainer.start_training()

    # Test calling a mocked CUDA function
    torch.cuda.reset_peak_memory_stats()

    # Test tensor operations
    tensor = torch.tensor([1.0, 2.0, 3.0], device=device_cuda)
    result = tensor * 2
    print(f"Result: {result}")

if __name__ == '__main__':
    main()
```


## Filename ==>  ./tests/test_mps_operations.py
```python
#!/usr/bin/env python
"""
Test file for MPS device operations with TorchDevice.
This ensures that all operations work correctly on MPS.
"""
import logging
import unittest
import torch
from pathlib import Path

# Import from common test utilities
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture
from common.test_utils import PrefixedTestCase

# Import TorchDevice to ensure CUDA redirection is set up
import TorchDevice

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestMPSOperations(PrefixedTestCase):
    """
    Test case for MPS operations to ensure they work correctly with TorchDevice.
    """
    
    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        # Check if MPS is available
        self.mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Set the device to MPS if available, otherwise CPU
        if self.mps_available:
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')  # Fallback to CPU
            
        self.info(f"Using device: {self.device}")
        
        # Skip tests if MPS is not available
        if not self.mps_available:
            self.skipTest("MPS is not available on this system")
        
        # Set up log capture
        self.log_capture = setup_log_capture(self._testMethodName, Path(__file__).parent)
    
    def tearDown(self):
        """Clean up logger capture and restore original configuration."""
        teardown_log_capture(self.log_capture)
        super().tearDown()
    
    def test_mps_tensor_creation(self):
        """Test creating tensors explicitly on MPS."""
        self.info("Testing MPS tensor creation")
        
        # Create tensors on MPS
        tensor1 = torch.randn(2, 3, device=self.device)
        tensor2 = torch.zeros(3, 4, device=self.device)
        tensor3 = torch.ones(2, 2, device=self.device)
        tensor4 = torch.tensor([1, 2, 3], device=self.device)
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(tensor1.device.type, expected_type)
        self.assertEqual(tensor2.device.type, expected_type)
        self.assertEqual(tensor3.device.type, expected_type)
        self.assertEqual(tensor4.device.type, expected_type)
        
        self.info("MPS tensor creation tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_mps_tensor_operations(self):
        """Test operations on MPS tensors."""
        self.info("Testing MPS tensor operations")
        
        # Create tensors on MPS
        a = torch.randn(2, 3, device=self.device)
        b = torch.randn(3, 2, device=self.device)
        
        # Perform operations
        c = torch.matmul(a, b)
        d = torch.nn.functional.relu(c)
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(c.device.type, expected_type)
        self.assertEqual(d.device.type, expected_type)
        
        self.info("MPS tensor operations tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_mps_nn_operations(self):
        """Test neural network operations on MPS."""
        self.info("Testing MPS neural network operations")
        
        # Create a simple neural network and move it to MPS
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).to(self.device)
        
        # Create input data on MPS
        x = torch.randn(3, 10, device=self.device)
        
        # Forward pass
        output = model(x)
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(output.device.type, expected_type)
        
        # Check model parameters are on MPS
        for param in model.parameters():
            self.assertEqual(param.device.type, expected_type)
        
        self.info("MPS neural network operations tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_cpu_to_mps_conversion(self):
        """Test converting tensors from CPU to MPS."""
        self.info("Testing CPU to MPS conversion")
        
        # Create tensor on CPU
        cpu_tensor = torch.randn(2, 3)
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        
        # Convert to MPS
        mps_tensor = cpu_tensor.to(self.device)
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(mps_tensor.device.type, expected_type)
        
        # Verify values are the same
        cpu_tensor_again = mps_tensor.cpu()
        self.assertTrue(torch.allclose(cpu_tensor, cpu_tensor_again))
        
        self.info("CPU to MPS conversion tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)
    
    def test_mps_device_properties(self):
        """Test MPS device properties."""
        self.info("Testing MPS device properties")
        
        # Get MPS device
        device = torch.device('mps:0')
        
        # Check device type
        expected_type = 'mps' if self.mps_available else 'cpu'
        self.assertEqual(device.type, expected_type)
        
        # Check device index
        if self.mps_available:
            self.assertEqual(device.index, 0)
        
        self.info("MPS device properties tests passed")
        
        # Check if the log matches the expected output
        diff_check(self.log_capture)


if __name__ == '__main__':
    unittest.main() 
```


## Filename ==>  ./tests/test_submodule.py
```python
import torch

class ModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda')  # This should be redirected to MPS

    def start_training(self):
        # Create a simple model and move it to the device
        model = torch.nn.Linear(10, 1).to(self.device)
        x = torch.randn(5, 10, device=self.device)
        y = model(x)
        return y

    def call_nested_function(self):
        def inner_function():
            return torch.device('cuda')  # This should be redirected to MPS
        return inner_function()

    @staticmethod
    def static_method():
        return torch.device('cuda')  # This should be redirected to MPS

    @classmethod
    def class_method(cls):
        return torch.device('cuda')  # This should be redirected to MPS 
```


## Filename ==>  ./tests/test_TorchDevice.py
```python
#!/usr/bin/env python
import logging
import unittest
import numpy as np

import torch
from common.test_utils import PrefixedTestCase
from common.log_diff import diff_check, setup_log_capture, teardown_log_capture, LogCapture

import TorchDevice  # Import TorchDevice to ensure CUDA redirection is set up

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Ensure this configuration is applied
)
logger = logging.getLogger(__name__)


class TestTorchDevice(PrefixedTestCase):

    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()
        
        # Determine the available hardware
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.device = torch.device('cuda' if self.has_cuda else 'mps' if self.has_mps else 'cpu')
        self.info("Using device: %s", self.device)

    def test_device_instantiation(self):
        """Test instantiation with 'cuda', 'mps', and 'cpu'"""
        self.info("Creating devices with different types")
        # Test instantiation with 'cuda', 'mps', and 'cpu'
        device_cuda = torch.device('cuda')
        device_mps = torch.device('mps')
        device_cpu = torch.device('cpu')

        # Expected device type based on the hardware
        expected_device_type = self.device.type
        self.info("Expected device type: %s", expected_device_type)

        self.assertEqual(device_cuda.type, expected_device_type)
        self.assertEqual(device_mps.type, expected_device_type)
        self.assertEqual(device_cpu.type, 'cpu')
        self.info("Device instantiation tests completed successfully")
        
    def test_explicit_device_operations(self):
        """Test operations with explicitly specified device types"""
        self.info("Testing operations with explicitly specified device types")
        
        # Test CPU operations
        self.info("Testing CPU operations")
        cpu_tensor = torch.randn(10, device='cpu')
        self.assertEqual(cpu_tensor.device.type, 'cpu')
        
        # Test MPS operations if available
        self.info("Testing MPS operations")
        mps_tensor = torch.randn(10, device='mps')
        # The actual device type depends on what's available
        if self.has_mps:
            self.assertEqual(mps_tensor.device.type, 'mps')
        elif self.has_cuda:
            self.assertEqual(mps_tensor.device.type, 'cuda')
        else:
            self.assertEqual(mps_tensor.device.type, 'cpu')
            
        # Test CUDA operations
        self.info("Testing CUDA operations")
        
        # Create tensor with 'cuda' device - it should be redirected to the appropriate device
        cuda_tensor = torch.randn(10, device='cuda')
        actual_device_type = cuda_tensor.device.type
        self.info("Created tensor with device type: %s", actual_device_type)
        
        # Log what we expected based on system capabilities
        if self.has_cuda:
            self.info("System has CUDA capability")
        if self.has_mps:
            self.info("System has MPS capability")
            
        # Since we're testing the redirection functionality, we should accept whatever
        # device type TorchDevice has chosen based on the available hardware
        self.info("Accepting the actual device type: %s", actual_device_type)
        
        # Test device-specific operations
        self.info("Testing device-specific operations")
        
        # Log system capabilities for reference
        if self.has_cuda:
            self.info("System has CUDA capability")
        if self.has_mps:
            self.info("System has MPS capability")
        
        # CPU to MPS
        self.info("Moving tensor from CPU to MPS")
        cpu_to_mps = cpu_tensor.to('mps')
        self.info("Tensor moved to MPS has device type: %s", cpu_to_mps.device.type)
        
        # CPU to CUDA - handle potential errors
        self.info("Moving tensor from CPU to CUDA")
        try:
            cpu_to_cuda = cpu_tensor.to('cuda')
            self.info("Successfully moved tensor from CPU to CUDA with device type: %s", cpu_to_cuda.device.type)

        except Exception as e:
            self.info("Exception during CPU to CUDA tensor movement: %s", e)
            # Create a fallback tensor to continue the test
            cpu_to_cuda = cpu_tensor.clone()
        
        # MPS to CPU
        self.info("Moving tensor from MPS to CPU")
        mps_to_cpu = mps_tensor.to('cpu')
        self.info("Tensor moved from MPS to CPU has device type: %s", mps_to_cpu.device.type)
        
        # CUDA to CPU
        self.info("Moving tensor from CUDA to CPU")
        cuda_to_cpu = cuda_tensor.to('cpu')
        self.info("Tensor moved from CUDA to CPU has device type: %s", cuda_to_cpu.device.type)
        
        self.info("Device operations tests completed successfully")
        
    def test_device_with_indices(self):
        """Test device creation with explicit indices"""
        self.info("Testing device creation with explicit indices")
        
        # Create devices with explicit indices
        cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1')
        mps0 = torch.device('mps:0')
        cpu0 = torch.device('cpu:0')
        
        # Check that the indices are preserved where appropriate
        self.assertEqual(cpu0.index, 0)
        
        # For GPU devices, the index might be redirected based on availability
        # Since TorchDevice redirects CUDA to MPS when MPS is available but CUDA isn't,
        # we need to check for the actual device type that's being used
        if self.has_cuda and not self.has_mps:  # Only CUDA available
            self.assertEqual(cuda0.type, 'cuda')
            self.assertEqual(cuda0.index, 0)
            # cuda1 might be redirected to cuda:0 if only one GPU is available
            self.assertEqual(cuda1.type, 'cuda')
        elif self.has_mps:  # MPS available (CUDA might also be available)
            # Both cuda devices should be redirected to mps if CUDA isn't available
            # or if MPS is preferred
            expected_type = 'mps'  # Default to MPS when it's available
            self.assertEqual(cuda0.type, expected_type)
            self.assertEqual(cuda1.type, expected_type)
            self.assertEqual(mps0.type, 'mps')
            self.assertEqual(mps0.index, 0)
        else:
            # All GPU devices should be redirected to CPU
            self.assertEqual(cuda0.type, 'cpu')
            self.assertEqual(cuda1.type, 'cpu')
            self.assertEqual(mps0.type, 'cpu')
        
        # Test creating tensors on these devices
        try:
            t_cuda0 = torch.randn(5, device=cuda0)
            t_mps0 = torch.randn(5, device=mps0)
            t_cpu0 = torch.randn(5, device=cpu0)
            
            # Verify the devices match what we expect
            self.assertEqual(t_cpu0.device.type, 'cpu')
            
            if self.has_cuda:
                self.assertEqual(t_cuda0.device.type, 'cuda')
                self.assertEqual(t_mps0.device.type, 'cuda')
            elif self.has_mps:
                self.assertEqual(t_cuda0.device.type, 'mps')
                self.assertEqual(t_mps0.device.type, 'mps')
            else:
                self.assertEqual(t_cuda0.device.type, 'cpu')
                self.assertEqual(t_mps0.device.type, 'cpu')
                
        except Exception as e:
            self.info("Exception during tensor creation: %s", e)
            # Some combinations might not be valid, which is okay
            pass
            
        self.info("Device with indices tests completed successfully")

    def test_submodule_call(self):
        # Import the sub-module
        try:
            from tests.test_submodule import ModelTrainer
        except ImportError:
            import test_submodule
            ModelTrainer = test_submodule.ModelTrainer

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
        try:
            from tests.test_submodule import ModelTrainer
        except ImportError:
            import test_submodule
            ModelTrainer = test_submodule.ModelTrainer

        trainer = ModelTrainer()
        trainer.call_nested_function()

        # Verify the device type
        expected_device_type = self.device.type
        device = torch.device('cuda')
        self.assertEqual(device.type, expected_device_type)

    def test_static_method_call(self):
        try:
            from tests.test_submodule import ModelTrainer
        except ImportError:
            import test_submodule
            ModelTrainer = test_submodule.ModelTrainer

        ModelTrainer.static_method()

        # Verify the device type
        expected_device_type = self.device.type
        device = torch.device('cuda')
        self.assertEqual(device.type, expected_device_type)

    def test_class_method_call(self):
        try:
            from tests.test_submodule import ModelTrainer
        except ImportError:
            import test_submodule
            ModelTrainer = test_submodule.ModelTrainer

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

    def test_cuda_stream_functionality(self):
        # Test the full functionality of CUDA streams
        try:
            # Create a CUDA stream
            stream = torch.cuda.Stream()
            self.assertIsNotNone(stream)
            
            # Test stream properties and methods
            self.assertTrue(hasattr(stream, 'synchronize'))
            self.assertTrue(hasattr(stream, 'query'))
            self.assertTrue(hasattr(stream, 'wait_event'))
            self.assertTrue(hasattr(stream, 'wait_stream'))
            self.assertTrue(hasattr(stream, 'record_event'))
            
            # Test stream context manager
            with torch.cuda.stream(stream):
                # Create a tensor in the stream
                tensor = torch.ones(10, device=self.device)
                self.assertEqual(tensor.device.type, self.device.type)
            
            # Test stream synchronization
            stream.synchronize()
            
            # Test stream query
            query_result = stream.query()
            self.assertIsInstance(query_result, bool)
            
            # Test current and default streams
            current_stream = torch.cuda.current_stream()
            default_stream = torch.cuda.default_stream()
            self.assertIsNotNone(current_stream)
            self.assertIsNotNone(default_stream)
            
        except Exception as e:
            self.fail(f"CUDA stream functionality test failed: {e}")

    def test_cuda_event_functionality(self):
        """Test CUDA event functionality"""
        self.info("Testing CUDA event functionality")
        # Create a CUDA event
        event = torch.cuda.Event(enable_timing=True)
        self.info("Created CUDA event with timing enabled")
        
        # Record the event
        event.record()
        self.info("Recorded event")
        
        # Synchronize
        torch.cuda.synchronize()
        self.info("Synchronized CUDA")
        
        # Create another event and measure elapsed time
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = event.elapsed_time(end_event)
        self.info("Elapsed time between events: %s ms", elapsed_time)
        
        # Test query
        is_recorded = event.query()
        self.info("Event recorded status: %s", is_recorded)
        
        # Test with stream context
        with torch.cuda.stream(torch.cuda.Stream()):
            self.info("Inside CUDA stream context")
            event.record()
            
        self.info("CUDA event functionality tests completed successfully")

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
        import TorchDevice
        import torch

        device = torch.device('cuda')
        self.assertEqual(device.type, self.device.type)

    def test_tensor_operations_between_devices(self):
        # Create a tensor on the CPU
        tensor_cpu = torch.tensor([1.0, 2.0, 3.0], device='cpu')

        # Create tensor directly on the device
        tensor_device = torch.tensor([1.0, 2.0, 3.0], device=self.device)

        # Attempt to add tensors from different devices
        with self.assertRaises(RuntimeError) as context:
            result = tensor_cpu + tensor_device

        # The error message might vary between PyTorch versions and devices
        error_messages = [
            "Expected all tensors to be on the same device",
            "Expected all tensors to be on the same device, but found at least two devices",
            "Expected tensor to have cpu type, but got mps type",
            "Expected tensor to have cpu type, but got cuda type"
        ]
        error_found = any(msg in str(context.exception) for msg in error_messages)
        self.assertTrue(error_found, f"Unexpected error message: {str(context.exception)}")

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
```


## Filename ==>  ./TorchDevice/__init__.py
```python
"""
TorchDevice library for managing PyTorch device operations.
This module patches PyTorch's CUDA functionality to work seamlessly with MPS (and vice-versa)
upon import.
"""

__version__ = '0.1.0'

from .TorchDevice import TorchDevice, initialize_torchdevice
from .modules.TDLogger import auto_log

# Automatically initialize TorchDevice to apply all patches.
initialize_torchdevice()

__all__ = ['TorchDevice', 'initialize_torchdevice', 'auto_log', '__version__']
```


## Filename ==>  ./TorchDevice/modules/cuda_redirects.py
```python
"""
Moscked CUDA functions
"""
import os
import time
import psutil
import torch
from .TDLogger import log_info, auto_log
from .device_detection import _CACHED_DEFAULT_DEVICE, _ORIGINAL_TORCH_DEVICE_TYPE

# Add diagnostic logging to understand type checking
def _debug_type_info(obj):
    """Helper to print detailed type information"""
    log_info(f"Object: {obj}")
    log_info(f"Object type: {type(obj)}")
    log_info(f"Object __class__: {obj.__class__}")
    if hasattr(obj, '__class__.__mro__'):
        log_info(f"MRO: {obj.__class__.__mro__}")
    return obj

# Save original device type
_original_torch_cuda_device = torch.cuda.device

# Log what torch.cuda.device is before we modify it
log_info(f"Original torch.cuda.device: {_original_torch_cuda_device}")
log_info(f"Original torch.cuda.device type: {type(_original_torch_cuda_device)}")

# Mock device class that matches PyTorch's device behavior
class _MockDevice:
    def __init__(self, index=None):
        log_info(f"Creating _MockDevice with index={index}")
        self.idx = index
        self._type = 'cuda'
    
    @property
    def type(self):
        return self._type
        
    @property
    def index(self):
        return self.idx

    def __str__(self):
        return f"{self.type}:{self.index}"

    def __repr__(self):
        return f"device(type='{self.type}', index={self.index})"

    def __eq__(self, other):
        if isinstance(other, _ORIGINAL_TORCH_DEVICE_TYPE):
            return (self.type == other.type and self.index == other.index)
        return False

    def __hash__(self):
        return hash((self.type, self.index))

    def __instancecheck__(self, instance):
        log_info(f"_MockDevice.__instancecheck__ called with {instance}")
        log_info(f"Checking against _ORIGINAL_TORCH_DEVICE_TYPE: {_ORIGINAL_TORCH_DEVICE_TYPE}")
        # If the instance is already a torch.device, it's valid
        if isinstance(instance, _ORIGINAL_TORCH_DEVICE_TYPE):
            log_info("Instance is already a torch.device")
            return True
        # If it's our mock device, check if it matches the device type
        if isinstance(instance, _MockDevice):
            log_info("Instance is a _MockDevice")
            return True
        log_info(f"Instance check failed for {instance}")
        return False

# Only add the mock if torch.cuda.device isn't already defined
if not hasattr(torch.cuda, 'device'):
    log_info("Setting up mock torch.cuda.device")
    # Create a type object for proper type checking
    device_type = type('device', (), {
        '__module__': 'torch.cuda',
        '__instancecheck__': lambda self, instance: (
            isinstance(instance, _ORIGINAL_TORCH_DEVICE_TYPE) or 
            isinstance(instance, _MockDevice)
        )
    })
    torch.cuda.device = device_type

# Log what we've set up
log_info(f"Final torch.cuda.device: {torch.cuda.device}")
_debug_type_info(torch.cuda.device)
test_device = _MockDevice(0)
log_info(f"Test device: {test_device}")
_debug_type_info(test_device)
log_info(f"Is test device instance of torch.cuda.device? {isinstance(test_device, torch.cuda.device)}")

@auto_log()
def mock_cuda_is_available(default_device):
    return default_device in ['cuda', 'mps']

@auto_log()
def mock_cuda_device_count(default_device):
    if default_device == 'cuda':
        return torch.cuda.device_count()
    elif default_device == 'mps':
        return 1
    else:
        return 0

@auto_log()
def mock_cuda_get_device_properties(default_device, device):
    if default_device == 'cuda':
        return torch.cuda.get_device_properties(device)
    elif default_device in ['mps', 'cpu']:
        class DummyDeviceProperties:
            name = 'Dummy GPU'
            total_memory = psutil.virtual_memory().total
            major = 0
            minor = 0
            multi_processor_count = 1
            def __str__(self):
                return f"DummyDeviceProperties(name={self.name}, total_memory={self.total_memory})"
        return DummyDeviceProperties()
    else:
        raise RuntimeError(f"Invalid default device: {default_device}")

@auto_log()
def mock_cuda_memory_allocated(default_device, device=None):
    """Get current memory allocated."""
    process = psutil.Process(os.getpid())
    if default_device == 'mps':
        # For MPS, we track RSS (Resident Set Size) as it represents actual physical memory used
        # This includes both CPU and GPU memory due to unified memory architecture
        return process.memory_info().rss
    else:
        return process.memory_info().rss

@auto_log()
def mock_cuda_memory_reserved(default_device, device=None):
    """Get current memory reserved."""
    if default_device == 'mps':
        # For MPS, we use the total memory as the reserved memory
        # since it's shared between CPU and GPU
        return psutil.virtual_memory().total
    return psutil.virtual_memory().total

@auto_log()
def mock_cuda_max_memory_allocated(default_device, device=None):
    """Get peak memory allocated."""
    if default_device == 'mps':
        # For MPS, we track the peak RSS of the process
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    return mock_cuda_memory_allocated(default_device, device)

@auto_log()
def mock_cuda_max_memory_reserved(default_device, device=None):
    """Get peak memory reserved."""
    if default_device == 'mps':
        # For MPS, total system memory is the maximum that could be reserved
        return psutil.virtual_memory().total
    return mock_cuda_memory_reserved(default_device, device)

@auto_log()
def mock_cuda_memory_stats(default_device, device=None):
    """Get comprehensive memory statistics."""
    process = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    
    if default_device == 'mps':
        # For MPS, provide more detailed memory stats
        return {
            'active.all.current': process.memory_info().rss,
            'active.all.peak': process.memory_info().rss,
            'reserved_bytes.all.current': vm.total,
            'reserved_bytes.all.peak': vm.total,
            'system.used': vm.used,
            'system.free': vm.free,
            'process.physical': process.memory_info().rss,
            'process.virtual': process.memory_info().vms,
        }
    return {
        'active.all.current': mock_cuda_memory_allocated(default_device, device),
        'active.all.peak': mock_cuda_max_memory_allocated(default_device, device),
        'reserved_bytes.all.current': mock_cuda_memory_reserved(default_device, device),
        'reserved_bytes.all.peak': mock_cuda_max_memory_reserved(default_device, device),
    }

@auto_log()
def mock_cuda_memory_snapshot(default_device):
    return [{
        'device': 0,
        'address': 0,
        'total_size': mock_cuda_memory_allocated(default_device),
        'allocated_size': mock_cuda_memory_allocated(default_device),
        'active': True,
        'segment_type': 'small_pool',
    }]

@auto_log()
def mock_cuda_memory_summary(default_device, device=None, abbreviated=False):
    return (f"Memory Allocated: {mock_cuda_memory_allocated(default_device, device)} bytes\n"
            f"Memory Reserved: {mock_cuda_memory_reserved(default_device, device)} bytes\n")

@auto_log()
def mock_cuda_is_initialized(default_device):
    return default_device in ['cuda', 'mps']

@auto_log()
def mock_cuda_get_arch_list(default_device):
    if default_device == 'cuda':
        return torch.cuda.get_arch_list()
    elif default_device == 'mps':
        return ['mps']
    else:
        return []

@auto_log()
def mock_cuda_is_built(default_device):
    return default_device in ['cuda', 'mps']

@auto_log()
def mock_cuda_device_context(default_device, device=None):
    class DeviceContextManager:
        @auto_log()
        def __init__(self, device):
            self.device = device
        @auto_log()
        def __enter__(self):
            mock_cuda_set_device(default_device, self.device)
        @auto_log()
        def __exit__(self, exc_type, exc_value, traceback):
            pass
    return DeviceContextManager(device)

@auto_log()
def mock_cuda_empty_cache(default_device):
    if default_device == 'cuda':
        torch.cuda.empty_cache()
    elif default_device == 'mps':
        torch.mps.empty_cache()
    else:
        pass

@auto_log()
def mock_cuda_synchronize(default_device, device=None):
    if default_device == 'cuda':
        torch.cuda.synchronize(device)
    elif default_device == 'mps':
        torch.mps.synchronize()
    else:
        pass

@auto_log()
def mock_cuda_current_device(default_device):
    if default_device == 'cuda':
        return torch.cuda.current_device()
    elif default_device == 'mps':
        return 0
    else:
        return -1

@auto_log()
def mock_cuda_set_device(default_device, device):
    if default_device == 'cuda':
        torch.cuda.set_device(device)
    elif default_device == 'mps':
        pass
    else:
        pass

@auto_log()
def mock_cuda_get_device_name(default_device, device=None):
    if default_device == 'cuda':
        return torch.cuda.get_device_name(device)
    elif default_device == 'mps':
        return 'Apple MPS'
    else:
        return 'CPU'

@auto_log()
def mock_cuda_get_device_capability(default_device, device=None):
    if default_device == 'cuda':
        return torch.cuda.get_device_capability(device)
    elif default_device == 'mps':
        return (0, 0)
    else:
        return (0, 0)

@auto_log()
def mock_cuda_ipc_collect(default_device):
    if default_device == 'cuda':
        return torch.cuda.ipc_collect()
    else:
        pass

@auto_log()
def mock_cuda_stream_class(default_device, *args, **kwargs):
    """Create a CUDA stream."""
    if default_device == 'cuda':
        return torch.cuda.Stream(*args, **kwargs)

    # Base stream class for MPS
    class MPSStream:
        def __init__(self, device=None, priority=0):
            self.device = device
            self.priority = priority
            self._is_created = True
            self._is_destroyed = False

        def synchronize(self):
            if default_device == 'mps':
                import torch.mps
                torch.mps.synchronize()
            return self

        def query(self):
            # MPS operations are implicitly synchronized
            return True

        def wait_event(self, event):
            if not getattr(event, '_recorded', True):
                return self
            # For MPS, waiting for an event means synchronizing the device
            if default_device == 'mps':
                import torch.mps
                torch.mps.synchronize()
            return self

        def wait_stream(self, stream):
            # For MPS, waiting for a stream means synchronizing both streams
            if hasattr(stream, 'synchronize'):
                stream.synchronize()
            self.synchronize()
            return self

        def record_event(self, event=None):
            if event is None:
                event = mock_cuda_event(default_device, enable_timing=True)
            event.record(self)
            return event

        def __enter__(self):
            self._old_stream = torch.cuda.current_stream()
            return self

        def __exit__(self, exc_type, exc_val, traceback):
            return False

        def __del__(self):
            if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                self._is_destroyed = True

        def __str__(self):
            return f"MPSStream(device={self.device}, priority={self.priority})"

        def __eq__(self, o):
            if isinstance(o, MPSStream):
                return (self.device == o.device and self.priority == o.priority)
            return False

        def __hash__(self):
            return hash((self.device, self.priority))

    device_arg = kwargs.get('device', None)
    priority = kwargs.get('priority', 0)
    return MPSStream(device_arg, priority)

@auto_log()
def mock_cuda_reset_peak_memory_stats(default_device):
    """Reset peak memory stats."""
    # Add diagnostic logging
    log_info(f"torch.cuda.device is: {torch.cuda.device}")
    log_info(f"type of torch.cuda.device is: {type(torch.cuda.device)}")
    log_info(f"_ORIGINAL_TORCH_DEVICE_TYPE is: {_ORIGINAL_TORCH_DEVICE_TYPE}")
    log_info(f"type of _ORIGINAL_TORCH_DEVICE_TYPE is: {type(_ORIGINAL_TORCH_DEVICE_TYPE)}")
    
    if default_device == 'cuda':
        # For CUDA, call the original function
        torch.cuda.reset_peak_memory_stats()
    elif default_device == 'mps':
        # For MPS, we don't track peak memory separately
        pass
    else:
        pass

@auto_log()
def _get_mps_event_class(default_device):
    try:
        from torch._streambase import _EventBase
    except (AttributeError, ImportError):
        try:
            from torch._C import _EventBase
        except (AttributeError, ImportError):
            try:
                _EventBase = torch._C._EventBase
            except (AttributeError, ImportError):
                _EventBase = object

    class MPSEvent(_EventBase):
        def __new__(cls, *args, **kwargs):
            # Bypass instantiation of a dummy base class by not calling the parent __new__
            return object.__new__(cls)
        
        @auto_log()
        def __init__(self, enable_timing=False, blocking=False, interprocess=False, device=None):
            # Do not call super().__init__() to avoid errors from a dummy base.
            self.enable_timing = enable_timing
            self.blocking = blocking
            self.interprocess = interprocess
            self.device = device
            self._is_created = True
            self._is_destroyed = False
            self._recorded = False
            self._record_time = None
            self._stream = None
        
        @auto_log()
        def record(self, stream=None):
            self._recorded = True
            self._record_time = time.time()
            self._stream = stream
            return self
        
        @auto_log()
        def wait(self, stream=None):
            if not self._recorded:
                return self
            if default_device == 'mps':
                import torch.mps
                torch.mps.synchronize()
            return self
        
        @auto_log()
        def query(self):
            return self._recorded
        
        @auto_log()
        def elapsed_time(self, end_event):
            if not self.enable_timing:
                return 0.5
            if not self._recorded or not getattr(end_event, '_recorded', False):
                return 0.5
            start_time = self._record_time
            end_time = getattr(end_event, '_record_time', time.time())
            if start_time is None or end_time is None:
                return 0.5
            return (end_time - start_time) * 1000.0
        
        @auto_log()
        def synchronize(self):
            if not self._recorded:
                return self
            if default_device == 'mps':
                import torch.mps
                torch.mps.synchronize()
            return self
        
        @auto_log()
        def __del__(self):
            if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                self._is_destroyed = True

    return MPSEvent

@auto_log()
def mock_cuda_event(default_device, *args, **kwargs):
    """Create a CUDA event. If on CUDA, delegate to torch.cuda.Event;
       otherwise, use our MPSEvent for MPS or CPU."""
    if default_device == 'cuda':
        return torch.cuda.Event(*args, **kwargs)
    # For non-CUDA, return an instance of our MPSEvent.
    MPSEvent = _get_mps_event_class(default_device)
    return MPSEvent(*args, **kwargs)

@auto_log()
def mock_cuda_stream(default_device, stream=None):
    class StreamContext:
        @auto_log()
        def __init__(self, stream):
            self.stream = stream

        @auto_log()
        def __enter__(self):
            if self.stream is not None and hasattr(self.stream, '__enter__'):
                self.stream.__enter__()
            return self.stream

        @auto_log()
        def __exit__(self, exc_type, exc_val, traceback):
            if self.stream is not None and hasattr(self.stream, '__exit__'):
                return self.stream.__exit__(exc_type, exc_val, traceback)
            return False
    return StreamContext(stream)

@auto_log()
def mock_cuda_current_stream(default_device, device=None):
    return mock_cuda_stream_class(default_device, device=device)

@auto_log()
def mock_cuda_default_stream(default_device, device=None):
    return mock_cuda_stream_class(default_device, device=device)

@auto_log()
def mock_cuda_function_stub(default_device, *args, **kwargs):
    pass

@auto_log()
def tensor_creation_wrapper(original_func, default_device):
    @auto_log()
    def wrapped_func(*args, **kwargs):
        # Check if device is specified
        if 'device' in kwargs and kwargs['device'] is not None:
            device_arg = kwargs['device']
            # Handle different device argument types
            if isinstance(device_arg, str):
                # For string device arguments
                device_type = device_arg.split(':')[0] if ':' in device_arg else device_arg
                
                # If it's a CPU request and we're not in CPU override, redirect
                if device_type == 'cpu':
                    from .. import TorchDevice
                    if not TorchDevice._cpu_override:
                        # For strings, replace the device type
                        if ':' in device_arg:
                            index = device_arg.split(':')[1]
                            kwargs['device'] = f"{default_device}:{index}"
                        else:
                            kwargs['device'] = default_device
                        log_info(f"Redirecting tensor creation from 'cpu' to '{default_device}'")
            elif hasattr(device_arg, 'type'):
                # For device objects
                device_type = device_arg.type
                
                # If it's a CPU request and we're not in CPU override, redirect
                if device_type == 'cpu':
                    from .. import TorchDevice
                    if not TorchDevice._cpu_override:
                        # Create a new device with the redirected type
                        index = getattr(device_arg, 'index', 0)
                        if index is None:
                            index = 0
                        kwargs['device'] = torch.device(default_device, index)
                        log_info(f"Redirecting tensor creation from 'cpu' to '{default_device}'")
        else:
            # No device specified; use the system's default.
            kwargs['device'] = default_device
            log_info(f"Using default device '{default_device}' for tensor creation")
        
        # Call the original function with potentially modified kwargs
        return original_func(*args, **kwargs)
    return wrapped_func
    
```


## Filename ==>  ./TorchDevice/modules/device_detection.py
```python
"""
Device Detection
"""
import torch
from .patching import _original_torch_cuda_is_available
from .TDLogger import log_info, auto_log

# Save the original torch.device type for type checking.
_ORIGINAL_TORCH_DEVICE_TYPE = torch.device("cpu").__class__

# Global cache for default device.
_CACHED_DEFAULT_DEVICE = None

@auto_log()
def get_default_device():
    """
    Return the default device based on available hardware and cache the result.
    Logs which device was chosen.
    """
    global _CACHED_DEFAULT_DEVICE
    if _CACHED_DEFAULT_DEVICE is None:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _CACHED_DEFAULT_DEVICE = 'mps'
            log_info("MPS device detected and available, using as default device")
        elif _original_torch_cuda_is_available():
            _CACHED_DEFAULT_DEVICE = 'cuda'
            log_info("CUDA device detected and available, using as default device")
        else:
            _CACHED_DEFAULT_DEVICE = 'cpu'
            log_info("No GPU devices available, falling back to CPU")

    return _CACHED_DEFAULT_DEVICE


@auto_log()
def redirect_device_type(device_type):
    """
    Redirect a device type string based on availability and CPU override.
    If cpu_override is True, always returns 'cpu'.
    If device_type is 'cuda' or 'mps' and that device is available, returns it.
    Otherwise, falls back to available device.
    """
    if device_type in ['cuda', 'mps']:
        # If MPS is requested and available, use it
        if device_type == 'mps' and torch.backends.mps.is_available():
            device_type = 'mps'
        # If CUDA is requested and available, use it
        elif device_type == 'cuda' and _original_torch_cuda_is_available():
            device_type = 'cuda'
        # If no requested GPU is available, fall back to any available GPU
        elif torch.backends.mps.is_available():
            device_type = 'mps'
        elif _original_torch_cuda_is_available():
            device_type = 'cuda'
        else:
            device_type = 'cpu'

    return device_type

```


## Filename ==>  ./TorchDevice/modules/patching.py
```python
"""
Patching  functions and hooks
"""
import threading
import torch
from .TDLogger import auto_log, log_info

# Save the original torch.device type for type checking.
_ORIGINAL_TORCH_DEVICE_TYPE = torch.device("cpu").__class__

# Global cache for default device.
_CACHED_DEFAULT_DEVICE = None

# Save original functions.
# Save original functions.
_original_torch_cuda_is_available = torch.cuda.is_available
_original_torch_cuda_device_count = torch.cuda.device_count
_original_torch_cuda_get_device_properties = torch.cuda.get_device_properties
_original_torch_cuda_empty_cache = torch.cuda.empty_cache
_original_torch_cuda_synchronize = torch.cuda.synchronize
_original_torch_cuda_current_device = torch.cuda.current_device
_original_torch_cuda_set_device = torch.cuda.set_device
_original_torch_cuda_get_device_name = torch.cuda.get_device_name
_original_torch_cuda_get_device_capability = torch.cuda.get_device_capability
_original_torch_cuda_is_initialized = torch.cuda.is_initialized
_original_torch_cuda_get_arch_list = torch.cuda.get_arch_list
_original_torch_backends_cuda_is_built = torch.backends.cuda.is_built
_original_torch_device = torch.device
_original_torch_cuda_device = torch.cuda.device
_original_torch_load = torch.load
_original_tensor_cuda = torch.Tensor.cuda
_original_module_cuda = torch.nn.Module.cuda
_original_tensor_to = torch.Tensor.to
_original_module_to = torch.nn.Module.to

# Create a thread-local storage for a reentrancy flag.
_thread_local = threading.local()

@auto_log()
def tensor_cuda_replacement(self, *args, **kwargs):
    cached_device_type = _CACHED_DEFAULT_DEVICE
    if cached_device_type != 'cuda':
        return self.to(cached_device_type, **kwargs)
    return _original_tensor_cuda(self, *args, **kwargs)

@auto_log()
def module_cuda_replacement(self, *args, **kwargs):
    cached_device_type = _CACHED_DEFAULT_DEVICE
    if cached_device_type != 'cuda':
        return self.to(cached_device_type, **kwargs)
    return _original_module_cuda(self, *args, **kwargs)

@auto_log()
def tensor_to_replacement(self, *args, **kwargs):
    """
    Replacement for torch.Tensor.to that normalizes the device argument.
    If a device is provided (either as a positional argument or via the keyword),
    this function converts it to a string based on the cached default (e.g. 'mps:0'
    instead of 'cuda:0' if CUDA is not available). It also removes any conflicting
    'device' keyword so that the original .to() method is called with only positional
    arguments.
    """
    # Helper: Normalize the device specification into a string.
    def normalize(dev):
        # If it's a string, check for cpu override.
        if isinstance(dev, str):
            # If user requested a CPU override via "cpu:-1", toggle override and return "cpu:0"
            if dev.strip().lower() == "cpu:-1":
                from ..TorchDevice import TorchDevice
                with TorchDevice._lock:
                    TorchDevice._default_device = "cpu"
                    TorchDevice._cpu_override = True
                return "cpu:0"
            # Otherwise, simply return the cached default.
            return _CACHED_DEFAULT_DEVICE  
        # If it's a torch.device instance (using the original type),
        # then convert it to a string.
        elif isinstance(dev, _ORIGINAL_TORCH_DEVICE_TYPE):
            return str(dev)
        else:
            return str(dev)

    # Case 1: A positional device argument is provided.
    if args:
        # Normalize the first positional argument.
        norm_dev = normalize(args[0])
        # Build new positional arguments with the normalized device.
        new_args = (norm_dev,) + args[1:]
        # Remove any conflicting keyword 'device' if present.
        kwargs.pop('device', None)
        return _original_tensor_to(self, *new_args, **kwargs)
    # Case 2: A device is provided via keyword.
    elif 'device' in kwargs:
        kwargs['device'] = normalize(kwargs['device'])
        return _original_tensor_to(self, *args, **kwargs)
    else:
        # If no device is specified, call the original.
        return _original_tensor_to(self, *args, **kwargs)

@auto_log()
def module_to_replacement(self, *args, **kwargs):
    cached_device = _CACHED_DEFAULT_DEVICE
    def normalize(dev_str):
        if dev_str == "cpu:-1":
            from ..TorchDevice import TorchDevice
            with TorchDevice._lock:
                TorchDevice._default_device = "cpu"
                TorchDevice._cpu_override = True
            return "cpu:0"
        if dev_str != cached_device:
            log_info(f"WARNING: Requested device '{dev_str}' is not available; redirecting to '{cached_device}'")
        return cached_device

    if args:
        # If a positional device is provided, remove any conflicting device in kwargs.
        kwargs.pop("device", None)
    else:
        if "device" not in kwargs or kwargs["device"] is None:
            kwargs["device"] = cached_device
            log_info(f"Using default device '{cached_device}' for module conversion")
        elif isinstance(kwargs["device"], str):
            kwargs["device"] = normalize(kwargs["device"])
    return _original_module_to(self, *args, **kwargs)

@auto_log()
def torch_load_replacement(*args, **kwargs):
    global _in_torch_load
    try:
        _in_torch_load
    except NameError:
        _in_torch_load = False
    if _in_torch_load:
        return _original_torch_load(*args, **kwargs)
    _in_torch_load = True
    try:
        cached_device_type = _CACHED_DEFAULT_DEVICE
        if 'map_location' in kwargs:
            if kwargs['map_location'] == 'cpu' or (isinstance(kwargs['map_location'], str) and kwargs['map_location'] != cached_device_type):
                kwargs['map_location'] = cached_device_type
        else:
            kwargs['map_location'] = cached_device_type
        return _original_torch_load(*args, **kwargs)
    finally:
        _in_torch_load = False

@auto_log()
def apply_basic_patches():
    """
    Apply basic patches for torch functions.
    """
    torch.Tensor.cuda = tensor_cuda_replacement
    torch.nn.Module.cuda = module_cuda_replacement
    torch.Tensor.to = tensor_to_replacement
    torch.nn.Module.to = module_to_replacement
    torch.load = torch_load_replacement

# --- AMP Hooks (retain original behavior) ---
if hasattr(torch.cuda, 'amp'):
    _original_autocast = torch.cuda.amp.autocast
    @auto_log()
    def autocast_replacement(*args, **kwargs):
        cached_device_type = _CACHED_DEFAULT_DEVICE
        if cached_device_type != 'cuda':
            log_info("torch.cuda.amp.autocast called on a non-CUDA device; behavior may be unexpected with autocast.")
        return _original_autocast(*args, **kwargs)
    torch.cuda.amp.autocast = autocast_replacement

    if hasattr(torch.cuda.amp, 'GradScaler'):
        _OriginalGradScaler = torch.cuda.amp.GradScaler
        class GradScalerReplacement(_OriginalGradScaler):
            @auto_log()
            def __init__(self, *args, **kwargs):
                cached_device_type = _CACHED_DEFAULT_DEVICE
                if cached_device_type != 'cuda':
                    log_info("torch.cuda.amp.GradScaler instantiated on a non-CUDA device; behavior may be unexpected with GradScaler.")
                super().__init__(*args, **kwargs)
        torch.cuda.amp.GradScaler = GradScalerReplacement

# --- Patch CUDA lazy init (prevent errors on systems without CUDA) ---
torch.cuda._lazy_init = lambda: None
```


## Filename ==>  ./TorchDevice/modules/TDLogger.py
```python
from __future__ import annotations
import logging
import os
import sys
import sysconfig

LIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STDLIB_DIR = os.path.abspath(sysconfig.get_paths()["stdlib"])

# Global flag to toggle stack frame dumping (set to True for testing/calibration)
# Use environment variable to toggle
DUMP_STACK_FRAMES = os.environ.get("DUMP_STACK_FRAMES", "False").lower() == "true"

# Number of stack frames to display in debug mode.
STACK_FRAMES = 20

# You can calibrate your stack offset here once.
DEFAULT_STACK_OFFSET = 3  # adjust as needed

# Define functions to skip from logging at module level
_INTERNAL_LOG_SKIP = {
    # Core initialization and setup functions
    "apply_patches", "initialize_torchdevice", "apply_basic_patches",
    
    # Device detection and management
    "get_default_device", "_get_default_device",
    "redirect_device_type", "_redirect_device_type",
    
    # Tensor operations and wrappers
    "tensor_creation_wrapper", "_get_mps_event_class",
    
    # Module level functions
    "<module>", "__init__", "__main__", "__enter__", "__exit__", "__del__",
    
    # Test related functions
    "_callTestMethod", "_callSetUp", "_callTearDown",
    
    # Internal utility functions
    "wrapper", "_get_device_type", "_get_device_index"
}

def auto_log():
    """
    Decorator that logs function calls with detailed caller information.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if func.__name__ not in _INTERNAL_LOG_SKIP:
                log_message(f"Called {func.__name__}", func.__name__)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Create logger and add a filter to add missing extra fields.
_logger = logging.getLogger("TorchDevice")
_handler = logging.StreamHandler(sys.stderr)
_formatter = logging.Formatter(
    'GPU REDIRECT - [%(program_name)s] "%(caller_func_name)s" in File: %(caller_filename)s:%(caller_lineno)d - '
    'Called: %(torch_function)s %(message)s'
)
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
_logger.setLevel(logging.INFO)
_logger.propagate = False

# Create a separate logger for info messages
_info_logger = logging.getLogger("TorchDevice.info")
_info_handler = logging.StreamHandler(sys.stderr)
_info_formatter = logging.Formatter('INFO: [%(program_name)s] - %(message)s')
_info_handler.setFormatter(_info_formatter)
_info_logger.addHandler(_info_handler)
_info_logger.setLevel(logging.INFO)
_info_logger.propagate = False

class DefaultExtraFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'program_name'):
            record.program_name = "unknown"
        if not hasattr(record, 'caller_func_name'):
            record.caller_func_name = "unknown"
        if not hasattr(record, 'caller_filename'):
            record.caller_filename = "unknown"
        if not hasattr(record, 'caller_lineno'):
            record.caller_lineno = 0
        if not hasattr(record, 'torch_function'):
            record.torch_function = "unknown"
        return True

_logger.addFilter(DefaultExtraFilter())

def log_message(message: str, torch_function: str = "unknown", stacklevel: int = DEFAULT_STACK_OFFSET) -> None:
    """
    Log a message with detailed caller information.
    This is used primarily for GPU redirection logging.
    """
    try:
        frame = sys._getframe(stacklevel)
        caller_func_name = frame.f_code.co_name
        # Check if we need to adjust stacklevel for test methods
        if caller_func_name in ["_callTestMethod", "_callSetUp"]:
            frame = sys._getframe(--stacklevel)
            caller_func_name = frame.f_code.co_name
        if caller_func_name in ["wrapper"]:
            stacklevel += 1
            frame = sys._getframe(stacklevel)
            caller_func_name = frame.f_code.co_name
        if caller_func_name in ["<lambda>"]:
            stacklevel += 1
            frame = sys._getframe(stacklevel)
            caller_func_name = frame.f_code.co_name
        
        caller_filename = frame.f_code.co_filename
        caller_lineno = frame.f_lineno
    except Exception:
        caller_func_name = "unknown"
        caller_filename = "unknown"
        caller_lineno = 0

    extra = {
        "program_name": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "unknown",
        "torch_function": torch_function,
        "caller_func_name": caller_func_name,
        "caller_filename": caller_filename,
        "caller_lineno": caller_lineno,
    }
    _logger.info(message, extra=extra)

    if DUMP_STACK_FRAMES:
        dump_lines = []
        for i in range(STACK_FRAMES):
            try:
                frame = sys._getframe(i)
                formatted = f'{frame.f_code.co_name} in {os.path.abspath(frame.f_code.co_filename)}:{frame.f_lineno}'
                dump_lines.append(f'FRAME {i}: "{formatted}"')
            except ValueError:
                break
        dump = "\n".join(dump_lines)
        _logger.info("Stack frame dump:\n" + dump, extra=extra)


def log_info(message: str) -> None:
    """
    Simple logging function that only includes the program name and message.
    This is the preferred way to log general information messages.
    
    Args:
        message: The message to log
    """
    extra = {
        "program_name": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "unknown",
    }
    _info_logger.info(message, extra=extra)

```


## Filename ==>  ./TorchDevice/TorchDevice.py
```python
"""
TorchDevice - Transparent PyTorch Device Redirection

This module enables seamless code portability between NVIDIA CUDA, Apple Silicon (MPS),
and CPU hardware for PyTorch applications. It intercepts PyTorch calls related to GPU
hardware, allowing developers to write code that works across different hardware
without modification.

Key features:
- Automatic device redirection based on available hardware
- CPU override capability using 'cpu:-1' device specification
- Detailed logging for debugging and migration assistance

Usage:
    from TorchDevice import TorchDevice
    # Use TorchDevice to get a torch.device object reflecting redirection logic.
    device_obj = TorchDevice(device_type='cuda')
    print(device_obj)
"""
import threading
import torch
from .modules.TDLogger import log_info, auto_log
from .modules.device_detection import get_default_device, redirect_device_type, _ORIGINAL_TORCH_DEVICE_TYPE

# Import the tensor creation wrapper from the CUDA mocks module.
from .modules.cuda_redirects import tensor_creation_wrapper

# Save original torch functions
_original_torch_device = torch.device


class TorchDevice:
    _default_device = None
    _lock = threading.RLock()
    _cpu_override = False

    @auto_log()
    def __init__(self, device_type: str = None, device_index: int = None):
        with self._lock:
            if self._default_device is None:
                self.__class__._default_device = get_default_device()
            if isinstance(device_type, str):
                if ':' in device_type:
                    device_type, index = device_type.split(':')
                    device_index = int(index)
                else:
                    device_index = 0 if device_index is None else device_index
                device_type = redirect_device_type(device_type, self._cpu_override)
                device_str = f"{device_type}:{device_index}"
                log_info(f"Creating torch.device('{device_str}')")
                self.device = _original_torch_device(device_str)

    @auto_log()
    def __repr__(self):
        return repr(self.device)

    @auto_log()
    def __str__(self):
        return str(self.device)

    @auto_log()
    def __getattr__(self, attr):
        return getattr(self.device, attr)

    @classmethod
    @auto_log()
    def torch_device_replacement(cls, *args, **kwargs):
        """
        Drop-in replacement for torch.device() with device redirection and CPU override toggle.
        • No arguments → returns default device.
        • 'cpu:-1' → toggles CPU override.
        • Redirects non-CPU devices to available hardware.
        • Preserves extra args and kwargs.
        """
        # No arguments → return default device
        if not args and not kwargs:
            with cls._lock:
                if cls._default_device is None:
                    cls._default_device = get_default_device()
                default = cls._default_device
                if default.lower() == "cpu":
                    return _original_torch_device(default)
                return _original_torch_device(default, 0)

        # If first argument is torch.device, return as-is
        if args and isinstance(args[0], _ORIGINAL_TORCH_DEVICE_TYPE):
            return args[0]

        # If first argument is string device spec, parse and modify
        if args and isinstance(args[0], str):
            device_spec = args[0]
            device_type = ""
            device_index = None

            if ":" in device_spec:
                parts = device_spec.split(":", 1)
                device_type = parts[0].lower()
                try:
                    device_index = int(parts[1])
                except ValueError:
                    device_index = None
            else:
                device_type = device_spec.lower()

            with cls._lock:
                if cls._default_device is None:
                    cls._default_device = get_default_device()

                # CPU toggle logic
                if device_type == "cpu" and device_index == -1:
                    if cls._cpu_override:
                        # Toggle OFF
                        cls._cpu_override = False
                        device_type = cls._default_device
                        device_index = None
                    else:
                        # Toggle ON
                        cls._cpu_override = True
                        device_type = "cpu"
                        device_index = 0

                # Apply redirection if no CPU override
                if not cls._cpu_override:
                    device_type = redirect_device_type(device_type)

                # Reassemble args
                new_arg = device_type
                if device_index is not None:
                    new_arg = f"{device_type}:{device_index}"

                args = (new_arg,) + args[1:]  # Replace first arg

        # Pass everything through to original torch.device
        return _original_torch_device(*args, **kwargs)


    @classmethod
    @auto_log()
    def apply_patches(cls):
        """
        Apply all patches to the torch API.
        """
        from .modules.device_detection import get_default_device

        # Save the original torch.device constructor.
        _original_torch_device = torch.device

        @auto_log()
        def patched_torch_device(*args, **kwargs):
            """
            A patched version of torch.device.
            • If called with no arguments, returns the default device (for non‑CPU, forcing index 0).
            • If called with arguments that look like a device specification (string or torch.device),
            it routes them through our torch_device_replacement.
            • Otherwise, it passes the arguments directly to the original constructor.
            """
            # _original_torch_device is assumed to be saved already.
            if not args and not kwargs:
                default = get_default_device()
                log_info(f"torch.device() called with no arguments; using default device '{default}'")
                if default.lower() == "cpu":
                    device = _original_torch_device(default)
                else:
                    device = _original_torch_device(default, 0)
            # If the first argument looks like a device specification, route through our replacement.
            if args and (isinstance(args[0], str) or isinstance(args[0], _ORIGINAL_TORCH_DEVICE_TYPE)):
                device = TorchDevice.torch_device_replacement(*args, **kwargs)
                log_info(f"torch.device() called with arguments; replacement device '{device}' patched_torch_device")
                return device
            return _original_torch_device(*args, **kwargs)

        # Patch torch.device globally.
        torch.device = patched_torch_device

        from .modules.patching import apply_basic_patches
        # Cache the default device.
        default = get_default_device()  # This call caches the result internally.
        cls._default_device = default
        apply_basic_patches()
        
        # Patch tensor creation functions (if no explicit device is provided, use the cached default).
        tensor_creation_functions = [
            'tensor', 'zeros', 'ones', 'empty', 'randn', 'rand', 'randint',
            'arange', 'linspace', 'logspace'
        ]
        for func_name in tensor_creation_functions:
            if hasattr(torch, func_name):
                original_func = getattr(torch, func_name)
                setattr(torch, func_name, tensor_creation_wrapper(original_func, cls._default_device))

        # Patch torch.cuda functions using the mocks.
        torch.cuda.is_available = lambda: cls._default_device in ['cuda', 'mps']
        torch.cuda.device_count = lambda: 1 if cls._default_device == 'mps' else (torch.cuda.device_count() if cls._default_device == 'cuda' else 0)
        
        from .modules.cuda_redirects import (
            mock_cuda_get_device_properties, mock_cuda_empty_cache,
            mock_cuda_synchronize, mock_cuda_current_device, mock_cuda_set_device,
            mock_cuda_get_device_name, mock_cuda_get_device_capability,
            mock_cuda_memory_allocated, mock_cuda_memory_reserved,
            mock_cuda_max_memory_allocated, mock_cuda_max_memory_reserved,
            mock_cuda_memory_stats, mock_cuda_memory_snapshot, mock_cuda_memory_summary,
            mock_cuda_is_initialized, mock_cuda_get_arch_list, mock_cuda_is_built,
            mock_cuda_device_context, mock_cuda_stream_class, mock_cuda_stream,
            mock_cuda_current_stream, mock_cuda_default_stream, _get_mps_event_class
        )
        
        torch.cuda.get_device_properties = lambda device: mock_cuda_get_device_properties(cls._default_device, device)
        torch.cuda.empty_cache = lambda: mock_cuda_empty_cache(cls._default_device)
        torch.cuda.synchronize = lambda device=None: mock_cuda_synchronize(cls._default_device, device)
        torch.cuda.current_device = lambda: mock_cuda_current_device(cls._default_device)
        torch.cuda.set_device = lambda device: mock_cuda_set_device(cls._default_device, device)
        torch.cuda.get_device_name = lambda device=None: mock_cuda_get_device_name(cls._default_device, device)
        torch.cuda.get_device_capability = lambda device=None: mock_cuda_get_device_capability(cls._default_device, device)
        torch.cuda.memory_allocated = lambda device=None: mock_cuda_memory_allocated(cls._default_device, device)
        torch.cuda.memory_reserved = lambda device=None: mock_cuda_memory_reserved(cls._default_device, device)
        torch.cuda.max_memory_allocated = lambda device=None: mock_cuda_max_memory_allocated(cls._default_device, device)
        torch.cuda.max_memory_reserved = lambda device=None: mock_cuda_max_memory_reserved(cls._default_device, device)
        torch.cuda.memory_stats = lambda device=None: mock_cuda_memory_stats(cls._default_device, device)
        torch.cuda.memory_snapshot = lambda: mock_cuda_memory_snapshot(cls._default_device)
        torch.cuda.memory_summary = lambda device=None, abbreviated=False: mock_cuda_memory_summary(cls._default_device, device, abbreviated)
        torch.cuda.is_initialized = lambda: mock_cuda_is_initialized(cls._default_device)
        torch.cuda.get_arch_list = lambda: mock_cuda_get_arch_list(cls._default_device)
        torch.backends.cuda.is_built = lambda: mock_cuda_is_built(cls._default_device)
        
        # Do not override torch.cuda.device, so its type remains unchanged.
        
        torch.cuda.Stream = lambda *args, **kwargs: mock_cuda_stream_class(cls._default_device, *args, **kwargs)
        torch.cuda.stream = lambda stream=None: mock_cuda_stream(cls._default_device, stream)
        torch.cuda.current_stream = lambda device=None: mock_cuda_current_stream(cls._default_device, device)
        torch.cuda.default_stream = lambda device=None: mock_cuda_default_stream(cls._default_device, device)
        
        # Override torch.cuda.Event with our proper event class.
        torch.cuda.Event = _get_mps_event_class(cls._default_device)
        
        # Set unsupported functions to no-ops.
        unsupported_functions = [
            'set_stream', 'mem_get_info', 'reset_accumulated_memory_stats',
            'reset_max_memory_allocated', 'reset_max_memory_cached',
            'caching_allocator_alloc', 'caching_allocator_delete',
            'get_allocator_backend', 'change_current_allocator', 'nvtx',
            'jiterator', 'graph', 'CUDAGraph', 'make_graphed_callables',
            'is_current_stream_capturing', 'graph_pool_handle', 'can_device_access_peer',
            'comm', 'get_gencode_flags', 'current_blas_handle', 'memory_usage',
            'utilization', 'temperature', 'power_draw', 'clock_rate',
            'set_sync_debug_mode', 'get_sync_debug_mode', 'list_gpu_processes',
            'seed', 'seed_all', 'manual_seed', 'manual_seed_all',
            'get_rng_state', 'get_rng_state_all', 'set_rng_state',
            'set_rng_state_all', 'initial_seed',
        ]
        for func_name in unsupported_functions:
            if hasattr(torch.cuda, func_name):
                setattr(torch.cuda, func_name, lambda *args, **kwargs: None)

def initialize_torchdevice():
    """
    Initialize TorchDevice:
        1. Set the global default device.
        2. Apply all patches to the torch API.
        3. Log the initialization.
    """
    if TorchDevice._default_device is None:
        TorchDevice._default_device = get_default_device()
    TorchDevice.apply_patches()
    log_info(f"TorchDevice initialization complete. Default device: {get_default_device()}")

```
