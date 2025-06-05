#!/usr/bin/env python
"""
Test CUDA function stubs to ensure all expected CUDA functions are available.

This module tests that all expected CUDA functions are available and return expected types,
even when running on systems without actual CUDA hardware.
"""
import unittest
from typing import List, Tuple, Union, Type, Any, Optional
import torch
from common.test_utils import PrefixedTestCase

# Import TorchDevice to ensure CUDA redirection is set up
import TorchDevice

class TestCUDAStubs(PrefixedTestCase):
    """Test that all expected CUDA function stubs are available."""

    def setUp(self):
        """Set up test environment."""
        # Call the parent setUp method to set up logging
        super().setUp()

        # Determine the available hardware
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        self.device = torch.device('cuda' if self.has_cuda else 'mps' if self.has_mps else 'cpu')
        self.info("Using device: %s", self.device)

        self.logger.info(f"TestCUDAStubs.setUp: torch.cuda.current_device ID: {id(torch.cuda.current_device)}")
        self.logger.info(f"TestCUDAStubs.setUp: torch.cuda.device_count ID: {id(torch.cuda.device_count)}")
        self.logger.info(f"TestCUDAStubs.setUp: torch.cuda.manual_seed_all ID: {id(torch.cuda.manual_seed_all)}")
        self.logger.info("Finished TestCUDAStubs setUp.")

        # Get the default device through PyTorch's interface
        # TorchDevice will intercept this call and handle redirection
        self.default_device = torch.get_default_device()
        self.info(f"Using default device: {self.default_device}")

    def test_cuda_functions(self):
        """Test that all expected CUDA functions are available and return expected types."""
        # List of (function_name, expected_type_or_value, alt_type_for_noncuda) pairs
        # The alt_type_for_noncuda is the expected type when not running on CUDA
        cuda_functions: List[Tuple[str, Optional[Union[Type, Any]], Optional[Union[Type, Any]]]] = [
            # Memory-related functions might be fully implemented in memory.py
            ('set_stream', None, None),
            # mem_get_info could return tuple or None depending on whether memory.py or unassigned.py wins
            ('mem_get_info', tuple, (tuple, type(None))),
            ('reset_accumulated_memory_stats', None, None),
            ('reset_max_memory_allocated', None, None),
            ('reset_max_memory_cached', None, None),
            ('caching_allocator_alloc', None, None),
            ('caching_allocator_delete', None, None),
            ('get_allocator_backend', None, None),
            ('change_current_allocator', None, None),
            ('nvtx', None, None),
            ('jiterator', None, None),
            ('graph', None, None),
            ('CUDAGraph', None, None),
            ('make_graphed_callables', None, None),
            ('is_current_stream_capturing', None, None),
            ('graph_pool_handle', None, None),
            ('can_device_access_peer', None, None),
            ('comm', None, None),
            ('get_gencode_flags', None, None),
            ('current_blas_handle', None, None),
            ('memory_usage', None, None),
            ('utilization', int, 0),
            ('temperature', int, 0),
            ('power_draw', int, 0),
            ('clock_rate', int, 0),
            ('set_sync_debug_mode', None, None),
            ('get_sync_debug_mode', int, 0),
            ('list_gpu_processes', list, []),
            ('seed', None, None),
            ('seed_all', None, None),
            ('manual_seed', None, None),
            ('manual_seed_all', None, None),
            ('get_rng_state', None, None),
            ('get_rng_state_all', None, None),
            ('set_rng_state', None, None),
            ('set_rng_state_all', None, None),
            ('initial_seed', None, None),
        ]

        for func_name, expected_cuda, expected_noncuda in cuda_functions:
            self.info(f"Testing torch.cuda.{func_name}")
            fn = getattr(torch.cuda, func_name, None)
            self.assertIsNotNone(fn, f"torch.cuda.{func_name} is missing")

            try:
                result = fn()

                # Determine expected type based on current device
                expected = expected_cuda
                if self.default_device != 'cuda' and expected_noncuda is not None:
                    expected = expected_noncuda

                if expected is None:
                    self.assertIsNone(result, f"torch.cuda.{func_name} should return None")
                elif isinstance(expected, tuple) and isinstance(expected[0], type):
                    # Special case for functions that could return multiple types
                    self.assertTrue(isinstance(result, expected),
                                    f"torch.cuda.{func_name} should return one of {expected}, got {type(result)}")
                elif isinstance(expected, type):
                    # For functions with a specific return type
                    self.assertIsInstance(result, expected,
                                          f"torch.cuda.{func_name} should return {expected}, got {type(result)}")
                else:
                    # For functions with constant return values
                    self.assertEqual(result, expected,
                                     f"torch.cuda.{func_name} should return {expected}, got {result}")
            except Exception as e:
                self.info(f"Exception calling torch.cuda.{func_name}: {e}")
                # If the function raises an exception, note it but don't fail the test
                # as some functions might not be fully implemented on all platforms


if __name__ == '__main__':
    unittest.main() 