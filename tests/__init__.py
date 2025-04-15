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
