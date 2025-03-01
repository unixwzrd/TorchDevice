#!/usr/bin/env python
"""
Script to run TorchDevice tests, build, and install.

This script:
1. Runs all tests in the tests directory
2. If tests pass, optionally builds the package
3. Optionally installs the package in development mode

Usage:
    python run_tests_and_install.py             # Run tests, build, and install
    python run_tests_and_install.py --test-only # Run only tests
    python run_tests_and_install.py --help      # Show help message
"""

import os
import sys
import subprocess
import logging
import unittest
import time
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prevent duplicate logging from TorchDevice
logging.getLogger("TorchDevice").propagate = False

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def run_tests():
    """Run all tests in the tests directory."""
    logger.info("Running TorchDevice tests...")
    
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Load tests from the tests directory
    test_suite = loader.discover(str(PROJECT_ROOT / 'tests'), pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Check if all tests passed
    if result.wasSuccessful():
        logger.info("✅ All tests passed!")
        return True
    else:
        logger.error("❌ Tests failed!")
        return False


def build_package():
    """Build the TorchDevice package."""
    logger.info("Building TorchDevice package...")
    
    # Change to the project root directory
    os.chdir(PROJECT_ROOT)
    
    # Run the build command
    build_cmd = [sys.executable, 'setup.py', 'build']
    build_process = subprocess.run(build_cmd, capture_output=True, text=True)
    
    # Check if the build was successful
    if build_process.returncode == 0:
        logger.info(build_process.stdout)
        logger.info("✅ Package built successfully!")
        return True
    else:
        logger.error(build_process.stderr)
        logger.error("❌ Package build failed!")
        return False


def install_package():
    """Install the TorchDevice package in development mode."""
    logger.info("Installing TorchDevice package...")
    
    # Change to the project root directory
    os.chdir(PROJECT_ROOT)
    
    # Run the install command
    install_cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.']
    install_process = subprocess.run(install_cmd, capture_output=True, text=True)
    
    # Check if the installation was successful
    if install_process.returncode == 0:
        logger.info(install_process.stdout)
        logger.info("✅ Package installed successfully!")
        return True
    else:
        logger.error(install_process.stderr)
        logger.error("❌ Package installation failed!")
        return False


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run TorchDevice tests, build, and install.'
    )
    parser.add_argument(
        '--test-only', 
        action='store_true',
        help='Run only tests without building and installing'
    )
    return parser.parse_args()


def main():
    """Main function to run tests, build, and install."""
    args = parse_arguments()
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("Starting TorchDevice test, build, and install process")
    logger.info("=" * 80)
    
    # Run tests
    tests_passed = run_tests()
    
    # If tests passed and not in test-only mode, build and install
    if tests_passed and not args.test_only:
        # Build the package
        build_success = build_package()
        
        # If build was successful, install the package
        if build_success:
            install_success = install_package()
            if not install_success:
                logger.error("Failed to install the package.")
                return 1
        else:
            logger.error("Failed to build the package.")
            return 1
    elif not tests_passed:
        logger.error("Tests failed. Skipping build and install.")
        return 1
    elif args.test_only:
        logger.info("Test-only mode. Skipping build and install.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info("=" * 80)
    logger.info(f"TorchDevice test, build, and install process completed in {elapsed_time:.2f} seconds")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
