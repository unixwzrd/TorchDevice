#!/usr/bin/env python
"""
Script to run TorchDevice tests, build, and install.

This script:
1. Runs all tests in the tests directory or specific tests if provided
2. If tests pass, optionally builds the package
3. Optionally installs the package in development mode

Usage:
    python run_tests_and_install.py                        # Run all tests, build, and install
    python run_tests_and_install.py --test-only            # Run only all tests
    python run_tests_and_install.py --update-expected      # Update expected log output files
    python run_tests_and_install.py tests/test_file.py     # Run specific test(s)
"""

import os
import sys
import subprocess
import logging
import argparse
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


def run_test(test_path, update_expected=False):
    """
    Run a single test file or directory.
    
    Args:
        test_path: Path to the test file or directory
        update_expected: Whether to update expected output files
    
    Returns:
        True if the test passed, False otherwise
    """
    # Convert to Path object if it's a string
    test_path = Path(test_path)
    
    # Set environment variable for update_expected
    env = os.environ.copy()
    env['UPDATE_EXPECTED_OUTPUT'] = '1' if update_expected else '0'
    
    # Determine the command to run
    if test_path.is_file():
        # Run a specific test file
        cmd = [sys.executable, str(test_path)]
        if update_expected:
            cmd.append('--update-expected')
    else:
        # For directories, find all test files and run them individually
        # This ensures consistent environment variable handling
        logger.info(f"Discovering tests in: {test_path}")
        all_passed = True
        
        # Find all Python files that start with "test_"
        test_files = list(test_path.glob("**/test_*.py"))
        if not test_files:
            logger.warning(f"No test files found in {test_path}")
            return True
        
        logger.info(f"Found {len(test_files)} test files")
        
        # Run each test file individually
        for test_file in test_files:
            logger.info(f"Running test file: {test_file}")
            file_cmd = [sys.executable, str(test_file)]
            if update_expected:
                file_cmd.append('--update-expected')
            
            process = subprocess.run(file_cmd, env=env)
            if process.returncode != 0:
                all_passed = False
        
        return all_passed
    
    # Run the test
    logger.info(f"Running test: {test_path}")
    process = subprocess.run(cmd, env=env)
    
    # Return True if the test passed
    return process.returncode == 0


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
        logger.info("✅ Package built successfully!")
        return True
    else:
        logger.error(f"❌ Package build failed: {build_process.stderr}")
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
        logger.info("✅ Package installed successfully!")
        return True
    else:
        logger.error(f"❌ Package installation failed: {install_process.stderr}")
        return False


def main():
    """Main function to run tests, build, and install."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run TorchDevice tests, build, and install')
    parser.add_argument('--test-only', action='store_true',
                        help='Run only tests without building and installing')
    parser.add_argument('--update-expected', action='store_true',
                        help='Update expected log output files')
    parser.add_argument('test_paths', nargs='*',
                        help='Specific test files or directories to run')
    args = parser.parse_args()
    
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("Starting TorchDevice test, build, and install process")
    logger.info("=" * 80)
    
    # If no test paths are provided, run all tests
    if not args.test_paths:
        test_paths = [PROJECT_ROOT / 'tests']
    else:
        test_paths = [PROJECT_ROOT / path for path in args.test_paths]
        # Force test-only mode if specific tests are provided
        if not args.test_only:
            logger.info("Forcing test-only mode since specific tests were provided.")
            args.test_only = True
    
    # Run each test
    all_passed = True
    for test_path in test_paths:
        if not run_test(test_path, args.update_expected):
            all_passed = False
    
    # If tests passed and not in test-only mode, build and install
    if all_passed and not args.test_only:
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
    elif not all_passed:
        logger.error("Tests failed. Skipping build and install.")
        return 1
    elif args.test_only:
        logger.info("Test-only mode. Skipping build and install.")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info("=" * 80)
    logger.info(f"TorchDevice process completed in {elapsed_time:.2f} seconds")
    logger.info("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
