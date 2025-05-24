#!/usr/bin/env python
"""
Custom runner script to run TorchDevice tests, build, and install.

Usage examples:
    python run_tests_and_install.py                      # Run all tests, build, and install
    python run_tests_and_install.py --test-only          # Run tests only
    python run_tests_and_install.py --update-expected    # Update expected output files
    python run_tests_and_install.py tests/some_test.py    # Run specific test file(s)
"""

import os
import sys
import argparse
import subprocess
import time
import logging
from pathlib import Path

# Configure basic logging for the runner
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Determine the project root directory (where run_tests_and_install.py is located).
PROJECT_ROOT = Path(__file__).parent.absolute()

# Add the tests directory (which contains your common package) to sys.path.
tests_dir = PROJECT_ROOT / "tests"
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


def discover_test_files(test_root: Path) -> list:
    """Recursively discover test files in test_root that match 'test_*.py'."""
    return list(test_root.rglob("test_*.py"))


def run_test_file(test_file: Path, update_expected: bool) -> bool:
    """
    Run a test file via subprocess.
    
    Args:
        test_file: Path to the test file.
        update_expected: Whether to pass the update flag.
        
    Returns:
        True if the test passed, False otherwise.
    """
    env = os.environ.copy()
    
    # Add the tests directory to PYTHONPATH so that "common" can be imported.
    tests_dir = PROJECT_ROOT / "tests"
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{str(tests_dir)}{os.pathsep}{current_pythonpath}"
    
    # Build the command.
    cmd = [sys.executable, str(test_file)]
    if update_expected:
        # Append the flag so that the test file's sys.argv contains it.
        cmd.append("--update-expected")
    
    logger.info(f"Running test file: {test_file}")
    logger.info(f"Running command: {cmd}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        logger.error(f"Test file failed: {test_file}")
        return False
    else:
        logger.info(f"Test file passed: {test_file}")
        return True


def build_package() -> bool:
    """Build the TorchDevice package."""
    logger.info("Building TorchDevice package...")
    os.chdir(PROJECT_ROOT)
    build_cmd = [sys.executable, 'setup.py', 'build']
    process = subprocess.run(build_cmd, capture_output=True, text=True)
    if process.returncode == 0:
        logger.info("✅ Package built successfully!")
        return True
    else:
        logger.error(f"❌ Package build failed: {process.stderr}")
        return False

def install_package() -> bool:
    """Install the TorchDevice package in development mode."""
    logger.info("Installing TorchDevice package...")
    os.chdir(PROJECT_ROOT)
    install_cmd = [sys.executable, '-m', 'pip', 'install', '-e', '.']
    process = subprocess.run(install_cmd, capture_output=True, text=True)
    if process.returncode == 0:
        logger.info("✅ Package installed successfully!")
        return True
    else:
        logger.error(f"❌ Package installation failed: {process.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run TorchDevice tests, build, and install")
    parser.add_argument('--test-only', action='store_true', help="Run tests only")
    parser.add_argument('--update-expected', action='store_true', help="Update expected output files")
    parser.add_argument('test_paths', nargs='*', help="Test files or directories to run")
    args = parser.parse_args()

    # Remove custom args from sys.argv so unittest in test files doesn't complain.
    sys.argv = [sys.argv[0]]

    # Add the project root to sys.path so that tests can import modules correctly.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    start_time = time.time()

    # Determine test files: if none are provided, use the whole tests directory.
    if not args.test_paths:
        test_root = PROJECT_ROOT / "tests"
        test_files = discover_test_files(test_root)
    else:
        test_files = []
        for path in args.test_paths:
            p = PROJECT_ROOT / path
            if p.is_dir():
                test_files.extend(discover_test_files(p))
            else:
                test_files.append(p)

    if not test_files:
        logger.warning("No test files found.")
        return 0

    # Run tests.
    all_passed = True
    test_results = []
    for test_file in test_files:
        passed = run_test_file(test_file, args.update_expected)
        test_results.append((str(test_file), passed))
        if not passed:
            all_passed = False

    # Print a summary of test results
    print("\nTest Results Summary:")
    for test_file, passed in test_results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_file}: {status}")

    # If tests passed and we're not in test-only mode, build and install.
    if all_passed and not args.test_only:
        if build_package():
            if not install_package():
                logger.error("Package installation failed.")
                return 1
        else:
            logger.error("Package build failed.")
            return 1
    elif not all_passed:
        logger.error("Some tests failed. Skipping build and install.")
        return 1
    else:
        logger.info("Test-only mode: Skipping build and install.")

    elapsed_time = time.time() - start_time
    logger.info(f"Process completed in {elapsed_time:.2f} seconds")
    return 0

if __name__ == "__main__":
    sys.exit(main())
