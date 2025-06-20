# Python script to automate running Hugging Face Transformers tests with TorchDevice
import os
import subprocess
import argparse
from pathlib import Path
import datetime

# Define project roots dynamically and via arguments
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TORCHDEVICE_ROOT = SCRIPT_DIR.parent # Assumes script is in test_automation subdirectory
DEFAULT_PROJECT_ROOT = DEFAULT_TORCHDEVICE_ROOT.parent.parent / "transformers" # Default to transformers, but arg makes it generic
DEFAULT_LOG_BASE_DIR = SCRIPT_DIR / "logs" # Logs will be stored in test_automation/logs/

def discover_tests(test_path_str, project_abs_path):
    """Discover test files (.py) in the given path relative to project_abs_path."""
    # Convert test_path_str to a Path object for easier manipulation
    test_path_input = Path(test_path_str)

    # Check if the input path already starts with "tests"
    if test_path_input.parts and test_path_input.parts[0] == "tests":
        # If it starts with "tests", join it directly with project_abs_path
        full_test_path = project_abs_path / test_path_input
    else:
        # Otherwise, assume it's relative to the project's "tests" directory
        full_test_path = project_abs_path / "tests" / test_path_input
    test_files = []
    if full_test_path.is_file() and full_test_path.name.startswith("test_") and full_test_path.suffix == ".py":
        test_files.append(full_test_path)
    elif full_test_path.is_dir():
        for item in full_test_path.rglob("test_*.py"):
            if item.is_file():
                test_files.append(item)
    else:
        print(f"Error: Test path {full_test_path} is not a valid file or directory.")
    return test_files

def run_test(test_file_path, project_root_path, torchdevice_root_path, log_dir_base):
    """Run a single test file and log its output."""
    # Construct the module path for unittest (e.g., tests.pipelines.test_pipelines_common)
    relative_test_path = test_file_path.relative_to(project_root_path)
    module_path = str(relative_test_path.with_suffix('')).replace(os.sep, '.')

    # Create a log directory structure: log_dir_base / project_name / test_sub_path / test_file.log
    project_name = project_root_path.name # Get the name of the project being tested (e.g., "transformers")
    project_log_dir = log_dir_base / project_name
    
    # Path of the test file relative to the project's 'tests' directory
    # e.g., if test_file_path is .../some_project/tests/utils/test_this.py
    # and project_root_path is .../some_project
    # then relative_to_tests_dir will be utils/test_this.py
    try:
        relative_to_tests_dir = test_file_path.relative_to(project_root_path / "tests")
    except ValueError:
        # This can happen if test_file_path is not under project_root_path/tests, 
        # which shouldn't occur with current discovery but good to be safe.
        # Or if a single test file is given directly at the root of 'tests'
        relative_to_tests_dir = test_file_path.name

    log_file_subpath_dir = project_log_dir / relative_to_tests_dir.parent
    log_file_subpath_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_file_subpath_dir / (test_file_path.name + ".log")

    command = [
        "python",
        "-m",
        "unittest",
        "-v",
        module_path
    ]

    env = os.environ.copy()
    env["ACTIVATE_TORCH_DEVICE"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Ensure TorchDevice is discoverable, and also project src for some utils
    # Also add the project_root_path itself to PYTHONPATH for general imports
    python_path_parts = [
        str(torchdevice_root_path),       # Essential for TorchDevice import
        str(project_root_path / 'src')  # For projects like Transformers that have a src dir
                                        # project_root_path itself should be covered by CWD
    ]
    existing_python_path = env.get('PYTHONPATH', '')
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

    print(f"Running test: {module_path}")
    # Log the command being run
    command_str = ' '.join(command)
    print(f"  Command: {command_str}")
    print(f"  Log file: {log_file_path}")
    print(f"  PYTHONPATH: {env.get('PYTHONPATH')}")
    print(f"  PYTORCH_ENABLE_MPS_FALLBACK: {env.get('PYTORCH_ENABLE_MPS_FALLBACK')}")
    print(f"  Working directory: {project_root_path}")

    start_time = datetime.datetime.now()
    try:
        with open(log_file_path, 'w') as lf:
            lf.write(f"Test: {module_path}\n")
            lf.write(f"Command: {' '.join(command)}\n")
            lf.write(f"PYTHONPATH: {env['PYTHONPATH']}\n")
            lf.write(f"CWD: {str(project_root_path)}\n")
            lf.write(f"Timestamp: {start_time.isoformat()}\n\n")
            lf.flush()

            process = subprocess.Popen(
                command,
                cwd=str(project_root_path), # Ensure CWD is string
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT # Redirect stderr to stdout, then to file
            )
            process.wait() # Wait for the process to complete
            return_code = process.returncode
            
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            lf.write(f"\nFinished at: {end_time.isoformat()}\n")
            lf.write(f"Duration: {duration}\n")
            lf.write(f"Return code: {return_code}\n")

        if return_code == 0:
            print(f"  SUCCESS: {module_path} (Return code: {return_code}) Duration: {duration}")
        else:
            print(f"  FAILURE: {module_path} (Return code: {return_code}) Duration: {duration}")
        return module_path, log_file_path, return_code, duration, command_str

    except Exception as e:
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        error_message = f"  CRITICAL FAILURE executing {module_path}: {e} Duration: {duration}"
        print(error_message)
        with open(log_file_path, 'a') as lf:
            lf.write(f"\nCRITICAL FAILURE: {e}\n")
            lf.write(f"Finished at: {end_time.isoformat()}\n")
            lf.write(f"Duration: {duration}\n")
        return module_path, log_file_path, -1, duration # Indicate critical failure

def log_and_print(message, overall_log_file_handle):
    """Prints a message to stdout and writes it to the overall log file."""
    print(message)
    if overall_log_file_handle:
        overall_log_file_handle.write(message + "\n")
        overall_log_file_handle.flush() # Ensure it's written immediately

def main():
    parser = argparse.ArgumentParser(description="Run Hugging Face Transformers tests with TorchDevice enabled.")
    parser.add_argument(
        "test_targets", 
        nargs='+', 
        help="Relative path(s) to test files or directories from 'project/tests/'. E.g., 'pipelines/test_pipelines_common.py' or 'utils'"
    )
    parser.add_argument(
        "--project_root", 
        default=str(DEFAULT_PROJECT_ROOT),
        type=Path,
        help=f"Absolute path to the target project repo. Default: {DEFAULT_PROJECT_ROOT} (Transformers)"
    )
    parser.add_argument(
        "--torchdevice_root", 
        default=str(DEFAULT_TORCHDEVICE_ROOT),
        type=Path,
        help=f"Absolute path to the TorchDevice repo. Default: {DEFAULT_TORCHDEVICE_ROOT}"
    )
    parser.add_argument(
        "--log_dir",
        default=str(DEFAULT_LOG_BASE_DIR),
        type=Path,
        help=f"Base directory to store log files. Default: {DEFAULT_LOG_BASE_DIR}"
    )

    args = parser.parse_args()

    project_abs_path = args.project_root.resolve()
    torchdevice_abs_path = args.torchdevice_root.resolve()
    log_dir_base = args.log_dir.resolve()

    log_dir_base.mkdir(parents=True, exist_ok=True) # Ensure log base dir exists

    # --- Overall Log File Setup ---
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a descriptive series name for the overall log file
    series_name_base = "unknown_series"
    if args.test_targets:
        first_target_sanitized = args.test_targets[0].replace('/', '-').replace('.py', '')
        if len(args.test_targets) > 1:
            series_name_base = f"{first_target_sanitized}_plus_{len(args.test_targets) - 1}_more"
        else:
            series_name_base = first_target_sanitized
    
    overall_log_filename = f"{series_name_base}_{timestamp}.log"
    overall_log_filepath = log_dir_base / overall_log_filename
    overall_log_file_handle = None
    try:
        overall_log_file_handle = open(overall_log_filepath, 'w', buffering=1) # Enable line buffering
        # Display paths relative to torchdevice_abs_path for cleaner output
        if torchdevice_abs_path in overall_log_filepath.parents or torchdevice_abs_path == overall_log_filepath:
            display_overall_log_filepath = overall_log_filepath.relative_to(torchdevice_abs_path)
        else:
            display_overall_log_filepath = overall_log_filepath # Fallback if not relative, though unlikely
        print(f"Overall run log: {display_overall_log_filepath}") # Print to console only
    except IOError as e:
        print(f"Error: Could not open overall log file {overall_log_filepath}: {e}")
        # Continue without overall log if it fails, or exit if critical

    # Display the user-provided project root argument for "Target project repo"
    log_and_print(f"Target project repo: {args.project_root}", overall_log_file_handle)
    # Display TorchDevice repo relative to its parent directory
    log_and_print(f"TorchDevice repo: {torchdevice_abs_path.relative_to(torchdevice_abs_path.parent)} (CWD)", overall_log_file_handle) # Keep absolute for clarity

    log_dir_project_specific = log_dir_base / project_abs_path.name
    if torchdevice_abs_path in log_dir_project_specific.parents or torchdevice_abs_path == log_dir_project_specific:
        display_log_dir_project_specific = log_dir_project_specific.relative_to(torchdevice_abs_path)
    else:
        display_log_dir_project_specific = log_dir_project_specific
    log_and_print(f"Log directory for individual tests: {display_log_dir_project_specific}", overall_log_file_handle)

    if torchdevice_abs_path in overall_log_filepath.parents or torchdevice_abs_path == overall_log_filepath:
        display_overall_log_filepath_logged = overall_log_filepath.relative_to(torchdevice_abs_path)
    else:
        display_overall_log_filepath_logged = overall_log_filepath
    log_and_print(f"Overall run log file: {display_overall_log_filepath_logged}", overall_log_file_handle)

    all_tests_to_run = []
    for target in args.test_targets:
        discovered = discover_tests(target, project_abs_path)
        if discovered:
            all_tests_to_run.extend(discovered)
        else:
            log_and_print(f"Warning: No tests found for target '{target}' at {project_abs_path / 'tests' / target}", overall_log_file_handle)
    
    # Remove duplicates if any, preserving order
    seen = set()
    unique_tests = []
    for test_file in all_tests_to_run:
        if test_file not in seen:
            seen.add(test_file)
            unique_tests.append(test_file)
    all_tests_to_run = unique_tests

    if not all_tests_to_run:
        log_and_print("No test files found to run. Exiting.", overall_log_file_handle)
        if overall_log_file_handle: overall_log_file_handle.close()
        return

    log_and_print(f"\nFound {len(all_tests_to_run)} test files to run:", overall_log_file_handle)
    for test_file in all_tests_to_run:
        log_and_print(f"  - {test_file.relative_to(project_abs_path)}", overall_log_file_handle)
    log_and_print("-"*50, overall_log_file_handle)

    results_summary = []
    overall_start_time = datetime.datetime.now()

    for i, test_file_path in enumerate(all_tests_to_run):
        # The run_test function already prints its own progress to console.
        # We can log a pre-run message here to the overall log.
        log_and_print(f"\nPreparing to run test {i+1}/{len(all_tests_to_run)}: {test_file_path.name}", overall_log_file_handle)
        
        # run_test prints to console; its detailed output goes to its specific log file.
        # We capture its summary for the overall log.
        module_path, log_file_for_test, return_code, duration, test_command_str = run_test(test_file_path, project_abs_path, torchdevice_abs_path, log_dir_base)
        
        status_message = f"Finished test: {module_path} - Status: {'SUCCESS' if return_code == 0 else ('FAILURE' if return_code > 0 else 'CRITICAL_FAILURE')} (RC: {return_code}) Duration: {duration}"
        log_and_print(status_message, overall_log_file_handle)
        if torchdevice_abs_path in log_file_for_test.parents or torchdevice_abs_path == log_file_for_test:
            display_log_file_for_test = log_file_for_test.relative_to(torchdevice_abs_path)
        else:
            display_log_file_for_test = log_file_for_test
        log_and_print(f"  Log file: {display_log_file_for_test}", overall_log_file_handle)
        # Also log the command to the overall summary log
        log_and_print(f"  Test Command: {test_command_str}", overall_log_file_handle) 

        results_summary.append({
            "module": module_path,
            "log_file": str(log_file_for_test),
            "test_command": test_command_str, # Add test command here
            "status": "SUCCESS" if return_code == 0 else ("FAILURE" if return_code > 0 else "CRITICAL_FAILURE"),
            "return_code": return_code,
            "duration": str(duration)
        })
        log_and_print("-"*50, overall_log_file_handle)

    overall_end_time = datetime.datetime.now()
    overall_duration = overall_end_time - overall_start_time

    log_and_print("\n" + "="*20 + " Test Run Summary " + "="*20, overall_log_file_handle)
    successful_tests = 0
    failed_tests = 0
    critical_failures = 0

    for result in results_summary:
        log_and_print(f"Module: {result['module']}", overall_log_file_handle)
        log_and_print(f"  Status: {result['status']} (Return Code: {result['return_code']})", overall_log_file_handle)
        # Ensure result['log_file'] is a Path object for relative_to
        result_log_path = Path(result['log_file'])
        if torchdevice_abs_path in result_log_path.parents or torchdevice_abs_path == result_log_path:
            display_result_log_file = result_log_path.relative_to(torchdevice_abs_path)
        else:
            display_result_log_file = result_log_path
        log_and_print(f"  Log: {display_result_log_file}", overall_log_file_handle)
        if "test_command" in result: # Print command if available
            log_and_print(f"  Test Command: {result['test_command']}", overall_log_file_handle)
        log_and_print(f"  Duration: {result['duration']}", overall_log_file_handle)
        if result['status'] == "SUCCESS":
            successful_tests += 1
        elif result['status'] == "FAILURE":
            failed_tests += 1
        else:
            critical_failures += 1
    
    log_and_print("\n" + "-"*50, overall_log_file_handle)
    log_and_print(f"Total test files processed: {len(results_summary)}", overall_log_file_handle)
    log_and_print(f"Successful: {successful_tests}", overall_log_file_handle)
    log_and_print(f"Failed (test failures): {failed_tests}", overall_log_file_handle)
    log_and_print(f"Critical failures (script/execution errors): {critical_failures}", overall_log_file_handle)
    log_and_print(f"Overall execution time: {overall_duration}", overall_log_file_handle)
    log_and_print("="*58, overall_log_file_handle)

    if overall_log_file_handle:
        overall_log_file_handle.close()

if __name__ == "__main__":
    main()

