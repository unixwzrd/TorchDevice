# Test Automation for TorchDevice Integration

## Purpose

This directory contains `run_transformer_tests.py`, a Python script designed to automate the execution of test suites for Python projects, specifically with `TorchDevice` integration enabled. It facilitates running tests, capturing detailed logs, and summarizing results, streamlining the process of testing `TorchDevice` against various codebases like Hugging Face Transformers.

## Prerequisites

* **Python:** Python 3.8+ is recommended.
* **Virtual Environment:** It's highly recommended to run tests within a Python virtual environment where the target project's dependencies are installed.
* **TorchDevice:** The `TorchDevice` library must be accessible. The script attempts to locate it by default in the parent directory of `test_automation/` (i.e., `../`). This can be overridden with the `--torchdevice_root` argument.
* **Target Project:** The Python project you intend to test should be available locally.

### Specific Prerequisites for Testing Hugging Face Transformers

When targeting the Hugging Face Transformers library (`--project_root path/to/transformers`), the following additional setup steps and considerations are crucial:

1. **Activate TorchDevice in Transformers Tests:**
    You **must** modify the `tests/__init__.py` file within your local clone of the Hugging Face Transformers library. This ensures `TorchDevice` is imported and activated early in the test discovery process.

    Add the following conditional import logic to `your_transformers_clone/tests/__init__.py`:

    ```python
    import os
    import importlib

    # Check for TorchDevice activation environment variable
    if os.environ.get("ACTIVATE_TORCH_DEVICE") == "1":
        try:
            # Attempt to import TorchDevice
            import TorchDevice
            print("[transformers.tests.__init__] TorchDevice activated.")
        except ImportError:
            print("[transformers.tests.__init__] TorchDevice import failed. Ensure TorchDevice is in PYTHONPATH.")
            # Optionally, re-raise the error or handle as appropriate for your setup
            # raise
    ```

    This modification allows the `ACTIVATE_TORCH_DEVICE=1` environment variable (set by `run_transformer_tests.py`) to correctly hook `TorchDevice` into the Transformers test suite.

2. **Additional Dependencies for Transformers Tests:**
    The full Transformers test suite may require additional packages beyond the core library. Ensure your Transformers project environment includes the following (these were previously listed in a `test_projects/requirements.txt`):
    * `parameterized`
    * `timeout-decorator`
    * `huggingface_hub` (usually a core dependency, but ensure it's up-to-date)
    * `hf_xet` (for specific XetHub related tests, may be optional if not running those)

    Install these into your Transformers virtual environment if they are missing (e.g., `pip install parameterized timeout-decorator hf_xet`).

## Usage

The script is run from the command line.

```bash
python run_transformer_tests.py [options] <test_target_1> [test_target_2 ...]
```

### Command-Line Arguments

* **`test_targets`** (Positional Arguments):
  * One or more paths to test files or directories relative to the target project's `tests/` directory.
  * Example: `utils/test_my_util.py`, `pipelines`

* **`--project_root <PATH_TO_PROJECT>`** (Optional):
  * Specifies the absolute or relative path to the root directory of the Python project you want to test.
  * Defaults to `../transformers` (assuming a sibling `transformers` directory relative to the `TorchDevice` project root).
  * Example: `--project_root /path/to/your/custom_project`

* **`--torchdevice_root <PATH_TO_TORCHDEVICE>`** (Optional):
  * Specifies the absolute or relative path to the `TorchDevice` library's root directory.
  * Defaults to `../` (the parent directory of `test_automation/`).
  * Example: `--torchdevice_root /path/to/TorchDevice`

### Examples

1. **Run a single test file from the default `transformers` project:**

    ```bash
    python run_transformer_tests.py utils/test_versions_utils.py
    ```

2. **Run all tests in the `pipelines` directory of the default `transformers` project:**

    ```bash
    python run_transformer_tests.py pipelines
    ```

3. **Run tests in a custom project:**

    ```bash
    python run_transformer_tests.py --project_root /path/to/my_custom_project core_tests/test_module.py
    ```

4. **Run multiple specific test files:**

    ```bash
    python run_transformer_tests.py utils/test_one.py utils/test_two.py
    ```

## Managing Test Target Projects

While you can specify any project path using `--project_root`, for frequently tested projects, it can be convenient to organize them.

**Using Symbolic Links (Recommended for `test_projects/`):**
If you maintain local clones of projects you regularly test (like Hugging Face Transformers), you can create symbolic links (symlinks) within the `TorchDevice/test_projects/` directory. These symlinks should point to the actual root directories of your target projects.

*Example:*
Suppose your main Transformers clone is at `/path/to/my/transformers_clone`. You can create a symlink:

```bash
# Navigate to your TorchDevice project directory
cd /path/to/TorchDevice
# Create the test_projects directory if it doesn't exist
mkdir -p test_projects
# Create the symlink
ln -s /path/to/my/transformers_clone test_projects/transformers
```

Then, you can run tests using:

```bash
python run_transformer_tests.py --project_root ../test_projects/transformers <test_target>
```

(Note: `../test_projects/transformers` is relative to the `test_automation` directory where the script resides).

This approach keeps your `TorchDevice` repository clean (as the large project clones are external) and makes it easy to reference test targets. The script's default `--project_root` is `../transformers` (relative to the script's directory), which would point to `TorchDevice/transformers`. If you use the `test_projects/transformers` symlink structure, you must specify it with `--project_root`.

## Output

The script generates comprehensive logs for each test run:

* **Overall Run Log:**
  * Named in the format: `<series_name>_<timestamp>.log` (e.g., `pipelines_2025-06-08_11-30-00.log`).
  * Located in: `test_automation/logs/`
  * Contains a summary of all tests run, their status, execution times, and any errors encountered by the script itself. This log is updated in real-time (line-buffered).

* **Individual Test Logs:**
  * Named after the test file: `<test_file_path>.log` (e.g., `utils/test_versions_utils.py.log`).
  * Located in: `test_automation/logs/<project_name>/<relative_test_path_in_project>/` (e.g., `test_automation/logs/transformers/utils/test_versions_utils.py.log`).
  * Contains the full STDOUT and STDERR output from the execution of that specific test file.

### Log Parsing with `parse_transformer_log_summary.py`

The `parse_transformer_log_summary.py` script processes the summary log files generated by `run_transformer_tests.py` (or similarly formatted logs) to produce a structured JSON output.

* **Input:** A summary log file (e.g., `test_automation/logs/utils_2025-06-08_18-55-39.log`).
* **Output JSON File:**
  * By default, if no `--output-dir` is specified, the JSON file will be created in the same directory as the input summary log file, with the same base name but a `.json` extension.
  * If `--output-dir <PATH_TO_DIR>` is specified, the JSON file will be created in that directory, named after the input summary log file (e.g., `PATH_TO_DIR/utils_2025-06-08_18-55-39.json`).
  * **Recommended Usage:** It's recommended to use a dedicated output directory for clarity, for example:

    ```bash
    python test_automation/parse_transformer_log_summary.py <path_to_summary_log.log> --output-dir test_automation/logs/summary_output/
    ```

* **JSON Content:** The JSON file provides a structured list of results, with one entry per test module processed from the summary log. Each entry contains:
  * **Module Information:** `module` (e.g., `tests.utils.test_activations`), `test_file_name`, `test_script_path`, and `component`.
  * **Execution Summary:** `status_from_summary` (the high-level status from the main log), `return_code`, `log_file` (path to the individual test log), and `duration`.
  * **Detailed Test Cases:** A `test_cases` array, where each object represents a single test (`test_...`) or a runner-level issue (`RUNNER_ERROR_...`). Each test case includes its `name`, `class_path`, `status` (`SUCCESS`, `FAILURE`, `SKIPPED`), and `error_details` (containing tracebacks if applicable).
  * **Individual Log Summary:** An `individual_log_summary` object, which is the result of parsing the detailed individual log file. This provides a definitive `overall_status` ("OK" or "FAILED") and counts for `failures` and `skipped` tests.
  * **Final Status:** A final `status` field for the module (e.g., `SUCCESS`, `FAILURE`). This status is derived from the `individual_log_summary`, ensuring it accurately reflects the detailed test outcomes, including runner errors that the main summary might miss.

## Environment Variables

* **`ACTIVATE_TORCH_DEVICE=1`**: The script automatically sets this environment variable for each test subprocess it runs. This is the trigger that `TorchDevice` (and projects configured to use it, like `transformers/tests/__init__.py`) uses to enable `TorchDevice` functionality.
* **`PYTHONPATH`**: The script dynamically constructs and sets `PYTHONPATH` for each test subprocess to ensure:
    1. The `TorchDevice` library (from `--torchdevice_root`) is importable.
    2. The target project's `src/` directory (if it exists, e.g., `project_root/src/`) is importable, which is common for projects like Hugging Face Transformers.
    The script also preserves any existing `PYTHONPATH` from your environment. The Current Working Directory (CWD) for each test is set to the `project_root`, which typically handles imports relative to the project base.

- **`PYTORCH_ENABLE_MPS_FALLBACK=1`**: The script sets this environment variable for each test subprocess to enable fallback to CPU for operations not supported by MPS. This seems to be in the module "aten::"  This needs more examination.
