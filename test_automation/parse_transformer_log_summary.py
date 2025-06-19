import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum

import datetime
import traceback

# --- Regex Patterns ---

# For the main summary log (human-readable section)
SUMMARY_SECTION_START_RE = re.compile(r"^=+\s*Test Run Summary\s*=+$")
MODULE_RE = re.compile(r"^Module: (.*)$")
STATUS_RE = re.compile(r"^Status: (\w+)(?: \(Return Code: (-?[\dNA]+)\))?$")
LOG_FILE_RE = re.compile(r"^Log: (\S+\.log)$")
TEST_COMMAND_RE = re.compile(r"^Test Command: (.*)$")
DURATION_RE = re.compile(r"^Duration: (\S+)$")

# For parsing "batch" summary log files (multi-line entries from run_transformer_tests.py)
BATCH_FINISHED_TEST_RE = re.compile(
    r"^Finished test: (?P<module_name>[\w\.\-]+)"
    r"\s+-\s+Status:\s+(?P<status_text>SUCCESS|FAILURE|CRITICAL_FAILURE)"
    r"\s+\(RC:\s+(?P<return_code>-?\d+)\)"
    r"\s+Duration:\s+(?P<duration>\S+)$")
BATCH_LOG_FILE_RE = re.compile(r"^\s+Log file:\s+(?P<log_file_path>\S+\.log)$")
BATCH_TEST_COMMAND_RE = re.compile(r"^\s+Test Command:\s+(?P<test_command_str>.+)$")

# For parsing individual test log files
INDIVIDUAL_TEST_LINE_RE = re.compile(
    r"^(?P<test_method>\w+)\s+"  # Captures the test method name
    r"\((?P<test_class_path>[^)]+)\)\s+\.\.\."  # Captures the class path like "module.TestClass"
    r"(?:.*?\s)?"  # Non-capturing group for any optional text (e.g., timing) before status
    r"(?P<status_text>ok|FAIL|ERROR|skipped|expected failure|unexpected success|PASSED|FAILED|SKIPPED)"  # Captures various status strings
    r"(?:\s+'(?P<skip_reason_text>[^']*)')?"  # Optional capturing group for skip reason, e.g., 'reason for skip'
    r"(?:\s+\(.*\))?"  # Optional non-capturing group for extra details like (XTREAM_PASS)
    r"\s*$"  # Matches end of line, allowing trailing whitespace
)

# Regex for the final summary block in individual logs (e.g., "Ran 29 tests in 0.567s")
INDIVIDUAL_LOG_FINAL_RAN_TESTS_RE = re.compile(r"^Ran (?P<total>\d+) tests in \S+s$")
# Regex for the outcome line, e.g., "OK" or "FAILED (failures=1, errors=2, skipped=3)"
INDIVIDUAL_LOG_SUMMARY_RE = re.compile(r"^(?P<status_text>OK|FAILED)(?: \((?P<details>.*)\))?$")
PYTHON_TEST_SUMMARY_LINE_RE = re.compile(r"^(Ran \d+ tests? in \d+\.\d+s|OK(?: \(.*\))?|FAILED(?: \(.*\))?)$", re.IGNORECASE)
# Regex to extract counts from the details string of a FAILED outcome
INDIVIDUAL_LOG_FINAL_COUNTS_RE = re.compile(r"(?:failures=(?P<failures>\d+))?|\s*(?:errors=(?P<errors>\d+))?|\s*(?:skipped=(?P<skipped>\d+))?")

# Regex for simple OK/FAILED lines that might appear mid-log for a block of tests
MID_LOG_STATUS_RE = re.compile(r"^(OK|FAILED)$")

INDIVIDUAL_LOG_SUMMARY_RE = re.compile(
    r"^(?P<overall_status>OK|FAILED)"
    r"(?:\s+\((?:skipped=(?P<skipped>\d+))?(?:,\s*)?(?:failures=(?P<failures>\d+))?(?:,\s*)?(?:errors=(?P<errors>\d+))?\))?$"
) # Kept for now, but new regexes above are preferred for final summary

TRACEBACK_HEADER = "======================================================================"

# For parsing error details from tracebacks
TRACEBACK_START_RE = re.compile(r"Traceback \(most recent call last\):")
TEST_FAILURE_HEADER_RE = re.compile(r"^(?:ERROR|FAIL): (?P<test_method>\w+) \((?P<test_class_path>[\w\.]+)\)$")
INTERNAL_DYNAMO_ERROR_RE = re.compile(r"InternalTorchDynamoError: (.*)")
JIT_ERROR_RE = re.compile(r"Compiled functions can't take variable number of arguments or use keyword-only arguments with defaults:")
ATTRIBUTE_ERROR_RE = re.compile(r"AttributeError: (.*)")
ASSERTION_ERROR_RE = re.compile(r"AssertionError: (.*)")


# For parsing error details from tracebacks
PYTHON_TRACEBACK_START_RE = re.compile(r"Traceback \(most recent call last\):")
# Matches typical error lines like "ValueError: ...", "torch.cuda.OutOfMemoryError: ..."
PYTHON_TRACEBACK_END_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)?Error:.*$")
PYTHON_EXCEPTION_RE = re.compile(r"^((?:[a-zA-Z_][a-zA-Z0-9_]*)(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*Error):") # Extracts specific exception type

# Component diagnosis keywords and patterns
SUB_PATTERNS = {
    "RuntimeError": {
        "CUDA Generator Error": [re.compile(r"Cannot get CUDA generator", re.IGNORECASE)],
        "Stream Sync Error": [re.compile(r"Backend doesn't support synchronizing streams", re.IGNORECASE)],
        "Distributed Process Group Error": [re.compile(r"Default process group has not been initialized", re.IGNORECASE)],
    },
    "ValueError": {
        "Tokenizer Input Error": [re.compile(r"Asking to pad but the tokenizer returned", re.IGNORECASE)],
        "Shape Mismatch Error": [re.compile(r"Shape of tensors do not match", re.IGNORECASE)],
    },
    "AssertionError": {
        "Device Mismatch": [re.compile(r"device\(type='mps', index=0\) != device\(type='mps'\)", re.IGNORECASE)],
        "Tensors Not Close": [re.compile(r"Tensor-likes are not close!", re.IGNORECASE)],
    },
    "ValueError": {
        "Numeric Precision (hidden_states)": [
            re.compile(r"mean relative difference for hidden_states: nan", re.IGNORECASE)
        ],
        "Numeric Precision (audio_values)": [
            re.compile(r"mean relative difference for audio_values: nan", re.IGNORECASE)
        ],
        "PyTorch Tensor Output Only": [
            re.compile(r"Only returning PyTorch tensors is currently supported", re.IGNORECASE)
        ]
    }
}

COMPONENT_PATTERNS = {
    "Missing Module": [
        re.compile(r"ModuleNotFoundError: No module named '(?P<module_name>[\w.]+)'", re.IGNORECASE),
        re.compile(r"No module named '(?P<module_name>[\w.]+)'", re.IGNORECASE), # For cases where ModuleNotFoundError isn't explicitly stated but the pattern is clear
    ],
    "ImportError / Environment Issue": [
        re.compile(r"ImportError:", re.IGNORECASE), # General ImportError
        re.compile(r"DLL load failed", re.IGNORECASE),
        re.compile(r"cannot open shared object file", re.IGNORECASE),
    ],
    "Tensor Replacement": [
        re.compile(r"FakeTensor", re.IGNORECASE),
        re.compile(r"meta-only", re.IGNORECASE),
    ],
    "CUDA Generator Error": [
        re.compile(r"CUDA Generator", re.IGNORECASE),
    ],
    "NotImplementedError on MPS": [
        re.compile(r"not currently implemented for the MPS", re.IGNORECASE),
        re.compile(r"not implemented for 'mps'", re.IGNORECASE),        
    ],
    "CUDA Internals / Mocking Issues (on MPS)": [
        re.compile(r"AttributeError.*torch[._]C[._]_cuda", re.IGNORECASE), # e.g., _cuda_getDevice
        re.compile(r"Expected a cuda device but got mps", re.IGNORECASE),
        re.compile(r"CUDA-specific call on MPS", re.IGNORECASE), # General note for this category
    ],
    "TorchDevice Core / Patching": [
        re.compile(r"TorchDevice/core/patch\.py", re.IGNORECASE),
        re.compile(r"TorchDevice/core/wrappers\.py", re.IGNORECASE),
        re.compile(r"TorchDevice/core/device\.py", re.IGNORECASE),
        re.compile(r"torch_device_replacement", re.IGNORECASE),
    ],
    "Critical Failure / System Error": [
        re.compile(r"SIGBUS", re.IGNORECASE), re.compile(r"Bus error", re.IGNORECASE),
        re.compile(r"segmentation fault", re.IGNORECASE), re.compile(r"SIGSEGV", re.IGNORECASE),
        re.compile(r"Return Code: -10", re.IGNORECASE), re.compile(r"Return Code: -11", re.IGNORECASE),
        re.compile(r"illegal hardware instruction", re.IGNORECASE),
    ],
    "PyTorch Testing Framework / Assertion": [
        re.compile(r"torch/testing/_comparison\.py", re.IGNORECASE),
        re.compile(r"self[.]assertEqual"), # Common in test failures
        re.compile(r"self[.]assertTrue"),
        re.compile(r"self[.]assertFalse"),
        re.compile(r"self[.]assertRaises"),
    ],
    "PyTorch JIT / Compilation": [
        re.compile(r"CompiledFunction", re.IGNORECASE),
        re.compile(r"torch[._]jit[._]script", re.IGNORECASE),
        re.compile(r"torch[._]jit[._]trace", re.IGNORECASE),
        re.compile(r"Failed to compile", re.IGNORECASE),
        re.compile(r"can't take variable number of arguments or use keyword-only arguments with defaults", re.IGNORECASE),
    ],
    "Tensor Data Type Conversion": [
        re.compile(r"Cannot convert a MPS Tensor to float64", re.IGNORECASE),
        re.compile(r"result type .* can't be cast to the desired output type", re.IGNORECASE),
    ],
    "MPS Backend Specific": [
        re.compile(r"mps (?:runtime|driver) error", re.IGNORECASE),
        re.compile(r"falling back to cpu.*MPS", re.IGNORECASE),
        re.compile(r"not supported on MPS backend", re.IGNORECASE),
        re.compile(r"not implemented for 'mps'", re.IGNORECASE),
        re.compile(r"'_pack_padded_sequence' not implemented for 'mps'", re.IGNORECASE),
        re.compile(r"'_unpack_padded_sequence' not implemented for 'mps'", re.IGNORECASE),
    ],
    "CUDA Backend Specific": [ # For when CUDA is actually the target or expected
        re.compile(r"CUDA error:", re.IGNORECASE),
        re.compile(r"torch[._]cuda[._]OutOfMemoryError", re.IGNORECASE),
        re.compile(r"invalid device function", re.IGNORECASE),
        re.compile(r"cudaSuccess", re.IGNORECASE), # Often in error messages
    ],
    "ImportError / Environment Issue": [
        re.compile(r"ImportError:", re.IGNORECASE),
        re.compile(r"ModuleNotFoundError:", re.IGNORECASE),
    ],
    "General PyTorch Error": [
        re.compile(r"RuntimeError:"), # Catch-all for other PyTorch runtime issues
        re.compile(r"ValueError:"),
        re.compile(r"TypeError:"),
        re.compile(r"IndexError:"),
        re.compile(r"AttributeError:"), # Generic attribute error not caught by others
        re.compile(r"KeyError:"),
    ]
}

def extract_error_details(log_content_str: str, max_snippet_lines: int = 30) -> Dict[str, Any]:
    """
    Parses log content (which could be a Python traceback or general log output)
    to extract key error information, diagnose components, and provide summaries.
    Ensures 'traceback' and 'display_traceback_snippet' are lists of strings.
    'traceback' contains full Python TB if found, otherwise a log snippet.
    """
    extracted_tb_lines, key_error_line_from_tb = _extract_python_traceback(log_content_str)

    text_for_diagnosis = "\n".join(extracted_tb_lines) if extracted_tb_lines else log_content_str
    
    final_key_error_line = key_error_line_from_tb
    if not final_key_error_line and not extracted_tb_lines:
        log_lines_for_error_search = log_content_str.splitlines()[-20:]
        for line in reversed(log_lines_for_error_search):
            if any(err_indicator in line for err_indicator in ["Error:", "failed:", "exception:", "Traceback", "SIGBUS", "AssertionError"]):
                final_key_error_line = line.strip()
                break
        if not final_key_error_line and log_lines_for_error_search:
            final_key_error_line = log_lines_for_error_search[-1].strip()

    diagnosed_component, diagnostic_notes, identified_failure_type = _diagnose_component_and_notes(
        text_for_diagnosis, 
        final_key_error_line, 
        log_content_str
    )

    traceback_list: List[str]
    display_snippet_list: List[str]

    log_lines_full = log_content_str.splitlines()

    if extracted_tb_lines:
        traceback_list = extracted_tb_lines
        if len(extracted_tb_lines) > 15: # For display snippet from Python TB
            display_snippet_list = extracted_tb_lines[:5] + ["..."] + extracted_tb_lines[-10:]
        else:
            display_snippet_list = list(extracted_tb_lines)
    else:
        # No Python TB found, traceback becomes a snippet of the original log
        if len(log_lines_full) > max_snippet_lines:
            traceback_list = ["... (log start truncated) ..."] + log_lines_full[-max_snippet_lines+1:]
        else:
            traceback_list = list(log_lines_full)
        display_snippet_list = list(traceback_list) # Display snippet is the same as raw snippet here

    test_run_command = None
    command_match = re.search(r"^Command: (.*)$", log_content_str, re.MULTILINE)
    if command_match:
        test_run_command = command_match.group(1).strip()

    if extracted_tb_lines:
        # We have a Python traceback
        summary_notes_str = final_key_error_line if final_key_error_line else "Key error line not extracted from traceback."
    else:
        # No Python traceback, this is likely a module-level/runner error
        summary_notes_str = "Module-level issue or runner error. Details derived from log snippet."
        # Check if final_key_error_line is informative before appending
        if final_key_error_line and \
           final_key_error_line.strip() not in ["Error details not automatically summarized.", "======================================================================"] and \
           not final_key_error_line.strip().isdigit(): # Avoid simple numbers that might be RCs
            summary_notes_str += f" Last significant line from log: '{final_key_error_line.strip()}'"

    # Prepend component or failure type if determined and informative
    if diagnosed_component != "Undetermined PyTorch/TorchDevice Component":
        summary_notes_str = f"[{diagnosed_component}] {summary_notes_str}"
    elif identified_failure_type not in ["GenericTraceback", "UnknownRunnerError", "ProcessExit_0"]:
        # Add identified_failure_type if it's specific and not redundant with component
        summary_notes_str = f"[{identified_failure_type}] {summary_notes_str}"

    return {
        "diagnosed_component": diagnosed_component,
        "diagnostic_notes": diagnostic_notes,
        "summary_notes": summary_notes_str,
        "traceback": traceback_list, # Always List[str], (full Python TB or log snippet)
        "display_traceback_snippet": display_snippet_list, # Always List[str], for report display
        "key_error_line": final_key_error_line,
        "identified_failure_type": identified_failure_type,
        "test_run_command": test_run_command,
        "raw_log_for_error_len": len(log_content_str)
    }


def _extract_python_traceback(log_content: str) -> tuple[List[str] | None, str | None]:
    """
    Extracts the most recent Python traceback from a log string.
    Returns a tuple: (full_traceback_lines: List[str] | None, key_error_line_str: str | None).
    """
    lines = log_content.splitlines()
    tb_start_indices = [i for i, line in enumerate(lines) if PYTHON_TRACEBACK_START_RE.match(line)]

    if not tb_start_indices:
        return None, None

    tb_start_index = tb_start_indices[-1]  # Start with the last traceback found
    tb_end_index = -1
    key_error_line = None

    # Find the end of this traceback (the error line)
    for i in range(tb_start_index + 1, len(lines)):
        if PYTHON_TRACEBACK_START_RE.match(lines[i]) or INDIVIDUAL_TEST_LINE_RE.match(lines[i]) or INDIVIDUAL_LOG_SUMMARY_RE.match(lines[i]):
            break
        match_tb_end = PYTHON_TRACEBACK_END_RE.match(lines[i])
        if match_tb_end:
            tb_end_index = i
            key_error_line = match_tb_end.group(0).strip() # Initialize with the full matched error line
            # Try to append subsequent lines if they are part of a multi-line error message
            for j in range(i + 1, min(i + 5, len(lines))): # Look ahead a few lines
                current_line_stripped = lines[j].strip()
                if current_line_stripped and \
                   not PYTHON_TRACEBACK_START_RE.match(lines[j]) and \
                   not INDIVIDUAL_TEST_LINE_RE.match(lines[j]) and \
                   not INDIVIDUAL_LOG_SUMMARY_RE.match(current_line_stripped) and \
                   not PYTHON_TEST_SUMMARY_LINE_RE.match(current_line_stripped) and \
                   not current_line_stripped.startswith("===") and \
                   not current_line_stripped.startswith("---"):
                    key_error_line += "\n" + current_line_stripped
                else:
                    break # Stop if the line doesn't look like a continuation
            break
        tb_end_index = i

    traceback_lines_to_return = None
    if tb_start_index != -1 and tb_end_index != -1: # A valid block was identified
        traceback_lines_to_return = lines[tb_start_index : tb_end_index + 1]
        if not key_error_line: # If PYTHON_TRACEBACK_END_RE didn't match clearly
             key_error_line = lines[tb_end_index].strip()
    elif tb_start_index != -1: # Found start but no clear end, heuristic for block and key error line
        heuristic_end_index = min(len(lines), tb_start_index + 30)
        traceback_lines_to_return = lines[tb_start_index:heuristic_end_index]
        for i_rev in range(len(traceback_lines_to_return) - 1, -1, -1):
            line_content = traceback_lines_to_return[i_rev]
            if ":" in line_content and not line_content.strip().startswith("File "):
                key_error_line = line_content.strip()
                break
        if not key_error_line and traceback_lines_to_return: 
            key_error_line = traceback_lines_to_return[-1].strip()
    
    return traceback_lines_to_return, key_error_line


def _extract_meaningful_snippets_from_lines(log_lines: List[str], max_lines: int = 20) -> Tuple[List[str], str]:
    """
    Extracts a meaningful snippet from log lines when a full traceback is not available.
    This is a simple heuristic that returns the last `max_lines` non-empty lines.
    """
    non_empty_lines = [line.strip() for line in log_lines if line.strip()]
    if not non_empty_lines:
        return [], "no non-empty lines found"
    
    snippet = non_empty_lines[-max_lines:]
    return snippet, f"last {len(snippet)} non-empty lines"


def _diagnose_component_and_notes(text_to_diagnose: str, key_error_line: str | None, full_log_context: str) -> tuple[str, str, str]:
    """
    Diagnoses component and creates notes based on text (preferably a Python traceback or key error line).
    Returns (diagnosed_component, diagnostic_notes, identified_failure_type)
    """
    identified_failure_type = "GenericTraceback" if key_error_line else "UnknownRunnerError"
    diagnosed_component = "Undetermined PyTorch/TorchDevice Component"
    diagnostic_notes = "No specific component pattern matched."

    # 1. Prioritize direct Python exception type extraction from key_error_line
    if key_error_line:
        exception_match = PYTHON_EXCEPTION_RE.match(key_error_line)
        if exception_match:
            specific_exception_type = exception_match.group(1).rstrip(':') # Get like "TypeError" or "torch.cuda.OutOfMemoryError"
            # Make it more readable, e.g., "Python TypeError", "PyTorch CUDA OutOfMemoryError"
            base_readable_exception_name = ""
            if specific_exception_type.startswith("torch."):
                base_readable_exception_name = "PyTorch " + specific_exception_type.split('.')[-1].replace("Error", " Error")
            else:
                base_readable_exception_name = "Python " + specific_exception_type.replace("Error", " Error")

            # Make diagnosed_component more granular by default by including a snippet of the error message
            error_message_part = key_error_line[len(specific_exception_type):].lstrip(':').lstrip()
            snippet_length = 60  # Max length of the snippet from the error message
            error_snippet = (error_message_part[:snippet_length] + '...') if len(error_message_part) > snippet_length else error_message_part
            
            if not error_snippet.strip() or error_snippet == "...":
                # Fallback if snippet is empty or just ellipsis
                diagnosed_component = base_readable_exception_name
            else:
                diagnosed_component = f"{base_readable_exception_name}: {error_snippet}"
            
            identified_failure_type = specific_exception_type # Store the raw exception type (e.g., "ValueError")
            diagnostic_notes = f"Identified Python Exception. Key error: {key_error_line}"

            # --- Sub-pattern matching for more granular diagnosis ---
            # SUB_PATTERNS uses the raw specific_exception_type (e.g., "RuntimeError") as its primary key
            if specific_exception_type in SUB_PATTERNS:
                for sub_component_name, sub_patterns_list in SUB_PATTERNS[specific_exception_type].items():
                    for sub_pattern_item in sub_patterns_list:
                        if (isinstance(sub_pattern_item, re.Pattern) and sub_pattern_item.search(key_error_line)) or \
                           (isinstance(sub_pattern_item, str) and sub_pattern_item in key_error_line):
                            # Override diagnosed_component with the more specific, curated sub-pattern category
                            # Format: "ValueError: Curated Sub-Pattern Name"
                            exception_name_for_display = specific_exception_type.split('.')[-1].replace("Error", " Error") # e.g. ValueError, OutOfMemoryError
                            diagnosed_component = f"{exception_name_for_display.strip()}: {sub_component_name}"
                            diagnostic_notes = f"Identified sub-pattern '{sub_component_name}'. Key error: {key_error_line}"
                            break # Found specific sub-pattern for this sub_component_name
                    if diagnosed_component != f"{base_readable_exception_name}: {error_snippet}" and not (not error_snippet.strip() or error_snippet == "..." and diagnosed_component != base_readable_exception_name) : # If sub-match found and component changed from its initial snippet form
                        break
            # Now, check if this specific error also matches a more contextual COMPONENT_PATTERN
            for component, patterns in COMPONENT_PATTERNS.items():
                for pattern in patterns:
                    match_found_in_pattern = False
                    if isinstance(pattern, re.Pattern) and pattern.search(key_error_line):
                        match_found_in_pattern = True
                    elif isinstance(pattern, str) and pattern in key_error_line:
                        match_found_in_pattern = True
                    
                    if match_found_in_pattern:
                        # If a pattern matches, we can refine the component or add to notes
                        # For now, let's prioritize the specific Python exception as the component
                        # but add the pattern match to the notes for more context.
                        diagnostic_notes += f" Also matched component pattern '{component}' (pattern: '{pattern.pattern if isinstance(pattern, re.Pattern) else pattern}')."
                        # Optionally, you could decide to override diagnosed_component here if the pattern is more specific
                        # e.g., if component == "Missing Module" and specific_exception_type == "ModuleNotFoundError"
                        if component == "Missing Module" and specific_exception_type == "ModuleNotFoundError":
                            m = pattern.search(key_error_line) # Re-match to get group
                            if m and 'module_name' in m.groupdict():
                                diagnosed_component = f"Missing Module: {m.group('module_name')}"
                                diagnostic_notes = f"Module '{m.group('module_name')}' not found. Key error: {key_error_line}"
                            else: # Fallback for missing module if group not found
                                mnf_match_generic = re.search(r"No module named '(?P<mod_name>[\w.]+)'", key_error_line)
                                if mnf_match_generic:
                                    diagnosed_component = f"Missing Module: {mnf_match_generic.group('mod_name')}"
                                    diagnostic_notes = f"Module '{mnf_match_generic.group('mod_name')}' not found. Key error: {key_error_line}"
                        break # Found a pattern match for this component, move to next component or finish
                if match_found_in_pattern and component == "Missing Module": # Ensure we don't overwrite specific missing module diagnosis
                    break
            return diagnosed_component, diagnostic_notes, identified_failure_type

    # 2. If no specific Python exception from key_error_line, or no key_error_line, use COMPONENT_PATTERNS on broader text
    search_order = [key_error_line, text_to_diagnose] if key_error_line else [text_to_diagnose]
    for text_source_idx, text_source in enumerate(search_order):
        if not text_source: continue
        source_name = "key error line" if text_source_idx == 0 and key_error_line else "traceback/log"

        for component, patterns in COMPONENT_PATTERNS.items():
            for pattern in patterns:
                match_found = False
                if isinstance(pattern, re.Pattern) and pattern.search(text_source):
                    match_found = True
                    diagnostic_notes = f"Pattern '{pattern.pattern}' matched in {source_name}."
                elif isinstance(pattern, str) and pattern in text_source:
                    match_found = True
                    diagnostic_notes = f"Keyword '{pattern}' found in {source_name}."
                
                if match_found:
                    diagnosed_component = component
                    # Update identified_failure_type based on component (similar to original logic)
                    if "Dynamo" in component: identified_failure_type = "TorchDynamoError"
                    elif "JIT" in component: identified_failure_type = "JITCompilationError"
                    elif "FakeTensor" in component or "Tensor Replacement" in component: identified_failure_type = "FakeTensorError"
                    elif "Critical Failure" in component: identified_failure_type = "SystemError"
                    elif "CUDA Internals" in component: identified_failure_type = "CUDAInternalError"
                    elif "MPS Backend" in component: identified_failure_type = "MPSBackendError"
                    elif "CUDA Backend" in component: identified_failure_type = "CUDABackendError"
                    elif "ImportError" in component: identified_failure_type = "ImportError"
                    elif component == "Missing Module":
                        identified_failure_type = "MissingModuleError"
                        match = pattern.search(text_source)
                        if match and 'module_name' in match.groupdict():
                            module_name_extracted = match.group('module_name')
                            diagnosed_component = f"Missing Module: {module_name_extracted}"
                            diagnostic_notes = f"Module '{module_name_extracted}' not found. {diagnostic_notes}"
                        elif key_error_line: # Fallback to key_error_line if available
                            mnf_match = re.search(r"No module named '(?P<mod_name>[\w.]+)'", key_error_line)
                            if mnf_match:
                                diagnosed_component = f"Missing Module: {mnf_match.group('mod_name')}"
                                diagnostic_notes = f"Module '{mnf_match.group('mod_name')}' not found. {diagnostic_notes}"
                    
                    if key_error_line:
                        diagnostic_notes += f" Key error: {key_error_line}"
                    elif not key_error_line and text_to_diagnose:
                        for line_content_snippet in text_to_diagnose.splitlines()[-5:]:
                            if any(err_type_snip in line_content_snippet for err_type_snip in ["Error:", "Exception:", "failed:", "warning:"]):
                                diagnostic_notes += f" Context: ...{line_content_snippet[-100:]}"
                                break
                    return diagnosed_component, diagnostic_notes, identified_failure_type

    # 3. Final fallback if still undetermined (mostly for cases without key_error_line and no pattern matches)
    if key_error_line: # This part is less likely to be hit if the first block works well
        diagnostic_notes = f"Key error: {key_error_line}"
        if "AttributeError:" in key_error_line: diagnosed_component = "Python AttributeError"; identified_failure_type = "AttributeError"
        elif "AssertionError:" in key_error_line: diagnosed_component = "Python AssertionError"; identified_failure_type = "AssertionError"
        elif "RuntimeError:" in key_error_line: diagnosed_component = "General RuntimeError"; identified_failure_type = "RuntimeError"
        elif "TypeError:" in key_error_line: diagnosed_component = "General TypeError"; identified_failure_type = "TypeError"
        elif "IndexError:" in key_error_line: diagnosed_component = "General IndexError"; identified_failure_type = "IndexError"
        elif "KeyError:" in key_error_line: diagnosed_component = "General KeyError"; identified_failure_type = "KeyError"
        elif "ValueError:" in key_error_line: diagnosed_component = "General ValueError"; identified_failure_type = "ValueError"
        # ImportError is handled by COMPONENT_PATTERNS
    
    if identified_failure_type == "UnknownRunnerError":
        notes_parts = []
        rc_match = re.search(r"Return Code: (-?\d+)", full_log_context)
        if rc_match:
            rc_str = rc_match.group(1)
            notes_parts.append(f"Process exited with return code {rc_str}.")
            identified_failure_type = f"ProcessExit_{rc_str}"
            if rc_str in ["-10", "-11"]:  # SIGBUS, SIGSEGV
                diagnosed_component = "Critical Failure / System Error"
                identified_failure_type = "SystemError" # More specific than ProcessExit
        else:
            notes_parts.append("No specific Python traceback or return code identified in the log.")

        snippet_lines, extraction_method = _extract_meaningful_snippets_from_lines(full_log_context.splitlines())
        if snippet_lines:
            notes_parts.append(f"Relevant log snippet ({extraction_method}):\n" + "\n".join(snippet_lines))
        else:
            if not rc_match: # Only add this if we don't even have RC info
                notes_parts.append("Additionally, no meaningful snippet could be extracted from the log.")
            # If we have RC info but no snippet, the RC info itself is the primary note from this section.
        
        diagnostic_notes = " ".join(notes_parts).strip()
        if not diagnostic_notes: # Fallback, though unlikely with the above logic
            diagnostic_notes = "Unable to determine specific cause from log. Consult the full log for details."

    return diagnosed_component, diagnostic_notes, identified_failure_type


# Ensure these regexes are defined at the top of your file, or adjust if they already exist
# with these exact definitions.
TRACEBACK_SEPARATOR_LINE = "=" * 70 # Used by unittest
TEST_FAILURE_HEADER_RE = re.compile(r"^(ERROR|FAIL):\s+(?P<test_method>\w+)\s+\((?P<test_class_path>[^)]+)\)")

# Regexes for parsing the *final* summary block of an individual log
INDIVIDUAL_LOG_FINAL_RAN_TESTS_RE = re.compile(r"^Ran\s+(?P<total>\d+)\s+tests? in .*s$")
INDIVIDUAL_LOG_FINAL_OUTCOME_RE = re.compile(r"^(?P<final_status>OK|FAILED)\s*(?:\((?P<details>.*?)\))?$")
INDIVIDUAL_LOG_FINAL_COUNTS_RE = re.compile(r"(?:failures=(?P<failures>\d+))|(?:errors=(?P<errors>\d+))|(?:skipped=(?P<skipped>\d+))|(?:expected failures=(?P<expected_failures>\d+))|(?:unexpected successes=(?P<unexpected_successes>\d+))")

def _parse_final_summary_from_log_end(log_lines: List[str]) -> Dict[str, Any]:
    """
    Parses the final summary block from the end of the log file.
    Extracts total tests, failures, errors, skipped, and overall status.
    Looks at the last ~15 non-empty lines for this information.
    """
    summary_details = {
        "total_tests_from_log": 0,
        "passed_from_log": 0,
        "failures_from_log": 0,
        "errors_from_log": 0,
        "skipped_from_log": 0,
        "raw_status_line_from_log": None,
        "derived_status_from_log": "UNKNOWN"  # Default status
    }

    # Get the last ~15 non-empty lines for parsing
    relevant_lines = []
    for i in range(len(log_lines) - 1, -1, -1):
        line = log_lines[i].strip()
        if line:
            relevant_lines.append(line)
            if len(relevant_lines) >= 15:  # Limit to roughly the last 15 non-empty lines
                break
    relevant_lines.reverse() # Process in chronological order

    found_ran_line = False
    found_outcome_line = False

    for line in relevant_lines:
        # Try to match "Ran X tests..."
        # We check this first but don't set found_ran_line until after outcome, 
        # to ensure outcome details (if on a later line) are captured.
        ran_match = INDIVIDUAL_LOG_FINAL_RAN_TESTS_RE.match(line)
        if ran_match:
            summary_details["total_tests_from_log"] = int(ran_match.group("total"))
            # Mark that we've processed a line that could be the 'Ran X tests' line
            # but don't set found_ran_line = True yet, as the outcome line might be later
            # and we want its details.

        # Try to match "OK/FAILED (details)"
        outcome_match = INDIVIDUAL_LOG_FINAL_OUTCOME_RE.match(line)
        if outcome_match:
            summary_details["raw_status_line_from_log"] = line
            parsed_final_status = outcome_match.group("final_status")
            details_str = outcome_match.group("details")

            # Reset counts before parsing details from this specific outcome line
            # This handles cases where multiple 'FAILED' lines might appear, taking the last one.
            current_failures = 0
            current_errors = 0
            current_skipped = 0

            if details_str:
                for count_match in INDIVIDUAL_LOG_FINAL_COUNTS_RE.finditer(details_str):
                    if count_match.group("failures"):
                        current_failures = int(count_match.group("failures"))
                    if count_match.group("errors"):
                        current_errors = int(count_match.group("errors"))
                    if count_match.group("skipped"):
                        current_skipped = int(count_match.group("skipped"))
            
            summary_details["failures_from_log"] = current_failures
            summary_details["errors_from_log"] = current_errors
            summary_details["skipped_from_log"] = current_skipped
            found_outcome_line = True
            
            # Determine derived_status_from_log based on parsed_final_status and current counts
            if parsed_final_status == "FAILED" or current_failures > 0 or current_errors > 0:
                summary_details["derived_status_from_log"] = "FAILURE"
            elif parsed_final_status == "OK":
                # total_tests_from_log might not be set yet if 'Ran X tests' is later in window
                # We will use the current counts for this decision point.
                # The final passed_from_log calculation later will use total_tests_from_log if found.
                if current_failures == 0 and current_errors == 0:
                    # If total_tests_from_log is known AND all were skipped
                    if summary_details["total_tests_from_log"] > 0 and \
                       summary_details["total_tests_from_log"] == current_skipped:
                        summary_details["derived_status_from_log"] = "SKIPPED_ALL"
                    else:
                        summary_details["derived_status_from_log"] = "SUCCESS"
                else: # OK but has failures/errors (defensive)
                    summary_details["derived_status_from_log"] = "FAILURE"
        
        # After checking both regexes on a line, if INDIVIDUAL_LOG_FINAL_RAN_TESTS_RE matched this line,
        # we can now confidently say we've found the ran line. This ensures that if outcome is on a later line,
        # we still capture it.
        if ran_match: # if ran_match was from the current line
            found_ran_line = True

    # Final calculation for passed_from_log, should happen after all details are gathered
    if found_ran_line: # Ensure total_tests_from_log is from the log
        # Calculate passed tests
        passed_count = summary_details["total_tests_from_log"] - \
                       (summary_details["failures_from_log"] + \
                        summary_details["errors_from_log"] + \
                        summary_details["skipped_from_log"])
        summary_details["passed_from_log"] = max(0, passed_count) # Ensure non-negative

    return summary_details

# --- NEW MAIN PARSING FUNCTION ---
# (This replaces the old parse_individual_log function)
def parse_individual_log_simplified(log_file_path: Path, module_name: str) -> Dict[str, Any]:
    """
    Simplified parser for an individual test log file.
    1. Extracts the final summary from the log's end using _parse_final_summary_from_log_end.
    2. Splits the log by '==================' to find traceback blocks.
    3. For each traceback block, extracts test name, status (FAIL/ERROR), and uses
       extract_error_details to get structured traceback info.
    """
    if not log_file_path.exists():
        return {
            "module_name": module_name,
            "test_cases": [], 
            "individual_log_summary": {"overall_status": "LOG_NOT_FOUND", "total": 0, "passed": 0, "failures": 0, "errors": 0, "skipped": 0},
            "log_reported_summary": {"derived_status_from_log": "LOG_NOT_FOUND"}, # Consistent with successful parse
            "error_message": f"Log file not found: {log_file_path}"
        }

    try:
        log_content = log_file_path.read_text(encoding='utf-8')
        log_lines = log_content.splitlines()
    except Exception as e:
        return {
            "module_name": module_name,
            "test_cases": [],
            "individual_log_summary": {"overall_status": "MALFORMED_LOG", "total": 0, "passed": 0, "failures": 0, "errors": 0, "skipped": 0},
            "log_reported_summary": {"derived_status_from_log": "MALFORMED_LOG"},
            "error_message": f"Could not read log file {log_file_path}: {e}"
        }

    final_summary_from_log = _parse_final_summary_from_log_end(log_lines)
    
    # Split by the official unittest traceback separator line.
    # We add a newline because split removes the delimiter, but the header (ERROR/FAIL line)
    # is on the *next* line.
    traceback_blocks_raw = log_content.split(TRACEBACK_SEPARATOR_LINE + "\n")
    
    parsed_test_failures_and_errors = []

    for i, block_text in enumerate(traceback_blocks_raw):
        if i == 0: # Skip content before the first separator
            continue
        if not block_text.strip():
            continue

        block_lines = block_text.splitlines()
        if not block_lines:
            continue
            
        header_line = block_lines[0].strip() # This should be "ERROR: test (class)" or "FAIL: test (class)"
        
        failure_header_match = TEST_FAILURE_HEADER_RE.match(header_line)
        if failure_header_match:
            status = failure_header_match.group(1).upper() 
            test_method = failure_header_match.group("test_method")
            test_class_path = failure_header_match.group("test_class_path")
            
            # The full traceback for analysis includes the "====...", the "FAIL/ERROR..." header,
            # and the subsequent lines of the traceback body.
            # Reconstruct the block as it appeared for extract_error_details.
            full_traceback_text_for_analysis = TRACEBACK_SEPARATOR_LINE + "\n" + block_text
            
            # Use the existing extract_error_details to process this isolated block
            error_details_dict = extract_error_details(full_traceback_text_for_analysis)
            
            parsed_test_failures_and_errors.append({
                "name": test_method,
                "class_path": test_class_path,
                "status": status, 
                "output": [], # Output is not captured per test in this simplified model,
                              # tracebacks are in error_details.
                "error_details": error_details_dict 
            })

    # Create the 'individual_log_summary' dictionary based on 'final_summary_from_log'
    calculated_summary = {
        "total": final_summary_from_log.get("total_tests_from_log", 0),
        "passed": final_summary_from_log.get("passed_from_log", 0),
        "failures": final_summary_from_log.get("failures_from_log", 0),
        "errors": final_summary_from_log.get("errors_from_log", 0),
        "skipped": final_summary_from_log.get("skipped_from_log", 0),
        "runner_errors": 0, # This concept is deprecated here
        "overall_status": final_summary_from_log.get("derived_status_from_log", "UNKNOWN")
    }
    
    # If the log's final summary indicates FAILURE, but we found no '====' tracebacks,
    # it implies a module-level failure (e.g., import error, syntax error before tests run).
    if calculated_summary["overall_status"] == "FAILURE" and not parsed_test_failures_and_errors:
        # Use the whole log content for error details in this case.
        # extract_error_details will try to find a Python TB or provide a snippet.
        module_error_details_dict = extract_error_details(log_content, max_snippet_lines=50)
        
        parsed_test_failures_and_errors.append({
            "name": f"MODULE_LEVEL_ERROR_{module_name.replace('.', '_')}",
            "class_path": module_name, 
            "status": "ERROR", # Categorize as ERROR
            "output": [],
            "error_details": module_error_details_dict
        })
        # If the log's counts didn't reflect this module error (e.g., said 0 errors/0 failures)
        # we adjust them to ensure the failure is represented.
        if calculated_summary["errors"] == 0 and calculated_summary["failures"] == 0:
             calculated_summary["errors"] = 1
             calculated_summary["total"] = max(1, calculated_summary["total"]) # Ensure total is at least 1
             # Note: This might make passed count negative if total was 0.
             # Re-calculate passed based on new error count if total was initially 0.
             if final_summary_from_log.get("total_tests_from_log", 0) == 0:
                 calculated_summary["passed"] = 0 # Cannot pass if total was 0 and we added an error
             else: # Recalculate passed if total was > 0
                 t = calculated_summary["total"]
                 f = calculated_summary["failures"]
                 e = calculated_summary["errors"] # now at least 1
                 s = calculated_summary["skipped"]
                 calculated_summary["passed"] = max(0, t - (f+e+s))


    return {
        "module_name": module_name,
        "test_cases": parsed_test_failures_and_errors,
        "individual_log_summary": calculated_summary,
        "log_reported_summary": final_summary_from_log, # Keep the direct parse for reference
        "error_message": None # No error in parsing the file itself at this stage
    }


def _finalize_test_entry(test_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finalizes a test entry by processing its individual log file using
    parse_individual_log_simplified and directly using its output.
    """
    log_file_path_str = test_info.get("log_file")
    module_name = test_info.get("module", "UnknownModule")

    if not log_file_path_str:
        test_info["status"] = "ERROR_MISSING_LOG_INFO"
        test_info.setdefault("error_details", {})
        test_info["error_details"]["summary_notes"] = (
            test_info["error_details"].get("summary_notes", "") +
            " Critical error: Log file path missing in test_info. Cannot process."
        ).strip()
        test_info.setdefault("test_cases", [{
            "name": "ProcessingError", "status": "ERROR",
            "output": ["Log file path missing."],
            "error_details": {"summary_notes": "Log file path missing."}
        }])
        test_info.setdefault("individual_log_summary", {
            "total": 0, "passed": 0, "failures": 0, "errors": 1, "skipped": 0,
            "overall_status": "ERROR",
            "log_reported_status_line": "N/A",
            "log_reported_overall_status": "N/A",
            "source_of_summary": "finalize_entry_error_missing_log"
        })
        return test_info

    log_file_path = Path(log_file_path_str)
    parsed_log_data = None # Initialize to ensure it's defined

    try:
        parsed_log_data = parse_individual_log_simplified(log_file_path, module_name)
    except Exception as e:
        error_message = f"Unexpected error calling parse_individual_log_simplified for {log_file_path}: {e}\n{traceback.format_exc()}"
        print(f"CRITICAL ERROR: {error_message}")
        test_info["status"] = "ERROR_UNHANDLED_PARSER_CRASH"
        test_info.setdefault("error_details", {})
        test_info["error_details"]["summary_notes"] = (
            test_info["error_details"].get("summary_notes", "") +
            f" Critical internal error during log parsing: {e}"
        ).strip()
        test_info["error_details"]["parser_crash_traceback"] = error_message
        
        test_info.setdefault("test_cases", [{
            "name": "ParserCrash", "status": "ERROR",
            "output": [f"Parser crashed: {e}"],
            "error_details": {"summary_notes": f"Parser crashed: {e}"}
        }])
        test_info.setdefault("individual_log_summary", {
            "total": 0, "passed": 0, "failures": 0, "errors": 1, "skipped": 0,
            "overall_status": "ERROR_UNHANDLED_PARSER_CRASH",
            "log_reported_status_line": "Parser Crash",
            "log_reported_overall_status": "ERROR_UNHANDLED_PARSER_CRASH",
            "source_of_summary": "finalize_entry_parser_crash"
        })
        return test_info

    # Directly use the results from parse_individual_log_simplified
    test_info["test_cases"] = parsed_log_data.get("test_cases", [])
    
    # Get the summary parsed directly from the individual log's footer
    log_footer_summary = parsed_log_data.get("log_reported_summary")
    
    # Get the summary calculated by parse_individual_log_simplified based on found test cases/tracebacks
    calculated_parser_summary = parsed_log_data.get("individual_log_summary")

    batch_status_str = str(test_info.get("status_from_summary", "UNKNOWN")).upper()

    final_module_status = "UNKNOWN_MODULE_STATUS"
    final_individual_summary_for_json = {}
    source_of_summary_info = "unknown_source"

    if log_footer_summary:
        # Use the direct parse of the log's own summary as the primary source
        final_individual_summary_for_json = {
            "total": log_footer_summary.get("total_tests_from_log", 0),
            "passed": log_footer_summary.get("passed_from_log", 0),
            "failures": log_footer_summary.get("failures_from_log", 0),
            "errors": log_footer_summary.get("errors_from_log", 0),
            "skipped": log_footer_summary.get("skipped_from_log", 0),
            "runner_errors": 0, # This concept is deprecated for this summary
            "overall_status": log_footer_summary.get("derived_status_from_log", "UNKNOWN_LOG_FOOTER_STATUS"),
            "raw_log_status_line": log_footer_summary.get("raw_status_line_from_log", "N/A")
            # "source_of_summary" will be set below
        }
        final_module_status = final_individual_summary_for_json["overall_status"]
        source_of_summary_info = "log_footer"

        # If log footer says SUCCESS, but batch says FAILURE, then module is FAILURE
        if final_module_status == "SUCCESS" and ("FAIL" in batch_status_str or "ERROR" in batch_status_str) :
            final_module_status = batch_status_str if batch_status_str != "UNKNOWN" else "FAILURE_AS_PER_BATCH"
            source_of_summary_info = "log_footer_overridden_by_batch"
        
        final_individual_summary_for_json["overall_status"] = final_module_status # Reflect potential override
        final_individual_summary_for_json["source_of_summary"] = source_of_summary_info


    elif calculated_parser_summary: # Fallback if log_footer_summary was missing
        warning_msg = f"Warning: Log footer summary missing for {log_file_path}. Using calculated summary from parser."
        print(warning_msg)
        final_individual_summary_for_json = calculated_parser_summary # This already has a 'source_of_summary'
        final_module_status = calculated_parser_summary.get("overall_status", "UNKNOWN_CALCULATED_STATUS")
        source_of_summary_info = calculated_parser_summary.get("source_of_summary", "calculated_by_parser")
        
        # Still check against batch status
        if final_module_status == "SUCCESS" and ("FAIL" in batch_status_str or "ERROR" in batch_status_str):
            final_module_status = batch_status_str if batch_status_str != "UNKNOWN" else "FAILURE_AS_PER_BATCH"
            source_of_summary_info = f"{source_of_summary_info}_overridden_by_batch"

        final_individual_summary_for_json["overall_status"] = final_module_status # Reflect potential override
        final_individual_summary_for_json["source_of_summary"] = source_of_summary_info
            
    else: # Should not happen if parse_individual_log_simplified guarantees a summary
        critical_msg = f"CRITICAL: No summary available (neither log_footer nor calculated) for {log_file_path}"
        print(critical_msg)
        final_individual_summary_for_json = {
            "total": 0, "passed": 0, "failures": 0, "errors": 1, "skipped": 0, "runner_errors": 0,
            "overall_status": "ERROR_NO_SUMMARY_FOUND",
            "raw_log_status_line": "N/A",
            "source_of_summary": "error_no_summary_found_in_finalize"
        }
        final_module_status = "ERROR_NO_SUMMARY_FOUND"

    test_info["individual_log_summary"] = final_individual_summary_for_json
    test_info["status"] = final_module_status

    # Propagate any parser-level error messages (e.g., file not found, parse crash handled by simplified_parser)
    if parsed_log_data.get("error_message"):
        test_info.setdefault("error_details", {})
        test_info["error_details"]["parser_internal_error_message"] = parsed_log_data["error_message"]
        # If the parser itself reported an error (like file not found),
        # this status should take precedence if it's more severe
        # than what was derived from log content (or lack thereof).
        parser_reported_status = parsed_log_data.get("individual_log_summary", {}).get("overall_status", "")
        if "ERROR" in parser_reported_status.upper() or "CRASH" in parser_reported_status.upper():
            if final_module_status != parser_reported_status:
                 print(f"Info: Overriding module status for {log_file_path} from '{final_module_status}' to '{parser_reported_status}' due to parser-level error message.")
                 test_info["status"] = parser_reported_status
                 test_info["individual_log_summary"]["overall_status"] = parser_reported_status # also update the summary status
                 test_info["individual_log_summary"]["source_of_summary"] = f"{test_info['individual_log_summary'].get('source_of_summary', 'unknown')}_overridden_by_parser_error"

    # Ensure test_cases is always a list, even if empty. (Should be redundant due to earlier assignment)
    if not isinstance(test_info.get("test_cases"), list):
        test_info["test_cases"] = [] 

    return test_info


def parse_summary_log(summary_log_path: Path) -> List[Dict[str, Any]]:
    """Parses batch summary log files (e.g., from run_transformer_tests.py)
    where module info is spread across multiple lines."""
    parsed_results = []
    
    if not summary_log_path.is_file():
        print(f"Error: Summary log file not found at {summary_log_path}")
        return [{"error": "Summary log file not found", "path": str(summary_log_path), "module": f"ErrorProcessing_{summary_log_path.name}"}]

    lines = []
    try:
        with open(summary_log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading summary log file {summary_log_path}: {e}")
        return [{"error": f"Could not read summary log file: {e}", "path": str(summary_log_path), "module": f"ErrorProcessing_{summary_log_path.name}"}]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue

        finished_match = BATCH_FINISHED_TEST_RE.match(line)
        if finished_match:
            current_test_info: Dict[str, Any] = {
                "module": finished_match.group("module_name").strip(),
                "status_from_summary": finished_match.group("status_text").strip(),
                "module_status_from_summary": finished_match.group("status_text").strip(),
                "return_code": finished_match.group("return_code").strip(),
                "duration": finished_match.group("duration").strip(),
            }

            raw_log_file_line = ""
            actual_log_file_line_for_warning = "<end of file>"
            if i < len(lines):
                raw_log_file_line = lines[i].rstrip('\n') # Use rstrip to remove only newline for matching
                actual_log_file_line_for_warning = lines[i].strip()
                i += 1 # Advance index after reading
            
            log_file_match = BATCH_LOG_FILE_RE.match(raw_log_file_line)
            if not log_file_match:
                print(f"Warning: Expected log file path after '{line}' in {summary_log_path}, but found '{actual_log_file_line_for_warning}'. Skipping module.")
                continue

            current_test_info["log_file"] = log_file_match.group("log_file_path").strip() # Strip the captured group

            raw_test_command_line = ""
            actual_test_command_line_for_warning = "<end of file>"
            if i < len(lines):
                raw_test_command_line = lines[i].rstrip('\n') # Use rstrip
                actual_test_command_line_for_warning = lines[i].strip()
                i += 1 # Advance index after reading
            
            test_command_match = BATCH_TEST_COMMAND_RE.match(raw_test_command_line)
            if not test_command_match:
                # Use the stripped version of the previous line (log_file_line) in the warning for context
                print(f"Warning: Expected test command after '{actual_log_file_line_for_warning}' in {summary_log_path}, but found '{actual_test_command_line_for_warning}'. Skipping module.")
                continue
            
            current_test_info["test_command"] = test_command_match.group("test_command_str").strip() # Strip the captured group

            parts = current_test_info["module"].split('.')
            current_test_info["test_file_name"] = (parts[-1] + ".py") if parts else "unknown.py"
            current_test_info["test_script_path"] = "/".join(parts) + ".py" if parts else "unknown.py"
            
            if parts and parts[0] == "tests" and len(parts) > 1:
                main_comp_parts = [p.capitalize() for p in parts[1:-1]]
                test_name_part = parts[-1].replace("test_", "").replace("_", " ").title()
                if main_comp_parts:
                    current_test_info["component"] = f"{' '.join(main_comp_parts)} - {test_name_part}"
                else:
                    current_test_info["component"] = test_name_part
            elif parts:
                component_prefix = parts[0] if len(parts) == 1 else (parts[-2] if len(parts) > 1 else "Unknown")
                test_name_part = parts[-1].replace("test_", "").replace("_", " ").title()
                current_test_info["component"] = f"{component_prefix.capitalize()} - {test_name_part}"
            else:
                current_test_info["component"] = "Unknown"

            finalized_entry = _finalize_test_entry(current_test_info)
            if finalized_entry:
                parsed_results.append(finalized_entry)
            else:
                current_test_info["error_details"] = current_test_info.get("error_details", {})
                current_test_info["error_details"]["summary_notes"] = (
                    current_test_info["error_details"].get("summary_notes", "") +
                    " Failed to finalize entry; individual log might be missing or empty."
                ).strip()
                current_test_info["status"] = "ERROR_IN_PARSING_INDIVIDUAL_LOG"
                current_test_info.setdefault("test_cases", [{"name": "LogProcessingError", "status": "ERROR", "output": ["Individual log could not be processed."], "error_details": current_test_info["error_details"]}])
                current_test_info.setdefault("individual_log_summary", {"total": 1, "passed": 0, "failures": 1, "skipped": 0, "errors": 1, "runner_errors": 0, "overall_status": "ERROR"})
                parsed_results.append(current_test_info)

    if not parsed_results:
        print(f"Warning: No modules successfully parsed from {summary_log_path}")
        return [{"error": "No modules parsed", "path": str(summary_log_path), "module": f"NoModulesParsed_{summary_log_path.name}"}]

    return parsed_results


def main():
    parser = argparse.ArgumentParser(description="Parse one or more Transformers test summary log files into a single structured JSON file.")
    parser.add_argument("summary_log_files", type=Path, nargs='+', help="Path(s) to the summary log file(s) to parse.")
    parser.add_argument("--output-dir", type=Path, default=Path("test_automation/data/summary_output"), help="Directory to save the output JSON file.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_parsed_data = []
    valid_files_processed = 0

    files_to_process = []
    for path_arg in args.summary_log_files:
        if path_arg.is_dir():
            # Find summary logs, which are directly in the logs dir, not in subdirs
            print(f"Searching for summary logs in directory: {path_arg}")
            found_files = list(path_arg.glob('*.log'))
            if not found_files:
                print(f"Warning: No summary log files (*.log) found in {path_arg}.")
            files_to_process.extend(found_files)
        elif path_arg.is_file():
            files_to_process.append(path_arg)
        else:
            print(f"Warning: Provided path is not a valid file or directory: {path_arg}, skipping.")

    if not files_to_process:
        print("Error: No log files found to process.")
        return

    for summary_log_file in files_to_process:
        print(f"Processing: {summary_log_file}...")
        parsed_data_list = parse_summary_log(summary_log_file)
        if parsed_data_list:
            all_parsed_data.extend(parsed_data_list)
            valid_files_processed += 1
        else:
            print(f"Warning: No data parsed from {summary_log_file}.")

    if not all_parsed_data:
        print("No data parsed from any of the provided log files. No output JSON will be generated.")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = output_dir / f"collated_transformers_test_summary_{timestamp}.json"
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(all_parsed_data, f, indent=4)
        print(f"Successfully parsed {valid_files_processed} log file(s) and saved aggregated JSON output to: {output_filename}")
    except IOError as e:
        print(f"Error: Could not write JSON output to {output_filename}: {e}")

if __name__ == "__main__":
    main()
