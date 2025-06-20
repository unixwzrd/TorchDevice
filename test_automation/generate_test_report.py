import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import os
import re

def dot_to_path(dot_notation_module: str) -> str:
    """Converts a module path from dot notation to OS-specific path notation with .py extension."""
    return os.path.join(*dot_notation_module.split('.')) + '.py'

def to_anchor_link(text: str) -> str:
    """Converts a string to a GitHub-style anchor link."""
    # Replace newlines and multiple spaces with a single hyphen, then other chars
    processed_text = text.lower().replace('\n', ' ').replace('.', '').replace('/', '').replace(':', '').replace('(', '').replace(')', '')
    # Consolidate multiple spaces/hyphens that might result from replacements
    processed_text = re.sub(r'\s+', '-', processed_text)
    processed_text = re.sub(r'-+', '-', processed_text) # Consolidate multiple hyphens
    return processed_text.strip('-')

def generate_markdown_report(json_files: list[Path]) -> str:
    """Generates a Markdown test report with an executive summary, failure analysis, and navigation."""
    all_module_dictionaries = []

    # --- Data Loading ---
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                loaded_content = json.load(f)
                if isinstance(loaded_content, dict):
                    all_module_dictionaries.append(loaded_content)
                elif isinstance(loaded_content, list):
                    all_module_dictionaries.extend([item for item in loaded_content if isinstance(item, dict)])
        except Exception as e:
            print(f"Error loading or processing {json_file.name}: {e}")
            continue

    if not all_module_dictionaries:
        return f"# PyTorch Transformers Test Suite - TorchDevice Integration Report\n\n**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nNo valid JSON data found to generate a report.\n"

    # --- Data Aggregation ---
    total_modules_processed = len(all_module_dictionaries)
    modules_passed_count = 0
    modules_failed_count = 0

    grand_total_tests_from_summaries = 0
    grand_total_passed_from_summaries = 0
    grand_total_failed_or_error_from_summaries = 0
    grand_total_skipped_from_summaries = 0
    
    failures_by_diagnosed_component = defaultdict(lambda: {"count": 0, "failures": []})
    passed_modules_details = []

    for module_data in all_module_dictionaries:
        module_name_dot = module_data.get("module", "Unknown.Module")
        module_overall_status = module_data.get("status_from_summary", module_data.get("status", "UNKNOWN")).upper()

        individual_summary = module_data.get("individual_log_summary", {})
        grand_total_tests_from_summaries += individual_summary.get("total", 0)
        grand_total_passed_from_summaries += individual_summary.get("passed", 0)
        grand_total_failed_or_error_from_summaries += individual_summary.get("failures", 0) + individual_summary.get("errors", 0)
        grand_total_skipped_from_summaries += individual_summary.get("skipped", 0)

        module_has_actual_test_failures = False
        test_cases_data = module_data.get("test_cases", [])
        if not isinstance(test_cases_data, list):
            test_cases_data = []

        for test_case in test_cases_data:
            test_status = test_case.get("status", "UNKNOWN").upper()
            
            if test_status in ["FAIL", "FAILURE", "ERROR"]:
                module_has_actual_test_failures = True
                error_details = test_case.get("error_details", {})
                diagnosed_component = error_details.get("diagnosed_component", "Unknown Failure Component")
                
                failure_entry = {
                    "module_name": module_name_dot,
                    "module_path": dot_to_path(module_name_dot),
                    "name": test_case.get("name", "UnknownTest"),
                    "status": test_status,
                    "diagnosed_component": diagnosed_component,
                    "key_error_line": error_details.get("key_error_line", "N/A"),
                    "diagnostic_notes": error_details.get("diagnostic_notes", "N/A"),
                    "traceback_snippet": error_details.get("display_traceback_snippet", []),
                    "test_command": module_data.get("test_command", "N/A"),
                    "module_duration": module_data.get("duration", "N/A")
                }
                failures_by_diagnosed_component[diagnosed_component]["count"] += 1
                failures_by_diagnosed_component[diagnosed_component]["failures"].append(failure_entry)

        if module_has_actual_test_failures:
            modules_failed_count += 1
        elif module_overall_status == "SUCCESS":
            modules_passed_count += 1
            passed_modules_details.append({
                "path": dot_to_path(module_name_dot),
                "summary_counts": individual_summary
            })

    # --- Report Generation ---
    report_lines = [
        f"# PyTorch Transformers Test Suite - TorchDevice Integration Report",
        f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ""
    ]

    # I. Executive Summary
    report_lines.extend([
        "## I. Executive Summary",
        "",
        "This report summarizes the results of the PyTorch Transformers test suite run with TorchDevice integration.",
        ""
    ])
    
    # Summary Table
    report_lines.append("| Category                          | Count      | Percentage |")
    report_lines.append("|:----------------------------------|:-----------|:-----------|")
    total_for_perc = grand_total_tests_from_summaries if grand_total_tests_from_summaries > 0 else 1
    total_modules_for_perc = total_modules_processed if total_modules_processed > 0 else 1
    report_lines.append(f"| Modules Processed                 | {total_modules_processed:<10} | N/A        |")
    report_lines.append(f"| Modules Passed                    | {modules_passed_count:<10} | {modules_passed_count/total_modules_for_perc:.1%} |")
    report_lines.append(f"| Modules with Failures/Errors      | {modules_failed_count:<10} | {modules_failed_count/total_modules_for_perc:.1%} |")
    report_lines.append(f"| Passed Tests                      | {grand_total_passed_from_summaries:<10} | {grand_total_passed_from_summaries/total_for_perc:.1%} |")
    report_lines.append(f"| Failed/Errored Tests              | {grand_total_failed_or_error_from_summaries:<10} | {grand_total_failed_or_error_from_summaries/total_for_perc:.1%} |")
    report_lines.append(f"| Skipped Tests                     | {grand_total_skipped_from_summaries:<10} | {grand_total_skipped_from_summaries/total_for_perc:.1%} |")
    report_lines.append(f"| **Total Tests (from log summaries)** | **{grand_total_tests_from_summaries}** | **100.0%**   |")
    report_lines.append("")

    # II. Failure Analysis by Component
    report_lines.extend([
        "## II. Failure Analysis by Component",
        "",
        "This section categorizes test failures by the diagnosed root cause or component. This helps prioritize debugging efforts.",
        ""
    ])
    
    if failures_by_diagnosed_component:
        report_lines.extend([
            "| Diagnosed Component/Failure Type | Failure Count | Jump to Details |",
            "|:---------------------------------|:--------------|:----------------|"
        ])
        sorted_failures_by_component = sorted(failures_by_diagnosed_component.items(), key=lambda item: item[1]['count'], reverse=True)
        for component_key, data in sorted_failures_by_component:
            anchor_link = f"#{to_anchor_link(component_key)}"
            # Sanitize component_key for display in the table
            component_display_name = component_key.replace('\n', ' ').strip()
            if len(component_display_name) > 100: # Truncate if too long for table cell
                component_display_name = component_display_name[:97] + "..."
            report_lines.append(f"| {component_display_name} | {data['count']} | [Link]({anchor_link}) |")
    else:
        report_lines.append("No component-specific failures were diagnosed.")
    report_lines.append("\n---\n")

    # III. Detailed Failure Report
    report_lines.extend([
        "## III. Detailed Failure Report",
        "",
        "This section provides a detailed breakdown of each test that failed or produced an error, grouped by component.",
        ""
    ])

    if not failures_by_diagnosed_component:
        report_lines.append("No failures or errors to report in detail.")
    else:
        failure_index = 0
        sorted_components_for_details = sorted(failures_by_diagnosed_component.items(), key=lambda item: item[1]['count'], reverse=True)
        for component, data in sorted_components_for_details:
            anchor = to_anchor_link(component) # Anchor should use original component name
            # Sanitize component for display as a header in Section III
            header_display_name = component.replace('\n', ' ').strip()
            report_lines.append(f'<a id="{anchor}"></a>\n### {header_display_name}\n')
            
            for failure in data['failures']:
                failure_index += 1
                report_lines.append(f"#### {failure_index}. Failure in `{failure['name']}` (Module: `{failure['module_name']}`)\n")
                # Prepend the path to the transformers test directory for correct relative linking from the report's location
                link_path = os.path.join('../../test_projects/transformers', failure['module_path'])
                report_lines.append(f"- **Test File Path:** [`{failure['module_path']}`]({link_path})")
                report_lines.append(f"- **Module Duration:** `{failure['module_duration']}`")
                report_lines.append(f"- **Status:** `{failure['status']}`")
                report_lines.append(f"- **Key Error Line:** `{failure['key_error_line']}`")
                report_lines.append(f"- **Test Run Command:** `{failure['test_command']}`")

                diagnostic_notes_content = failure.get('diagnostic_notes')
                if diagnostic_notes_content and diagnostic_notes_content != 'N/A':
                    report_lines.append(f"- **Diagnostic Details:**")
                    report_lines.append(f"  ```txt")
                    report_lines.append(f"  {diagnostic_notes_content}")
                    report_lines.append(f"  ```")
                
                report_lines.append("") # Add a blank line for spacing before the next block

                snippet_lines = failure.get('traceback_snippet', [])
                if snippet_lines:
                    report_lines.append(f"- **Traceback / Log Snippet:**")
                    report_lines.append(f"  ```python")
                    indented_snippet = "\n".join([f"  {line}" for line in snippet_lines])
                    report_lines.append(indented_snippet)
                    report_lines.append(f"  ```")
                
                report_lines.append("") # Add a blank line for spacing after the entry
    report_lines.append("\n---\n")

    # IV. Modules That Passed All Their Tests
    report_lines.append("## IV. Modules That Passed All Their Tests\n")
    if not passed_modules_details:
        report_lines.append("No modules passed all their tests according to parsed results.")
    else:
        report_lines.append(f"<details>")
        report_lines.append(f"<summary>Click to expand/collapse list of passed modules ({len(passed_modules_details)} modules)</summary>\n")
        for module_info in sorted(passed_modules_details, key=lambda x: x['path']):
            isum = module_info['summary_counts']
            report_lines.append(f"- `{module_info['path']}` (Parsed Summary - Passed: {isum.get('passed',0)}, Skipped: {isum.get('skipped',0)}, Total: {isum.get('total',0)})")
        report_lines.append("</details>")
    report_lines.append("\n---")

    return "\n".join(report_lines)

def main():
    parser = argparse.ArgumentParser(description="Generate a Markdown test report from JSON summary files produced by parse_transformer_log_summary.py.")
    parser.add_argument("json_inputs", nargs='+', help="Path to one or more JSON summary files or directories containing them (e.g., test_automation/data/summary_output/).")
    parser.add_argument("-o", "--output", default="test_automation/reports/transformers_test_report.md", help="Path to save the Markdown report. Defaults to test_automation/reports/transformers_test_report.md.")
    args = parser.parse_args()

    json_files = []
    for item_path_str in args.json_inputs:
        item_path = Path(item_path_str)
        if item_path.is_dir():
            json_files.extend(list(item_path.rglob("*.json"))) # Recursive glob for subdirectories
        elif item_path.is_file() and item_path.suffix == '.json':
            json_files.append(item_path)
        else:
            print(f"Warning: '{item_path_str}' is not a valid JSON file or directory. Skipping.", file=sys.stderr)

    if not json_files:
        print("No JSON files found to process.", file=sys.stderr)
        return
    
    json_files = sorted(list(set(json_files))) # Deduplicate and sort for consistent processing order

    print(f"Processing {len(json_files)} JSON file(s):", file=sys.stderr)
    for f_path in json_files: print(f"  - {f_path}", file=sys.stderr)
        
    markdown_content = generate_markdown_report(json_files)

    output_path_base = Path(args.output)
    output_dir = output_path_base.parent
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_filename = f"{output_path_base.stem}_{timestamp}{output_path_base.suffix}"
    final_output_path = output_dir / output_filename

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        with open(final_output_path, 'w') as f:
            f.write(markdown_content)
        print(f"\nReport successfully saved to {final_output_path}", file=sys.stderr)
    except Exception as e:
        print(f"\nError saving report to {final_output_path}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
