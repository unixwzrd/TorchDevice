# PyTorch Gap Analysis Strategy for TorchDevice

This document outlines a strategy for systematically comparing TorchDevice's patched functionalities against the official PyTorch source code to identify potential gaps, ensure comprehensive coverage, and maintain compatibility.

## 1. Preparation & Version Confirmation

* **Confirm Target PyTorch Version**:
  * Identify the specific PyTorch version (e.g., release tag `v2.1.0`, branch `release/2.1`, or commit hash) that TorchDevice aims to be compatible with. This should align with the version of the PyTorch source code linked into the project.
* **Establish Codebase Mapping**:
  * Create a preliminary mapping of TorchDevice modules (e.g., `TorchDevice.core.tensors`, `TorchDevice.ops.nn.attention`) to their corresponding modules in the PyTorch source (e.g., `torch.Tensor`, `torch.nn.functional`).

## 2. Scope Definition & Prioritization

* **Identify Key PyTorch Namespaces**:
  * Focus on namespaces critical for CUDA-to-MPS redirection and common PyTorch usage. Examples:
    * `torch` (top-level functions, tensor creation)
    * `torch.Tensor` (methods)
    * `torch.nn.Module` (methods)
    * `torch.cuda` (especially functions related to device management, streams, events, memory)
    * `torch.mps` (for understanding MPS backend specifics)
    * `torch.nn.functional`
    * `torch.nn.modules.*` (common layers)
    * `torch.optim`
    * `torch.autograd`
    * `torch.distributed` (if within scope for TorchDevice)
* **Prioritize Based on Usage & Impact**:
  * Prioritize functions and modules that are most frequently used in typical deep learning workflows.
  * Consider areas where CUDA-specific behavior is prominent and redirection is crucial.

## 3. Comparative Analysis Methodology

* **API Surface Review**:
  * For each prioritized PyTorch module/class, list all public functions, methods, and attributes (often indicated by `__all__` or lack of a leading underscore).
  * Compare this list against what TorchDevice currently patches, wraps, or mocks.
* **Signature Matching**:
  * For every function/method TorchDevice replaces or wraps, meticulously verify that its signature (parameter names, order, types, default values, `*args`, `**kwargs`) exactly matches the original PyTorch API for the target version.
* **Behavioral Deep Dive (Original PyTorch Code)**:
  * For each relevant PyTorch function/method, study its source code to understand:
    * Device-specific logic paths (CUDA, MPS, CPU).
    * Interactions with other PyTorch internal components.
    * Expected side effects and return values.
    * Error handling mechanisms and types of exceptions raised.
    * Use of global state or configurations.
* **Behavioral Deep Dive (TorchDevice Code)**:
  * Review TorchDevice's corresponding replacement/wrapper to ensure:
    * Correct redirection logic is applied (e.g., CUDA -> MPS, device argument handling).
    * All relevant parameters are passed through or handled appropriately.
    * Equivalent behavior is achieved, or deviations are intentional and documented.
    * Error conditions are handled gracefully, possibly mimicking PyTorch's error messages.
* **Identifying Unpatched Functionality**:
  * Note any public PyTorch APIs within the scoped modules that TorchDevice does not currently address but might be relevant for device redirection or could lead to unexpected behavior if called directly without TorchDevice's intervention.
* **Reviewing `__all__` and Documentation**:
  * Use `__all__` attributes in PyTorch modules and official PyTorch documentation as primary guides to what constitutes the public API.

## 4. Gap Identification & Documentation

* **Signature Mismatches**: Document any discrepancies found in function/method signatures.
* **Unpatched APIs**: List public PyTorch APIs within the defined scope that are not currently handled by TorchDevice and assess their relevance.
* **Behavioral Divergences**: Note any differences in behavior between the original PyTorch function and TorchDevice's version, distinguishing between intentional (and documented) differences and potential bugs.
* **MPS Limitations**: Identify functionalities where the PyTorch MPS backend itself has limitations compared to CUDA. TorchDevice cannot overcome these, but they should be documented for users.
* **Areas for Refinement**: Record parts of TorchDevice's redirection logic that could be improved for accuracy or robustness based on insights from the PyTorch source.
* **Maintain a Gap Log**: Use a structured format (e.g., a markdown table, spreadsheet, or issue tracker) to log identified gaps, including:
  * PyTorch Module/Function
  * TorchDevice Equivalent (if any)
  * Description of Gap/Issue
  * Severity/Priority
  * Proposed Action
  * Status

## 5. Action Plan & Test Case Development

* **Prioritize Fixes/Enhancements**: Based on the gap log, prioritize which items to address first.
* **Develop/Update Patches**: Implement new patches or modify existing ones in TorchDevice to address the identified gaps.
* **Create Targeted Unit Tests**: For each gap addressed or new functionality covered:
  * Write specific unit tests that verify the correct behavior of TorchDevice's patched version.
  * Ensure tests cover various device scenarios (CUDA calls redirected to MPS, direct MPS calls, CPU fallback, CPU override).
  * Aim for tests that would have failed if the gap was present.

## 6. Iteration and Ongoing Maintenance

* **Regular Reviews**: Periodically re-evaluate TorchDevice against new PyTorch releases to catch new APIs or changes to existing ones.
* **Integrate with CI**: Ensure comprehensive tests run automatically to catch regressions.
* **Community Feedback**: Actively monitor user feedback and bug reports, as these often highlight real-world gaps.

By following this strategy, TorchDevice can systematically improve its compatibility and transparency with the target PyTorch version.
