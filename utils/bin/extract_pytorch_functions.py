#!/usr/bin/env python3
"""
PyTorch Function Extractor
==========================

This script extracts all PyTorch functions from the installed environment
and saves them to a JSON database using Pydantic schema for type safety.

This is the data extraction phase - separate from report generation.
"""

import importlib
import inspect
import json
import os
import pkgutil
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

# Add schemas directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'schemas'))

try:
    from function_catalog import (
        DeviceSupport,
        FunctionCatalog,
        ImplementationStatus,
        MLXMapping,
        PyTorchFunction,
        VersionInfo,
        save_catalog_to_json,
    )
except ImportError as e:
    print(f"Error importing function_catalog: {e}")
    print("Make sure the schemas directory exists and contains function_catalog.py")
    sys.exit(1)

# Configurable parameters
JSON_OUTPUT = "utils/data/comprehensive_function_catalog.json"

# Function categorization mapping - organized by PyTorch's actual structure
MODULE_CATEGORIES = {
    # Core PyTorch modules
    "torch": "core_torch",
    "torch.nn": "neural_network",
    "torch.nn.functional": "neural_network_functional",
    "torch.optim": "optimization",
    "torch.autograd": "autograd",
    "torch.distributions": "distributions",
    "torch.linalg": "linear_algebra",
    "torch.fft": "signal_processing",
    "torch.sparse": "sparse_operations",
    "torch.quantization": "quantization",
    "torch.jit": "jit_compilation",
    "torch.utils": "utilities",
    "torch.cuda": "cuda_operations",
    "torch.mps": "mps_operations",
    "torch.cpu": "cpu_operations",
    "torch.xpu": "xpu_operations",
    "torch.distributed": "distributed_computing",
    "torch.multiprocessing": "multiprocessing",
    "torch.profiler": "profiling",
    "torch.package": "packaging",
    "torch.hub": "model_hub",
    "torch.fx": "graph_transformations",
    "torch.futures": "asynchronous",
    "torch.func": "functional_programming",
    "torch.export": "model_export",
    "torch.onnx": "onnx_export",
    "torch.signal": "signal_processing",
    "torch.special": "special_functions",
    "torch.storage": "storage_management",
    "torch.testing": "testing_utilities",
    "torch.types": "type_system",
    "torch.overrides": "customization",
    "torch.random": "random_generation",
    "torch.masked": "masked_operations",
    "torch.monitor": "monitoring",
    "torch.nested": "nested_tensors",
    "torch.return_types": "return_types",
    "torch.windows": "window_functions",
    "torch.backends": "backend_management",
    "torch.compiler": "compilation",
    "torch.accelerator": "accelerator_support",
    "torch.amp": "automatic_mixed_precision",
    "torch.ao": "ao_quantization",
    "torch.contrib": "contributions",
    "torch.mtia": "mtia_operations",
}


def categorize_function(function_name: str, module_name: str) -> str:
    """Categorize a function based on its PyTorch module structure."""
    # Use module-based categorization following PyTorch's organization
    if module_name in MODULE_CATEGORIES:
        return MODULE_CATEGORIES[module_name]
    
    # Handle submodules by finding the best matching module pattern
    for module_pattern, category in MODULE_CATEGORIES.items():
        if module_name.startswith(module_pattern):
            return category
    
    # Default category for unknown modules
    return "other"


def discover_torch_modules() -> List[str]:
    """Discover all available PyTorch modules."""
    modules = []
    
    try:
        import torch
        torch_dir = os.path.dirname(torch.__file__)
        
        # Get all modules in torch directory
        for _, name, is_pkg in pkgutil.iter_modules([torch_dir]):
            if is_pkg:
                module_name = f"torch.{name}"
                try:
                    # Try to import the module to verify it's accessible
                    importlib.import_module(module_name)
                    modules.append(module_name)
                except ImportError:
                    continue
        
        # Add the main torch module
        modules.append("torch")
        
    except ImportError:
        print("Warning: PyTorch not found. Cannot discover modules.")
    
    return sorted(modules)


def discover_mlx_modules() -> List[str]:
    """Discover all available MLX modules."""
    modules = []
    
    try:
        import mlx
        mlx_dir = os.path.dirname(mlx.__file__)
        
        # Get all modules in mlx directory
        for _, name, is_pkg in pkgutil.iter_modules([mlx_dir]):
            if is_pkg:
                module_name = f"mlx.{name}"
                try:
                    # Try to import the module to verify it's accessible
                    importlib.import_module(module_name)
                    modules.append(module_name)
                except ImportError:
                    continue
        
        # Add the main mlx module
        modules.append("mlx")
        
    except ImportError:
        print("Warning: MLX not found. Cannot discover modules.")
    
    return sorted(modules)


def extract_version_info(doc: str) -> Dict[str, Any]:
    """Extract version information from function docstring."""
    version_info = {
        "version": "unknown",
        "status": "current",
        "signature": "",
        "doc": doc,
        "added_in": None,
        "deprecated_in": None,
        "changed_in": None,
        "added_date": datetime.now().strftime("%Y-%m-%d")
    }
    
    if not doc:
        return version_info
    
    # Look for version patterns in docstring
    version_patterns = [
        r"Added in version (\d+\.\d+\.\d+)",
        r"New in version (\d+\.\d+\.\d+)",
        r"Deprecated since version (\d+\.\d+\.\d+)",
        r"Changed in version (\d+\.\d+\.\d+)"
    ]
    
    for pattern in version_patterns:
        match = re.search(pattern, doc, re.IGNORECASE)
        if match:
            version = match.group(1)
            if "Added" in pattern or "New" in pattern:
                version_info["added_in"] = version
            elif "Deprecated" in pattern:
                version_info["deprecated_in"] = version
            elif "Changed" in pattern:
                version_info["changed_in"] = version
    
    return version_info


def extract_functions_from_module(module_name: str) -> List[Dict[str, Any]]:
    """Extract all functions from a PyTorch module."""
    functions = []
    
    try:
        module = importlib.import_module(module_name)
        
        # Get all members of the module
        for name, obj in inspect.getmembers(module):
            # Skip private attributes and built-ins
            if name.startswith('_'):
                continue
            
            # Check if it's a callable function
            if callable(obj) and inspect.isfunction(obj):
                try:
                    # Get function signature
                    sig = inspect.signature(obj)
                    signature = f"{module_name}.{name}{sig}"
                    
                    # Get docstring
                    doc = obj.__doc__ or ""
                    
                    # Extract arguments
                    args = list(sig.parameters.keys())
                    
                    # Get return type annotation
                    return_type = "Any"
                    if sig.return_annotation != inspect.Signature.empty:
                        return_type = str(sig.return_annotation)
                    
                    # Create function data
                    func_data = {
                        "function": f"{module_name}.{name}",
                        "signature": signature,
                        "doc": doc,
                        "arguments": args,
                        "return_type": return_type,
                        "implementation_status": "not_started",
                        "implementation_notes": None,
                        "status_updated": None,
                        "versions": [extract_version_info(doc)],
                        "current_version": "unknown",
                        "last_updated": datetime.now().strftime("%Y-%m-%d"),
                        "device_support": None,
                        "mlx_mapping": None,
                        "deprecation_warnings": [],
                        "added_date": datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    functions.append(func_data)
                    
                except Exception as e:
                    print(f"Warning: Could not extract function {module_name}.{name}: {e}")
                    continue
    
    except ImportError as e:
        print(f"Warning: Could not import module {module_name}: {e}")
    except Exception as e:
        print(f"Warning: Error processing module {module_name}: {e}")
    
    return functions


def build_catalog() -> FunctionCatalog:
    """Build the function catalog and return as a FunctionCatalog object."""
    print("Building PyTorch function catalog...")
    
    # Discover modules
    torch_modules = discover_torch_modules()
    print(f"Discovered {len(torch_modules)} PyTorch modules")
    
    # Extract functions from each module
    catalog_by_category = defaultdict(list)
    
    for module_name in torch_modules:
        print(f"Extracting from {module_name}...")
        functions = extract_functions_from_module(module_name)
        
        # Categorize functions
        for func_data in functions:
            category = categorize_function(func_data["function"], module_name)
            catalog_by_category[category.upper()].append(func_data)
    
    # Convert to FunctionCatalog
    catalog = FunctionCatalog(categories=dict(catalog_by_category))
    
    # Update total functions count
    total_functions = sum(len(funcs) for funcs in catalog.categories.values())
    catalog.total_functions = total_functions
    
    print(f"Extracted {total_functions} functions from {len(torch_modules)} modules")
    
    return catalog


def main():
    """Main function to extract PyTorch functions and save to JSON."""
    print("PyTorch Function Extractor")
    print("=" * 40)
    
    # Build catalog
    catalog = build_catalog()
    
    # Save to JSON
    print(f"\nSaving catalog to {JSON_OUTPUT}...")
    save_catalog_to_json(catalog, JSON_OUTPUT)
    
    # Print summary
    print(f"\nExtraction complete!")
    print(f"Total functions: {catalog.total_functions}")
    print(f"Categories: {len(catalog.categories)}")
    print(f"JSON file: {JSON_OUTPUT}")
    
    # Print category breakdown
    print("\nCategory breakdown:")
    for category, functions in sorted(catalog.categories.items()):
        print(f"  {category}: {len(functions)} functions")


if __name__ == "__main__":
    main() 