#!/usr/bin/env python3
"""
Status Update Script for TorchDevice Migration Plan
Generated on 2025-07-10 10:47:40

Usage:
    python update_status.py --function "torch.tensor" --status "complete"
    python update_status.py --function "torch.cuda.specific_function" --status "not_implemented" --notes "CUDA-specific kernel"
    python update_status.py --category "neural_network" --status "in_progress"
    python update_status.py --list-categories
    python update_status.py --list-functions --category "neural_network"
"""

import argparse
import os
import sys

# Add schemas directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'schemas'))

try:
    from function_catalog import (
        FunctionCatalog,
        ImplementationStatus,
        load_catalog_from_json,
        save_catalog_to_json,
    )
except ImportError as e:
    print(f"Error importing function_catalog: {e}")
    print("Make sure the schemas directory exists and contains function_catalog.py")
    sys.exit(1)


def update_function_status(catalog: FunctionCatalog, function_name: str, status: ImplementationStatus, notes: str = None) -> bool:
    """Update status for a specific function using Pydantic schema."""
    success = catalog.update_function_status(function_name, status, notes)
    if success:
        print(f"Updated {function_name} status to {status.value}")
        if notes:
            print(f"Added notes: {notes}")
    else:
        print(f"Error: Function {function_name} not found")
    return success


def update_category_status(catalog: FunctionCatalog, category_name: str, status: ImplementationStatus, notes: str = None) -> bool:
    """Update status for all functions in a category using Pydantic schema."""
    if category_name not in catalog.categories:
        print(f"Error: Category {category_name} not found")
        return False
    
    updated_count = 0
    for func in catalog.categories[category_name]:
        func.update_status(status, notes)
        updated_count += 1
    
    print(f"Updated {updated_count} functions in {category_name} to status {status.value}")
    if notes:
        print(f"Added notes: {notes}")
    return True


def list_categories(catalog: FunctionCatalog) -> None:
    """List all available categories."""
    print("Available categories:")
    for category in sorted(catalog.categories.keys()):
        count = len(catalog.categories[category])
        print(f"  {category}: {count} functions")


def list_functions(catalog: FunctionCatalog, category_name: str) -> None:
    """List all functions in a category."""
    if category_name not in catalog.categories:
        print(f"Error: Category {category_name} not found")
        return
    
    print(f"Functions in {category_name}:")
    for func in sorted(catalog.categories[category_name], key=lambda x: x.function):
        status = func.implementation_status.value
        notes = func.implementation_notes or ""
        if notes:
            print(f"  {func.function} ({status}) - {notes}")
        else:
            print(f"  {func.function} ({status})")


def main():
    parser = argparse.ArgumentParser(description="Update TorchDevice implementation status")
    parser.add_argument("--function", help="Function name to update")
    parser.add_argument("--category", help="Category name to update")
    parser.add_argument("--status", choices=[s.value for s in ImplementationStatus], help="New status")
    parser.add_argument("--notes", help="Implementation notes (especially useful for 'not_implemented' status)")
    parser.add_argument("--list-categories", action="store_true", help="List all categories")
    parser.add_argument("--list-functions", action="store_true", help="List functions in category")
    
    args = parser.parse_args()
    
    # Load catalog with Pydantic validation
    try:
        catalog = load_catalog_from_json("utils/data/comprehensive_function_catalog.json")
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return
    
    if args.list_categories:
        list_categories(catalog)
        return
    
    if args.list_functions and args.category:
        list_functions(catalog, args.category)
        return
    
    if args.function and args.status:
        status_enum = ImplementationStatus(args.status)
        if update_function_status(catalog, args.function, status_enum, args.notes):
            try:
                save_catalog_to_json(catalog, "utils/data/comprehensive_function_catalog.json")
            except Exception as e:
                print(f"Error saving catalog: {e}")
        return
    
    if args.category and args.status:
        status_enum = ImplementationStatus(args.status)
        if update_category_status(catalog, args.category, status_enum, args.notes):
            try:
                save_catalog_to_json(catalog, "utils/data/comprehensive_function_catalog.json")
            except Exception as e:
                print(f"Error saving catalog: {e}")
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
