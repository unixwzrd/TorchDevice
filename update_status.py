#!/usr/bin/env python3
"""
Status Update Script for TorchDevice Migration Plan
Generated on 2025-07-10 10:24:23

Usage:
    python update_status.py --function "torch.tensor" --status "complete"
    python update_status.py --category "neural_network" --status "in_progress"
    python update_status.py --list-categories
    python update_status.py --list-functions --category "neural_network"
"""

import json
import argparse
from datetime import datetime

STATUS_OPTIONS = ["not_started", "design", "in_progress", "testing", "complete"]

def load_catalog(filename="utils/data/comprehensive_function_catalog.json"):
    """Load the function catalog."""
    with open(filename, "r") as f:
        return json.load(f)

def save_catalog(catalog, filename="utils/data/comprehensive_function_catalog.json"):
    """Save the function catalog."""
    with open(filename, "w") as f:
        json.dump(catalog, f, indent=2)

def update_function_status(catalog, function_name, status):
    """Update status for a specific function."""
    if status not in STATUS_OPTIONS:
        print(f"Error: Status must be one of {STATUS_OPTIONS}")
        return False
    
    for category, functions in catalog.items():
        for func in functions:
            if func["function"] == function_name:
                func["implementation_status"] = status
                func["status_updated"] = datetime.now().strftime("%Y-%m-%d")
                print(f"Updated {function_name} status to {status}")
                return True
    
    print(f"Error: Function {function_name} not found")
    return False

def update_category_status(catalog, category_name, status):
    """Update status for all functions in a category."""
    if status not in STATUS_OPTIONS:
        print(f"Error: Status must be one of {STATUS_OPTIONS}")
        return False
    
    if category_name not in catalog:
        print(f"Error: Category {category_name} not found")
        return False
    
    updated_count = 0
    for func in catalog[category_name]:
        func["implementation_status"] = status
        func["status_updated"] = datetime.now().strftime("%Y-%m-%d")
        updated_count += 1
    
    print(f"Updated {updated_count} functions in {category_name} to status {status}")
    return True

def list_categories(catalog):
    """List all available categories."""
    print("Available categories:")
    for category in sorted(catalog.keys()):
        count = len(catalog[category])
        print(f"  {category}: {count} functions")

def list_functions(catalog, category_name):
    """List all functions in a category."""
    if category_name not in catalog:
        print(f"Error: Category {category_name} not found")
        return
    
    print(f"Functions in {category_name}:")
    for func in sorted(catalog[category_name], key=lambda x: x["function"]):
        status = func.get("implementation_status", "not_started")
        print(f"  {func['function']} ({status})")

def main():
    parser = argparse.ArgumentParser(description="Update TorchDevice implementation status")
    parser.add_argument("--function", help="Function name to update")
    parser.add_argument("--category", help="Category name to update")
    parser.add_argument("--status", choices=STATUS_OPTIONS, help="New status")
    parser.add_argument("--list-categories", action="store_true", help="List all categories")
    parser.add_argument("--list-functions", action="store_true", help="List functions in category")
    
    args = parser.parse_args()
    
    catalog = load_catalog()
    
    if args.list_categories:
        list_categories(catalog)
        return
    
    if args.list_functions and args.category:
        list_functions(catalog, args.category)
        return
    
    if args.function and args.status:
        if update_function_status(catalog, args.function, args.status):
            save_catalog(catalog)
        return
    
    if args.category and args.status:
        if update_category_status(catalog, args.category, args.status):
            save_catalog(catalog)
        return
    
    parser.print_help()

if __name__ == "__main__":
    main()
