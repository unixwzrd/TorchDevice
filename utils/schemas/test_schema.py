#!/usr/bin/env python3
"""
Test script for Pydantic function catalog schema.
"""

from schemas.function_catalog import (
    FunctionCatalog,
    ImplementationStatus,
    PyTorchFunction,
    load_catalog_from_json,
    save_catalog_to_json,
)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_schema_creation():
    """Test creating a function with the schema."""
    print("Testing schema creation...")

    # Create a sample function
    func = PyTorchFunction(
        function="torch.device",
        signature="torch.device(device)",
        doc="Creates a device object",
        arguments=["device"],
        return_type="torch.device",
        implementation_status=ImplementationStatus.IN_PROGRESS,
        implementation_notes="Core device translation function"
    )

    print(f"✅ Created function: {func.function}")
    print(f"   Status: {func.implementation_status}")
    print(f"   Notes: {func.implementation_notes}")

    return func


def test_catalog_operations():
    """Test catalog operations."""
    print("\nTesting catalog operations...")

    # Create a catalog
    catalog = FunctionCatalog(categories={
        "TORCH_DEVICE": [
            PyTorchFunction(
                function="torch.device",
                implementation_status=ImplementationStatus.IN_PROGRESS,
                implementation_notes="Core device translation"
            ),
            PyTorchFunction(
                function="torch.tensor",
                implementation_status=ImplementationStatus.NOT_STARTED
            )
        ]
    })

    # Test getting function
    func = catalog.get_function("torch.device")
    print(f"✅ Found function: {func.function if func else 'Not found'}")

    # Test updating status
    success = catalog.update_function_status("torch.device", ImplementationStatus.COMPLETE, "Implemented!")
    print(f"✅ Status update: {'Success' if success else 'Failed'}")

    # Test getting statistics
    stats = catalog.get_statistics()
    print(f"✅ Statistics: {stats['total_functions']} total functions")
    print(f"   Completed: {stats['by_status']['complete']}")
    print(f"   In progress: {stats['by_status']['in_progress']}")

    return catalog


def test_json_operations():
    """Test JSON loading and saving."""
    print("\nTesting JSON operations...")

    # Create test catalog
    catalog = FunctionCatalog(categories={
        "TEST_CATEGORY": [
            PyTorchFunction(
                function="torch.test_function",
                implementation_status=ImplementationStatus.TESTING,
                implementation_notes="Test function"
            )
        ]
    })

    # Save to JSON
    test_file = "test_catalog.json"
    save_catalog_to_json(catalog, test_file)
    print(f"✅ Saved catalog to {test_file}")

    # Load from JSON
    loaded_catalog = load_catalog_from_json(test_file)
    print(f"✅ Loaded catalog with {loaded_catalog.get_statistics()['total_functions']} functions")

    # Clean up
    os.remove(test_file)
    print(f"✅ Cleaned up {test_file}")

    return loaded_catalog


def test_validation():
    """Test schema validation."""
    print("\nTesting schema validation...")

    try:
        # This should fail - invalid function name
        func = PyTorchFunction(
            function="invalid_function",  # Should start with torch.
            implementation_status=ImplementationStatus.NOT_STARTED
        )
        print("❌ Should have failed validation")
    except ValueError as e:
        print(f"✅ Validation caught error: {e}")

    try:
        # This should work
        func = PyTorchFunction(
            function="torch.valid_function",
            implementation_status=ImplementationStatus.NOT_STARTED
        )
        print(f"✅ Valid function created: {func.function}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    """Run all tests."""
    print("🧪 Testing Pydantic Function Catalog Schema\n")

    test_schema_creation()
    test_catalog_operations()
    test_json_operations()
    test_validation()

    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    main()
