#!/usr/bin/env python3
"""
Extract Device-Related Functions for Translation System
======================================================

This script extracts the most important device-related functions from our
PyTorch catalog and creates a focused analysis for the fake device system.
Updated to work with our current migration plan JSON structure and Pydantic schema.
"""

import os
import sys
from typing import Any, Dict

# Add schemas directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils', 'schemas'))

try:
    from function_catalog import FunctionCatalog, load_catalog_from_json
except ImportError as e:
    print(f"Error importing function_catalog: {e}")
    print("Make sure the schemas directory exists and contains function_catalog.py")
    sys.exit(1)


def load_catalog(filename: str) -> FunctionCatalog:
    """Load the PyTorch function catalog using Pydantic schema."""
    return load_catalog_from_json(filename)


def analyze_device_functions(catalog: FunctionCatalog) -> Dict[str, Any]:
    """Analyze device-related functions and create a focused analysis."""

    # Categories of functions we need to handle
    critical_functions = {
        'device_creation': [],
        'tensor_creation': [],
        'device_management': [],
        'neural_network': [],
        'device_specific': [],
        'events': [],
        'streams': [],
        'memory': []
    }

    # Process our Pydantic catalog structure
    for category_name, functions in catalog.categories.items():
        for func in functions:
            func_name = func.function
            description = func.doc or ""

            # Check if function has device parameters or is device-related
            has_device_param = any(
                keyword in func_name.lower() 
                for keyword in ['device', 'cuda', 'cpu', 'mps', 'xpu', 'xla']
            )

            # Check description for device-related keywords
            description_device_related = any(
                keyword in description.lower() 
                for keyword in ['device', 'cuda', 'cpu', 'mps', 'tensor', 'gpu']
            )

            # Categorize based on function name, category, and description
            if 'device' in func_name.lower() or 'cuda' in func_name.lower():
                critical_functions['device_creation'].append({
                    'name': func_name,
                    'category': category_name,
                    'description': description,
                    'info': func.model_dump()
                })
            elif category_name == 'tensor_creation' or 'tensor' in func_name.lower():
                critical_functions['tensor_creation'].append({
                    'name': func_name,
                    'category': category_name,
                    'description': description,
                    'info': func.model_dump()
                })
            elif category_name == 'device_management' or 'device' in func_name.lower():
                critical_functions['device_management'].append({
                    'name': func_name,
                    'category': category_name,
                    'description': description,
                    'info': func.model_dump()
                })
            elif category_name == 'neural_network' or 'nn' in func_name.lower():
                critical_functions['neural_network'].append({
                    'name': func_name,
                    'category': category_name,
                    'description': description,
                    'info': func.model_dump()
                })
            elif category_name == 'events' or 'event' in func_name.lower():
                critical_functions['events'].append({
                    'name': func_name,
                    'category': category_name,
                    'description': description,
                    'info': func.model_dump()
                })
            elif category_name == 'streams' or 'stream' in func_name.lower():
                critical_functions['streams'].append({
                    'name': func_name,
                    'category': category_name,
                    'description': description,
                    'info': func.model_dump()
                })
            elif category_name == 'memory' or 'memory' in func_name.lower():
                critical_functions['memory'].append({
                    'name': func_name,
                    'category': category_name,
                    'description': description,
                    'info': func.model_dump()
                })
            elif has_device_param or description_device_related:
                critical_functions['device_specific'].append({
                    'name': func_name,
                    'category': category_name,
                    'description': description,
                    'info': func.model_dump()
                })

    return critical_functions


def generate_implementation_guide(critical_functions: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an implementation guide for the fake device system."""

    guide = {
        'implementation_priority': {
            'phase_1': [],
            'phase_2': [],
            'phase_3': []
        },
        'translation_strategies': {},
        'code_examples': {}
    }

    # Phase 1: Core device functions (highest priority)
    for func in critical_functions['device_creation']:
        guide['implementation_priority']['phase_1'].append({
            'function': func['name'],
            'reason': 'Core device creation - fundamental to fake device system',
            'strategy': 'Intercept and return FakeDevice objects',
            'category': func['category']
        })

    for func in critical_functions['tensor_creation']:
        guide['implementation_priority']['phase_1'].append({
            'function': func['name'],
            'reason': 'Tensor creation with device parameters - high impact',
            'strategy': 'Translate device parameters, create on actual device',
            'category': func['category']
        })

    # Phase 2: Device management (medium priority)
    for func in critical_functions['device_management']:
        guide['implementation_priority']['phase_2'].append({
            'function': func['name'],
            'reason': 'Device management - affects default behavior',
            'strategy': 'Translate device specifications',
            'category': func['category']
        })

    for func in critical_functions['events']:
        guide['implementation_priority']['phase_2'].append({
            'function': func['name'],
            'reason': 'Event management - important for CUDA compatibility',
            'strategy': 'Translate CUDA events to MPS equivalents',
            'category': func['category']
        })

    for func in critical_functions['streams']:
        guide['implementation_priority']['phase_2'].append({
            'function': func['name'],
            'reason': 'Stream management - important for CUDA compatibility',
            'strategy': 'Translate CUDA streams to MPS equivalents',
            'category': func['category']
        })

    # Phase 3: Neural network and device-specific (lower priority)
    for func in critical_functions['neural_network']:
        guide['implementation_priority']['phase_3'].append({
            'function': func['name'],
            'reason': 'Neural network functions - may need tensor device translation',
            'strategy': 'Handle tensor device translation',
            'category': func['category']
        })

    for func in critical_functions['device_specific']:
        guide['implementation_priority']['phase_3'].append({
            'function': func['name'],
            'reason': 'Device-specific operations - may need special handling',
            'strategy': 'Device-specific translation or fallback',
            'category': func['category']
        })

    for func in critical_functions['memory']:
        guide['implementation_priority']['phase_3'].append({
            'function': func['name'],
            'reason': 'Memory management - may need device-specific handling',
            'strategy': 'Device-specific memory management',
            'category': func['category']
        })

    return guide


def create_implementation_guide(critical_functions: Dict[str, Any], guide: Dict[str, Any]) -> str:
    """Create a comprehensive implementation guide."""

    content = """# Device Translation Implementation Guide
*Generated from PyTorch Function Analysis*

## Critical Functions by Category

### 1. Device Creation Functions (Phase 1 - HIGHEST PRIORITY)
"""

    for func in critical_functions['device_creation']:
        content += f"""
#### {func['name']}
- **Category**: {func['category']}
- **Strategy**: Intercept and return FakeDevice objects
- **Implementation**: Replace torch.device() calls
- **Description**: {func['description'][:200]}...
"""

    content += """
### 2. Tensor Creation Functions (Phase 1 - HIGHEST PRIORITY)
"""

    for func in critical_functions['tensor_creation']:
        content += f"""
#### {func['name']}
- **Category**: {func['category']}
- **Strategy**: Translate device parameters, create on actual device
- **Implementation**: Intercept device keyword arguments
- **Description**: {func['description'][:200]}...
"""

    content += """
### 3. Device Management Functions (Phase 2 - MEDIUM PRIORITY)
"""

    for func in critical_functions['device_management']:
        content += f"""
#### {func['name']}
- **Category**: {func['category']}
- **Strategy**: Translate device specifications
- **Implementation**: Intercept device arguments
- **Description**: {func['description'][:200]}...
"""

    content += """
### 4. Events Functions (Phase 2 - MEDIUM PRIORITY)
"""

    for func in critical_functions['events']:
        content += f"""
#### {func['name']}
- **Category**: {func['category']}
- **Strategy**: Translate CUDA events to MPS equivalents
- **Implementation**: Map CUDA event operations to MPS
- **Description**: {func['description'][:200]}...
"""

    content += """
### 5. Streams Functions (Phase 2 - MEDIUM PRIORITY)
"""

    for func in critical_functions['streams']:
        content += f"""
#### {func['name']}
- **Category**: {func['category']}
- **Strategy**: Translate CUDA streams to MPS equivalents
- **Implementation**: Map CUDA stream operations to MPS
- **Description**: {func['description'][:200]}...
"""

    content += """
### 6. Neural Network Functions (Phase 3 - LOWER PRIORITY)
"""

    for func in critical_functions['neural_network']:
        content += f"""
#### {func['name']}
- **Category**: {func['category']}
- **Strategy**: Handle tensor device translation
- **Implementation**: Translate tensor devices before function calls
- **Description**: {func['description'][:200]}...
"""

    content += """
### 7. Device-Specific Functions (Phase 3 - LOWER PRIORITY)
"""

    for func in critical_functions['device_specific']:
        content += f"""
#### {func['name']}
- **Category**: {func['category']}
- **Strategy**: Device-specific translation or fallback
- **Implementation**: Custom handling based on device type
- **Description**: {func['description'][:200]}...
"""

    content += """
### 8. Memory Functions (Phase 3 - LOWER PRIORITY)
"""

    for func in critical_functions['memory']:
        content += f"""
#### {func['name']}
- **Category**: {func['category']}
- **Strategy**: Device-specific memory management
- **Implementation**: Handle memory operations per device
- **Description**: {func['description'][:200]}...
"""

    content += """
## Implementation Phases Summary

### Phase 1: Core Infrastructure (Highest Priority)
- Device creation functions
- Tensor creation functions
- **Total**: """ + str(len(guide['implementation_priority']['phase_1'])) + """ functions

### Phase 2: Device Management (Medium Priority)
- Device management functions
- Events and streams
- **Total**: """ + str(len(guide['implementation_priority']['phase_2'])) + """ functions

### Phase 3: Advanced Features (Lower Priority)
- Neural network functions
- Device-specific operations
- Memory management
- **Total**: """ + str(len(guide['implementation_priority']['phase_3'])) + """ functions

## Next Steps

1. **Start with Phase 1** - Implement core device and tensor creation
2. **Move to Phase 2** - Add device management and CUDA compatibility
3. **Complete Phase 3** - Handle advanced features and optimizations
4. **Test thoroughly** - Ensure all functions work correctly
5. **Update status** - Mark functions as implemented in migration plan
"""

    return content


def main():
    """Main function to extract device functions and create implementation guide."""

    # Default catalog file location
    catalog_file = "utils/data/comprehensive_function_catalog.json"

    # Check if file exists
    if not os.path.exists(catalog_file):
        print(f"Error: Catalog file not found: {catalog_file}")
        print("Please run generate_migration_plan.py first to create the catalog.")
        return

    print("Device Function Extractor")
    print("=" * 40)

    # Load catalog
    print(f"Loading catalog from: {catalog_file}")
    catalog = load_catalog(catalog_file)

    # Analyze device functions
    print("Analyzing device-related functions...")
    critical_functions = analyze_device_functions(catalog)

    # Generate implementation guide
    print("Generating implementation guide...")
    guide = generate_implementation_guide(critical_functions)

    # Create implementation guide document
    print("Creating implementation guide document...")
    implementation_guide = create_implementation_guide(critical_functions, guide)

    # Save implementation guide
    output_dir = "utils/docs/implementation_guide"
    os.makedirs(output_dir, exist_ok=True)

    guide_file = f"{output_dir}/device_implementation_guide.md"
    with open(guide_file, 'w') as f:
        f.write(implementation_guide)

    print(f"Saved implementation guide: {guide_file}")

    # Print summary
    total_critical = sum(len(funcs) for funcs in critical_functions.values())
    print("\nAnalysis complete!")
    print(f"Total critical functions identified: {total_critical}")
    print("Functions by category:")
    for category, funcs in critical_functions.items():
        if funcs:
            print(f"  - {category}: {len(funcs)} functions")

    print("\nImplementation phases:")
    for phase, funcs in guide['implementation_priority'].items():
        print(f"  - {phase}: {len(funcs)} functions")


if __name__ == "__main__":
    main()