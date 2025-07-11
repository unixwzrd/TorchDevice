#!/usr/bin/env python3
"""
Report Generator for TorchDevice Migration Plan
==============================================

This script generates all reports from the JSON database using Pydantic schema.
This is the report generation phase - separate from data extraction.

Reports generated:
- Migration plan markdown files
- Device compatibility analysis
- Implementation guides
- Status summaries
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# Add schemas directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'schemas'))

try:
    from function_catalog import (
        FunctionCatalog,
        ImplementationStatus,
        load_catalog_from_json,
    )
except ImportError as e:
    print(f"Error importing function_catalog: {e}")
    print("Make sure the schemas directory exists and contains function_catalog.py")
    sys.exit(1)

# Configurable parameters
JSON_INPUT = "utils/data/comprehensive_function_catalog.json"
OUTPUT_DIR = "utils/docs/migration_plan_unified"
LINE_LIMIT = 75000  # Max lines per Markdown file

# Migration plan header and footer
HEADER = """# TorchDevice Migration Plan
*Generated from PyTorch Function Analysis*

## Overview
This document contains the complete migration plan for implementing TorchDevice, 
a PyTorch device translation layer that enables seamless switching between 
CUDA, MPS, and CPU backends.

## Migration Strategy

### Priority Matrix
- **🔴 Critical**: Core device functions (torch.device, torch.tensor, etc.)
- **🟡 High**: Neural network operations, optimization functions
- **🟢 Medium**: Mathematical operations, utilities
- **🔵 Low**: Specialized operations, experimental features

### Implementation Phases
1. **Phase 1**: Core device management and tensor creation
2. **Phase 2**: Neural network operations and optimization
3. **Phase 3**: Mathematical operations and utilities
4. **Phase 4**: Specialized operations and edge cases

### Architecture Overview
- Device translation layer intercepts PyTorch calls
- Automatic fallback from CUDA → MPS → CPU
- Transparent API compatibility
- Performance monitoring and optimization

### Risk Mitigation
- Comprehensive testing across device combinations
- Gradual rollout with feature flags
- Performance benchmarking and optimization
- Backward compatibility maintenance

### Success Metrics
- 100% API compatibility with PyTorch
- <5% performance overhead
- Zero breaking changes for existing code
- Successful migration of test projects

## Function Catalog

| Category | CPU | CUDA | MPS | MLX | Status | Arguments | Return Type | Notes |
|:---------|:---:|:----:|:---:|:---:|:------:|-----------|-------------|-------|
"""

FOOTER = """
## Summary

- **Total Functions**: {total_functions}
- **Completed**: {completed}
- **In Progress**: {in_progress}
- **Not Started**: {not_started}

## Implementation Status Tracking

### Overall Progress
- **Total Functions**: {total_functions}
- **Completed**: {completed}
- **In Progress**: {in_progress}
- **Not Started**: {not_started}

### Priority Implementation Order
1. **Core Device Operations** (CUDA, MPS, CPU)
2. **Tensor Operations** (creation, manipulation, math)
3. **Neural Network Modules** (nn, functional)
4. **Optimization** (optim, autograd)
5. **Utilities** (utils, types, storage)

## Migration Strategy

### Priority Matrix

| Priority | Category | Functions | Rationale |
|----------|----------|-----------|-----------|
| **P0 (Critical)** | Device Management | 15 functions | Core functionality required |
| **P1 (High)** | Tensor Creation | 20 functions | Basic tensor operations |
| **P2 (Medium)** | Neural Network | 25 functions | ML model support |
| **P3 (Low)** | Mathematical | 30 functions | Advanced operations |
| **P4 (Nice-to-have)** | Utility | 40 functions | Helper functions |

### Implementation Phases

#### Phase 1: Foundation (Weeks 1-2)
- Device management functions
- Basic tensor creation
- Core mathematical operations

#### Phase 2: Neural Networks (Weeks 3-4)
- Activation functions
- Loss functions
- Basic neural network operations

#### Phase 3: Advanced Operations (Weeks 5-6)
- Complex mathematical operations
- Linear algebra functions
- Signal processing

#### Phase 4: Optimization (Weeks 7-8)
- Performance optimization
- Memory management
- Stream operations

## Architecture Overview

### Function Router Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PyTorch Call  │───▶│  TorchDevice    │───▶│  Device Router  │
│                 │    │  Interceptor    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CPU Device    │◀───│  Translation    │◀───│  Compatibility  │
│   Operations    │    │  Engine         │    │  Matrix         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CUDA Device   │◀───│  Fallback       │◀───│  Error Handler  │
│   Operations    │    │  Manager        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MPS Device    │◀───│  Performance    │◀───│  Monitoring     │
│   Operations    │    │  Optimizer      │    │  System         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

1. **Function Interceptor**: Hooks into PyTorch calls
2. **Device Router**: Determines optimal device for operation
3. **Translation Engine**: Converts between device types
4. **Compatibility Matrix**: Stores device support information
5. **Fallback Manager**: Handles unsupported operations
6. **Performance Optimizer**: Optimizes for specific devices
7. **Monitoring System**: Tracks performance and errors

### Success Metrics

- **Function Coverage**: 95% of PyTorch functions supported
- **Performance**: <5% overhead compared to native PyTorch
- **Compatibility**: 100% API compatibility with PyTorch
- **Error Rate**: <1% translation errors
- **Memory Usage**: <10% additional memory overhead

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API Changes | Medium | High | Version pinning, compatibility tests |
| Performance Degradation | High | Medium | Profiling, optimization |
| Memory Leaks | Low | High | Memory monitoring, cleanup |
| Device Compatibility | Medium | High | Comprehensive testing |

### Implementation Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Scope Creep | High | Medium | Phased approach, clear priorities |
| Testing Complexity | Medium | High | Automated testing, CI/CD |
| Documentation | Low | Medium | Auto-generated docs |
| Maintenance | Medium | Medium | Modular design, clear interfaces |

## Next Steps

1. Review function status and update implementation priorities
2. Implement core device functions (Phase 1)
3. Test with real-world projects
4. Iterate and optimize based on feedback

---
*Generated on {date}*
"""


def get_implementation_stats(catalog: FunctionCatalog) -> Dict[str, Any]:
    """Get implementation statistics from the catalog."""
    stats = {
        "total_functions": 0,
        "completed": 0,
        "in_progress": 0,
        "not_started": 0,
        "testing": 0,
        "design": 0,
        "by_category": {}
    }
    
    for category, functions in catalog.categories.items():
        category_stats = {
            "total": len(functions),
            "completed": 0,
            "in_progress": 0,
            "not_started": 0,
            "testing": 0,
            "design": 0
        }
        
        for func in functions:
            status = func.implementation_status.value
            stats["total_functions"] += 1
            category_stats["total"] += 1
            
            if status == "complete":
                stats["completed"] += 1
                category_stats["completed"] += 1
            elif status == "in_progress":
                stats["in_progress"] += 1
                category_stats["in_progress"] += 1
            elif status == "testing":
                stats["testing"] += 1
                category_stats["testing"] += 1
            elif status == "design":
                stats["design"] += 1
                category_stats["design"] += 1
            else:  # not_started
                stats["not_started"] += 1
                category_stats["not_started"] += 1
        
        stats["by_category"][category] = category_stats
    
    return stats


def write_markdown_files(catalog: FunctionCatalog, line_limit=LINE_LIMIT, output_dir=OUTPUT_DIR):
    """Write Markdown files with proper sectioning."""
    os.makedirs(output_dir, exist_ok=True)
    file_idx = 1
    line_count = 0
    first_section_in_file = True

    def start_new_file():
        return [HEADER]

    def end_file(lines, idx):
        # Calculate total functions for this file
        total_functions = catalog.total_functions
        completed = len(catalog.get_functions_by_status(ImplementationStatus.COMPLETE))
        
        # Replace placeholders in footer
        footer_with_stats = FOOTER.format(
            total_functions=total_functions,
            completed=completed,
            in_progress=len(catalog.get_functions_by_status(ImplementationStatus.IN_PROGRESS)),
            not_started=len(catalog.get_functions_by_status(ImplementationStatus.NOT_STARTED)),
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        lines.append(footer_with_stats)
        filename = os.path.join(output_dir, f"migration_plan_section_{idx:02d}.md")
        with open(filename, "w", encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"Wrote {filename} ({len(lines)} lines)")

    def estimate_section_lines(section, functions):
        """Estimate how many lines this section will add."""
        lines_needed = 0
        # Section header
        lines_needed += 1
        # Separator line (if first section in file)
        if first_section_in_file:
            lines_needed += 1
        # Blank separator row (if not first section)
        if not first_section_in_file:
            lines_needed += 1
        # Function rows
        lines_needed += len(functions)
        return lines_needed

    lines = start_new_file()
    
    for section, functions in catalog.categories.items():
        # Estimate how many lines this section will add
        section_lines = estimate_section_lines(section, functions)
        
        # Check if adding this section would exceed the line limit
        if line_count > 0 and (line_count + section_lines) > line_limit:
            # Start a new file
            end_file(lines, file_idx)
            file_idx += 1
            lines = start_new_file()
            line_count = 0
            first_section_in_file = True
            # Recalculate section lines for new file
            section_lines = estimate_section_lines(section, functions)
        
        # Add blank separator row before new section (except first section)
        if not first_section_in_file:
            lines.append("| | | | | | | | | |")
            line_count += 1
        
        # Section header
        section_header = (
            f"| 🟦 {section} | CPU | CUDA | MPS | MLX | Status | Arguments | "
            f"Return Type | Notes |"
        )
        lines.append(section_header)
        line_count += 1
        
        # Only add separator line for the first section in the file
        if first_section_in_file:
            section_sep = (
                "|:-----------------|:---:|:----:|:---:|:---:|:------:|-----------|"
                "-------------|-------|"
            )
            lines.append(section_sep)
            line_count += 1
            first_section_in_file = False
        
        for func in functions:
            args = ", ".join(func.arguments[:3])
            if len(func.arguments) > 3:
                args += ", ..."
            return_type = func.return_type
            description = func.doc.replace("\n", " ").replace("\r", " ").strip()
            description = " ".join(description.split())  # Remove extra spaces
            
            # Add version info to description if available
            if func.versions:
                current_version = func.versions[0]
                if current_version.added_in:
                    description += f" [Added: {current_version.added_in}]"
                if current_version.deprecated_in:
                    description += f" [Deprecated: {current_version.deprecated_in}]"
            
            # Only truncate if extremely long
            if len(description) > 200:
                description = description[:197] + "..."
            
            # Default status and device support
            status_emoji = "🔴"
            if func.implementation_status == ImplementationStatus.COMPLETE:
                status_emoji = "✅"
            elif func.implementation_status == ImplementationStatus.IN_PROGRESS:
                status_emoji = "🟡"
            elif func.implementation_status == ImplementationStatus.TESTING:
                status_emoji = "🟠"
            elif func.implementation_status == ImplementationStatus.DESIGN:
                status_emoji = "🔵"
            elif func.implementation_status == ImplementationStatus.NOT_IMPLEMENTED:
                status_emoji = "❌"
            
            cpu = cuda = mps = mlx = "❓"
            if func.device_support:
                cpu = "✅" if func.device_support.cpu else "❌"
                cuda = "✅" if func.device_support.cuda else "❌"
                mps = "✅" if func.device_support.mps else "❌"
                mlx = "✅" if func.device_support.mlx else "❌"
            
            row = (
                f"| `{func.function}` | {cpu} | {cuda} | {mps} | {mlx} | {status_emoji} | "
                f"`{args}` | `{return_type}` | {description} |"
            )
            lines.append(row)
            line_count += 1
    
    # Write any remaining lines
    if len(lines) > 1:
        end_file(lines, file_idx)


def generate_status_summary(catalog: FunctionCatalog) -> str:
    """Generate a status summary report."""
    stats = get_implementation_stats(catalog)
    
    summary = f"""# TorchDevice Implementation Status Summary
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

## Overall Progress

- **Total Functions**: {stats['total_functions']}
- **Completed**: {stats['completed']} ({stats['completed']/stats['total_functions']*100:.1f}%)
- **In Progress**: {stats['in_progress']} ({stats['in_progress']/stats['total_functions']*100:.1f}%)
- **Testing**: {stats['testing']} ({stats['testing']/stats['total_functions']*100:.1f}%)
- **Design**: {stats['design']} ({stats['design']/stats['total_functions']*100:.1f}%)
- **Not Started**: {stats['not_started']} ({stats['not_started']/stats['total_functions']*100:.1f}%)

## Progress by Category

"""
    
    # Sort categories by completion percentage
    category_progress = []
    for category, cat_stats in stats['by_category'].items():
        if cat_stats['total'] > 0:
            completion_pct = (cat_stats['completed'] / cat_stats['total']) * 100
            category_progress.append((category, completion_pct, cat_stats))
    
    category_progress.sort(key=lambda x: x[1], reverse=True)
    
    for category, completion_pct, cat_stats in category_progress:
        summary += f"### {category}\n"
        summary += f"- **Progress**: {completion_pct:.1f}% ({cat_stats['completed']}/{cat_stats['total']})\n"
        summary += f"- **In Progress**: {cat_stats['in_progress']}\n"
        summary += f"- **Testing**: {cat_stats['testing']}\n"
        summary += f"- **Design**: {cat_stats['design']}\n"
        summary += f"- **Not Started**: {cat_stats['not_started']}\n\n"
    
    return summary


def main():
    """Main function to generate all reports."""
    print("Report Generator for TorchDevice Migration Plan")
    print("=" * 50)
    
    # Check if JSON file exists
    if not os.path.exists(JSON_INPUT):
        print(f"Error: JSON catalog not found: {JSON_INPUT}")
        print("Please run extract_pytorch_functions.py first to create the catalog.")
        return
    
    # Load catalog
    print(f"Loading catalog from: {JSON_INPUT}")
    try:
        catalog = load_catalog_from_json(JSON_INPUT)
    except Exception as e:
        print(f"Error loading catalog: {e}")
        return
    
    print(f"Loaded catalog with {catalog.total_functions} functions")
    
    # Generate reports
    print("\nGenerating reports...")
    
    # 1. Migration plan markdown files
    print("1. Generating migration plan markdown files...")
    write_markdown_files(catalog)
    
    # 2. Status summary
    print("2. Generating status summary...")
    status_summary = generate_status_summary(catalog)
    status_file = "utils/docs/STATUS_SUMMARY.md"
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    with open(status_file, "w", encoding='utf-8') as f:
        f.write(status_summary)
    print(f"   Saved: {status_file}")
    
    # 3. Run other report generators
    print("3. Running device compatibility analyzer...")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "utils/bin/device_compatibility_analyzer.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   Device compatibility analysis complete")
        else:
            print(f"   Warning: Device compatibility analysis failed: {result.stderr}")
    except Exception as e:
        print(f"   Warning: Could not run device compatibility analyzer: {e}")
    
    print("4. Running device function extractor...")
    try:
        result = subprocess.run([sys.executable, "utils/bin/extract_device_functions.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   Device function extraction complete")
        else:
            print(f"   Warning: Device function extraction failed: {result.stderr}")
    except Exception as e:
        print(f"   Warning: Could not run device function extractor: {e}")
    
    # Print final statistics
    stats = get_implementation_stats(catalog)
    print(f"\nReport generation complete!")
    print(f"Total functions: {stats['total_functions']}")
    print(f"Completed: {stats['completed']} ({stats['completed']/stats['total_functions']*100:.1f}%)")
    print(f"In Progress: {stats['in_progress']} ({stats['in_progress']/stats['total_functions']*100:.1f}%)")
    print(f"Not Started: {stats['not_started']} ({stats['not_started']/stats['total_functions']*100:.1f}%)")
    
    print(f"\nReports saved to:")
    print(f"  - Migration plan: {OUTPUT_DIR}/")
    print(f"  - Status summary: utils/docs/STATUS_SUMMARY.md")
    print(f"  - Device compatibility: utils/docs/compatibility_analysis/")
    print(f"  - Implementation guide: utils/docs/implementation_guide/")


if __name__ == "__main__":
    main() 