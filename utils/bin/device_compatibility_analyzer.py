#!/usr/bin/env python3
"""
Device Compatibility Analyzer
=============================

This script analyzes PyTorch function compatibility across CPU, MPS, and CUDA
to create comparison tables for the fake device system. Updated to work with
our current migration plan JSON structure and Pydantic schema.
"""

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

# Add schemas directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils', 'schemas'))

try:
    from function_catalog import FunctionCatalog, load_catalog_from_json
except ImportError as e:
    print(f"Error importing function_catalog: {e}")
    print("Make sure the schemas directory exists and contains function_catalog.py")
    sys.exit(1)


class DeviceCompatibilityAnalyzer:
    """Analyze device compatibility patterns across CPU, MPS, and CUDA."""
    
    def __init__(self):
        self.devices = ['cpu', 'cuda', 'mps']
        self.compatibility_patterns = defaultdict(list)
        self.function_categories = defaultdict(list)
        
    def load_catalog(self, filename: str) -> FunctionCatalog:
        """Load the PyTorch function catalog using Pydantic schema."""
        return load_catalog_from_json(filename)
    
    def analyze_compatibility_patterns(self, catalog: FunctionCatalog) -> Dict[str, Any]:
        """Analyze compatibility patterns across devices using our Pydantic structure."""
        
        # Initialize pattern tracking
        patterns = {
            'universal': [],        # Works on all devices
            'cpu_only': [],         # CPU only
            'cuda_only': [],        # CUDA only
            'mps_only': [],         # MPS only
            'cpu_cuda': [],         # CPU + CUDA
            'cpu_mps': [],          # CPU + MPS
            'cuda_mps': [],         # CUDA + MPS
            'device_specific': [],  # Device-specific functions
            'unknown': []           # Unknown compatibility
        }
        
        # Track function categories
        categories = {
            'tensor_creation': [],
            'device_management': [],
            'neural_network': [],
            'mathematical': [],
            'utility': [],
            'device_specific': [],
            'events': [],
            'streams': [],
            'memory': [],
            'random': [],
            'autograd': []
        }
        
        # Process our Pydantic catalog structure
        for category_name, functions in catalog.categories.items():
            for func in functions:
                func_name = func.function
                device_support = func.device_support or {}
                
                # Determine compatibility pattern from our status fields
                cpu_ok = device_support.get('cpu', '❓') == '✅'
                cuda_ok = device_support.get('cuda', '❓') == '✅'
                mps_ok = device_support.get('mps', '❓') == '✅'
                
                # Categorize by compatibility pattern
                if cpu_ok and cuda_ok and mps_ok:
                    patterns['universal'].append(func_name)
                elif cpu_ok and not cuda_ok and not mps_ok:
                    patterns['cpu_only'].append(func_name)
                elif cuda_ok and not cpu_ok and not mps_ok:
                    patterns['cuda_only'].append(func_name)
                elif mps_ok and not cpu_ok and not cuda_ok:
                    patterns['mps_only'].append(func_name)
                elif cpu_ok and cuda_ok and not mps_ok:
                    patterns['cpu_cuda'].append(func_name)
                elif cpu_ok and mps_ok and not cuda_ok:
                    patterns['cpu_mps'].append(func_name)
                elif cuda_ok and mps_ok and not cpu_ok:
                    patterns['cuda_mps'].append(func_name)
                elif any([cpu_ok, cuda_ok, mps_ok]):
                    patterns['device_specific'].append(func_name)
                else:
                    patterns['unknown'].append(func_name)
                
                # Categorize by function type
                if category_name in categories:
                    categories[category_name].append(func_name)
                else:
                    categories['utility'].append(func_name)
        
        return {
            'patterns': patterns,
            'categories': categories,
            'summary': self._generate_summary(patterns, categories)
        }
    
    def _generate_summary(self, patterns: Dict[str, List], categories: Dict[str, List]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_functions = sum(len(funcs) for funcs in patterns.values())
        
        pattern_summary = {}
        for pattern, funcs in patterns.items():
            pattern_summary[pattern] = {
                'count': len(funcs),
                'percentage': (len(funcs) / total_functions * 100) if total_functions > 0 else 0
            }
        
        category_summary = {}
        for category, funcs in categories.items():
            category_summary[category] = {
                'count': len(funcs),
                'percentage': (len(funcs) / total_functions * 100) if total_functions > 0 else 0
            }
        
        return {
            'total_functions': total_functions,
            'patterns': pattern_summary,
            'categories': category_summary
        }
    
    def create_compatibility_table(self, analysis: Dict[str, Any]) -> str:
        """Create a markdown compatibility table."""
        
        patterns = analysis['patterns']
        summary = analysis['summary']
        
        table = """# Device Compatibility Analysis
*Generated from PyTorch Function Analysis*

## Compatibility Patterns Summary

| Pattern | Count | Percentage | Description |
|---------|-------|------------|-------------|
"""
        
        # Add pattern rows
        for pattern, funcs in patterns.items():
            count = len(funcs)
            percentage = summary['patterns'][pattern]['percentage']
            description = self._get_pattern_description(pattern)
            table += f"| **{pattern.replace('_', ' ').title()}** | {count} | {percentage:.1f}% | {description} |\n"
        
        table += "\n## Detailed Function Lists\n\n"
        
        # Add detailed lists for each pattern
        for pattern, funcs in patterns.items():
            if funcs:  # Only show patterns with functions
                table += f"### {pattern.replace('_', ' ').title()} Functions ({len(funcs)})\n\n"
                for func in sorted(funcs):
                    table += f"- `{func}`\n"
                table += "\n"
        
        return table
    
    def _get_pattern_description(self, pattern: str) -> str:
        """Get description for compatibility pattern."""
        descriptions = {
            'universal': 'Works on CPU, CUDA, and MPS',
            'cpu_only': 'CPU only - may need translation',
            'cuda_only': 'CUDA only - needs MPS translation',
            'mps_only': 'MPS only - may need fallback',
            'cpu_cuda': 'CPU + CUDA - may need MPS translation',
            'cpu_mps': 'CPU + MPS - good compatibility',
            'cuda_mps': 'CUDA + MPS - good compatibility',
            'device_specific': 'Device-specific operations',
            'unknown': 'Unknown compatibility - needs investigation'
        }
        return descriptions.get(pattern, 'Unknown pattern')
    
    def create_translation_strategy_table(self, analysis: Dict[str, Any]) -> str:
        """Create a translation strategy table."""
        
        patterns = analysis['patterns']
        
        table = """# Translation Strategy by Compatibility Pattern

## Translation Approaches

| Pattern | Translation Strategy | Priority | Notes |
|---------|---------------------|----------|-------|
"""
        
        strategies = {
            'universal': ('Pass through', 'Low', 'No translation needed'),
            'cpu_only': ('CPU fallback', 'Medium', 'May need performance optimization'),
            'cuda_only': ('MPS translation', 'High', 'Core translation target'),
            'mps_only': ('MPS native', 'Low', 'Already compatible'),
            'cpu_cuda': ('MPS translation', 'High', 'Important for CUDA compatibility'),
            'cpu_mps': ('MPS native', 'Medium', 'Good compatibility'),
            'cuda_mps': ('MPS native', 'Medium', 'Good compatibility'),
            'device_specific': ('Special handling', 'High', 'May need custom implementation'),
            'unknown': ('Investigation needed', 'Medium', 'Requires testing')
        }
        
        priorities = {'High': 1, 'Medium': 2, 'Low': 3}
        
        # Sort by priority
        sorted_patterns = sorted(
            strategies.items(),
            key=lambda x: priorities.get(x[1][1], 4)
        )
        
        for pattern, (strategy, priority, notes) in sorted_patterns:
            count = len(patterns.get(pattern, []))
            table += f"| **{pattern.replace('_', ' ').title()}** | {strategy} | {priority} | {notes} ({count} functions) |\n"
        
        return table
    
    def create_device_comparison_matrix(self, analysis: Dict[str, Any]) -> str:
        """Create a device comparison matrix."""
        
        patterns = analysis['patterns']
        
        matrix = """# Device Compatibility Matrix

## Function Count by Device Support

| Device Support | Count | Functions |
|----------------|-------|-----------|
"""
        
        # Create device support combinations
        device_combinations = {
            'CPU + CUDA + MPS': patterns['universal'],
            'CPU + CUDA': patterns['cpu_cuda'],
            'CPU + MPS': patterns['cpu_mps'],
            'CUDA + MPS': patterns['cuda_mps'],
            'CPU Only': patterns['cpu_only'],
            'CUDA Only': patterns['cuda_only'],
            'MPS Only': patterns['mps_only'],
            'Device Specific': patterns['device_specific'],
            'Unknown': patterns['unknown']
        }
        
        for support, funcs in device_combinations.items():
            if funcs:  # Only show combinations with functions
                sample_funcs = funcs[:5]  # Show first 5 as examples
                sample_text = ', '.join([f'`{f}`' for f in sample_funcs])
                if len(funcs) > 5:
                    sample_text += f' ... and {len(funcs) - 5} more'
                
                matrix += f"| {support} | {len(funcs)} | {sample_text} |\n"
        
        return matrix
    
    def generate_complete_analysis(self, catalog_file: str) -> Dict[str, str]:
        """Generate complete analysis and save to files."""
        
        print(f"Loading catalog from: {catalog_file}")
        catalog = self.load_catalog(catalog_file)
        
        print("Analyzing compatibility patterns...")
        analysis = self.analyze_compatibility_patterns(catalog)
        
        # Generate reports
        reports = {}
        
        print("Generating compatibility table...")
        reports['compatibility_table'] = self.create_compatibility_table(analysis)
        
        print("Generating translation strategy...")
        reports['translation_strategy'] = self.create_translation_strategy_table(analysis)
        
        print("Generating device comparison matrix...")
        reports['device_matrix'] = self.create_device_comparison_matrix(analysis)
        
        print("Generating summary report...")
        reports['summary'] = self._create_summary_report(analysis)
        
        return reports
    
    def _create_summary_report(self, analysis: Dict[str, Any]) -> str:
        """Create a summary report."""
        
        summary = analysis['summary']
        patterns = analysis['patterns']
        
        report = f"""# Device Compatibility Summary Report

## Overview
- **Total Functions Analyzed**: {summary['total_functions']}
- **Functions with Known Compatibility**: {summary['total_functions'] - len(patterns['unknown'])}
- **Functions Needing Investigation**: {len(patterns['unknown'])}

## Priority Analysis

### High Priority (CUDA Translation)
- **CUDA Only**: {len(patterns['cuda_only'])} functions
- **CPU + CUDA**: {len(patterns['cpu_cuda'])} functions
- **Device Specific**: {len(patterns['device_specific'])} functions

### Medium Priority (Optimization)
- **CPU Only**: {len(patterns['cpu_only'])} functions
- **CPU + MPS**: {len(patterns['cpu_mps'])} functions
- **CUDA + MPS**: {len(patterns['cuda_mps'])} functions

### Low Priority (Already Compatible)
- **Universal**: {len(patterns['universal'])} functions
- **MPS Only**: {len(patterns['mps_only'])} functions

## Implementation Recommendations

1. **Start with CUDA-only functions** - These are the core translation targets
2. **Focus on CPU+CUDA functions** - Important for CUDA compatibility
3. **Investigate unknown functions** - Determine compatibility status
4. **Optimize CPU-only functions** - May need performance improvements
5. **Leverage universal functions** - No translation needed

## Next Steps

1. Update function status in migration plan JSON
2. Implement translation for high-priority functions
3. Test compatibility for unknown functions
4. Create implementation guides for each pattern
"""
        
        return report


def main():
    """Main function to run the compatibility analysis."""
    
    # Default catalog file location
    catalog_file = "utils/data/comprehensive_function_catalog.json"
    
    # Check if file exists
    if not os.path.exists(catalog_file):
        print(f"Error: Catalog file not found: {catalog_file}")
        print("Please run generate_migration_plan.py first to create the catalog.")
        return
    
    analyzer = DeviceCompatibilityAnalyzer()
    
    print("Device Compatibility Analyzer")
    print("=" * 40)
    
    # Generate complete analysis
    reports = analyzer.generate_complete_analysis(catalog_file)
    
    # Save reports
    output_dir = "utils/docs/compatibility_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    for report_name, content in reports.items():
        filename = f"{output_dir}/{report_name}.md"
        with open(filename, 'w') as f:
            f.write(content)
        print(f"Saved: {filename}")
    
    print(f"\nAnalysis complete! Reports saved to: {output_dir}")
    print("\nFiles generated:")
    for report_name in reports.keys():
        print(f"  - {report_name}.md")


if __name__ == "__main__":
    main() 