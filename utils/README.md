# TorchDevice Utils Directory

This directory contains utilities for managing the TorchDevice migration plan and PyTorch function catalog.

## Directory Structure

```
utils/
├── bin/                    # Scripts for data management and report generation
│   ├── extract_pytorch_functions.py    # Extract PyTorch functions to JSON database
│   ├── generate_reports.py             # Generate all reports from JSON database
│   ├── update_status.py                # Update function implementation status
│   ├── extract_device_functions.py     # Extract device-specific functions
│   └── device_compatibility_analyzer.py # Analyze device compatibility
├── data/                   # JSON database and data files
│   └── comprehensive_function_catalog.json  # Main function catalog (Pydantic schema)
├── docs/                   # Generated documentation and reports
│   ├── migration_plan_unified/         # Migration plan markdown files
│   ├── compatibility_analysis/         # Device compatibility reports
│   ├── implementation_guide/           # Implementation guides
│   └── STATUS_SUMMARY.md               # Current implementation status
└── schemas/                # Pydantic schemas for type safety
    └── function_catalog.py             # Main schema for function catalog
```

## Scripts Overview

### Data Extraction
- **[extract_pytorch_functions.py](bin/extract_pytorch_functions.py)**: Extracts all PyTorch functions from the installed environment and saves them to the JSON database using Pydantic schema for type safety.

### Report Generation
- **[generate_reports.py](bin/generate_reports.py)**: Generates all reports from the JSON database, including migration plan markdown files, status summaries, and device compatibility analysis.

### Status Management
- **[update_status.py](bin/update_status.py)**: Update function implementation status and notes in the JSON database.

### Analysis Tools
- **[extract_device_functions.py](bin/extract_device_functions.py)**: Extract device-specific functions for focused analysis.
- **[device_compatibility_analyzer.py](bin/device_compatibility_analyzer.py)**: Analyze device compatibility across CPU, CUDA, MPS, and MLX.

## Pydantic Schema Integration

All scripts now use the Pydantic schema defined in [schemas/function_catalog.py](schemas/function_catalog.py) for:
- **Type Safety**: All data is validated against the schema
- **Consistency**: Ensures data structure consistency across all scripts
- **Validation**: Automatic validation of function data, status updates, and device support
- **Serialization**: Proper JSON serialization with Unicode support

## Usage

### Generate Migration Plan
```bash
cd utils/bin
python extract_pytorch_functions.py
```
This creates/updates the JSON database with all PyTorch functions from your current environment.

### Update Implementation Status
```bash
cd utils/bin
python update_status.py --list-categories
python update_status.py --function "torch.tensor" --status "complete"
python update_status.py --category "neural_network" --status "in_progress"
```

### Generate Reports
```bash
cd utils/bin
python generate_reports.py
```

### Data Management
- The JSON database in [data/comprehensive_function_catalog.json](data/comprehensive_function_catalog.json) is persistent and preserves:
  - Function status updates
  - Version history
  - Implementation progress
  - Timestamps for changes
- New runs of [bin/extract_pytorch_functions.py](bin/extract_pytorch_functions.py) merge with existing data
- Status updates are preserved across regeneration

## Key Features

1. **No Duplication**: Functions are categorized uniquely (no overlap)
2. **Version Tracking**: Tracks PyTorch version changes and deprecations
3. **Status Persistence**: Implementation status survives regeneration
4. **Dynamic Discovery**: Auto-discovers PyTorch modules
5. **Comprehensive Coverage**: 1,299 functions across 29 categories
6. **Organized Output**: Clean directory structure with clear separation
7. **Type Safety**: Full Pydantic schema validation
8. **Device Support**: Track compatibility across CPU, CUDA, MPS, and MLX

## Migration Plan Structure

The generated migration plan includes:
- Executive summary and architecture overview
- Function routing matrix with device compatibility
- Implementation phases and priority matrix
- Risk mitigation strategies
- Success metrics and progress tracking
- Complete function catalog with status tracking

### Implementation Phases (Accelerated Timeline)

#### Phase 1: Foundation (Days 1-3)
- Device management functions (torch.device, torch.cuda.is_available, etc.)
- Basic tensor creation (torch.tensor, torch.zeros, torch.ones, etc.)
- Core mathematical operations (torch.add, torch.mul, torch.matmul, etc.)

#### Phase 2: Neural Networks (Days 4-7)
- Activation functions (torch.nn.functional.relu, torch.nn.functional.sigmoid, etc.)
- Loss functions (torch.nn.functional.mse_loss, torch.nn.functional.cross_entropy, etc.)
- Basic neural network operations (torch.nn.Linear, torch.nn.Conv2d, etc.)

#### Phase 3: Advanced Operations (Days 8-10)
- Complex mathematical operations (torch.linalg.inv, torch.linalg.eig, etc.)
- Linear algebra functions (torch.matrix_rank, torch.svd, etc.)
- Signal processing (torch.fft.fft, torch.stft, etc.)

#### Phase 4: Optimization (Days 11-14)
- Performance optimization
- Memory management
- Stream operations
- Final testing and validation

### Architecture Overview

The system uses a function router design with:
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

## Data Files

- **[comprehensive_function_catalog.json](data/comprehensive_function_catalog.json)**: Main function catalog with 1,299 PyTorch functions across 29 categories

## Reports

- **Migration Plan**: Split into manageable markdown files in [docs/migration_plan_unified/migration_plan_section_01.md](docs/migration_plan_unified/migration_plan_section_01.md)
- **Status Summary**: Current implementation progress in [docs/STATUS_SUMMARY.md](docs/STATUS_SUMMARY.md)
- **Device Compatibility**: Analysis reports in [docs/compatibility_analysis/compatibility_table.md](docs/compatibility_analysis/compatibility_table.md)
- **Implementation Guide**: Guides in [docs/implementation_guide/device_implementation_guide.md](docs/implementation_guide/device_implementation_guide.md)

## Quick Navigation

### Main Files:
- **[Main Script](bin/extract_pytorch_functions.py)** - Extract PyTorch functions
- **[Status Manager](bin/update_status.py)** - Update implementation status
- **[Report Generator](bin/generate_reports.py)** - Generate all reports
- **[Main Database](data/comprehensive_function_catalog.json)** - Function catalog with status
- **[Migration Plan](docs/migration_plan_unified/migration_plan_section_01.md)** - Complete migration plan
- **[Status Summary](docs/STATUS_SUMMARY.md)** - Current implementation status

### Key Directories:
- **[Scripts](bin/)** - All active tools and utilities
- **[Data](data/)** - JSON databases and data files
- **[Documentation](docs/)** - Generated plans and analysis
- **[Schemas](schemas/)** - Pydantic schema definitions 