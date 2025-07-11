#!/usr/bin/env python3
"""
Pydantic Schemas for TorchDevice Function Catalog
Provides type safety and validation for function catalog data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ImplementationStatus(str, Enum):
    """Valid implementation status values."""
    NOT_STARTED = "not_started"
    DESIGN = "design"
    IN_PROGRESS = "in_progress"
    TESTING = "testing"
    COMPLETE = "complete"
    NOT_IMPLEMENTED = "not_implemented"


class DeviceSupport(BaseModel):
    """Device support information for a function."""
    cpu: bool = Field(default=False, description="Supports CPU operations")
    cuda: bool = Field(default=False, description="Supports CUDA operations")
    mps: bool = Field(default=False, description="Supports MPS operations")
    mlx: bool = Field(default=False, description="Supports MLX operations")
    
    @field_validator('cpu', 'cuda', 'mps', 'mlx', mode='before')
    @classmethod
    def validate_device_support(cls, v):
        """Convert emoji or string values to boolean."""
        if isinstance(v, str):
            # Convert emoji or text to boolean
            if v in ['❓', '❌', 'false', 'False', '0', '']:
                return False
            elif v in ['✅', 'true', 'True', '1']:
                return True
            else:
                return False  # Default to False for unknown values
        return bool(v)


class VersionInfo(BaseModel):
    """Version information for a function."""
    version: str = Field(..., description="PyTorch version")
    status: str = Field(default="current", description="Version status")
    signature: str = Field(default="", description="Function signature")
    doc: str = Field(default="", description="Function documentation")
    device_support: Optional[DeviceSupport] = Field(default=None, description="Device support for this version")
    added_in: Optional[str] = Field(default=None, description="Version where function was added")
    deprecated_in: Optional[str] = Field(default=None, description="Version where function was deprecated")
    changed_in: Optional[str] = Field(default=None, description="Version where function was changed")
    added_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), description="Date added to catalog")


class MLXMapping(BaseModel):
    """MLX function mapping information."""
    mlx_function: str = Field(default="", description="Corresponding MLX function name")
    confidence: str = Field(default="", description="Mapping confidence (exact, mapped, partial)")
    module: str = Field(default="", description="MLX module name")
    notes: str = Field(default="", description="Mapping notes")
    
    @field_validator('mlx_function', 'confidence', 'module', 'notes', mode='before')
    @classmethod
    def validate_mlx_fields(cls, v):
        """Convert None values to empty strings."""
        if v is None:
            return ""
        return str(v)


class PyTorchFunction(BaseModel):
    """Schema for a single PyTorch function."""
    function: str = Field(..., description="Function name (e.g., 'torch.device')")
    signature: str = Field(default="", description="Function signature")
    doc: str = Field(default="", description="Function documentation")
    arguments: List[str] = Field(default_factory=list, description="Function arguments")
    return_type: str = Field(default="Any", description="Function return type")

    # Implementation tracking
    implementation_status: ImplementationStatus = Field(
        default=ImplementationStatus.NOT_STARTED,
        description="Current implementation status"
    )
    implementation_notes: Optional[str] = Field(
        default=None,
        description="Custom notes about implementation (especially for not_implemented)"
    )
    status_updated: Optional[str] = Field(
        default=None,
        description="Date when status was last updated"
    )

    # Version tracking
    versions: List[VersionInfo] = Field(
        default_factory=list,
        description="Version history of the function"
    )
    current_version: Optional[str] = Field(
        default=None,
        description="Current PyTorch version"
    )
    last_updated: Optional[str] = Field(
        default=None,
        description="Last update date"
    )

    # Device support
    device_support: Optional[DeviceSupport] = Field(
        default=None,
        description="Current device support information"
    )

    # MLX cross-mapping
    mlx_mapping: Optional[MLXMapping] = Field(
        default=None,
        description="MLX function mapping information"
    )

    # Deprecation
    deprecation_warnings: List[str] = Field(
        default_factory=list,
        description="Deprecation warnings"
    )

    # Metadata
    added_date: Optional[str] = Field(
        default=None,
        description="Date function was added to catalog"
    )

    @field_validator('function')
    @classmethod
    def validate_function_name(cls, v):
        """Validate function name format."""
        if not v or not isinstance(v, str):
            raise ValueError("Function name must be a non-empty string")
        if not v.startswith('torch.'):
            raise ValueError("Function name must start with 'torch.'")
        return v

    @field_validator('implementation_status', mode='before')
    @classmethod
    def validate_status(cls, v):
        """Validate implementation status."""
        if isinstance(v, str):
            v = v.lower()
        return v

    def update_status(self, status: ImplementationStatus, notes: Optional[str] = None) -> None:
        """Update implementation status and notes."""
        self.implementation_status = status
        self.status_updated = datetime.now().strftime("%Y-%m-%d")
        if notes:
            self.implementation_notes = notes

    def add_version(self, version_info: VersionInfo) -> None:
        """Add a new version entry."""
        self.versions.append(version_info)
        self.current_version = version_info.version
        self.last_updated = version_info.added_date


class FunctionCatalog(BaseModel):
    """Schema for the complete function catalog."""
    categories: Dict[str, List[PyTorchFunction]] = Field(
        default_factory=dict,
        description="Functions organized by category"
    )

    # Metadata
    total_functions: int = Field(default=0, description="Total number of functions")
    generated_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        description="Date when catalog was generated"
    )
    pytorch_version: Optional[str] = Field(
        default=None,
        description="PyTorch version used for extraction"
    )
    mlx_version: Optional[str] = Field(
        default=None,
        description="MLX version used for cross-mapping"
    )

    @field_validator('categories')
    @classmethod
    def validate_categories(cls, v):
        """Validate categories structure."""
        if not isinstance(v, dict):
            raise ValueError("Categories must be a dictionary")
        return v

    def get_function(self, function_name: str) -> Optional[PyTorchFunction]:
        """Get a function by name."""
        for functions in self.categories.values():
            for func in functions:
                if func.function == function_name:
                    return func
        return None

    def update_all_function_status(self, function_name: str, status: ImplementationStatus, notes: Optional[str] = None) -> int:
        """Update status for all instances of a function by name in all categories. Returns number of updates."""
        count = 0
        for category, functions in self.categories.items():
            for func in functions:
                if func.function == function_name:
                    func.update_status(status, notes)
                    count += 1
        return count

    def update_function_status(self, function_name: str, status: ImplementationStatus, notes: Optional[str] = None) -> bool:
        """Update status for a specific function (all instances). Warn if duplicates."""
        count = self.update_all_function_status(function_name, status, notes)
        if count > 1:
            print(f"Warning: Function '{function_name}' found in {count} categories. All instances updated.")
        return count > 0

    def get_functions_by_status(self, status: ImplementationStatus) -> List[PyTorchFunction]:
        """Get all functions with a specific status."""
        result = []
        for functions in self.categories.values():
            for func in functions:
                if func.implementation_status == status:
                    result.append(func)
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get implementation statistics."""
        stats = {
            'total_functions': 0,
            'by_status': {status.value: 0 for status in ImplementationStatus},
            'by_category': {}
        }

        for category, functions in self.categories.items():
            stats['total_functions'] += len(functions)
            stats['by_category'][category] = {
                'total': len(functions),
                'by_status': {status.value: 0 for status in ImplementationStatus}
            }

            for func in functions:
                status = func.implementation_status.value
                stats['by_status'][status] += 1
                stats['by_category'][category]['by_status'][status] += 1

        return stats


# Convenience functions for working with the catalog
def load_catalog_from_json(filename: str) -> FunctionCatalog:
    """Load function catalog from JSON file."""
    import json
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert the flat structure to our schema
    categories = {}
    total_functions = 0
    for category_name, functions in data.items():
        categories[category_name] = [PyTorchFunction(**func) for func in functions]
        total_functions += len(functions)

    return FunctionCatalog(categories=categories, total_functions=total_functions)


def save_catalog_to_json(catalog: FunctionCatalog, filename: str) -> None:
    """Save function catalog to JSON file."""
    import json
    import os

    # Convert to flat structure for JSON
    data = {}
    for category_name, functions in catalog.categories.items():
        data[category_name] = [func.model_dump() for func in functions]

    # Ensure directory exists (only if filename has a directory)
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_empty_catalog() -> FunctionCatalog:
    """Create an empty function catalog."""
    return FunctionCatalog()


# Example usage
if __name__ == "__main__":
    # Create a sample function
    sample_func = PyTorchFunction(
        function="torch.device",
        signature="torch.device(device)",
        doc="Creates a device object",
        arguments=["device"],
        return_type="torch.device",
        implementation_status=ImplementationStatus.IN_PROGRESS,
        implementation_notes="Core device translation function"
    )

    # Create a catalog
    catalog = FunctionCatalog(categories={
        "TORCH_DEVICE": [sample_func]
    })

    print("Sample catalog created successfully!")
    print(f"Total functions: {catalog.total_functions}")
    print(f"Categories: {list(catalog.categories.keys())}")
