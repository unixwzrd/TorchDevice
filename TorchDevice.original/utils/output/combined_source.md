## genmd Settings

| Variable               | Value                                                                 |
|------------------------|-----------------------------------------------------------------------|
|add_line_numbers|"true"|
|compress|"false"|
|compression_tool|"gzip"|
|count_tokens|"false"|
|create_date|"2025-05-11 08:52:39"|
|debug_level|"20"|
|dir_excludes|(".git" "tmp" "log" "__pycache__" ".vscode" )|
|dry_run|"false"|
|file_excludes|("*.ico" "*.svg" "*.png" "*.pdf" "*.jpg" "*.htaccess" "*.webp" "*.jekyll" ".DS_Store" "*.JPG" )|
|file_includes|(".py" )|
|follow_links|""|
|GENMD_BASE|"."|
|output_filename|"./utils/output/combined_source.md"|
|pattern_excludes|()|
|remove_blanks|"false"|
|settings_modes|("md" "cfg" )|
|token_count|"0"|
|use_gitignore|"true"|


## Project filesystem directory structure
```text
filetree -l 20 -i .py -e tmp .git .git tmp log __pycache__ .vscode *.ico *.svg *.png *.pdf *.jpg *.htaccess *.webp *.jekyll .DS_Store *.JPG
Root Directory
├── TorchDevice.py
├── __init__.py
├── _deferred_patches.py
├── cuda/
│   ├── device.py
│   ├── memory.py
│   ├── random.py
│   ├── streams.py
│   └── unassigned.py
├── modules/
│   ├── TDLogger.py
│   ├── __init__.py
│   ├── compile.py
│   └── patch.py
└── patch.py

```

## Files included in final output
- ./__init__.py
- ./_deferred_patches.py
- ./cuda/device.py
- ./cuda/memory.py
- ./cuda/random.py
- ./cuda/streams.py
- ./cuda/unassigned.py
- ./modules/__init__.py
- ./modules/compile.py
- ./modules/patch.py
- ./modules/TDLogger.py
- ./patch.py
- ./TorchDevice.py

---


## Filename ==>  ./__init__.py
```python
     1	"""
     2	TorchDevice library for managing PyTorch device operations.
     3	This module patches PyTorch's CUDA functionality to work seamlessly with MPS (and vice-versa)
     4	upon import. Users should never need to call patch functions directly—patching is automatic.
     5	"""
     6	
     7	__version__ = '0.2.0'
     8	
     9	from .TorchDevice import TorchDevice
    10	from .modules.TDLogger import auto_log
    11	from .modules import patch
    12	from .modules import compile
    13	
    14	# Apply all monkey-patches automatically on import
    15	# Users should never call patch functions directly.
    16	patch.apply_all_patches()
    17	TorchDevice.get_default_device()
    18	
    19	
    20	# Expose a function to apply deferred patches - these must be run after core patching
    21	def apply_deferred_patches():
    22	    """Apply patches that must be run after the core system is initialized."""
    23	    compile.patch_dynamo_config()
    24	
    25	
    26	# Run deferred patches - but only after import is complete
    27	# This prevents circular import issues
    28	from . import _deferred_patches  # noqa
    29	
    30	__all__ = ['TorchDevice', 'auto_log', '__version__']

```


## Filename ==>  ./_deferred_patches.py
```python
     1	"""
     2	TorchDevice._deferred_patches - Run patches that need to be applied after main initialization
     3	
     4	This module is imported at the end of __init__.py to ensure that these patches run
     5	after all the core initialization is complete. This prevents circular import issues
     6	and helps avoid premature trigger of imports that could cause initialization problems.
     7	"""
     8	
     9	from .modules.TDLogger import log_info
    10	from .modules import compile
    11	
    12	# Apply deferred patches
    13	log_info("Applying deferred patches for PyTorch compiler")
    14	compile.patch_dynamo_config()
    15	log_info("Deferred patching complete") 

```


## Filename ==>  ./cuda/device.py
```python
     1	import torch
     2	from typing import Optional, Any
     3	import psutil
     4	from ..modules.TDLogger import auto_log
     5	import contextlib
     6	
     7	# --- Device-related Emulation Functions ---
     8	
     9	def _set_device(device: Optional[Any]) -> None:
    10	    """Set the current device (only mps:0 is supported)."""
    11	    if device not in (0, "mps", "mps:0", None):
    12	        pass
    13	
    14	
    15	def _current_device() -> int:
    16	    """Return the current device index (always 0 for MPS/CPU)."""
    17	    return 0
    18	
    19	
    20	def _device_count() -> int:
    21	    """Return the number of available devices (1 for MPS, 0 for CPU)."""
    22	    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    23	        return 1
    24	    return 0
    25	
    26	
    27	def _is_available() -> bool:
    28	    """Return True if a CUDA/MPS device is available, False otherwise."""
    29	    return (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
    30	
    31	
    32	def _get_device_name(device: Optional[Any] = None) -> str:
    33	    """Return the name of the current device."""
    34	    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    35	        return 'Apple MPS'
    36	    return 'CPU'
    37	
    38	
    39	def _get_device_capability(device: Optional[Any] = None) -> tuple:
    40	    """Return the device capability (mocked for MPS/CPU)."""
    41	    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    42	        return (0, 0)
    43	    return (0, 0)
    44	
    45	
    46	def _get_device_properties(device: Optional[Any] = None):
    47	    """Return device properties (mocked for MPS/CPU)."""
    48	    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    49	        class MPSDeviceProperties:
    50	            name = 'Apple MPS'
    51	            total_memory = psutil.virtual_memory().total
    52	
    53	            def __str__(self):
    54	                return f'MPSDeviceProperties(name={self.name}, total_memory={self.total_memory})'
    55	        return MPSDeviceProperties()
    56	    else:
    57	        raise RuntimeError("No GPU device available")
    58	
    59	
    60	def _is_initialized() -> bool:
    61	    """Return True if CUDA/MPS is initialized."""
    62	    return _is_available()
    63	
    64	
    65	def _get_arch_list() -> list:
    66	    """Return architecture list (mocked for MPS/CPU)."""
    67	    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    68	        return ['mps']
    69	    return []
    70	
    71	
    72	def _is_built() -> bool:
    73	    """Return True if CUDA/MPS is built (mocked)."""
    74	    return _is_available()
    75	
    76	
    77	def _cuda_device_context(device=0):
    78	    # On MPS, masquerade as CUDA context manager (no-op)
    79	    yield
    80	
    81	
    82	# --- Patch Application ---
    83	
    84	def apply_patches() -> None:
    85	    import torch
    86	    # Patch CUDA context manager to masquerade as CUDA on MPS
    87	    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
    88	        torch.cuda.device = contextlib.contextmanager(_cuda_device_context)
    89	        # Patch is_built as a function, not a bool
    90	        if hasattr(torch.backends.cuda, 'is_built'):
    91	            torch.backends.cuda.is_built = lambda: True
    92	        else:
    93	            setattr(torch.backends.cuda, 'is_built', lambda: True)
    94	        torch.cuda.is_available = lambda: True
    95	    torch.cuda.set_device = _set_device
    96	    torch.cuda.current_device = _current_device
    97	    torch.cuda.device_count = _device_count
    98	    torch.cuda.is_available = _is_available
    99	    torch.cuda.get_device_name = _get_device_name
   100	    torch.cuda.get_device_capability = _get_device_capability
   101	    torch.cuda.get_device_properties = _get_device_properties
   102	    torch.cuda.is_initialized = _is_initialized
   103	    torch.cuda.get_arch_list = _get_arch_list
   104	    # Patch torch.cuda.is_built as a function for API consistency
   105	    torch.cuda.is_built = _is_built
   106	
   107	
   108	# Device-related mock_* functions migrated from TorchDevice.py
   109	
   110	# Remove from here:
   111	@auto_log()
   112	def mock_cuda_is_available(cls):
   113	    return cls._default_device in ['cuda', 'mps']
   114	
   115	
   116	@auto_log()
   117	def mock_cuda_device_count(cls):
   118	    if cls._default_device == 'cuda':
   119	        return cls._original_torch_cuda_device_count()
   120	    elif cls._default_device == 'mps':
   121	        return 1
   122	    else:
   123	        return 0
   124	
   125	
   126	@auto_log()
   127	def mock_cuda_get_device_properties(cls, device):
   128	    if cls._default_device == 'cuda':
   129	        return cls._original_torch_cuda_get_device_properties(device)
   130	    elif cls._default_device == 'mps':
   131	        class MPSDeviceProperties:
   132	            name = 'Apple MPS'
   133	            total_memory = psutil.virtual_memory().total
   134	
   135	            def __str__(self):
   136	                return f'MPSDeviceProperties(name={self.name}, total_memory={self.total_memory})'
   137	        return MPSDeviceProperties()
   138	    else:
   139	        raise RuntimeError("No GPU device available")
   140	
   141	
   142	@auto_log()
   143	def mock_cuda_is_initialized(cls):
   144	    return cls._default_device in ['cuda', 'mps']
   145	
   146	
   147	@auto_log()
   148	def mock_cuda_get_arch_list(cls):
   149	    if cls._default_device == 'cuda':
   150	        return cls._original_torch_cuda_get_arch_list()
   151	    elif cls._default_device == 'mps':
   152	        return ['mps']
   153	    else:
   154	        return []
   155	
   156	
   157	@auto_log()
   158	def mock_cuda_is_built(cls):
   159	    if cls._default_device in ['cuda', 'mps']:
   160	        return True
   161	    else:
   162	        return False
   163	
   164	
   165	@auto_log()
   166	def mock_cuda_get_device_name(cls, device=None):
   167	    if cls._default_device == 'cuda':
   168	        return cls._original_torch_cuda_get_device_name(device)
   169	    elif cls._default_device == 'mps':
   170	        return 'Apple MPS'
   171	    else:
   172	        return 'CPU'
   173	
   174	
   175	@auto_log()
   176	def mock_cuda_set_device(cls, device):
   177	    if cls._default_device == 'cuda':
   178	        cls._original_torch_cuda_set_device(device)
   179	
   180	
   181	@auto_log()
   182	def mock_cuda_synchronize(cls, device=None):
   183	    if cls._default_device == 'cuda':
   184	        cls._original_torch_cuda_synchronize(device)
   185	    elif cls._default_device == 'mps':
   186	        import torch
   187	        torch.mps.synchronize()
   188	
   189	
   190	@auto_log()
   191	def mock_cuda_get_device_capability(cls, device=None):
   192	    if cls._default_device == 'cuda':
   193	        return cls._original_torch_cuda_get_device_capability(device)
   194	    elif cls._default_device == 'mps':
   195	        return (0, 0)
   196	    else:
   197	        return (0, 0)
   198	
   199	
   200	@auto_log()
   201	def mock_cuda_ipc_collect(cls):
   202	    if cls._default_device == 'cuda':
   203	        import torch
   204	        return torch.cuda.ipc_collect()
   205	
   206	
   207	@auto_log()
   208	def mock_cuda_function_stub(cls, *args, **kwargs):
   209	    pass
   210	
   211	
   212	@auto_log()
   213	def mock_cuda_current_device(cls):
   214	    if cls._default_device == 'cuda':
   215	        return cls._original_torch_cuda_current_device()
   216	    elif cls._default_device == 'mps':
   217	        return 0
   218	    else:
   219	        return -1
   220	
   221	
   222	@auto_log()
   223	def mock_cuda_device_context(cls, device=None):
   224	    class DeviceContextManager:
   225	        @auto_log()
   226	        def __init__(self, device):
   227	            self.device = device
   228	
   229	        @auto_log()
   230	        def __enter__(self):
   231	            cls.mock_cuda_set_device(self.device)
   232	
   233	        @auto_log()
   234	        def __exit__(self, exc_type, exc_value, traceback):
   235	            pass
   236	    return DeviceContextManager(device)
   237	
   238	
   239	@auto_log()
   240	def mock_cuda_empty_cache(cls):
   241	    if cls._default_device == 'cuda':
   242	        cls._original_torch_cuda_empty_cache()
   243	    elif cls._default_device == 'mps':
   244	        import torch
   245	        torch.mps.empty_cache()
   246	
   247	
   248	# Remove up to here.
   249	# ... existing code ... 

```


## Filename ==>  ./cuda/memory.py
```python
     1	"""
     2	TorchDevice Memory Management and Patching
     3	-----------------------------------------
     4	All memory management, emulation, and patching logic for TorchDevice is centralized in this module.
     5	
     6	
     7	"""
     8	
     9	import psutil
    10	import os
    11	from typing import Optional, Tuple, Dict, Any
    12	from ..modules.TDLogger import auto_log
    13	
    14	# --- Internal Memory-related Emulation Functions (use leading underscore) ---
    15	
    16	def _memory_allocated(device: Optional[int] = None) -> int:
    17	    proc = psutil.Process(os.getpid())
    18	    return proc.memory_info().rss
    19	
    20	@auto_log()
    21	def _memory_reserved(device: Optional[int] = None) -> int:
    22	    return psutil.virtual_memory().total
    23	
    24	@auto_log()
    25	def _max_memory_allocated(device: Optional[int] = None) -> int:
    26	    return _memory_allocated(device)
    27	
    28	@auto_log()
    29	def _max_memory_reserved(device: Optional[int] = None) -> int:
    30	    return _memory_reserved(device)
    31	
    32	@auto_log()
    33	def _mem_get_info(device: Optional[int] = None) -> Tuple[int, int]:
    34	    vm = psutil.virtual_memory()
    35	    return vm.available, vm.total
    36	
    37	@auto_log()
    38	def _memory_stats(device: Optional[int] = None) -> Dict[str, Any]:
    39	    allocated = _memory_allocated(device)
    40	    reserved = _memory_reserved(device)
    41	    stats = {
    42	        'active.all.current': allocated,
    43	        'active.all.peak': _max_memory_allocated(device),
    44	        'reserved_bytes.all.current': reserved,
    45	        'reserved_bytes.all.peak': _max_memory_reserved(device),
    46	        'allocated': allocated,
    47	        'reserved': reserved,
    48	        'free': psutil.virtual_memory().available,
    49	        'total': psutil.virtual_memory().total,
    50	    }
    51	    return stats
    52	
    53	@auto_log()
    54	def _memory_snapshot(device: Optional[int] = None):
    55	    return [{
    56	        'device': 0,
    57	        'address': 0,
    58	        'total_size': _memory_allocated(device),
    59	        'allocated_size': _memory_allocated(device),
    60	        'active': True,
    61	        'segment_type': 'small_pool',
    62	    }]
    63	
    64	@auto_log()
    65	def _memory_summary(device: Optional[int] = None, abbreviated: bool = False) -> str:
    66	    stats = _memory_stats(device)
    67	    return (f"Memory Allocated: {stats['allocated']} bytes\n"
    68	            f"Memory Reserved: {stats['reserved']} bytes\n"
    69	            f"Memory Free: {stats['free']} bytes\n"
    70	            f"Memory Total: {stats['total']} bytes\n")
    71	
    72	@auto_log()
    73	def _reset_peak_memory_stats(device: Optional[int] = None) -> None:
    74	    pass
    75	
    76	@auto_log()
    77	def _reset_accumulated_memory_stats(device: Optional[int] = None) -> None:
    78	    pass
    79	
    80	@auto_log()
    81	def _reset_max_memory_allocated(device: Optional[int] = None) -> None:
    82	    pass
    83	
    84	@auto_log()
    85	def _reset_max_memory_reserved(device: Optional[int] = None) -> None:
    86	    pass
    87	
    88	@auto_log()
    89	def _empty_cache(device: Optional[int] = None):
    90	    import torch
    91	    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
    92	        torch.mps.empty_cache()
    93	
    94	@auto_log()
    95	def mock_cuda_reset_peak_memory_stats(cls, device=None):
    96	    pass
    97	
    98	@auto_log()
    99	def mock_cuda_reset_accumulated_memory_stats(cls, device=None):
   100	    pass
   101	
   102	@auto_log()
   103	def mock_cuda_reset_max_memory_allocated(cls, device=None):
   104	    pass
   105	
   106	@auto_log()
   107	def mock_cuda_reset_max_memory_reserved(cls, device=None):
   108	    pass
   109	
   110	# --- Public API functions (patched onto torch.cuda) ---
   111	def memory_allocated(device: Optional[int] = None) -> int:
   112	    return _memory_allocated(device)
   113	def memory_reserved(device: Optional[int] = None) -> int:
   114	    return _memory_reserved(device)
   115	def max_memory_allocated(device: Optional[int] = None) -> int:
   116	    return _max_memory_allocated(device)
   117	def max_memory_reserved(device: Optional[int] = None) -> int:
   118	    return _max_memory_reserved(device)
   119	def mem_get_info(device: Optional[int] = None) -> Tuple[int, int]:
   120	    return _mem_get_info(device)
   121	def memory_stats(device: Optional[int] = None) -> Dict[str, Any]:
   122	    return _memory_stats(device)
   123	def memory_snapshot(device: Optional[int] = None):
   124	    return _memory_snapshot(device)
   125	def memory_summary(device: Optional[int] = None, abbreviated: bool = False) -> str:
   126	    return _memory_summary(device, abbreviated)
   127	def reset_peak_memory_stats(device: Optional[int] = None) -> None:
   128	    return _reset_peak_memory_stats(device)
   129	def reset_accumulated_memory_stats(device: Optional[int] = None) -> None:
   130	    return _reset_accumulated_memory_stats(device)
   131	def reset_max_memory_allocated(device: Optional[int] = None) -> None:
   132	    return _reset_max_memory_allocated(device)
   133	def reset_max_memory_reserved(device: Optional[int] = None) -> None:
   134	    return _reset_max_memory_reserved(device)
   135	def empty_cache(device: Optional[int] = None):
   136	    return _empty_cache(device)
   137	
   138	def apply_patches() -> None:
   139	    import torch
   140	    torch.cuda.memory_allocated = memory_allocated
   141	    torch.cuda.memory_reserved = memory_reserved
   142	    torch.cuda.max_memory_allocated = max_memory_allocated
   143	    torch.cuda.max_memory_reserved = max_memory_reserved
   144	    torch.cuda.mem_get_info = mem_get_info
   145	    torch.cuda.memory_stats = memory_stats
   146	    torch.cuda.memory_summary = memory_summary
   147	    torch.cuda.memory_snapshot = memory_snapshot
   148	    torch.cuda.empty_cache = empty_cache
   149	    torch.cuda.reset_peak_memory_stats = reset_peak_memory_stats
   150	    torch.cuda.reset_accumulated_memory_stats = reset_accumulated_memory_stats
   151	    torch.cuda.reset_max_memory_allocated = reset_max_memory_allocated
   152	    torch.cuda.reset_max_memory_reserved = reset_max_memory_reserved 

```


## Filename ==>  ./cuda/random.py
```python
     1	"""
     2	TorchDevice RNG/Seed Logic
     3	-------------------------
     4	All random number generation and seed management logic for TorchDevice is centralized in this module.
     5	This includes patching for torch and torch.cuda, and device-aware tensor creation wrappers.
     6	"""
     7	
     8	import torch
     9	from typing import Callable, Any, List, Optional
    10	from ..modules.TDLogger import auto_log, log_info
    11	from ..TorchDevice import TorchDevice
    12	
    13	def tensor_creation_wrapper(original_func: Callable) -> Callable:
    14	    """
    15	    Wrapper for tensor creation functions to enforce default device redirection and CPU override.
    16	    Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
    17	    """
    18	    @auto_log()
    19	    def wrapped_func(*args, **kwargs):
    20	        device_arg = kwargs.get('device', None)
    21	        # If device is not specified, inject the current device (default or override)
    22	        if device_arg is None:
    23	            device = TorchDevice.torch_device_replacement()
    24	            log_info(f"[tensor_creation_wrapper] Injecting device: {device}")
    25	            kwargs['device'] = device
    26	        else:
    27	            # Always pass through torch_device_replacement to handle override logic
    28	            device = TorchDevice.torch_device_replacement(device_arg)
    29	            log_info(f"[tensor_creation_wrapper] Normalized device: {device}")
    30	            kwargs['device'] = device
    31	        return original_func(*args, **kwargs)
    32	    return wrapped_func
    33	
    34	# RNG/seed logic implementations
    35	
    36	def manual_seed(seed: int) -> None:
    37	    """Set the random seed for PyTorch and MPS if available."""
    38	    torch.manual_seed(seed)
    39	    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
    40	        torch.mps.manual_seed(seed)
    41	
    42	def manual_seed_all(seed: int) -> None:
    43	    """Set the random seed for all devices (only one device in MPS/CPU)."""
    44	    manual_seed(seed)
    45	
    46	def seed() -> int:
    47	    """Set a random seed for PyTorch and MPS if available, and return it."""
    48	    s = torch.seed()
    49	    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
    50	        torch.mps.manual_seed(s)
    51	    return s
    52	
    53	def seed_all() -> int:
    54	    """Set a random seed for all devices (only one device in MPS/CPU)."""
    55	    return seed()
    56	
    57	def get_rng_state(device: Optional[Any] = None) -> torch.Tensor:
    58	    """Get RNG state for the current device (MPS or CPU)."""
    59	    return torch.get_rng_state()
    60	
    61	def set_rng_state(state: torch.Tensor, device: Optional[Any] = None) -> None:
    62	    """Set RNG state for the current device (MPS or CPU)."""
    63	    torch.set_rng_state(state)
    64	
    65	def get_rng_state_all() -> List[torch.Tensor]:
    66	    """Get RNG state for all devices (only one device in MPS/CPU)."""
    67	    return [torch.get_rng_state()]
    68	
    69	def set_rng_state_all(states: List[torch.Tensor]) -> None:
    70	    """Set RNG state for all devices (only one device in MPS/CPU)."""
    71	    if states:
    72	        torch.set_rng_state(states[0])
    73	
    74	def initial_seed() -> int:
    75	    """Return the initial seed for PyTorch."""
    76	    return torch.initial_seed() if hasattr(torch, "initial_seed") else torch.seed()
    77	
    78	def apply_patches() -> None:
    79	    # Patch tensor creation functions
    80	    tensor_creation_functions = [
    81	        'tensor', 'zeros', 'ones', 'empty', 'randn', 'rand', 'randint', 'arange', 'linspace', 'logspace'
    82	    ]
    83	    for func_name in tensor_creation_functions:
    84	        if hasattr(torch, func_name):
    85	            original_func = getattr(torch, func_name)
    86	            setattr(torch, func_name, tensor_creation_wrapper(original_func))
    87	    # Patch RNG/seed logic for torch and torch.cuda
    88	    torch.manual_seed = manual_seed
    89	    torch.seed = seed
    90	    torch.get_rng_state = get_rng_state
    91	    torch.set_rng_state = set_rng_state
    92	    if hasattr(torch, "cuda"):
    93	        torch.cuda.manual_seed = manual_seed
    94	        torch.cuda.manual_seed_all = manual_seed_all
    95	        torch.cuda.seed = seed
    96	        torch.cuda.seed_all = seed_all
    97	        torch.cuda.get_rng_state = get_rng_state
    98	        torch.cuda.set_rng_state = set_rng_state
    99	        torch.cuda.get_rng_state_all = get_rng_state_all
   100	        torch.cuda.set_rng_state_all = set_rng_state_all
   101	        torch.cuda.initial_seed = initial_seed 

```


## Filename ==>  ./cuda/streams.py
```python
     1	import torch
     2	from typing import Any, Optional
     3	import time
     4	from ..modules.TDLogger import auto_log, log_info
     5	from ..TorchDevice import TorchDevice
     6	
     7	# --- Stream and Event Replacement Classes/Functions ---
     8	
     9	class _cuda_Stream:
    10	    """Replacement for torch.cuda.Stream (MPS/CPU)."""
    11	    @auto_log()
    12	    def __init__(self, device: Optional[Any] = None, priority: int = 0):
    13	        if device is None:
    14	            device = TorchDevice.get_default_device()
    15	        self.device = torch.device(device)
    16	        self.priority = priority
    17	        self._is_created = True
    18	        self._is_destroyed = False
    19	
    20	        # --- NEW: alias for demo code's stream.cuda_stream access ---
    21	        self.cuda_stream = self
    22	
    23	    @auto_log()
    24	    def synchronize(self):
    25	        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
    26	            torch.mps.synchronize()
    27	        return self
    28	
    29	    @auto_log()
    30	    def query(self):
    31	        return True
    32	
    33	    @auto_log()
    34	    def wait_event(self, event=None):
    35	        if event is not None:
    36	            event_device = getattr(event, '_device', getattr(event, 'device', None))
    37	            log_info(f"[DEBUG] Stream.wait_event: self.device={self.device}, event.device={event_device}")
    38	            if event_device is not None and event_device != self.device:
    39	                raise RuntimeError(f"Stream and event device mismatch: {self.device} vs {event_device}")
    40	        return self
    41	
    42	    @auto_log()
    43	    def wait_stream(self, stream=None):
    44	        return self
    45	
    46	    @auto_log()
    47	    def record_event(self, event=None):
    48	        return self
    49	
    50	# --- Event Replacement ---
    51	
    52	def _get_mps_event_class():
    53	    try:
    54	        from torch._streambase import _EventBase
    55	    except (AttributeError, ImportError):
    56	        try:
    57	            from torch._C import _EventBase
    58	        except (AttributeError, ImportError):
    59	            try:
    60	                _EventBase = torch._C._EventBase
    61	            except (AttributeError, ImportError):
    62	                _EventBase = object
    63	
    64	    class _cuda_Event(_EventBase):
    65	        @auto_log()
    66	        def __init__(self, enable_timing=False, blocking=False, interprocess=False, device=None):
    67	            try:
    68	                super().__init__()
    69	            except Exception:
    70	                pass
    71	            if device is None:
    72	                device = TorchDevice.get_default_device()
    73	            self._device = torch.device(device)
    74	            self.enable_timing = enable_timing
    75	            self.blocking = blocking
    76	            self.interprocess = interprocess
    77	            self._is_created = True
    78	            self._is_destroyed = False
    79	            self._recorded = False
    80	            self._record_time = None
    81	            self._stream = None
    82	
    83	        @auto_log()
    84	        def record(self, stream=None):
    85	            if stream is not None:
    86	                stream_device = getattr(stream, 'device', None)
    87	                log_info(f"[DEBUG] Event.record: self._device={self._device}, stream.device={stream_device}")
    88	                if stream_device is not None and self._device != stream_device:
    89	                    raise RuntimeError(f"Event and stream device mismatch: {self._device} vs {stream_device}")
    90	                self._stream = stream
    91	            self._recorded = True
    92	            self._record_time = time.time()
    93	            return self
    94	
    95	        @auto_log()
    96	        def wait(self, stream=None):
    97	            return self
    98	
    99	        @auto_log()
   100	        def query(self):
   101	            return self._recorded
   102	
   103	        @auto_log()
   104	        def elapsed_time(self, end_event):
   105	            if not self.enable_timing:
   106	                return 0.5
   107	            if not self._recorded or not getattr(end_event, '_recorded', False):
   108	                return 0.5
   109	            start_time = self._record_time
   110	            end_time = getattr(end_event, '_record_time', time.time())
   111	            if start_time is None or end_time is None:
   112	                return 0.5
   113	            elapsed_ms = (end_time - start_time) * 1000.0
   114	            return elapsed_ms
   115	
   116	        @auto_log()
   117	        def synchronize(self):
   118	            return self
   119	
   120	        @auto_log()
   121	        def __del__(self):
   122	            if hasattr(self, '_is_destroyed') and not self._is_destroyed:
   123	                self._is_destroyed = True
   124	
   125	    return _cuda_Event
   126	
   127	# --- Stream/Event Factory Functions ---
   128	
   129	def _cuda_stream_class(device=None, priority=0):
   130	    return _cuda_Stream(device)
   131	
   132	def _cuda_event(*args, **kwargs):
   133	    enable_timing = kwargs.get('enable_timing', False)
   134	    blocking = kwargs.get('blocking', False)
   135	    interprocess = kwargs.get('interprocess', False)
   136	    device = kwargs.get('device', None)
   137	    _Event = _get_mps_event_class()
   138	    return _Event(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess, device=device)
   139	
   140	def _cuda_stream(stream=None):
   141	    class StreamContext:
   142	        @auto_log()
   143	        def __init__(self, stream):
   144	            self.stream = stream
   145	
   146	        @auto_log()
   147	        def __enter__(self):
   148	            if self.stream is not None and hasattr(self.stream, '__enter__'):
   149	                self.stream.__enter__()
   150	            return self.stream
   151	
   152	        @auto_log()
   153	        def __exit__(self, exc_type, exc_val, exc_tb):
   154	            if self.stream is not None and hasattr(self.stream, '__exit__'):
   155	                return self.stream.__exit__(exc_type, exc_val, exc_tb)
   156	            return False
   157	
   158	        @auto_log()
   159	        def query(self):
   160	            if self.stream is not None and hasattr(self.stream, 'query'):
   161	                return self.stream.query()
   162	            return True
   163	
   164	        @auto_log()
   165	        def synchronize(self):
   166	            if self.stream is not None and hasattr(self.stream, 'synchronize'):
   167	                return self.stream.synchronize()
   168	            return self
   169	
   170	        @auto_log()
   171	        def wait_event(self, event=None):
   172	            if self.stream is not None and hasattr(self.stream, 'wait_event'):
   173	                return self.stream.wait_event(event)
   174	            return self
   175	
   176	        @auto_log()
   177	        def wait_stream(self, stream=None):
   178	            if self.stream is not None and hasattr(self.stream, 'wait_stream'):
   179	                return self.stream.wait_stream(stream)
   180	            return self
   181	
   182	        @auto_log()
   183	        def record_event(self, event=None):
   184	            if self.stream is not None and hasattr(self.stream, 'record_event'):
   185	                return self.stream.record_event(event)
   186	            return self
   187	
   188	    return StreamContext(stream)
   189	
   190	def _cuda_stream_class(device=None, priority=0):
   191	    # FORWARD 'priority' into the constructor instead of dropping it:
   192	    return _cuda_Stream(device, priority)
   193	
   194	
   195	def _cuda_current_stream(device=None):
   196	    return _cuda_stream_class(device=device)
   197	
   198	
   199	def _cuda_default_stream(device=None):
   200	    return _cuda_stream_class(device=device)
   201	
   202	
   203	def _cuda_synchronize(device=None):
   204	    if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
   205	        torch.mps.synchronize()
   206	
   207	# --- Patch Application ---
   208	
   209	def apply_patches():
   210	    import torch
   211	
   212	    torch.cuda.Stream = _cuda_stream_class  # type: ignore[assignment]
   213	    torch.cuda.Event = _get_mps_event_class()  # type: ignore[assignment]
   214	    torch.cuda.current_stream = _cuda_current_stream  # type: ignore[assignment]
   215	    torch.cuda.default_stream = _cuda_default_stream  # type: ignore[assignment]
   216	    torch.cuda.synchronize = _cuda_synchronize  # type: ignore[assignment]
   217	    torch.cuda.stream = _cuda_stream  # type: ignore[assignment]
   218	
   219	    # Add global Stream class patch if it exists
   220	    if hasattr(torch, 'Stream'):
   221	        class StreamWrapper:
   222	            def __new__(cls, device=None, *args, **kwargs):
   223	                # Redirect to our stream implementation
   224	                if isinstance(device, str) and device.startswith('cuda'):
   225	                    device = TorchDevice.redirect_device_type(device)
   226	                return _cuda_stream_class(device)
   227	
   228	        torch.Stream = StreamWrapper  # type: ignore[assignment]

```


## Filename ==>  ./cuda/unassigned.py
```python
     1	"""
     2	TorchDevice CUDA Unassigned/Stub Functions
     3	-----------------------------------------
     4	This module centralizes all unsupported or stubbed CUDA functions for non-CUDA backends (e.g., MPS, CPU).
     5	All functions listed in CUDA_STUBS are patched as no-ops on torch.cuda.
     6	
     7	Naming Convention:
     8	- Internal helper functions use the same name as the public API but with a leading underscore (e.g., _set_stream for torch.cuda.set_stream).
     9	- Public API functions (patched onto torch.cuda) do not use the underscore and are thin wrappers around the internal helpers.
    10	"""
    11	
    12	from typing import Callable, Dict
    13	
    14	
    15	def make_noop(name: str) -> Callable:
    16	    """Return a no-op function for the given CUDA function name."""
    17	    def _noop(*args, **kwargs):
    18	        """Stub for torch.cuda.{0}(). No-op on non-CUDA backends.""".format(name)
    19	        return None
    20	    _noop.__name__ = name
    21	    return _noop
    22	
    23	
    24	# List of unsupported CUDA function names to stub (expanded from TorchDevice.py)
    25	CUDA_STUBS = [
    26	    'set_stream', 'mem_get_info', 'reset_accumulated_memory_stats', 'reset_max_memory_allocated',
    27	    'reset_max_memory_cached', 'caching_allocator_alloc', 'caching_allocator_delete', 'get_allocator_backend',
    28	    'change_current_allocator', 'nvtx', 'jiterator', 'graph', 'CUDAGraph', 'make_graphed_callables',
    29	    'is_current_stream_capturing', 'graph_pool_handle', 'can_device_access_peer', 'comm', 'get_gencode_flags',
    30	    'current_blas_handle', 'memory_usage', 'utilization', 'temperature', 'power_draw', 'clock_rate',
    31	    'set_sync_debug_mode', 'get_sync_debug_mode', 'list_gpu_processes', 'seed', 'seed_all', 'manual_seed',
    32	    'manual_seed_all', 'get_rng_state', 'get_rng_state_all', 'set_rng_state', 'set_rng_state_all', 'initial_seed',
    33	    'ipc_collect'
    34	]
    35	
    36	# Create a dictionary of stub functions
    37	cuda_stub_functions: Dict[str, Callable] = {name: make_noop(name) for name in CUDA_STUBS}
    38	
    39	def apply_patches() -> None:
    40	    import torch
    41	    for name, fn in cuda_stub_functions.items():
    42	        setattr(torch.cuda, name, fn)
    43	    # Patch additional no-ops for unsupported CUDA features with plausible return types
    44	    torch.cuda.utilization = lambda *a, **kw: 0
    45	    torch.cuda.temperature = lambda *a, **kw: 0
    46	    torch.cuda.power_draw = lambda *a, **kw: 0
    47	    torch.cuda.clock_rate = lambda *a, **kw: 0
    48	    torch.cuda.set_sync_debug_mode = lambda *a, **kw: None
    49	    torch.cuda.get_sync_debug_mode = lambda *a, **kw: 0
    50	    torch.cuda.list_gpu_processes = lambda *a, **kw: [] 

```


## Filename ==>  ./modules/__init__.py
```python
     1	"""
     2	TorchDevice modules package.
     3	Contains utility modules used by the main TorchDevice package.
     4	"""
     5	
     6	from .TDLogger import auto_log, log_info
     7	from . import compile
     8	
     9	__all__ = ['auto_log', 'log_info', 'compile']

```


## Filename ==>  ./modules/compile.py
```python
     1	"""
     2	TorchDevice.modules.compile - Smart PyTorch compiler management for different devices
     3	
     4	This module provides device-aware configuration of PyTorch's compilation features:
     5	
     6	1. TorchDynamo (the torch.compile tracer):
     7	   - Works on all devices (CPU, CUDA, MPS)
     8	   - Kept enabled but configured appropriately for each device
     9	
    10	2. TorchInductor (the default backend):
    11	   - Only works properly on CUDA and CPU devices
    12	   - For MPS devices, we configure to use 'aot_eager' backend instead
    13	"""
    14	
    15	from typing import Any
    16	from .TDLogger import log_info
    17	
    18	
    19	def _device_aware_compile(model: Any, *args: Any, **kwargs: Any) -> Any:
    20	    """
    21	    A device-aware wrapper for torch.compile that selects appropriate backends.
    22	    
    23	    On CUDA: Uses the default 'inductor' backend (or whatever was specified)
    24	    On MPS: Switches to 'aot_eager' backend which works better on Metal
    25	    On CPU: Uses the default backend (typically 'inductor')
    26	    """
    27	    # Import here to avoid circular imports
    28	    import torch
    29	    
    30	    # Get the current device being used
    31	    curr_device = None
    32	    
    33	    # Try to get current device from model if possible
    34	    if hasattr(model, 'parameters'):
    35	        try:
    36	            params = list(model.parameters())
    37	            if params and hasattr(params[0], 'device'):
    38	                curr_device = params[0].device
    39	        except Exception:
    40	            pass
    41	    
    42	    # If we couldn't get device from model, check global default device
    43	    if curr_device is None:
    44	        try:
    45	            # Import the main package rather than using relative imports
    46	            import TorchDevice
    47	            curr_device = TorchDevice.TorchDevice.get_default_device()
    48	        except Exception:
    49	            # Fall back to CPU as the safest option
    50	            curr_device = torch.device('cpu')
    51	    
    52	    # Check if we're on MPS
    53	    is_mps = (hasattr(curr_device, 'type') and curr_device.type == 'mps')
    54	    
    55	    # For MPS, use 'aot_eager' backend which works better
    56	    if is_mps and 'backend' not in kwargs:
    57	        log_info("Using 'aot_eager' backend for torch.compile on MPS device")
    58	        kwargs['backend'] = 'aot_eager'
    59	    
    60	    # Call the original compile function with our adjusted arguments
    61	    original_compile = getattr(_device_aware_compile, '_original_compile', None)
    62	    if original_compile:
    63	        try:
    64	            return original_compile(model, *args, **kwargs)
    65	        except Exception as e:
    66	            log_info(f"Error in torch.compile: {e}")
    67	            log_info("Returning uncompiled model")
    68	            return model
    69	    else:
    70	        # Original compile wasn't saved - return model uncompiled
    71	        log_info("Original torch.compile not found, returning uncompiled model")
    72	        return model
    73	
    74	
    75	def patch_compile() -> None:
    76	    """
    77	    Patch the torch.compile function for device-aware operation.
    78	    This is separate from _dynamo config patching to avoid import issues.
    79	    """
    80	    import torch
    81	    
    82	    # Save the original compile function
    83	    if hasattr(torch, "compile"):
    84	        _original = torch.compile
    85	        _device_aware_compile._original_compile = _original  # type: ignore
    86	        
    87	        # Replace the compile function
    88	        torch.compile = _device_aware_compile  # type: ignore
    89	        log_info("Patched torch.compile with device-aware version")
    90	
    91	
    92	def patch_dynamo_config() -> None:
    93	    """
    94	    Carefully configure torch._dynamo if it's already loaded.
    95	    This is called separately to avoid forcing _dynamo to load too early.
    96	    """
    97	    import torch
    98	    
    99	    # ONLY get _dynamo if it's already loaded/imported
   100	    if "_dynamo" in torch.__dict__:
   101	        dynamo = torch._dynamo
   102	        try:
   103	            # Configure _dynamo in a way that works across devices
   104	            if hasattr(dynamo, "config"):
   105	                # Enable dynamic shapes which helps with different devices
   106	                dynamo.config.dynamic_shapes = True
   107	                # Don't try to use cudagraphs on non-CUDA devices
   108	                dynamo.config.automatic_dynamic_shapes = True
   109	                # Increase tolerance for numerical differences between devices
   110	                dynamo.config.tolerance_for_precision = 1e-4
   111	                log_info("Configured torch._dynamo for cross-device compatibility")
   112	        except Exception as e:
   113	            log_info(f"Error configuring torch._dynamo: {e}")
   114	
   115	
   116	def apply_patches() -> None:
   117	    """
   118	    Apply patches to optimize PyTorch's compilation system for the current device.
   119	    
   120	    This patches:
   121	    1. torch.compile - to use the appropriate backend based on device
   122	    2. _dynamo config - to safely configure torch._dynamo if it's loaded
   123	    """
   124	    log_info("Applying PyTorch compiler patches")
   125	    
   126	    # Patch torch.compile first
   127	    patch_compile()
   128	    
   129	    # We'll patch _dynamo config LAST, after all other patches are applied
   130	    # to avoid triggering imports too early
   131	    
   132	    log_info("Primary compiler patching complete - _dynamo patching deferred until later") 

```


## Filename ==>  ./modules/patch.py
```python
     1	"""
     2	TorchDevice Central Patch Application
     3	-------------------------------------
     4	Orchestrate all TorchDevice patches (device, memory, RNG, streams)
     5	and disable Dynamo’s placeholder‐tests entirely on MPS.
     6	"""
     7	
     8	def apply_all_patches() -> None:
     9	    # 0) If running on MPS, turn Dynamo’s placeholder‐test into a no-op
    10	    import torch
    11	    from ..TorchDevice import TorchDevice
    12	    if TorchDevice.get_default_device() == 'mps':
    13	        try:
    14	            from torch._dynamo.variables import torch_function as _tf
    15	            _tf.populate_builtin_to_tensor_fn_map = lambda: None
    16	        except ImportError:
    17	            # Dynamo not present or too old—nothing to do
    18	            pass
    19	
    20	    # 1) core patches—these must come first
    21	    from ..cuda import device, memory, random, streams, unassigned
    22	    device.apply_patches()
    23	    memory.apply_patches()
    24	    random.apply_patches()
    25	    streams.apply_patches()
    26	    unassigned.apply_patches()
    27	
    28	    # 2) now install the compile/Dynamo patches if not pure CPU
    29	    from ..TorchDevice import TorchDevice as _TD
    30	    if _TD.get_default_device() != torch.device('cpu'):
    31	        # temporarily force all torch.device(...) calls to pick the MPS device
    32	        _orig = _TD.torch_device_replacement
    33	        _TD.torch_device_replacement = lambda *a, **k: _TD.get_default_device()
    34	        try:
    35	            from ..modules import compile
    36	            compile.apply_patches()
    37	        finally:
    38	            _TD.torch_device_replacement = _orig

```


## Filename ==>  ./modules/TDLogger.py
```python
     1	import functools  # Added import for functools import logging
     2	import logging
     3	import os
     4	import sys
     5	import sysconfig
     6	
     7	LIB_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
     8	STDLIB_DIR = os.path.abspath(sysconfig.get_paths()["stdlib"])
     9	
    10	# Global flag to toggle stack frame dumping (set to True for testing/calibration)
    11	# Use environment variable to toggle
    12	DUMP_STACK_FRAMES = os.environ.get("DUMP_STACK_FRAMES", "False").lower() == "true"
    13	
    14	# Number of stack frames to display in debug mode.
    15	STACK_FRAMES = 30
    16	
    17	# You can calibrate your stack offset here once.
    18	DEFAULT_STACK_OFFSET = 3  # adjust as needed
    19	
    20	# Define functions to skip from logging at module level
    21	_INTERNAL_LOG_SKIP = {
    22	    # Core initialization and setup functions
    23	    "apply_patches", "initialize_torchdevice", "apply_basic_patches",
    24	    
    25	    # Device detection and management
    26	    "get_default_device", "redirect_device_type", "_redirect_device_type",
    27	    
    28	    # Tensor operations and wrappers
    29	    "tensor_creation_wrapper", "_get_mps_event_class",
    30	    
    31	    # Module level functions
    32	    "<module>", "__init__", "__main__", "__enter__", "__exit__", "__del__",
    33	    
    34	    # Test related functions
    35	    "_callTestMethod", "_callSetUp", "_callTearDown",
    36	    
    37	    # Internal utility functions
    38	    "wrapper", "_get_device_type", "_get_device_index"
    39	}
    40	
    41	def auto_log():
    42	    """
    43	    Decorator that logs function calls with detailed caller information.
    44	    """
    45	    def decorator(func):
    46	        @functools.wraps(func)
    47	        def wrapper(*args, **kwargs):
    48	            result = None
    49	            if func.__name__ not in _INTERNAL_LOG_SKIP:
    50	                log_message(f"Called {func.__name__}", "calling the entry now")
    51	                result = func(*args, **kwargs)
    52	                log_message(f"{func.__name__} returned {result}", func.__name__)
    53	            else:
    54	                result = func(*args, **kwargs)
    55	            return result
    56	        return wrapper
    57	    return decorator
    58	
    59	# Create logger and add a filter to add missing extra fields.
    60	_logger = logging.getLogger("TorchDevice")
    61	_handler = logging.StreamHandler(sys.stderr)
    62	_formatter = logging.Formatter(
    63	    'GPU REDIRECT - [%(program_name)s] "%(caller_func_name)s" in File: %(caller_filename)s:%(caller_lineno)d - '
    64	    'Called: %(torch_function)s %(message)s'
    65	)
    66	_handler.setFormatter(_formatter)
    67	_logger.addHandler(_handler)
    68	_logger.setLevel(logging.INFO)
    69	_logger.propagate = False
    70	
    71	# Create a separate logger for info messages
    72	_info_logger = logging.getLogger("TorchDevice.info")
    73	_info_handler = logging.StreamHandler(sys.stderr)
    74	_info_formatter = logging.Formatter('INFO: [%(program_name)s] - %(message)s')
    75	_info_handler.setFormatter(_info_formatter)
    76	_info_logger.addHandler(_info_handler)
    77	_info_logger.setLevel(logging.INFO)
    78	_info_logger.propagate = False
    79	
    80	class DefaultExtraFilter(logging.Filter):
    81	    def filter(self, record):
    82	        if not hasattr(record, 'program_name'):
    83	            record.program_name = "unknown"
    84	        if not hasattr(record, 'caller_func_name'):
    85	            record.caller_func_name = "unknown"
    86	        if not hasattr(record, 'caller_filename'):
    87	            record.caller_filename = "unknown"
    88	        if not hasattr(record, 'caller_lineno'):
    89	            record.caller_lineno = 0
    90	        if not hasattr(record, 'torch_function'):
    91	            record.torch_function = "unknown"
    92	        return True
    93	
    94	_logger.addFilter(DefaultExtraFilter())
    95	
    96	def log_message(message: str, torch_function: str = "unknown", stacklevel: int = DEFAULT_STACK_OFFSET) -> None:
    97	    """
    98	    Log a message with detailed caller information.
    99	    This is used primarily for GPU redirection logging.
   100	    """
   101	    try:
   102	        frame = sys._getframe(stacklevel)
   103	        caller_func_name = frame.f_code.co_name
   104	        # Check if we need to adjust stacklevel for test methods
   105	        if caller_func_name in ["_callTestMethod", "_callSetUp"]:
   106	            stacklevel -= 1
   107	            frame = sys._getframe(stacklevel)
   108	            caller_func_name = frame.f_code.co_name
   109	        if caller_func_name in ["wrapper"]:
   110	            stacklevel += 1
   111	            frame = sys._getframe(stacklevel)
   112	            caller_func_name = frame.f_code.co_name
   113	        if caller_func_name in ["<lambda>"]:
   114	            stacklevel += 1
   115	            frame = sys._getframe(stacklevel)
   116	            caller_func_name = frame.f_code.co_name
   117	        
   118	        caller_filename = frame.f_code.co_filename
   119	        caller_lineno = frame.f_lineno
   120	    except Exception:
   121	        caller_func_name = "unknown"
   122	        caller_filename = "unknown"
   123	        caller_lineno = 0
   124	
   125	    extra = {
   126	        "program_name": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "unknown",
   127	        "torch_function": torch_function,
   128	        "caller_func_name": caller_func_name,
   129	        "caller_filename": caller_filename,
   130	        "caller_lineno": caller_lineno,
   131	    }
   132	    _logger.info(message, extra=extra)
   133	
   134	    if DUMP_STACK_FRAMES:
   135	        dump_lines = []
   136	        for i in range(STACK_FRAMES):
   137	            try:
   138	                frame = sys._getframe(i)
   139	                formatted = f'{frame.f_code.co_name} in {os.path.abspath(frame.f_code.co_filename)}:{frame.f_lineno}'
   140	                dump_lines.append(f'FRAME {i}: "{formatted}"')
   141	            except ValueError:
   142	                break
   143	        dump = "\n".join(dump_lines)
   144	        log_info(f"Stack frame dump:\n{dump}")
   145	        log_info(f"\n**** END OF STACKFRAME DUMP ****\n\n")
   146	
   147	
   148	def log_info(message: str) -> None:
   149	    """
   150	    Simple logging function that only includes the program name and message.
   151	    This is the preferred way to log general information messages.
   152	    
   153	    Args:
   154	        message: The message to log
   155	    """
   156	    extra = {
   157	        "program_name": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "unknown",
   158	    }
   159	    _info_logger.info(message, extra=extra)

```


## Filename ==>  ./patch.py
```python
     1	# This module is now redundant: patching is handled automatically in TorchDevice/__init__.py
     2	
     3	
     4	# Remove up to here.
     5	# ... existing code ...
     6	# Kept for future internal use or as a placeholder for additional patch logic if needed.
     7	
     8	from TorchDevice.cuda.streams import apply_patches as _apply_streams_patches
     9	
    10	def _patch_all() -> None:
    11	    """Internal: Apply all TorchDevice monkey-patches. Not for public use."""
    12	    _apply_streams_patches()
    13	    # Add additional patch calls here as you modularize more functionality 

```


## Filename ==>  ./TorchDevice.py
```python
     1	"""
     2	TorchDevice - Transparent PyTorch Device Redirection
     3	
     4	This module enables seamless code portability between NVIDIA CUDA, Apple Silicon (MPS),
     5	and CPU hardware for PyTorch applications. It intercepts PyTorch calls related to GPU
     6	hardware, allowing developers to write code that works across different hardware
     7	without modification.
     8	
     9	Key features:
    10	- Automatic device redirection based on available hardware
    11	- CPU override capability using 'cpu:-1' device specification
    12	- Mocked CUDA functions for MPS and CPU compatibility
    13	- Stream and Event support across all device types
    14	- Unified memory handling and reporting
    15	- Detailed logging for debugging and migration assistance
    16	
    17	Usage:
    18	    import TorchDevice  # Import before torch to apply patches
    19	    import torch
    20	    
    21	    # Regular device selection (will be redirected based on available hardware)
    22	    device = torch.device('cuda')  # Redirects to MPS on Apple Silicon
    23	    
    24	    # Force CPU usage with the override feature
    25	    device = torch.device('cpu:-1')  # Forces CPU regardless of available GPUs
    26	    
    27	    # All subsequent operations respect the CPU override
    28	    tensor = torch.randn(5, 5)  # Will be created on CPU
    29	    model = torch.nn.Linear(10, 5).to('cuda')  # Still uses CPU due to override
    30	"""
    31	import torch
    32	from .modules.TDLogger import auto_log, log_info  # We now use only auto_log instead of log_info for debugging
    33	import TorchDevice.modules.patch as cuda_patch
    34	import threading
    35	from typing import Optional
    36	
    37	# Capture the original torch.device type.
    38	_ORIGINAL_TORCH_DEVICE_TYPE = torch.device("cpu").__class__
    39	
    40	_CACHED_DEFAULT_DEVICE = None
    41	_device_type = None
    42	
    43	# --- AMP Hooks ---
    44	if hasattr(torch.cuda, 'amp'):
    45	    _original_autocast = torch.cuda.amp.autocast
    46	
    47	    @auto_log()
    48	    def autocast_replacement(*args, **kwargs):
    49	        default_device = TorchDevice.get_default_device()
    50	        if default_device != 'cuda':
    51	            return _original_autocast(*args, **kwargs)
    52	
    53	    torch.cuda.amp.autocast = autocast_replacement
    54	
    55	    if hasattr(torch.cuda.amp, 'GradScaler'):
    56	        _OriginalGradScaler = torch.cuda.amp.GradScaler
    57	
    58	        class GradScalerReplacement(_OriginalGradScaler):
    59	            @auto_log()
    60	            def __init__(self, *args, **kwargs):
    61	                if TorchDevice.get_default_device() != 'cuda':
    62	                    pass
    63	                super().__init__(*args, **kwargs)
    64	        torch.cuda.amp.GradScaler = GradScalerReplacement
    65	
    66	
    67	# --- TorchDevice Class with Patched CUDA Functions ---
    68	class TorchDevice:
    69	    _default_device = None
    70	    _previous_default_device = None
    71	    _lock = threading.RLock()
    72	    _cpu_override = False  # Flag for explicit CPU override
    73	    _in_torch_load = False
    74	    _patches_applied = False
    75	
    76	    _original_tensor_to = torch.Tensor.to
    77	    _original_module_to = torch.nn.Module.to
    78	    _original_module_cpu = torch.nn.Module.cpu
    79	    _original_module_cuda = torch.nn.Module.cuda
    80	    _original_tensor_cuda = torch.Tensor.cuda
    81	    _original_torch_backends_cuda_is_built = torch.backends.cuda.is_built
    82	    _original_torch_cuda_current_device = torch.cuda.current_device
    83	    _original_torch_cuda_device = torch.cuda.device  # Context manager
    84	    _original_torch_cuda_device_count = torch.cuda.device_count
    85	    _original_torch_cuda_empty_cache = torch.cuda.empty_cache
    86	    _original_torch_cuda_get_arch_list = torch.cuda.get_arch_list
    87	    _original_torch_cuda_get_device_capability = torch.cuda.get_device_capability
    88	    _original_torch_cuda_get_device_name = torch.cuda.get_device_name
    89	    _original_torch_cuda_get_device_properties = torch.cuda.get_device_properties
    90	    _original_torch_cuda_is_available = torch.cuda.is_available
    91	    _original_torch_cuda_is_initialized = torch.cuda.is_initialized
    92	    _original_torch_cuda_set_device = torch.cuda.set_device
    93	    _original_torch_cuda_synchronize = torch.cuda.synchronize
    94	    _original_torch_device = torch.device
    95	    _original_torch_load = torch.load
    96	
    97	    @auto_log()
    98	    def __init__(self, device_type: Optional[str] = None, device_index: int = 0):
    99	        with self._lock:
   100	            if self._default_device is None:
   101	                self.__class__._detect_default_device()
   102	            if isinstance(device_type, str):
   103	                if ':' in device_type:
   104	                    device_type, index = device_type.split(':')
   105	                    device_index = int(index)
   106	                else:
   107	                    device_index = 0 if device_index is None else device_index
   108	                device_type = self.__class__.redirect_device_type(device_type)
   109	                device_str = f"{device_type}:{device_index}"
   110	                self.device = self.__class__._original_torch_device(device_str)
   111	
   112	    @auto_log()
   113	    def __repr__(self):
   114	        return repr(self.device)
   115	
   116	    @auto_log()
   117	    def __str__(self):
   118	        return str(self.device)
   119	
   120	    class TorchDeviceWrapper(object):
   121	        @auto_log()
   122	        def __init__(self, device):
   123	            self._device = device
   124	            
   125	        def __getattr__(self, name):
   126	            return getattr(self._device, name)
   127	            
   128	        def __repr__(self):
   129	            return repr(self._device)
   130	            
   131	        def __str__(self):
   132	            return str(self._device)
   133	            
   134	        def __instancecheck__(self, instance):
   135	            return isinstance(instance, _ORIGINAL_TORCH_DEVICE_TYPE)
   136	    
   137	    @classmethod
   138	    @auto_log()
   139	    def get_default_device(cls):
   140	        """
   141	        Return the default device based on available hardware and cache the result.
   142	        """
   143	        global _CACHED_DEFAULT_DEVICE
   144	        if _CACHED_DEFAULT_DEVICE is None:
   145	            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
   146	                _CACHED_DEFAULT_DEVICE = 'mps'
   147	            elif cls._original_torch_cuda_is_available():
   148	                _CACHED_DEFAULT_DEVICE = 'cuda'
   149	            else:
   150	                _CACHED_DEFAULT_DEVICE = 'cpu'
   151	        return _CACHED_DEFAULT_DEVICE
   152	
   153	    @classmethod
   154	    def cpu_override_set(cls):
   155	        return cls._cpu_override
   156	
   157	    @classmethod
   158	    @auto_log()
   159	    def redirect_device_type(cls, device_type):
   160	        """
   161	        Redirect a device type string based on availability and CPU override.
   162	        If cpu_override is True, always returns 'cpu'.
   163	        For 'cuda' and 'mps' requests, return the type that is available.
   164	        """
   165	        # For explicit CPU requests, always return 'cpu'
   166	        if device_type == 'cpu':
   167	            return 'cpu'
   168	        
   169	        if device_type.startswith('cuda'):
   170	            if _CACHED_DEFAULT_DEVICE == 'cuda':
   171	                device_type = 'cuda'
   172	            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
   173	                device_type = 'mps'
   174	            else:
   175	                device_type = 'cpu'
   176	        elif device_type.startswith('mps'):
   177	            if _CACHED_DEFAULT_DEVICE == 'mps':
   178	                device_type = 'mps'
   179	            elif cls._original_torch_cuda_is_available():
   180	                device_type = 'cuda'
   181	            else:
   182	                device_type = 'cpu'
   183	        return device_type
   184	
   185	    @classmethod
   186	    @auto_log()
   187	    def tensor_creation_wrapper(cls, original_func):
   188	        """
   189	        Wrapper for tensor creation functions to enforce default device redirection and CPU override.
   190	        Always calls torch_device_replacement for the device argument, so the toggle state is always respected.
   191	        """
   192	        def wrapped_func(*args, **kwargs):
   193	            device_arg = kwargs.get('device', None)
   194	            # If device is not specified, inject the current device (default or override)
   195	            if device_arg is None:
   196	                device = cls.torch_device_replacement()
   197	                log_info(f"[tensor_creation_wrapper] Injecting device: {device}")
   198	                kwargs['device'] = device
   199	            else:
   200	                # Always pass through torch_device_replacement to handle override logic
   201	                device = cls.torch_device_replacement(device_arg)
   202	                log_info(f"[tensor_creation_wrapper] Normalized device: {device}")
   203	                kwargs['device'] = device
   204	            return original_func(*args, **kwargs)
   205	        return wrapped_func
   206	
   207	    @staticmethod
   208	    def tensor_to_replacement(t, *args, **kwargs):
   209	        if not isinstance(t, torch.Tensor):
   210	            raise TypeError(f"tensor_to_replacement called on non-tensor object: {type(t)}")
   211	        if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
   212	            # Always redirect through the TorchDevice policy
   213	            device = TorchDevice.torch_device_replacement(args[0])
   214	            new_args = (device,) + args[1:]
   215	            kwargs.pop('device', None)
   216	            return TorchDevice._original_tensor_to(t, *new_args, **kwargs)
   217	        elif 'device' in kwargs and isinstance(kwargs['device'], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
   218	            # Always redirect through the TorchDevice policy
   219	            device = TorchDevice.torch_device_replacement(kwargs['device'])
   220	            kwargs['device'] = device
   221	            return TorchDevice._original_tensor_to(t, *args, **kwargs)
   222	        else:
   223	            return TorchDevice._original_tensor_to(t, *args, **kwargs)
   224	
   225	    @staticmethod
   226	    def module_to_replacement(m, *args, **kwargs):
   227	        if not isinstance(m, torch.nn.Module):
   228	            raise TypeError(f"module_to_replacement called on non-module object: {type(m)}")
   229	        if args and isinstance(args[0], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
   230	            # Always redirect through the TorchDevice policy
   231	            device = TorchDevice.torch_device_replacement(args[0])
   232	            new_args = (device,) + args[1:]
   233	            kwargs.pop('device', None)
   234	            return TorchDevice._original_module_to(m, *new_args, **kwargs)
   235	        elif 'device' in kwargs and isinstance(kwargs['device'], (str, _ORIGINAL_TORCH_DEVICE_TYPE)):
   236	            # Always redirect through the TorchDevice policy
   237	            device = TorchDevice.torch_device_replacement(kwargs['device'])
   238	            kwargs['device'] = device
   239	            return TorchDevice._original_module_to(m, *args, **kwargs)
   240	        else:
   241	            return TorchDevice._original_module_to(m, *args, **kwargs)
   242	
   243	    @classmethod
   244	    @auto_log()
   245	    def torch_device_replacement(cls, *args, **kwargs) -> torch.device:
   246	        """
   247	        Drop-in replacement for torch.device() with device redirection and CPU override toggle.
   248	        • No arguments → returns default device (or CPU if override is active).
   249	        • 'cpu:-1' or torch.device('cpu', -1) → toggles CPU override.
   250	        • Redirects non-CPU devices to available hardware.
   251	        • Preserves extra args and kwargs.
   252	        Always returns a torch.device object.
   253	        """
   254	        global _CACHED_DEFAULT_DEVICE
   255	        device_type = ""
   256	        device_index = None
   257	        log_info(f"Called with args={args}, kwargs={kwargs}")
   258	        with cls._lock:
   259	            # If first argument is torch.device, check for override
   260	            if args and isinstance(args[0], _ORIGINAL_TORCH_DEVICE_TYPE):
   261	                return args[0]
   262	
   263	            # If first argument is string device spec, parse and modify
   264	            if args and isinstance(args[0], str):
   265	                device_spec = args[0].strip()
   266	                if ":" in device_spec:
   267	                    parts = device_spec.split(":", 1)
   268	                    device_type = parts[0].lower()
   269	                    try:
   270	                        device_index = int(parts[1])
   271	                    except ValueError:
   272	                        device_index = None
   273	                else:
   274	                    device_type = device_spec.lower()
   275	
   276	                # CPU override toggle logic
   277	                if device_type == "cpu":
   278	                    if device_index == -1:
   279	                        device_index = None
   280	                        if cls.cpu_override_set():
   281	                            # Toggle OFF
   282	                            cls._cpu_override = False
   283	                            _CACHED_DEFAULT_DEVICE = cls._previous_default_device
   284	                            cls._previous_default_device = None
   285	                            log_info("CPU override toggled OFF")
   286	                        else:
   287	                            # Toggle ON
   288	                            cls._cpu_override = True
   289	                            cls._previous_default_device = _CACHED_DEFAULT_DEVICE
   290	                            _CACHED_DEFAULT_DEVICE = 'cpu'
   291	                            log_info("CPU override toggled ON")
   292	                
   293	            device_type = _CACHED_DEFAULT_DEVICE
   294	            result = cls._original_torch_device(device_type, device_index)
   295	            return result
   296	    
   297	    @classmethod
   298	    @auto_log()
   299	    def torch_load_replacement(cls, *args, **kwargs):
   300	        if cls._in_torch_load:
   301	            return cls._original_torch_load(*args, **kwargs)
   302	        cls._in_torch_load = True
   303	        try:
   304	            default_device = cls.get_default_device()
   305	            if 'map_location' in kwargs:
   306	                if kwargs['map_location'] == 'cpu' or (
   307	                    isinstance(kwargs['map_location'], str) and kwargs['map_location'] != default_device
   308	                ):
   309	                    kwargs['map_location'] = default_device
   310	            else:
   311	                kwargs['map_location'] = default_device
   312	            return cls._original_torch_load(*args, **kwargs)
   313	        finally:
   314	            cls._in_torch_load = False
   315	
   316	    @classmethod
   317	    @auto_log()
   318	    def _detect_default_device(cls):
   319	        if torch.backends.mps.is_available():
   320	            cls._default_device = 'mps'
   321	        elif cls._original_torch_cuda_is_available():
   322	            cls._default_device = 'cuda'
   323	        else:
   324	            cls._default_device = 'cpu'
   325	
   326	    @staticmethod
   327	    @auto_log()
   328	    def tensor_cuda_replacement(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
   329	        default_device = TorchDevice.get_default_device()
   330	        return tensor.to(default_device, non_blocking=non_blocking, memory_format=memory_format)
   331	
   332	    @staticmethod
   333	    @auto_log()
   334	    def module_cuda_replacement(module, device=None):
   335	        default_device = TorchDevice.get_default_device()
   336	        return module.to(default_device)
   337	
   338	    @staticmethod
   339	    @auto_log()
   340	    def tensor_mps_replacement(tensor, device=None, non_blocking=False, memory_format=torch.preserve_format):
   341	        default_device = TorchDevice.get_default_device()
   342	        return tensor.to(default_device, non_blocking=non_blocking, memory_format=memory_format)
   343	
   344	    @staticmethod
   345	    @auto_log()
   346	    def module_mps_replacement(module, device=None):
   347	        default_device = TorchDevice.get_default_device()
   348	        return module.to(default_device)
   349	
   350	    @staticmethod
   351	    @auto_log()
   352	    def tensor_cpu_replacement(tensor):
   353	        """
   354	        Replacement for torch.Tensor.cpu() that follows device redirection policy.
   355	        If CPU override is active, moves to CPU, otherwise redirects to default device.
   356	        """
   357	        # If CPU override is active, actually use CPU
   358	        if TorchDevice.cpu_override_set():
   359	            return TorchDevice._original_tensor_to(tensor, 'cpu')
   360	        # Otherwise redirect to default device as per policy
   361	        default_device = TorchDevice.get_default_device()
   362	        return tensor.to(default_device)
   363	
   364	    @staticmethod
   365	    @auto_log()
   366	    def module_cpu_replacement(module):
   367	        default_device = TorchDevice.get_default_device()
   368	        return module.to(default_device)
   369	
   370	    @staticmethod
   371	    @auto_log()
   372	    def numpy_replacement(tensor):
   373	        """
   374	        Replacement for torch.Tensor.numpy() that moves tensor to CPU first if needed.
   375	        This always needs to go to CPU regardless of device policy since numpy() 
   376	        requires CPU tensors.
   377	        """
   378	        # Always move to CPU for numpy conversion - this is a special case
   379	        # that must bypass the device redirection policy
   380	        if tensor.device.type != 'cpu':
   381	            cpu_tensor = TorchDevice._original_tensor_to(tensor, 'cpu')
   382	            return TorchDevice._original_numpy(cpu_tensor)
   383	        return TorchDevice._original_numpy(tensor)
   384	
   385	    @classmethod
   386	    @auto_log()
   387	    def apply_patches(cls):
   388	        cls.get_default_device()
   389	        """Apply all patches to PyTorch."""
   390	        if cls._patches_applied:
   391	            return
   392	
   393	        # --- Original Method Storage ---
   394	        # Store references to original methods before patching
   395	        cls._original_tensor_to = torch.Tensor.to
   396	        cls._original_module_to = torch.nn.Module.to
   397	        cls._original_tensor_cuda = torch.Tensor.cuda
   398	        cls._original_module_cuda = torch.nn.Module.cuda
   399	        cls._original_torch_device = torch.device
   400	        cls._original_torch_load = torch.load
   401	        cls._original_numpy = torch.Tensor.numpy  # Store original numpy method
   402	
   403	        # --- Patch PyTorch Methods ---
   404	        torch.device = cls.torch_device_replacement
   405	        setattr(torch.Tensor, 'to', cls.tensor_to_replacement)  # type: ignore[attr-defined]
   406	        setattr(torch.nn.Module, 'to', cls.module_to_replacement)  # type: ignore[attr-defined]
   407	        setattr(torch.Tensor, 'cuda', cls.tensor_cuda_replacement)  # type: ignore[attr-defined]
   408	        setattr(torch.nn.Module, 'cuda', cls.module_cuda_replacement)  # type: ignore[attr-defined]
   409	        setattr(torch.Tensor, 'mps', cls.tensor_mps_replacement)  # type: ignore[attr-defined]
   410	        setattr(torch.nn.Module, 'mps', cls.module_mps_replacement)  # type: ignore[attr-defined]
   411	        setattr(torch.Tensor, 'cpu', cls.tensor_cpu_replacement)  # type: ignore[attr-defined]
   412	        setattr(torch.nn.Module, 'cpu', cls.module_cpu_replacement)  # type: ignore[attr-defined]
   413	        setattr(torch.Tensor, 'numpy', cls.numpy_replacement)  # type: ignore[attr-defined]
   414	        cuda_patch.apply_all_patches()
   415	        torch.load = cls.torch_load_replacement  # type: ignore[assignment]
   416	
   417	        cls._patches_applied = True
   418	
   419	
   420	# Apply patches when the module is imported
   421	TorchDevice.apply_patches()

```
