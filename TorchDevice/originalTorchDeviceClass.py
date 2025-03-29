class TorchDevice:
    _default_device = None
    _lock = threading.Lock()
    _cpu_override = False

    # Save original functions as class attributes.
    _original_torch_cuda_is_available = torch.cuda.is_available
    _original_torch_cuda_device_count = torch.cuda.device_count
    _original_torch_cuda_get_device_properties = torch.cuda.get_device_properties
    _original_torch_cuda_empty_cache = torch.cuda.empty_cache
    _original_torch_cuda_synchronize = torch.cuda.synchronize
    _original_torch_cuda_current_device = torch.cuda.current_device
    _original_torch_cuda_set_device = torch.cuda.set_device
    _original_torch_cuda_get_device_name = torch.cuda.get_device_name
    _original_torch_cuda_get_device_capability = torch.cuda.get_device_capability
    _original_torch_cuda_is_initialized = torch.cuda.is_initialized
    _original_torch_cuda_get_arch_list = torch.cuda.get_arch_list
    _original_torch_backends_cuda_is_built = torch.backends.cuda.is_built
    _original_torch_device = torch.device
    _original_torch_cuda_device = torch.cuda.device

    @auto_log()
    def __init__(self, device_type: str = None, device_index: int = None):
        with self._lock:
            if self._default_device is None:
                self.__class__._get_default_device()
            if device_type is None:
                device_type = self._default_device
            if isinstance(device_type, str):
                if ':' in device_type:
                    device_type, index = device_type.split(':')
                    device_index = int(index)
                else:
                    device_index = 0 if device_index is None else device_index
                device_type = self.__class__._redirect_device_type(device_type)
                device_str = f"{device_type}:{device_index}"
                log_message(f"Creating torch.device('{device_str}')", "torch.device", stacklevel=2)
                self.device = self.__class__._original_torch_device(device_str)
            else:
                self.device = self.__class__._original_torch_device(device_type)

    @auto_log()
    def __repr__(self):
        return repr(self.device)

    @auto_log()
    def __str__(self):
        return str(self.device)

    class TorchDeviceWrapper:
        @auto_log()
        def __init__(self, device):
            self._device = device
            
        @auto_log()
        def __getattr__(self, name):
            return getattr(self._device, name)
            
        @auto_log()
        def __repr__(self):
            return repr(self._device)
            
        @auto_log()
        def __str__(self):
            return str(self._device)
            
        @auto_log()
        def __instancecheck__(self, instance):
            return isinstance(instance, _ORIGINAL_TORCH_DEVICE_TYPE)

    @classmethod
    @auto_log()
    def torch_device_replacement(cls, device_type="", device_index=None):
        if not device_type:
            with cls._lock:
                if cls._default_device is None:
                    cls._get_default_device()
                device_type = cls._default_device
            return cls._original_torch_device(device_type, 0 if device_type == 'mps' else None)

        # Handle torch.device objects
        if isinstance(device_type, _ORIGINAL_TORCH_DEVICE_TYPE):
            name = device_type.type
            index = device_type.index
            if name == 'cpu' and index == -1:
                with cls._lock:
                    cls._cpu_override = True
                return cls._original_torch_device('cpu:0')
            # If MPS is requested and available, use it directly
            if name == 'mps' and torch.backends.mps.is_available():
                return cls._original_torch_device('mps', 0)
            redirected = cls._redirect_device_type(name)
            if redirected != name:
                return cls._original_torch_device(redirected, index if index is not None else 0)
            return device_type

        if isinstance(device_type, str):
            if ':' in device_type:
                name, index = device_type.split(":", 1)
                if name == 'cpu' and index == '-1':
                    with cls._lock:
                        cls._cpu_override = True
                    return cls._original_torch_device('cpu:0')
                # If MPS is requested and available, use it directly
                if name == 'mps' and torch.backends.mps.is_available():
                    return cls._original_torch_device('mps:0')
                redirected = cls._redirect_device_type(name)
                device_str = f"{redirected}:{index}" if redirected != name else device_type
            else:
                # If MPS is requested and available, use it directly
                if device_type == 'mps' and torch.backends.mps.is_available():
                    return cls._original_torch_device('mps:0')
                redirected = cls._redirect_device_type(device_type)
                device_str = redirected if redirected != device_type else device_type
            return cls._original_torch_device(device_str)
        else:
            with cls._lock:
                if cls._default_device is None:
                    cls._get_default_device()
            if device_type == 'cpu' and device_index == -1:
                with cls._lock:
                    cls._cpu_override = True
                return cls._original_torch_device('cpu', 0)
            # If MPS is requested and available, use it directly
            if device_type == 'mps' and torch.backends.mps.is_available():
                return cls._original_torch_device('mps', 0)
            if isinstance(device_type, str):
                redirected = cls._redirect_device_type(device_type)
                if redirected != device_type:
                    device_type = redirected
            if device_index is not None:
                try:
                    device_index = int(device_index)
                except ValueError:
                    pass
                return cls._original_torch_device(device_type, device_index)
            return cls._original_torch_device(device_type, 0 if device_type == 'mps' else None)

    @classmethod
    @auto_log()
    def _get_default_device(cls):
        # First check if we have a CPU override
        if cls._cpu_override:
            log_message("CPU override is set, using CPU as default device", "device_detection")
            return

        # Then check for MPS availability
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            cls._default_device = 'mps'
            cls._cpu_override = False
            log_message("MPS device detected and available, using as default device", "device_detection")
            return

        # Then check for CUDA availability
        if cls._original_torch_cuda_is_available():
            cls._default_device = 'cuda'
            cls._cpu_override = False
            log_message("CUDA device detected and available, using as default device", "device_detection")
            return

        # Finally fall back to CPU
        cls._default_device = 'cpu'
        cls._cpu_override = False
        log_message("No GPU devices available, falling back to CPU", "device_detection")

    @classmethod
    @auto_log()
    def _redirect_device_type(cls, device_type):
        # If requesting CPU explicitly, respect that
        if device_type == 'cpu':
            return device_type

        # If we have a CPU override and not explicitly requesting MPS, everything goes to CPU
        if cls._cpu_override and device_type != 'mps':
            return 'cpu'

        # For GPU requests (cuda or mps), redirect based on availability
        if device_type in ['cuda', 'mps']:
            # If MPS is requested and available, use it
            if device_type == 'mps' and torch.backends.mps.is_available():
                return 'mps'
            # If CUDA is requested and available, use it
            elif device_type == 'cuda' and cls._original_torch_cuda_is_available():
                return 'cuda'
            # If no requested GPU is available, fall back to any available GPU
            elif torch.backends.mps.is_available():
                return 'mps'
            elif cls._original_torch_cuda_is_available():
                return 'cuda'
            # If no GPU available, fall back to CPU
            return 'cpu'

        # For any other device type, return as is
        return device_type

    @auto_log()
    def __getattr__(self, attr):
        return getattr(self.device, attr)

    # --- Mock CUDA methods ---
    @classmethod
    @auto_log()
    def mock_cuda_is_available(cls):
        return cls._default_device in ['cuda', 'mps']

    @classmethod
    @auto_log()
    def mock_cuda_device_count(cls):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_device_count()
        elif cls._default_device == 'mps':
            return 1
        else:
            return 0

    @classmethod
    @auto_log()
    def mock_cuda_get_device_properties(cls, device):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_get_device_properties(device)
        elif cls._default_device in ['mps', 'cpu']:
            class DummyDeviceProperties:
                name = 'Dummy GPU'
                total_memory = psutil.virtual_memory().total
                major = 0
                minor = 0
                multi_processor_count = 1
                def __str__(self):
                    return f"DummyDeviceProperties(name={self.name}, total_memory={self.total_memory})"
            return DummyDeviceProperties()
        else:
            raise RuntimeError(f"Invalid default device: {cls._default_device}")

    @classmethod
    @auto_log()
    def mock_cuda_memory_allocated(cls, device=None):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    @classmethod
    @auto_log()
    def mock_cuda_memory_reserved(cls, device=None):
        return psutil.virtual_memory().total

    @classmethod
    @auto_log()
    def mock_cuda_max_memory_allocated(cls, device=None):
        return cls.mock_cuda_memory_allocated(device)

    @classmethod
    @auto_log()
    def mock_cuda_max_memory_reserved(cls, device=None):
        return cls.mock_cuda_memory_reserved(device)

    @classmethod
    @auto_log()
    def mock_cuda_memory_stats(cls, device=None):
        return {
            'active.all.current': cls.mock_cuda_memory_allocated(device),
            'active.all.peak': cls.mock_cuda_max_memory_allocated(device),
            'reserved_bytes.all.current': cls.mock_cuda_memory_reserved(device),
            'reserved_bytes.all.peak': cls.mock_cuda_max_memory_reserved(device),
        }

    @classmethod
    @auto_log()
    def mock_cuda_memory_snapshot(cls):
        return [{
            'device': 0,
            'address': 0,
            'total_size': cls.mock_cuda_memory_allocated(),
            'allocated_size': cls.mock_cuda_memory_allocated(),
            'active': True,
            'segment_type': 'small_pool',
        }]

    @classmethod
    @auto_log()
    def mock_cuda_memory_summary(cls, device=None, abbreviated=False):
        return (f"Memory Allocated: {cls.mock_cuda_memory_allocated(device)} bytes\n"
                f"Memory Reserved: {cls.mock_cuda_memory_reserved(device)} bytes\n")

    @classmethod
    @auto_log()
    def mock_cuda_is_initialized(cls):
        return cls._default_device in ['cuda', 'mps']

    @classmethod
    @auto_log()
    def mock_cuda_get_arch_list(cls):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_get_arch_list()
        elif cls._default_device == 'mps':
            return ['mps']
        else:
            return []

    @classmethod
    @auto_log()
    def mock_cuda_is_built(cls):
        return cls._default_device in ['cuda', 'mps']

    @classmethod
    @auto_log()
    def mock_cuda_device_context(cls, device=None):
        class DeviceContextManager:
            @auto_log()
            def __init__(self, device):
                self.device = device
            @auto_log()
            def __enter__(self):
                cls.mock_cuda_set_device(self.device)
            @auto_log()
            def __exit__(self, exc_type, exc_value, traceback):
                pass
        return DeviceContextManager(device)

    @classmethod
    @auto_log()
    def mock_cuda_empty_cache(cls):
        if cls._default_device == 'cuda':
            cls._original_torch_cuda_empty_cache()
        elif cls._default_device == 'mps':
            torch.mps.empty_cache()
        else:
            pass

    @classmethod
    @auto_log()
    def mock_cuda_synchronize(cls, device=None):
        if cls._default_device == 'cuda':
            cls._original_torch_cuda_synchronize(device)
        elif cls._default_device == 'mps':
            torch.mps.synchronize()
        else:
            pass

    @classmethod
    @auto_log()
    def mock_cuda_current_device(cls):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_current_device()
        elif cls._default_device == 'mps':
            return 0
        else:
            return -1

    @classmethod
    @auto_log()
    def mock_cuda_set_device(cls, device):
        if cls._default_device == 'cuda':
            cls._original_torch_cuda_set_device(device)
        elif cls._default_device == 'mps':
            pass
        else:
            pass

    @classmethod
    @auto_log()
    def mock_cuda_get_device_name(cls, device=None):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_get_device_name(device)
        elif cls._default_device == 'mps':
            return 'Apple MPS'
        else:
            return 'CPU'

    @classmethod
    @auto_log()
    def mock_cuda_get_device_capability(cls, device=None):
        if cls._default_device == 'cuda':
            return cls._original_torch_cuda_get_device_capability(device)
        elif cls._default_device == 'mps':
            return (0, 0)
        else:
            return (0, 0)

    @classmethod
    @auto_log()
    def mock_cuda_ipc_collect(cls):
        if cls._default_device == 'cuda':
            return torch.cuda.ipc_collect()
        else:
            pass

    @classmethod
    @auto_log()
    def mock_cuda_stream_class(cls, *args, **kwargs):
        try:
            from torch._streambase import _StreamBase
        except (AttributeError, ImportError):
            try:
                from torch._C import _StreamBase
            except (AttributeError, ImportError):
                try:
                    _StreamBase = torch._C._StreamBase
                except (AttributeError, ImportError):
                    _StreamBase = object
        
        class MPSStream(_StreamBase):
            @auto_log()
            def __init__(self, device=None, priority=0):
                if _StreamBase is not object:
                    try:
                        super().__init__()
                    except Exception as e:
                        log_message(f"Error calling _StreamBase.__init__: {e}",
                                    "torch.cuda.Stream.__init__")
                self.device = device
                self.priority = priority
                self._is_created = True
                self._is_destroyed = False
            
            @auto_log()
            def synchronize(self):
                if cls._default_device == 'mps':
                    torch.mps.synchronize()
                return self
            
            @auto_log()
            def query(self):
                return True
            
            @auto_log()
            def wait_event(self, event):
                if not getattr(event, '_recorded', True):
                    pass
                return self
            
            @auto_log()
            def wait_stream(self, stream):
                if hasattr(stream, 'synchronize'):
                    stream.synchronize()
                self.synchronize()
                return self
            
            @auto_log()
            def record_event(self, event=None):
                if event is None:
                    event = cls.mock_cuda_event(enable_timing=True)
                event.record(self)
                return event
            
            @auto_log()
            def __enter__(self):
                self._old_stream = torch.cuda.current_stream()
                return self
            
            @auto_log()
            def __exit__(self, exc_type, exc_val, traceback):
                return False
            
            @auto_log()
            def __del__(self):
                if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                    self._is_destroyed = True
            
            @auto_log()
            def __str__(self):
                return f"MPSStream(device={self.device}, priority={self.priority})"
            
            @auto_log()
            def __eq__(self, o):
                if isinstance(o, MPSStream):
                    return (self.device == o.device and self.priority == o.priority)
                return False
                
            @auto_log()
            def __hash__(self):
                return hash((self.device, self.priority))
        
        device = kwargs.get('device', None)
        priority = kwargs.get('priority', 0)
        return MPSStream(device, priority)

    @classmethod
    @auto_log()
    def mock_cuda_event(cls, *args, **kwargs):
        enable_timing = kwargs.get('enable_timing', False)
        blocking = kwargs.get('blocking', False)
        interprocess = kwargs.get('interprocess', False)
        device = kwargs.get('device', None)
        MPSEvent = cls._get_mps_event_class()
        return MPSEvent(enable_timing=enable_timing, blocking=blocking, interprocess=interprocess, device=device)
    
    @classmethod
    @auto_log()
    def _get_mps_event_class(cls):
        try:
            from torch._streambase import _EventBase
        except (AttributeError, ImportError):
            try:
                from torch._C import _EventBase
            except (AttributeError, ImportError):
                try:
                    _EventBase = torch._C._EventBase
                except (AttributeError, ImportError):
                    _EventBase = object
        
        class MPSEvent(_EventBase):
            @auto_log()
            def __init__(self, enable_timing=False, blocking=False, interprocess=False, device=None):
                if _EventBase is not object:
                    try:
                        super().__init__()
                    except Exception as e:
                        log_message(f"Error calling _EventBase.__init__: {e}",
                                    "torch.cuda.Event.__init__")
                self.enable_timing = enable_timing
                self.blocking = blocking
                self.interprocess = interprocess
                self.device = device
                self._is_created = True
                self._is_destroyed = False
                self._recorded = False
                self._record_time = None
                self._stream = None
            
            @auto_log()
            def record(self, stream=None):
                self._recorded = True
                self._record_time = time.time()
                self._stream = stream
                return self
            
            @auto_log()
            def wait(self, stream=None):
                if not self._recorded:
                    pass
                return self
            
            @auto_log()
            def query(self):
                return self._recorded
            
            @auto_log()
            def elapsed_time(self, end_event):
                if not self.enable_timing:
                    return 0.5
                if not self._recorded or not getattr(end_event, '_recorded', False):
                    return 0.5
                start_time = self._record_time
                end_time = getattr(end_event, '_record_time', time.time())
                if start_time is None or end_time is None:
                    return 0.5
                elapsed_ms = (end_time - start_time) * 1000.0
                return elapsed_ms
            
            @auto_log()
            def synchronize(self):
                if not self._recorded:
                    pass
                return self
            
            @auto_log()
            def __del__(self):
                if hasattr(self, '_is_destroyed') and not self._is_destroyed:
                    self._is_destroyed = True
        
        return MPSEvent
        
    @classmethod
    @auto_log()
    def mock_cuda_stream(cls, stream=None):
        class StreamContext:
            @auto_log()
            def __init__(self, stream):
                self.stream = stream
            
            @auto_log()
            def __enter__(self):
                if self.stream is not None and hasattr(self.stream, '__enter__'):
                    self.stream.__enter__()
                return self.stream
            
            @auto_log()
            def __exit__(self, exc_type, exc_val, traceback):
                if self.stream is not None and hasattr(self.stream, '__exit__'):
                    return self.stream.__exit__(exc_type, exc_val, traceback)
                return False
        
        return StreamContext(stream)

    @classmethod
    @auto_log()
    def mock_cuda_current_stream(cls, device=None):
        return cls.mock_cuda_stream_class(device=device)

    @classmethod
    @auto_log()
    def mock_cuda_default_stream(cls, device=None):
        return cls.mock_cuda_stream_class(device=device)

    @classmethod
    @auto_log()
    def mock_cuda_function_stub(cls, *args, **kwargs):
        pass

    @classmethod
    @auto_log()
    def mock_cuda_reset_peak_memory_stats(cls):
        if cls._default_device in ['cuda', 'mps']:
            pass
        else:
            pass

    @classmethod
    @auto_log()
    def tensor_creation_wrapper(cls, original_func):
        @auto_log()
        def wrapped_func(*args, **kwargs):
            # If we have a device argument, handle redirection
            if 'device' in kwargs:
                device_arg = kwargs['device']
                # Handle string device specifications
                if isinstance(device_arg, str):
                    device_type = device_arg.split(':')[0] if ':' in device_arg else device_arg
                    redirected_type = cls._redirect_device_type(device_type)
                    if redirected_type != device_type:
                        if ':' in device_arg:
                            index = device_arg.split(':')[1]
                            kwargs['device'] = f"{redirected_type}:{index}"
                        else:
                            kwargs['device'] = redirected_type
                        log_message(f"Redirecting tensor creation from '{device_type}' to '{redirected_type}'.",
                                    original_func.__name__)
                # Handle torch.device objects
                elif hasattr(device_arg, 'type'):
                    device_type = device_arg.type
                    redirected_type = cls._redirect_device_type(device_type)
                    if redirected_type != device_type:
                        index = getattr(device_arg, 'index', 0)
                        if index is None:
                            index = 0
                        kwargs['device'] = cls.torch_device_replacement(f"{redirected_type}:{index}")
                        log_message(f"Redirecting tensor creation from '{device_type}' to '{redirected_type}'.",
                                    original_func.__name__)
            # If no device is specified, use the default device
            else:
                with cls._lock:
                    if cls._default_device is None:
                        cls._get_default_device()
                    kwargs['device'] = cls._default_device
                    log_message(f"Using default device '{cls._default_device}' for tensor creation.",
                                original_func.__name__)
            return original_func(*args, **kwargs)
        return wrapped_func