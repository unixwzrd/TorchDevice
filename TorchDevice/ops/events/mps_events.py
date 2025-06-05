"""
TorchDevice MPS Events Module
-------------------------
MPS event handling and synchronization, including mocks for CUDA events.
"""
import time
import torch
from typing import Optional, Any # Added Any for stream type hint
from TorchDevice.core.logger import log_info, auto_log

log_info("Initializing TorchDevice MPS events module")

class MockCudaEvent:
    """
    A mock CUDA event for use on non-CUDA devices like MPS.
    Provides a compatible API for torch.cuda.Event.
    """
    @auto_log()
    def __init__(self, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False):
        self.enable_timing = enable_timing
        self.blocking = blocking  # Not strictly used in mock, but part of API
        self.interprocess = interprocess # Not strictly used, part of API
        self._recorded: bool = False
        self._record_time: Optional[float] = None
        self.stream: Optional[Any] = None # For compatibility, can be set by record
        log_info("MockCudaEvent created (enable_timing=%s, blocking=%s)", self.enable_timing, self.blocking)

    @auto_log()
    def record(self, stream: Optional[Any] = None) -> None:
        """Records the event in a given stream."""
        self.stream = stream 
        self._recorded = True
        if self.enable_timing:
            self._record_time = time.monotonic()
        log_info("MockCudaEvent recorded on stream: %s", stream)

    @auto_log()
    def synchronize(self) -> None:
        """Waits for the event to complete."""
        log_info("MockCudaEvent synchronize called (no-op)")
        pass

    @auto_log()
    def query(self) -> bool:
        """Checks if the event has been recorded."""
        log_info("MockCudaEvent query called, recorded: %s", self._recorded)
        return self._recorded

    @auto_log()
    def elapsed_time(self, end_event: 'MockCudaEvent') -> float:
        """Returns the time elapsed between two events."""
        if not self.enable_timing or not end_event.enable_timing:
            raise RuntimeError("Timing must be enabled for both events to measure elapsed time.")
        if not self._recorded or not end_event._recorded:
            raise RuntimeError("Both events must be recorded to measure elapsed time.")
        if self._record_time is None or end_event._record_time is None:
            raise RuntimeError("Internal error: record_time not set despite timing enabled and event recorded.")
        
        elapsed = (end_event._record_time - self._record_time) * 1000.0
        log_info("MockCudaEvent elapsed_time called, result: %s ms", elapsed)
        return elapsed

def apply_patches() -> None:
    """Apply MPS event patches."""
    log_info("Applying MPS event patches (currently no specific MPS patches for events)")
    log_info("MPS event patches applied")

__all__: list[str] = [
    'MockCudaEvent',
    'apply_patches'
]

log_info("TorchDevice MPS events module initialized")