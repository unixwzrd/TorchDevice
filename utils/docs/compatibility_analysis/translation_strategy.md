# Translation Strategy by Compatibility Pattern

## Translation Approaches

| Pattern | Translation Strategy | Priority | Notes |
|---------|---------------------|----------|-------|
| **Cuda Only** | MPS translation | High | Core translation target (0 functions) |
| **Cpu Cuda** | MPS translation | High | Important for CUDA compatibility (0 functions) |
| **Device Specific** | Special handling | High | May need custom implementation (0 functions) |
| **Cpu Only** | CPU fallback | Medium | May need performance optimization (0 functions) |
| **Cpu Mps** | MPS native | Medium | Good compatibility (0 functions) |
| **Cuda Mps** | MPS native | Medium | Good compatibility (0 functions) |
| **Unknown** | Investigation needed | Medium | Requires testing (1145 functions) |
| **Universal** | Pass through | Low | No translation needed (0 functions) |
| **Mps Only** | MPS native | Low | Already compatible (0 functions) |
