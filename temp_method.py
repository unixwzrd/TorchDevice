    @classmethod
    def mock_cuda_reset_peak_memory_stats(cls):
        if cls.get_default_device() in ['cuda', 'mps']:
            log_info("Resetting peak memory stats.", "torch.cuda.reset_peak_memory_stats", device_type=cls.get_default_device())
        else:
            log_warning("No GPU available to reset peak memory stats.", "torch.cuda.reset_peak_memory_stats", device_type='cpu')
