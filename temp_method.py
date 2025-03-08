    @classmethod
    def mock_cuda_reset_peak_memory_stats(cls):
        if cls.get_default_device() in ['cuda', 'mps']:
            log_message("Resetting peak memory stats.", "torch.cuda.reset_peak_memory_stats")
        else:
            log_message("No GPU available to reset peak memory stats.", "torch.cuda.reset_peak_memory_stats")
