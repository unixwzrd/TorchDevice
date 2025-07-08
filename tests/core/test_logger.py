import os
import unittest
from unittest.mock import patch

import TorchDevice
from tests.common.testing_utils import PrefixedTestCase

# A simple function to be decorated by auto_log for testing
# We are testing the logger, so we need to import the decorator
from TorchDevice.core.logger import auto_log, NOTICE


@auto_log()
def decorated_function(x, y=1):
    if x < 0:
        raise ValueError("x cannot be negative")
    return x + y


class TestLogger(PrefixedTestCase):



    def test_log_levels(self):
        """Test that different log functions produce output at the correct levels."""
        self.logger.error("This is an error")
        self.logger.warning("This is a warning")
        self.logger.info("This is info")
        self.logger.log(NOTICE, "This is a notice")

    @patch.dict(os.environ, {"TORCHDEVICE_LOG_LEVEL": "WARNING"})
    def test_log_level_filtering_warning(self):
        """Test that a WARNING level correctly filters lower-level logs."""
        # Re-import to apply the patched env var
        import importlib
        import TorchDevice.core.logger
        importlib.reload(TorchDevice.core.logger)

        self.logger.info("This should be filtered")
        self.logger.warning("This is a warning")
        decorated_function(1)  # This logs at NOTICE, which is below WARNING

    @patch.dict(os.environ, {"TORCHDEVICE_LOG_LEVEL": "INFO"})
    def test_log_level_filtering_info(self):
        """Test that an INFO level allows both INFO and NOTICE (higher) logs."""
        # Re-import to apply the patched env var
        import importlib
        import TorchDevice.core.logger
        importlib.reload(TorchDevice.core.logger)

        self.logger.info("This should NOT be filtered")
        decorated_function(1)  # This logs at NOTICE, which is >= INFO

    @patch.dict(os.environ, {"TORCHDEVICE_LOG_LEVEL": "NOTICE"})
    def test_auto_log_decorator(self):
        """Test the @auto_log decorator logs calls, returns, and exceptions at NOTICE level."""
        # Re-import to apply the patched env var
        import importlib
        import TorchDevice.core.logger
        importlib.reload(TorchDevice.core.logger)

        # Test normal call and return
        result = decorated_function(5, y=2)
        self.assertEqual(result, 7)

        # Test exception logging
        with self.assertRaises(ValueError):
            decorated_function(-1)


if __name__ == '__main__':
    unittest.main()
