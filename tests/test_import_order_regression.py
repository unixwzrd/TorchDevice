import unittest
import subprocess
import sys
# Remove any custom arguments that unittest doesn't recognize
sys.argv = [sys.argv[0]]


class TestImportOrderRegression(unittest.TestCase):
    def run_script(self, script):
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        return result

    def test_import_torch_then_torchdevice(self):
        script = """
import torch
import TorchDevice
try:
    d = torch.device('cpu:-1')
    print('device:', d.type, d.index)
    assert d.type == 'cpu' and (d.index == 0 or d.index is None)
    print('PASS')
except Exception as e:
    print('FAIL:', e)
    raise
"""
        result = self.run_script(script)
        self.assertIn('PASS', result.stdout, f"Output: {result.stdout}\nError: {result.stderr}")

    def test_import_torchdevice_then_torch(self):
        script = """
import TorchDevice
import torch
try:
    d = torch.device('cpu:-1')
    print('device:', d.type, d.index)
    assert d.type == 'cpu' and (d.index == 0 or d.index is None)
    print('PASS')
except Exception as e:
    print('FAIL:', e)
    raise
"""
        result = self.run_script(script)
        self.assertIn('PASS', result.stdout, f"Output: {result.stdout}\nError: {result.stderr}")


if __name__ == "__main__":
    unittest.main(argv=sys.argv[:1])