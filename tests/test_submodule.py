# test_submodule.py

import torch

class ModelTrainer:
    def start_training(self):
        device = torch.device('cuda')
        print(f"Using device in start_training: {device}")

    def call_nested_function(self):
        self._nested_function()

    def _nested_function(self):
        device = torch.device('cuda')
        print(f"Using device in _nested_function: {device}")

    @staticmethod
    def static_method():
        device = torch.device('cuda')
        print(f"Using device in static_method: {device}")

    @classmethod
    def class_method(cls):
        device = torch.device('cuda')
        print(f"Using device in class_method: {device}")
