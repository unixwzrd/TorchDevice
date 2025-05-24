import torch

class ModelTrainer:
    def __init__(self):
        self.device = torch.device('cuda')  # This should be redirected to MPS

    def start_training(self):
        # Create a simple model and move it to the device
        model = torch.nn.Linear(10, 1).to(self.device)
        x = torch.randn(5, 10, device=self.device)
        y = model(x)
        return y

    def call_nested_function(self):
        def inner_function():
            return torch.device('cuda')  # This should be redirected to MPS
        return inner_function()

    @staticmethod
    def static_method():
        return torch.device('cuda')  # This should be redirected to MPS

    @classmethod
    def class_method(cls):
        return torch.device('cuda')  # This should be redirected to MPS 