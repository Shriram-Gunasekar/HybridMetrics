import torch
import torch.nn as nn
from torchmetrics import Metric

class HybridRegressionMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        abs_error = torch.abs(preds - target).sum()
        squared_error = torch.pow(preds - target, 2).sum()
        self.abs_error += abs_error
        self.squared_error += squared_error
        self.total_samples += target.numel()

    def compute(self):
        mae = self.abs_error / self.total_samples
        mse = self.squared_error / self.total_samples
        return {'mae': mae, 'mse': mse}

# Example usage
class MyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

model = MyRegressor()
metric = HybridRegressionMetric()

# Dummy data
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# Training loop
for epoch in range(10):
    outputs = model(inputs)
    metric(outputs, targets
