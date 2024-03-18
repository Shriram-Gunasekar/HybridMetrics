import torch
import torch.nn as nn
from torchmetrics import Metric

class HybridMetric(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("false_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = (preds >= self.threshold).float()
        correct = torch.sum(preds == target)
        self.correct += correct
        self.total += target.numel()

        true_positives = torch.sum(preds * target)
        false_positives = torch.sum(preds * (1 - target))
        false_negatives = torch.sum((1 - preds) * target)

        self.true_positives += true_positives
        self.false_positives += false_positives
        self.false_negatives += false_negatives

    def compute(self):
        accuracy = self.correct.float() / self.total
        precision = self.true_positives.float() / (self.true_positives + self.false_positives + 1e-15)
        recall = self.true_positives.float() / (self.true_positives + self.false_negatives + 1e-15)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-15)

        return {'accuracy': accuracy, 'f1': f1}

# Example usage
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = MyModel()
metric = HybridMetric()

# Dummy data
inputs = torch.randn(100, 10)
targets = torch.randint(0, 2, (100,))

# Training loop
for epoch in range(10):
    outputs = model(inputs)
    metric(outputs.squeeze(), targets)

result = metric.compute()
print(result)
