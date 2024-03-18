import torch
import torch.nn as nn
from torchmetrics import Metric
from sklearn.metrics import precision_score, recall_score, f1_score

class HybridClassificationMetric(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("precision", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("recall", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("f1", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = torch.argmax(preds, dim=1)
        correct = torch.sum(preds == target)
        self.correct += correct
        self.total += target.numel()

        precision = precision_score(target.cpu().detach().numpy(), preds.cpu().detach().numpy(), average=None, zero_division=0)
        recall = recall_score(target.cpu().detach().numpy(), preds.cpu().detach().numpy(), average=None, zero_division=0)
        f1 = f1_score(target.cpu().detach().numpy(), preds.cpu().detach().numpy(), average=None, zero_division=0)

        self.precision += torch.tensor(precision)
        self.recall += torch.tensor(recall)
        self.f1 += torch.tensor(f1)

    def compute(self):
        accuracy = self.correct.float() / self.total
        precision = self.precision.float() / self.num_classes
        recall = self.recall.float() / self.num_classes
        f1 = self.f1.float() / self.num_classes

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Example usage
class MyClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(10, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x))

num_classes = 3  # Example: 3 classes
model = MyClassifier(num_classes)
metric = HybridClassificationMetric(num_classes)

# Dummy data
inputs = torch.randn(100, 10)
targets = torch.randint(0, num_classes, (100,))

# Training loop
for epoch in range(10):
    outputs = model(inputs)
    metric(outputs, targets)

result = metric.compute()
print(result)
