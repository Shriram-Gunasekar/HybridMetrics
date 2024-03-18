import torch
from torchmetrics import Accuracy, Precision, Recall, F1

class HybridNLPMetric:
    def __init__(self, num_classes):
        self.accuracy = Accuracy()
        self.precision = Precision(num_classes=num_classes, average='macro')
        self.recall = Recall(num_classes=num_classes, average='macro')
        self.f1 = F1(num_classes=num_classes, average='macro')
        self.num_classes = num_classes

    def update(self, predictions, targets):
        self.accuracy(predictions, targets)
        self.precision(predictions, targets)
        self.recall(predictions, targets)
        self.f1(predictions, targets)

    def compute(self):
        accuracy = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()

        return {'accuracy': accuracy, 'precision': precision,
                'recall': recall, 'f1': f1}

# Example usage
num_classes = 3  # Example: 3 classes
metric = HybridNLPMetric(num_classes)

# Dummy data (predictions and targets)
predictions = torch.tensor([0, 1, 2, 1, 0])
targets = torch.tensor([0, 1, 1, 2, 0])

# Update metric
metric.update(predictions, targets)

# Compute and print the hybrid metric
result = metric.compute()
print(result)
