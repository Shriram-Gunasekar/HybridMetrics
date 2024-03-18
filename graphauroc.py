import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torchmetrics import AUROC

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

# Example usage
model = GCN(in_channels=16, hidden_channels=64, out_channels=1)
auroc = AUROC()

# Dummy data (graph inputs)
x = torch.randn(100, 16)  # Node features
edge_index = torch.randint(0, 100, (2, 200))  # Example edge indices
y = torch.randint(0, 2, (100,))  # Binary labels

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(10):
    optimizer.zero_grad()
    pred = model(x, edge_index)
    loss = F.binary_cross_entropy(pred.view(-1), y.float())
    loss.backward()
    optimizer.step()

    # Update AUROC metric
    auroc(pred.view(-1), y)

result = auroc.compute()
print("AUROC:", result)
