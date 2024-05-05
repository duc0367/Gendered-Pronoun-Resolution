import torch.nn as nn
import torch.nn.functional as F


class GAPModel(nn.Module):
    def __init__(self):
        super(GAPModel, self).__init__()
        self.fc1 = nn.Linear(3*768, 4096)
        self.norm1 = nn.LayerNorm(4096)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 2048)
        self.norm2 = nn.LayerNorm(2048)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
