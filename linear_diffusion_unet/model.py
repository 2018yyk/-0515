import torch
import torch.nn as nn

class TimeEmbedding(nn.Module):
    def __init__(self, max_time=1000000, embedding_dim=64, output_dim=128):
        super(TimeEmbedding, self).__init__()
        self.embedding = nn.Embedding(max_time + 1, embedding_dim)
        self.project = nn.Sequential(
            nn.GELU(),
            nn.Linear(embedding_dim, output_dim)
        )

    def forward(self, times):
        return self.project(self.embedding(times))

class ExtendedLinearUNet(nn.Module):
    def __init__(self):
        super(ExtendedLinearUNet, self).__init__()
        self.expand = nn.Linear(25, 64)
        self.hidden1 = nn.Linear(64, 128)
        self.hidden2 = nn.Linear(128, 128)
        self.hidden3 = nn.Linear(128, 64)
        self.compress = nn.Linear(64, 25)
        self.gelu = nn.GELU()

    def forward(self, x, time_emb):
        skip1 = self.gelu(self.expand(x))
        x = self.gelu(self.hidden1(skip1))
        x = x + time_emb
        x = self.gelu(self.hidden2(x))
        x = self.gelu(self.hidden3(x))
        x = x + skip1
        return self.compress(x)

class TimeOrderClassifier(nn.Module):
    def __init__(self, input_size):
        super(TimeOrderClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
