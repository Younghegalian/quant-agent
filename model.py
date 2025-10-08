import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        w = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (x * w).sum(dim=1)               # (B, H)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = AttentionPooling(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, 2)  # action logits
        self.value_head = nn.Linear(hidden_dim, 1)   # state value

    def forward(self, x):
        # x: (B, T, input_dim)
        out, _ = self.gru(x)
        h = F.relu(self.fc(self.attn(out)))
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value