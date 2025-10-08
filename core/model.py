import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1)

    def forward(self, x):   # x: (B,T,H)
        w = F.softmax(self.query(x), dim=1)  # (B,T,1)
        ctx = torch.sum(w * x, dim=1)        # (B,H)
        return ctx


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, gru_layers=1, dropout=0.1, attention=True, action_dim=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        self.attn = AttentionPooling(hidden_dim) if attention else None
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)  # (B,2)
        self.value_head  = nn.Linear(hidden_dim, 1)           # (B,1)

    def forward(self, x):  # x: (B,T,F)
        out, _ = self.gru(x)                  # (B,T,H)
        h = self.attn(out) if self.attn else out[:, -1, :]
        z = self.fc(h)
        return self.policy_head(z), self.value_head(z)