import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Multi-Head Attention Pooling (확장형) ---
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):  # x: (B,T,H)
        B, T, H = x.shape
        head_dim = H // self.num_heads

        Q = self.query(x).view(B, T, self.num_heads, head_dim).transpose(1, 2)
        K = self.key(x).view(B, T, self.num_heads, head_dim).transpose(1, 2)
        V = self.value(x).view(B, T, self.num_heads, head_dim).transpose(1, 2)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        context = torch.matmul(attn_weights, V)  # (B, heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, T, H)
        pooled = context.mean(dim=1)  # (B,H)

        return self.out(pooled)


# --- Policy Network (개선형) ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, gru_layers=2, dropout=0.1,
                 attention=True, action_dim=2, num_heads=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        self.attn = MultiHeadAttentionPooling(hidden_dim, num_heads) if attention else None

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: (B,T,F)
        out, _ = self.gru(x)  # (B,T,H)
        if self.attn:
            h = self.attn(out)
        else:
            h = out[:, -1, :]  # 마지막 timestep
        z = self.fc(h)
        return self.policy_head(z), self.value_head(z)