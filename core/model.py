import datetime
from dataclasses import dataclass
from enum import Enum

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
    def __init__(
        self,
        input_dim=1, hidden_dim=128, gru_layers=2, dropout=0.1,
        attention=True, action_dim=2, num_heads=2,
        dual_timescale=True,
        short_input_dim=1,
        long_input_dim=2,
        account_dim=3   # ratio + current_price + PNL
    ):
        super().__init__()
        self.dual = dual_timescale
        self.attn_on = attention
        self.hidden_dim = hidden_dim

        # --- short-term (15 min) ---
        self.gru_s = nn.GRU(
            input_size=short_input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        self.attn_s = MultiHeadAttentionPooling(hidden_dim, num_heads) if attention else None

        # --- long-term (1 day + kimchi) ---
        self.gru_l = nn.GRU(
            input_size=long_input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        self.attn_l = MultiHeadAttentionPooling(hidden_dim, num_heads) if attention else None

        # --- fuse ---
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2 + account_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head  = nn.Linear(hidden_dim, 1)

        torch.nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        torch.nn.init.constant_(self.policy_head.bias, 0.0)
        torch.nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        torch.nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, x):
        """x = (x_short, x_long, x_acc)"""
        x_s, x_l, x_acc = x
        out_s, _ = self.gru_s(x_s)
        out_l, _ = self.gru_l(x_l)

        h_s = self.attn_s(out_s) if self.attn_s else out_s[:, -1, :]
        h_l = self.attn_l(out_l) if self.attn_l else out_l[:, -1, :]

        # 계좌상태(acc)는 (B,2) 형태니까 바로 concat
        h = torch.cat([h_s, h_l, x_acc], dim=-1)
        z = self.fuse(h)
        z = torch.tanh(z)

        return self.policy_head(z), self.value_head(z)