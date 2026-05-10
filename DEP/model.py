# =========================================================================== #
# conv_transformer_v7_model_only_template.py
# Only the model classes. All values replaced with placeholders:
#   ENTER_HERE1 for the first, ENTHER_HERE2 for the second.
# Keep variable NAMES the same; no training, no data, no metrics.
# =========================================================================== #

import math
from dataclasses import dataclass
import torch
import torch.nn as nn

# -------------------------- Placeholders --------------------------- #
ENTER_HERE1 = "ENTER_HERE1"
ENTHER_HERE2 = "ENTHER_HERE2"

# ---------------------------- Config ------------------------------- #
@dataclass
class ConfigBuzelin:
    seq_length:   int   = ENTER_HERE1
    hidden_dim:   int   = ENTHER_HERE2
    mlp_dim:      int   = ENTER_HERE1
    window_size:  int   = ENTHER_HERE2
    stride:       int   = ENTER_HERE1
    dropout_rate: float = ENTHER_HERE2

# ----------------------- Building Blocks --------------------------- #
class DilatedStem(nn.Module):
    def __init__(self, in_ch=ENTER_HERE1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, ENTER_HERE1, padding=ENTER_HERE1, dilation=ENTER_HERE1), nn.GELU(),
            nn.Conv1d(in_ch, in_ch, ENTHER_HERE2, padding=ENTHER_HERE2, dilation=ENTHER_HERE2), nn.GELU(),
            nn.Conv1d(in_ch, in_ch, ENTER_HERE1, padding=ENTER_HERE1, dilation=ENTER_HERE1), nn.GELU()
        )
    def forward(self, x):
        return self.net(x)

class WindowedAttention(nn.Module):
    def __init__(self, dim, kernel, stride, heads=ENTER_HERE1):
        super().__init__()
        self.norm     = nn.LayerNorm(dim)
        self.K        = nn.Conv1d(dim, dim, ENTER_HERE1, padding=ENTER_HERE1, bias=False)
        self.V        = nn.Conv1d(dim, dim, ENTER_HERE1, padding=ENTER_HERE1, bias=False)
        self.Qc       = nn.Conv1d(dim, dim, ENTER_HERE1, padding=ENTER_HERE1, bias=False)
        self.kernel   = kernel
        self.stride   = stride
        self.heads    = heads
        self.head_dim = dim // heads
        self.scale    = math.sqrt(self.head_dim)

    def forward(self, x):                             # x: (B,N,D)
        B, N, D = x.shape
        x_n     = self.norm(x)

        M = ENTER_HERE1  # number of windows (placeholder)
        Q = x_n.new_zeros(B, M, D)

        # (the exact windowing is omitted here since values are placeholders)
        K = self.K(x_n.permute(0,2,1)).permute(0,2,1)
        V = self.V(x_n.permute(0,2,1)).permute(0,2,1)

        Qh = Q.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        Kh = K.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        Vh = V.view(B, N, self.heads, self.head_dim).transpose(1, 2)

        out = (Qh @ Kh.transpose(-2, -1) / self.scale).softmax(-1) @ Vh
        return out.transpose(1, 2).contiguous().view(B, M, D) + Q

class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, win, stride):
        super().__init__()
        self.attn     = WindowedAttention(dim, win, stride)
        self.norm1    = nn.LayerNorm(dim)
        self.res_conv = nn.Conv1d(dim, dim, ENTER_HERE1)
        self.res_pool = nn.MaxPool1d(ENTER_HERE1, ENTHER_HERE2)
        self.norm2    = nn.LayerNorm(dim)
        self.mlp      = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(ENTER_HERE1),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x_n = self.norm1(x)
        y   = self.attn(x_n)

        # residual pool
        r   = self.res_pool(x_n.permute(0,2,1))
        r   = self.res_conv(r).permute(0,2,1)
        L   = ENTHER_HERE2  # placeholder for length alignment
        x2  = y[:, :L, :] + r[:, :L, :]
        return x2 + self.mlp(self.norm2(x2))

class SummaryAggregator(nn.Module):
    def __init__(self, hid, n_heads=ENTER_HERE1):
        super().__init__()
        self.summary_token = nn.Parameter(torch.randn(ENTER_HERE1, ENTER_HERE1, hid))
        self.attn_layer    = nn.TransformerEncoderLayer(
            d_model=hid, nhead=n_heads, batch_first=True
        )
    def forward(self, x):
        tok = self.summary_token.expand(x.size(0), -1, -1)
        seq = torch.cat([tok, x], dim=ENTHER_HERE2)
        out = self.attn_layer(seq)
        return out[:, ENTER_HERE1, :]

class GlobalAttentionPath(nn.Module):
    def __init__(self, in_dim=ENTER_HERE1, hid_dim=ENTHER_HERE2,
                 n_heads=ENTER_HERE1, n_layers=ENTHER_HERE2, dropout=ENTER_HERE1):
        super().__init__()
        self.proj   = nn.Linear(in_dim, hid_dim)
        enc_layer   = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=n_heads,
            dim_feedforward=hid_dim * ENTHER_HERE2,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm    = nn.LayerNorm(hid_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        return self.norm(x)[:, ENTER_HERE1, :]

class ConvTransformerCached_v7(nn.Module):
    def __init__(self, cfg: ConfigBuzelin):
        super().__init__()
        # local
        self.stem     = DilatedStem(in_ch=ENTER_HERE1)
        self.proj     = nn.Linear(ENTER_HERE1, cfg.hidden_dim)
        self.trf      = nn.Sequential(*[
            TransformerBlock(cfg.hidden_dim, cfg.mlp_dim * (ENTHER_HERE2 ** i),
                             cfg.window_size, cfg.stride)
            for i in range(ENTER_HERE1)
        ])
        self.loc_agg  = SummaryAggregator(cfg.hidden_dim, n_heads=ENTER_HERE1)
        # global
        self.glob     = GlobalAttentionPath(in_dim=ENTER_HERE1, hid_dim=cfg.hidden_dim,
                                            n_heads=ENTER_HERE1, n_layers=ENTHER_HERE2,
                                            dropout=cfg.dropout_rate)
        # head
        self.cls_head = nn.Linear(cfg.hidden_dim * ENTHER_HERE2, ENTER_HERE1)

    def forward(self, emb):
        emb = emb.float()
        # global
        g = self.glob(emb)
        # local (sem CLS)
        body  = emb[:, ENTHER_HERE2:, :]
        x     = self.stem(body.permute(ENTER_HERE1, ENTHER_HERE2, ENTER_HERE1)).permute(ENTER_HERE1, ENTHER_HERE2, ENTER_HERE1)
        x     = self.proj(x)
        x     = self.trf(x)
        l     = self.loc_agg(x)
        out   = torch.cat([l, g], dim=-1)
        return self.cls_head(out).squeeze(-1)
