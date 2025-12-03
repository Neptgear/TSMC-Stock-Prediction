import torch
from torch import nn
from typing import Optional, Sequence


class Gate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.sigmoid(self.linear(x))
        return g * x + (1.0 - g) * skip


class VariableSelection(nn.Module):
    """Simple variable selection network: per-feature gate + projection."""

    def __init__(self, input_size: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        self.gate = nn.Sequential(nn.Linear(input_size, input_size), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        gates = self.gate(x)  # (B, T, F)
        gated = x * gates
        return self.proj(gated)


class FullTFT(nn.Module):
    """
    A more faithful Temporal Fusion Transformer:
    - Variable selection for observed past and known future
    - Static context projection
    - LSTM encoder/decoder
    - Multi-head attention over decoder outputs
    - Quantile output per horizon step
    """

    def __init__(
        self,
        obs_size: int,
        known_size: int,
        static_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
        quantiles: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.horizon = int(max(1, horizon))
        self.quantiles = list(float(q) for q in (quantiles if quantiles is not None else (0.1, 0.5, 0.9)))
        self.n_quantiles = len(self.quantiles)

        self.vs_obs = VariableSelection(obs_size, d_model)
        self.vs_known = VariableSelection(known_size if known_size > 0 else 1, d_model)
        self.static_proj = nn.Linear(static_size, d_model) if static_size > 0 else None

        self.enc_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dec_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.gate = Gate(d_model)
        self.post_ff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.n_quantiles)

    def forward(
        self,
        enc_obs: torch.Tensor,
        dec_known: torch.Tensor,
        dec_start: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # enc_obs: (B, T_enc, F_obs), dec_known: (B, H, F_known), dec_start: (B, H, 1), static: (B, F_static)
        B, H = dec_known.shape[0], dec_known.shape[1]
        enc_in = self.vs_obs(enc_obs)
        if dec_known.size(-1) == 0:
            dec_known_use = torch.zeros(dec_known.shape[0], dec_known.shape[1], 1, device=dec_known.device, dtype=dec_known.dtype)
        else:
            dec_known_use = dec_known
        dec_in = self.vs_known(dec_known_use + dec_start)

        if self.static_proj is not None and static is not None and static.numel() > 0:
            s = self.static_proj(static).unsqueeze(1)
            enc_in = enc_in + s
            dec_in = dec_in + s

        enc_out, (h, c) = self.enc_lstm(enc_in)
        dec_out, _ = self.dec_lstm(dec_in, (h, c))

        attn_out, _ = self.attn(dec_out, enc_out, enc_out)
        fused = self.gate(attn_out, dec_out)
        fused = self.norm(fused + self.post_ff(fused))
        out = self.head(fused)  # (B, H, Q)
        return out
