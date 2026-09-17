import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass, asdict, replace
from typing import Tuple, Dict, Any, Optional, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from data_fetch import get_ohlcv
from data_fetch import get_fundamentals_timeseries, get_event_countdowns
from ta_features import (
    compute_features,
    compute_indicators_only,
    get_default_feature_columns,
    get_tft_feature_columns,
    make_sliding_windows,
    make_sliding_windows_with_base,
    time_series_train_test_split,
)
from tft_model import FullTFT


def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SequenceStandardScaler:
    """Standardize features across samples and time for 3D sequences.

    Fits mean and std over the flattened (n_samples * seq_len, n_features) matrix
    and applies the same scaling to each timestep.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SequenceStandardScaler":
        # X: (N, T, F) -> (N*T, F)
        N, T, F = X.shape
        flat = X.reshape(N * T, F)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted")
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted")
        return (X * self.std_) + self.mean_

    def state_dict(self) -> Dict[str, Any]:
        return {"mean_": self.mean_, "std_": self.std_}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean_ = state["mean_"]
        self.std_ = state["std_"]


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))  # (N, T, F)
        self.y = torch.from_numpy(y.astype(np.float32))  # (N,)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SeqFundDataset(Dataset):
    """Dataset with sequence (technical) and vector (fundamentals)."""
    def __init__(self, X_seq: np.ndarray, X_fund: np.ndarray, y: np.ndarray):
        self.X_seq = torch.from_numpy(X_seq.astype(np.float32))  # (N, T, F_ta)
        self.X_fund = torch.from_numpy(X_fund.astype(np.float32))  # (N, F_fund)
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.X_seq.shape[0]

    def __getitem__(self, idx: int):
        return self.X_seq[idx], self.X_fund[idx], self.y[idx]


class Seq2SeqDataset(Dataset):
    """Encoder-decoder dataset with observed past, known future, static, and full-horizon targets."""
    def __init__(
        self,
        enc_obs: np.ndarray,
        dec_known: np.ndarray,
        dec_start: np.ndarray,
        static: np.ndarray,
        y: np.ndarray,
    ):
        # Shapes:
        # enc_obs: (N, T_enc, F_obs)
        # dec_known: (N, T_dec, F_known)
        # dec_start: (N, T_dec, 1)   (shifted targets with start token)
        # static: (N, F_static)
        # y: (N, T_dec)
        self.enc_obs = torch.from_numpy(enc_obs.astype(np.float32))
        self.dec_known = torch.from_numpy(dec_known.astype(np.float32))
        self.dec_start = torch.from_numpy(dec_start.astype(np.float32))
        self.static = torch.from_numpy(static.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.enc_obs[idx],
            self.dec_known[idx],
            self.dec_start[idx],
            self.static[idx],
            self.y[idx],
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TimeSeriesBackbone(nn.Module):
    """Transformer encoder that outputs the last-token representation (no head)."""
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model)
        self.norm = nn.LayerNorm(d_model)
        # Lightweight attention pooling over the full window to capture
        # information from all timesteps, not just the last one.
        self.pool_attn = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)  # (B, T, D)
        # Last-timestep representation
        h_last = h[:, -1, :]  # (B, D)
        # Soft attention pooling over all timesteps
        attn_scores = self.pool_attn(h)  # (B, T, 1)
        attn_weights = attn_scores.softmax(dim=1)
        h_pool = (attn_weights * h).sum(dim=1)  # (B, D)
        # Fuse last state and pooled context then normalise
        fused = torch.cat([h_last, h_pool], dim=-1)  # (B, 2D)
        fused = self.out_proj(fused)  # (B, D)
        return self.norm(fused)


class TimeSeriesTransformer(nn.Module):
    """Backwards compatibility: backbone + linear head."""
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = TimeSeriesBackbone(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_last = self.backbone(x)
        return self.regressor(h_last).squeeze(-1)


class FundMLP(nn.Module):
    def __init__(self, input_size: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SeqFundRegressor(nn.Module):
    """Two-stream model: transformer over TA sequence + MLP over fund vector."""
    def __init__(
        self,
        ta_input_size: int,
        fund_input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        fund_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = TimeSeriesBackbone(
            input_size=ta_input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.fund = FundMLP(fund_input_size, hidden=fund_hidden, dropout=dropout) if fund_input_size > 0 else None
        # Fuse the sequence embedding with fund embedding
        fusion_in = d_model + (fund_hidden if fund_input_size > 0 else 0)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_fund: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_emb = self.backbone(x_seq)  # (B, D)
        if self.fund is not None and x_fund is not None and x_fund.numel() > 0:
            fund_out = self.fund(x_fund)
            fused = torch.cat([seq_emb, fund_out], dim=-1)
        else:
            fused = seq_emb
        return self.fusion(fused).squeeze(-1)


class Seq2SeqTransformer(nn.Module):
    """Encoder-decoder transformer for multi-step forecasting with known-future inputs."""
    def __init__(
        self,
        obs_size: int,
        known_size: int,
        static_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        horizon: int = 1,
    ) -> None:
        super().__init__()
        self.horizon = int(max(1, horizon))
        self.obs_proj = nn.Linear(obs_size, d_model)
        self.known_proj = nn.Linear(known_size, d_model)
        self.target_proj = nn.Linear(1, d_model)
        self.static_proj = nn.Linear(static_size, d_model) if static_size > 0 else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def _generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        enc_obs: torch.Tensor,
        dec_known: torch.Tensor,
        dec_start: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # enc_obs: (B, T_enc, F_obs)
        # dec_known: (B, T_dec, F_known)
        # dec_start: (B, T_dec, 1)   (shifted targets with start token)
        # static: (B, F_static)
        device = enc_obs.device
        enc_emb = self.obs_proj(enc_obs)  # (B, T_enc, D)
        if static is not None and self.static_proj is not None and static.numel() > 0:
            s_emb = self.static_proj(static).unsqueeze(1)  # (B, 1, D)
            enc_emb = enc_emb + s_emb
        memory = self.encoder(enc_emb)  # (B, T_enc, D)

        dec_in = self.known_proj(dec_known) + self.target_proj(dec_start)  # (B, T_dec, D)
        tgt_mask = self._generate_square_subsequent_mask(dec_in.size(1), device)
        out = self.decoder(tgt=dec_in, memory=memory, tgt_mask=tgt_mask)
        preds = self.head(out).squeeze(-1)  # (B, T_dec)
        return preds


class QuantileLoss(nn.Module):
    """
    Multi-quantile loss for sequences.

    Expects:
    - preds: (B, H, Q)
    - target: (B, H) or (B, H, 1)
    """
    def __init__(self, quantiles):
        super().__init__()
        q = torch.as_tensor(list(quantiles), dtype=torch.float32)
        self.register_buffer("quantiles", q.view(1, 1, -1))

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if preds.dim() != 3:
            raise ValueError("QuantileLoss expects preds with shape (B, H, Q)")
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        elif target.dim() != 3:
            raise ValueError("QuantileLoss expects target shape (B, H) or (B, H, 1)")
        q = self.quantiles.to(preds.device, dtype=preds.dtype)
        errors = target - preds
        loss = torch.maximum(q * errors, (q - 1.0) * errors)
        return loss.mean()


class VectorStandardScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "VectorStandardScaler":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler not fitted")
        return (X - self.mean_) / self.std_

    def state_dict(self) -> Dict[str, Any]:
        return {"mean_": self.mean_, "std_": self.std_}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.mean_ = state["mean_"]
        self.std_ = state["std_"]


@dataclass
class TrainConfig:
    ticker: str = "2330.TW"
    start: str = "2010-01-01"
    end: Optional[str] = None
    target: str = "next_close"  # or 'next_return_pct'
    horizon: int = 1
    window_size: int = 30
    batch_size: int = 64
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    scale_target: bool = False
    residualize: bool = True
    residual_base: str = "close"  # 'close' or 'ma20'
    loss: str = "mse"  # 'mse', 'huber', or 'mae'
    huber_delta: float = 1.0
    # TFT-specific: quantile loss configuration
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)
    clip_grad_norm: float = 0.0  # 0 disables clipping
    lr_scheduler: bool = True
    # Split controls
    train_ratio: float = 0.8
    split_mode: str = "ratio_time"  # 'ratio_samples', 'ratio_time', 'last_n_days'
    test_days: int = 0  # used when split_mode='last_n_days'
    # Explicit cutoff: train on target_dates <= split_date, test on > split_date
    split_date: Optional[str] = None
    # Hold out part of the training window for validation/early stopping
    val_ratio: float = 0.1
    # Window suggestion controls (always applied from horizon)
    min_window: int = 30
    max_window: int = 504
    # Fundamentals controls
    use_fundamentals: bool = True
    min_samples_with_fundamentals: int = 30
    require_fundamentals: bool = False
    # Evaluation split controls
    min_test_samples: int = 30
    # Persistence
    persist: bool = False
    # Model type: 'transformer' (SeqFundRegressor) or 'tft' (TemporalFusionRegressor)
    model_type: str = "transformer"
    # Walk-forward evaluation controls
    walkforward_splits: int = 0  # number of additional chronological folds for overfit checks (0 disables)
    walkforward_epochs: Optional[int] = None  # optional override for epochs per walk-forward fold


def suggest_window(h: int, min_w: int = 30, max_w: int = 504) -> int:
    """Heuristic window suggestion based on horizon.

    Mapping for common horizons, otherwise 6*h clamped.
    """
    table = {1: 60, 2: 90, 3: 120, 5: 120, 10: 180, 20: 252, 30: 252}
    if h in table:
        w = table[h]
    else:
        w = int(round(6 * max(1, h)))
    return max(min_w, min(max_w, w))


def standardize_ohlcv_columns(df_in):
    """Normalize OHLCV columns to standard names.

    Accepts inputs like 'Open_2330.TW', lowercase variants, or MultiIndex columns,
    and returns a DataFrame with columns among:
    ['Open','High','Low','Close','Adj Close','Volume'] when available.
    """
    import pandas as _pd

    df_local = df_in.copy()
    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if isinstance(df_local.columns, _pd.MultiIndex):
        first = df_local.columns.get_level_values(0)
        df_local.columns = first

    cols = list(map(str, df_local.columns))
    rename = {}

    def norm(s: str) -> str:
        return s.lower().replace(" ", "").replace("-", "").strip()

    for need in expected:
        if need in df_local.columns:
            continue
        nneed = norm(need)
        for c in cols:
            lc = norm(c)
            if lc == nneed or lc.endswith(nneed) or lc.startswith(nneed):
                rename[c] = need
                break
    if rename:
        df_local = df_local.rename(columns=rename)
    return df_local


def run_training_transformer_seq2seq(cfg: TrainConfig) -> dict:
    """Full seq2seq transformer training with observed/known/static splits and causal decoding."""
    seed_everything(42)
    df = get_ohlcv(ticker_yt=cfg.ticker, start=cfg.start, end=cfg.end, auto_adjust=True)
    data_quality_report = dict(df.attrs.get("data_quality", {}))
    df = standardize_ohlcv_columns(df)
    events_df = get_event_countdowns(cfg.ticker, df.index)
    feat_df = compute_features(df, target=cfg.target, horizon=cfg.horizon, include_extra=True, events_df=events_df)

    # Feature grouping
    known_future_candidates = {"year", "month", "day_of_week", "is_month_end", "is_quarter_end", "is_holiday_eve"}
    known_future_cols = [c for c in feat_df.columns if c in known_future_candidates or (isinstance(c, str) and c.startswith("f_days_"))]
    static_cols = [c for c in feat_df.columns if isinstance(c, str) and c.startswith("f_")]
    observed_cols = [c for c in feat_df.columns if c not in known_future_cols and c not in static_cols and c != "target"]

    # Forward fill uses only information available at or before each row. Do not
    # backfill, because it would copy future observations into earlier samples.
    if observed_cols:
        feat_df[observed_cols] = feat_df[observed_cols].ffill()
    clean_subset = observed_cols + ["target"]
    feat_df = feat_df.dropna(subset=clean_subset)
    if len(feat_df) == 0:
        raise RuntimeError("No rows remaining after cleaning features; try an earlier start, smaller window, or disable fundamentals.")

    # Known-future values may use a neutral fallback. Static/fundamental values
    # are forward-filled only; neither group is standardized before splitting.
    if known_future_cols:
        feat_df[known_future_cols] = feat_df[known_future_cols].fillna(0)
    if static_cols:
        feat_df[static_cols] = feat_df[static_cols].ffill().fillna(0)

    # Build tensors
    raw_close_series = df["Close"]
    enc_obs, dec_known, dec_start, static_vecs, y_all, base_last, anchor_dates, target_dates = build_seq2seq_tensors(
        feat_df,
        raw_close=raw_close_series,
        observed_cols=observed_cols,
        known_cols=known_future_cols,
        static_cols=static_cols,
        target_col="target",
        horizon=cfg.horizon,
        window_size=cfg.window_size,
    )
    num_samples = enc_obs.shape[0]
    if num_samples < 2:
        raise RuntimeError("Not enough samples after warm-up to train seq2seq transformer.")

    # Split by explicit date if provided
    import pandas as _pd
    if cfg.split_date:
        try:
            cut_date = _pd.to_datetime(cfg.split_date)
        except Exception:
            cut_date = cfg.split_date
        split_idx = int(_pd.Index(target_dates).searchsorted(cut_date, side="right"))
        split_rule = f"explicit_date<={cut_date}"
    else:
        split_idx = int(num_samples * float(cfg.train_ratio))
        split_rule = f"ratio_samples_{float(cfg.train_ratio):.2f}"
    min_test = int(max(1, cfg.min_test_samples))
    if num_samples <= min_test + 1:
        split_idx = max(1, int(num_samples * 0.8))
    else:
        split_idx = max(1, min(split_idx, num_samples - min_test))

    if split_idx <= 0 or split_idx >= num_samples:
        raise RuntimeError(f"Split produced empty train/test. samples={num_samples}, split_idx={split_idx}, min_test={min_test}, split_rule={split_rule}")

    enc_train, enc_test = enc_obs[:split_idx], enc_obs[split_idx:]
    dec_known_train, dec_known_test = dec_known[:split_idx], dec_known[split_idx:]
    dec_start_train, dec_start_test = dec_start[:split_idx], dec_start[split_idx:]
    static_train, static_test = static_vecs[:split_idx], static_vecs[split_idx:]
    y_train_full, y_test_full = y_all[:split_idx], y_all[split_idx:]
    base_train, base_test = base_last[:split_idx], base_last[split_idx:]
    anchor_train, anchor_test = anchor_dates[:split_idx], anchor_dates[split_idx:]
    target_dates_test = target_dates[split_idx:]

    val_ratio = float(max(0.0, min(1.0, getattr(cfg, "val_ratio", 0.1))))
    val_count = int(round(len(enc_train) * val_ratio)) if val_ratio > 0 else 0
    if val_count >= len(enc_train):
        val_count = max(0, len(enc_train) - 1)
    if val_count > 0:
        enc_val = enc_train[-val_count:]
        dec_known_val = dec_known_train[-val_count:]
        dec_start_val = dec_start_train[-val_count:]
        static_val = static_train[-val_count:]
        y_val_full = y_train_full[-val_count:]
        base_val = base_train[-val_count:]
        anchor_val = anchor_train[-val_count:]

        enc_train = enc_train[:-val_count]
        dec_known_train = dec_known_train[:-val_count]
        dec_start_train = dec_start_train[:-val_count]
        static_train = static_train[:-val_count]
        y_train_full = y_train_full[:-val_count]
        base_train = base_train[:-val_count]
        anchor_train = anchor_train[:-val_count]
    else:
        enc_val = enc_test
        dec_known_val = dec_known_test
        dec_start_val = dec_start_test
        static_val = static_test
        y_val_full = y_test_full
        base_val = base_test
        anchor_val = []

    # Fit all input scalers on the final training partition only. Validation,
    # test and future inputs must never influence preprocessing statistics.
    obs_scaler = SequenceStandardScaler().fit(enc_train)
    enc_train = obs_scaler.transform(enc_train)
    enc_val = obs_scaler.transform(enc_val)
    enc_test = obs_scaler.transform(enc_test)

    known_scaler = None
    if dec_known_train.shape[-1] > 0:
        known_scaler = SequenceStandardScaler().fit(dec_known_train)
        dec_known_train = known_scaler.transform(dec_known_train)
        dec_known_val = known_scaler.transform(dec_known_val)
        dec_known_test = known_scaler.transform(dec_known_test)

    static_scaler = None
    if static_train.shape[-1] > 0:
        static_scaler = VectorStandardScaler().fit(static_train)
        static_train = static_scaler.transform(static_train)
        static_val = static_scaler.transform(static_val)
        static_test = static_scaler.transform(static_test)

    # Residualize targets if configured
    if cfg.target == "next_close" and cfg.residualize:
        y_train_full = y_train_full - base_train[:, None]
        y_test_full = y_test_full - base_test[:, None]
        dec_start_train = dec_start_train - base_train[:, None, None]
        dec_start_test = dec_start_test - base_test[:, None, None]
        if val_count > 0 or (isinstance(y_val_full, np.ndarray) and y_val_full.size):
            y_val_full = y_val_full - base_val[:, None]
            dec_start_val = dec_start_val - base_val[:, None, None]

    # Target scaling (learn mean/std on train residual targets for stability)
    y_mean = float(y_train_full.mean())
    y_std = float(y_train_full.std() if y_train_full.std() != 0 else 1.0)
    y_train_scaled = (y_train_full - y_mean) / y_std
    y_test_scaled = (y_test_full - y_mean) / y_std
    dec_start_train_scaled = dec_start_train / y_std
    dec_start_test_scaled = dec_start_test / y_std
    if val_count > 0 or (isinstance(y_val_full, np.ndarray) and y_val_full.size):
        y_val_scaled = (y_val_full - y_mean) / y_std
        dec_start_val_scaled = dec_start_val / y_std
    else:
        y_val_scaled = y_test_scaled
        dec_start_val_scaled = dec_start_test_scaled

    train_ds = Seq2SeqDataset(enc_train, dec_known_train, dec_start_train_scaled, static_train, y_train_scaled)
    test_ds = Seq2SeqDataset(enc_test, dec_known_test, dec_start_test_scaled, static_test, y_test_scaled)
    val_ds = Seq2SeqDataset(enc_val, dec_known_val, dec_start_val_scaled, static_val, y_val_scaled)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = Seq2SeqTransformer(
        obs_size=enc_obs.shape[-1],
        known_size=dec_known.shape[-1],
        static_size=static_vecs.shape[-1],
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        horizon=cfg.horizon,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(2, cfg.patience // 2), min_lr=1e-6
    ) if cfg.lr_scheduler else None
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            enc_b, dec_k, dec_s, stc, yb = batch
            enc_b = enc_b.to(cfg.device)
            dec_k = dec_k.to(cfg.device)
            dec_s = dec_s.to(cfg.device)
            stc = stc.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad()
            pred = model(enc_b, dec_k, dec_s, stc)
            loss = criterion(pred, yb)
            loss.backward()
            if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            opt.step()
            train_loss += loss.item() * enc_b.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(float(train_loss))
        train_losses.append(float(train_loss))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                enc_b, dec_k, dec_s, stc, yb = batch
                enc_b = enc_b.to(cfg.device)
                dec_k = dec_k.to(cfg.device)
                dec_s = dec_s.to(cfg.device)
                stc = stc.to(cfg.device)
                yb = yb.to(cfg.device)
                pred = model(enc_b, dec_k, dec_s, stc)
                loss = criterion(pred, yb)
                val_loss += loss.item() * enc_b.size(0)
        val_loss /= len(val_loader.dataset) if len(val_loader.dataset) > 0 else val_loss
        val_losses.append(float(val_loss))
        val_losses.append(float(val_loss))
        if scheduler is not None:
            scheduler.step(val_loss)
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on train and test sets for overfit checks
    model.eval()
    train_eval_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    val_eval_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_eval_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    def _eval_loader(
        loader: DataLoader, base_slice: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        preds_seq_local: list[np.ndarray] = []
        trues_seq_local: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                enc_b, dec_k, dec_s, stc, yb = batch
                pred = model(enc_b.to(cfg.device), dec_k.to(cfg.device), dec_s.to(cfg.device), stc.to(cfg.device))
                preds_seq_local.append(pred.cpu().numpy())
                trues_seq_local.append(yb.numpy())
        if preds_seq_local:
            y_pred_seq_local = np.concatenate(preds_seq_local, axis=0)
            y_true_seq_local = np.concatenate(trues_seq_local, axis=0)
        else:
            return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

        # Inverse target scaling
        y_pred_seq_local = (y_pred_seq_local * y_std) + y_mean
        y_true_seq_local = (y_true_seq_local * y_std) + y_mean

        # Metrics on last step to align with UI
        if cfg.target == "next_close" and cfg.residualize:
            y_pred_seq_local = y_pred_seq_local + base_slice[:, None]
            y_true_seq_local = y_true_seq_local + base_slice[:, None]
        y_pred_last_local = y_pred_seq_local[:, -1]
        y_true_last_local = y_true_seq_local[:, -1]
        return y_true_last_local, y_pred_last_local, y_true_seq_local, y_pred_seq_local

    y_true_train_last, y_pred_train_last, y_true_seq_train, y_pred_seq_train = _eval_loader(train_eval_loader, base_train)
    _y_true_val_last, _y_pred_val_last, _y_true_seq_val, _y_pred_seq_val = _eval_loader(val_eval_loader, base_val if (val_count > 0) else base_test)
    y_true_last, y_pred_last, y_true_seq, y_pred_seq = _eval_loader(test_eval_loader, base_test)

    if y_pred_last.size == 0:
        raise RuntimeError("No predictions produced; check dataset sizes.")

    metrics_train = compute_metrics(y_true_train_last, y_pred_train_last) if y_pred_train_last.size else {}
    metrics = compute_metrics(y_true_last, y_pred_last)
    eps_dir = 0.001
    try:
        med_price = float(np.nanmedian(y_true_last)) if np.isfinite(np.nanmedian(y_true_last)) else float("nan")
        if med_price and np.isfinite(med_price) and med_price != 0:
            metrics["rmse_pct_price"] = 100.0 * metrics["rmse"] / med_price
            if metrics_train:
                metrics_train["rmse_pct_price"] = 100.0 * metrics_train["rmse"] / med_price
    except Exception:
        pass
    try:
        if metrics_train and metrics_train.get("rmse"):
            metrics["rmse_ratio"] = float(metrics["rmse"]) / float(metrics_train["rmse"])
    except Exception:
        pass
    try:
        med_price = float(np.nanmedian(y_true_last)) if np.isfinite(np.nanmedian(y_true_last)) else float("nan")
        if med_price and np.isfinite(med_price) and med_price != 0:
            metrics["rmse_pct_price"] = 100.0 * metrics["rmse"] / med_price
            if metrics_train:
                metrics_train["rmse_pct_price"] = 100.0 * metrics_train["rmse"] / med_price
    except Exception:
        pass
    try:
        if metrics_train.get("rmse"):
            metrics["rmse_ratio"] = float(metrics["rmse"]) / float(metrics_train["rmse"])
    except Exception:
        pass
    try:
        dirm = compute_directional_metrics(
            y_true_last,
            y_pred_last,
            base=base_test if cfg.target == "next_close" else None,
            target_kind=cfg.target,
            eps=eps_dir,
        )
        metrics.update(dirm)
    except Exception:
        pass
    try:
        dirm_train = compute_directional_metrics(
            y_true_train_last,
            y_pred_train_last,
            base=base_train if cfg.target == "next_close" else None,
            target_kind=cfg.target,
            eps=eps_dir,
        )
        metrics_train.update(dirm_train)
    except Exception:
        pass
    try:
        tr_rmse = metrics_train.get("rmse")
        te_rmse = metrics.get("rmse")
        if tr_rmse is not None and te_rmse is not None:
            metrics["overfit_gap_rmse"] = float(te_rmse) - float(tr_rmse)
    except Exception:
        pass
    try:
        f1_gap = compute_f1_stability(metrics_train, metrics) if metrics_train else None
        if f1_gap is not None:
            metrics["f1_stability"] = f1_gap
    except Exception:
        pass
    for sup_key in ("support_up", "support_down", "support_total"):
        metrics.pop(sup_key, None)
        if metrics_train:
            metrics_train.pop(sup_key, None)

    f1_by_horizon = compute_f1_by_horizon(
        y_true_seq,
        y_pred_seq,
        base=base_test if cfg.target == "next_close" else None,
        steps=(1, 5, 14),
        target_kind=cfg.target,
        eps=eps_dir,
    )
    rolling_f1 = compute_rolling_f1(
        target_dates_test,
        y_true_last,
        y_pred_last,
        base=base_test if cfg.target == "next_close" else None,
        target_kind=cfg.target,
        eps=eps_dir,
    )

    diagnostics = build_diagnostics(
        y_true_last=y_true_last,
        y_pred_last=y_pred_last,
        model_rmse=metrics.get("rmse"),
        base_values=base_test,
        anchor_dates=anchor_test,
        df=df,
        train_losses=train_losses,
        val_losses=val_losses,
        target_kind=cfg.target,
    )

    # Split meta
    def _to_date_str(val):
        try:
            return str(getattr(val, "date", lambda: val)())
        except Exception:
            return str(val)
    test_dates_str = [_to_date_str(d) for d in target_dates_test]
    val_dates = [_to_date_str(d) for d in anchor_val] if anchor_val is not None else []
    split_meta = {
        "train_count": int(len(anchor_train)),
        "val_count": int(len(val_dates)),
        "test_count": int(len(anchor_test)),
        "train_start": _to_date_str(anchor_train[0]) if anchor_train else None,
        "train_end": _to_date_str(anchor_train[-1]) if anchor_train else None,
        "val_start": val_dates[0] if val_dates else None,
        "val_end": val_dates[-1] if val_dates else None,
        "test_start": _to_date_str(anchor_test[0]) if anchor_test else None,
        "test_end": _to_date_str(anchor_test[-1]) if anchor_test else None,
        "cutoff": _to_date_str(cfg.split_date) if cfg.split_date else None,
        "split_rule": split_rule,
        "requested_split_mode": cfg.split_mode,
        "requested_train_ratio": float(cfg.train_ratio),
        "effective_train_ratio": float(split_idx / num_samples),
        "separation_ok": bool(len(anchor_train) == 0 or len(anchor_test) == 0 or anchor_train[-1] <= anchor_test[0]),
        "data_quality": data_quality_report,
    }

    # Lightweight future forecast using the latest available window
    future_point = None
    actual_target = None
    future_point_error = None
    try:
        model.eval()
        enc_future = obs_scaler.transform(enc_obs[-1:])
        dec_known_future = dec_known[-1:] if dec_known.shape[1] else np.zeros((1, cfg.horizon, 0), dtype=np.float32)
        if known_scaler is not None:
            dec_known_future = known_scaler.transform(dec_known_future)
        dec_start_future = np.zeros_like(dec_start[-1:])
        static_future = static_vecs[-1:] if static_vecs.shape[1] else np.zeros((1, 0), dtype=np.float32)
        if static_scaler is not None:
            static_future = static_scaler.transform(static_future)
        with torch.no_grad():
            pred_future = model(
                torch.from_numpy(enc_future).to(cfg.device),
                torch.from_numpy(dec_known_future).to(cfg.device),
                torch.from_numpy(dec_start_future).to(cfg.device),
                torch.from_numpy(static_future).to(cfg.device),
            )
        y_future_scaled = float(pred_future.detach().cpu().numpy().reshape(1, -1)[0, -1])
        y_future_eff = (y_future_scaled * y_std) + y_mean
        if cfg.target == "next_close" and cfg.residualize:
            y_future_eff = y_future_eff + float(base_last[-1])

        last_date = df.index[-1]
        future_date = last_date + _pd.tseries.offsets.BDay(cfg.horizon)
        try:
            df_after = get_ohlcv(
                ticker_yt=cfg.ticker,
                start=str(getattr(last_date, "date", lambda: last_date)()),
                end=None,
                auto_adjust=True,
            )
            df_after = standardize_ohlcv_columns(df_after)
            next_trading = df_after.index[df_after.index > last_date]
            if len(next_trading) >= cfg.horizon:
                future_date = next_trading[cfg.horizon - 1]
            elif len(next_trading) > 0:
                future_date = next_trading[-1] + _pd.tseries.offsets.BDay(cfg.horizon - len(next_trading))
        except Exception:
            pass

        future_point = {
            "date": str(getattr(future_date, "date", lambda: future_date)()),
            "value": float(y_future_eff),
        }

        try:
            tgt_str = str(getattr(future_date, "date", lambda: future_date)())
            df_target = get_ohlcv(ticker_yt=cfg.ticker, start=cfg.start, end=tgt_str, auto_adjust=True)
            df_target = standardize_ohlcv_columns(df_target)
            if future_date in df_target.index:
                actual_target = {"date": tgt_str, "value": float(df_target["Close"].loc[future_date])}
        except Exception:
            actual_target = None
    except Exception as ex:
        future_point_error = f"{type(ex).__name__}: {ex}"

    return {
        "run_dir": None,
        "metrics": metrics,
        "metrics_test": metrics,
        "metrics_train": metrics_train if metrics_train else None,
        "f1_by_horizon": f1_by_horizon,
        "rolling_f1": rolling_f1,
        "feature_cols": observed_cols + known_future_cols + static_cols,
        "feature_cols_tech": observed_cols,
        "feature_cols_fund": static_cols,
        "X_shape": list(enc_obs.shape),
        "y_true": y_true_last,
        "y_pred": y_pred_last,
        "test_dates": test_dates_str,
        "target": cfg.target,
        "residualized": False,
        "residual_base": "Close",
        "future_point": future_point,
        "actual_target": actual_target,
        "future_point_error": future_point_error,
        "effective_window": int(cfg.window_size),
        "used_fundamentals": bool(static_cols),
        "split_info": split_meta,
        "diagnostics": diagnostics,
        "model_type": "transformer",
        "model_name": "Transformer Seq2Seq",
    }


def run_training_tft_full(cfg: TrainConfig) -> dict:
    """Full TFT-style training with observed/known/static splits and quantile outputs."""
    seed_everything(42)
    future_point_error = None
    df = get_ohlcv(ticker_yt=cfg.ticker, start=cfg.start, end=cfg.end, auto_adjust=True)
    data_quality_report = dict(df.attrs.get("data_quality", {}))
    df = standardize_ohlcv_columns(df)
    events_df = get_event_countdowns(cfg.ticker, df.index)
    feat_df = compute_features(df, target=cfg.target, horizon=cfg.horizon, include_extra=True, events_df=events_df)

    known_future_candidates = {"year", "month", "day_of_week", "is_month_end", "is_quarter_end", "is_holiday_eve"}
    known_future_cols = [c for c in feat_df.columns if c in known_future_candidates or (isinstance(c, str) and c.startswith("f_days_"))]
    static_cols = [c for c in feat_df.columns if isinstance(c, str) and c.startswith("f_")]
    observed_cols = [c for c in feat_df.columns if c not in known_future_cols and c not in static_cols and c != "target"]

    if observed_cols:
        feat_df[observed_cols] = feat_df[observed_cols].ffill()
    clean_subset = observed_cols + ["target"]
    feat_df = feat_df.dropna(subset=clean_subset)
    if len(feat_df) == 0:
        raise RuntimeError("No rows remaining after cleaning features; try an earlier start, smaller window, or disable fundamentals.")

    if known_future_cols:
        feat_df[known_future_cols] = feat_df[known_future_cols].fillna(0)
    if static_cols:
        feat_df[static_cols] = feat_df[static_cols].ffill().fillna(0)

    raw_close_series = df["Close"]
    enc_obs, dec_known, dec_start, static_vecs, y_all, base_last, anchor_dates, target_dates = build_seq2seq_tensors(
        feat_df,
        raw_close=raw_close_series,
        observed_cols=observed_cols,
        known_cols=known_future_cols,
        static_cols=static_cols,
        target_col="target",
        horizon=cfg.horizon,
        window_size=cfg.window_size,
    )
    num_samples = enc_obs.shape[0]
    if num_samples < 2:
        raise RuntimeError("Not enough samples after warm-up to train TFT.")

    import pandas as _pd
    if cfg.split_date:
        try:
            cut_date = _pd.to_datetime(cfg.split_date)
        except Exception:
            cut_date = cfg.split_date
        split_idx = int(_pd.Index(target_dates).searchsorted(cut_date, side="right"))
        split_rule = f"explicit_date<={cut_date}"
    else:
        split_idx = int(num_samples * float(cfg.train_ratio))
        split_rule = f"ratio_samples_{float(cfg.train_ratio):.2f}"
    min_test = int(max(1, cfg.min_test_samples))
    if num_samples <= min_test + 1:
        split_idx = max(1, int(num_samples * 0.8))
    else:
        split_idx = max(1, min(split_idx, num_samples - min_test))

    if split_idx <= 0 or split_idx >= num_samples:
        raise RuntimeError(f"Split produced empty train/test. samples={num_samples}, split_idx={split_idx}, min_test={min_test}, split_rule={split_rule}")

    enc_train, enc_test = enc_obs[:split_idx], enc_obs[split_idx:]
    dec_known_train, dec_known_test = dec_known[:split_idx], dec_known[split_idx:]
    dec_start_train, dec_start_test = dec_start[:split_idx], dec_start[split_idx:]
    static_train, static_test = static_vecs[:split_idx], static_vecs[split_idx:]
    y_train_full, y_test_full = y_all[:split_idx], y_all[split_idx:]
    base_train, base_test = base_last[:split_idx], base_last[split_idx:]
    anchor_train, anchor_test = anchor_dates[:split_idx], anchor_dates[split_idx:]
    target_dates_test = target_dates[split_idx:]

    val_ratio = float(max(0.0, min(1.0, getattr(cfg, "val_ratio", 0.1))))
    val_count = int(round(len(enc_train) * val_ratio)) if val_ratio > 0 else 0
    if val_count >= len(enc_train):
        val_count = max(0, len(enc_train) - 1)
    if val_count > 0:
        enc_val = enc_train[-val_count:]
        dec_known_val = dec_known_train[-val_count:]
        dec_start_val = dec_start_train[-val_count:]
        static_val = static_train[-val_count:]
        y_val_full = y_train_full[-val_count:]
        base_val = base_train[-val_count:]
        anchor_val = anchor_train[-val_count:]

        enc_train = enc_train[:-val_count]
        dec_known_train = dec_known_train[:-val_count]
        dec_start_train = dec_start_train[:-val_count]
        static_train = static_train[:-val_count]
        y_train_full = y_train_full[:-val_count]
        base_train = base_train[:-val_count]
        anchor_train = anchor_train[:-val_count]
    else:
        enc_val = enc_test
        dec_known_val = dec_known_test
        dec_start_val = dec_start_test
        static_val = static_test
        y_val_full = y_test_full
        base_val = base_test
        anchor_val = []

    # Fit input preprocessing on training data only to avoid look-ahead leakage.
    obs_scaler = SequenceStandardScaler().fit(enc_train)
    enc_train = obs_scaler.transform(enc_train)
    enc_val = obs_scaler.transform(enc_val)
    enc_test = obs_scaler.transform(enc_test)

    known_scaler = None
    if dec_known_train.shape[-1] > 0:
        known_scaler = SequenceStandardScaler().fit(dec_known_train)
        dec_known_train = known_scaler.transform(dec_known_train)
        dec_known_val = known_scaler.transform(dec_known_val)
        dec_known_test = known_scaler.transform(dec_known_test)

    static_scaler = None
    if static_train.shape[-1] > 0:
        static_scaler = VectorStandardScaler().fit(static_train)
        static_train = static_scaler.transform(static_train)
        static_val = static_scaler.transform(static_val)
        static_test = static_scaler.transform(static_test)

    # Residualize targets if configured
    if cfg.target == "next_close" and cfg.residualize:
        y_train_full = y_train_full - base_train[:, None]
        y_test_full = y_test_full - base_test[:, None]
        dec_start_train = dec_start_train - base_train[:, None, None]
        dec_start_test = dec_start_test - base_test[:, None, None]
        if val_count > 0 or (isinstance(y_val_full, np.ndarray) and y_val_full.size):
            y_val_full = y_val_full - base_val[:, None]
            dec_start_val = dec_start_val - base_val[:, None, None]

    # Target scaling for TFT
    y_mean = float(y_train_full.mean())
    y_std = float(y_train_full.std() if y_train_full.std() != 0 else 1.0)
    y_train_scaled = (y_train_full - y_mean) / y_std
    y_test_scaled = (y_test_full - y_mean) / y_std
    dec_start_train_scaled = dec_start_train / y_std
    dec_start_test_scaled = dec_start_test / y_std
    if val_count > 0 or (isinstance(y_val_full, np.ndarray) and y_val_full.size):
        y_val_scaled = (y_val_full - y_mean) / y_std
        dec_start_val_scaled = dec_start_val / y_std
    else:
        y_val_scaled = y_test_scaled
        dec_start_val_scaled = dec_start_test_scaled

    train_ds = Seq2SeqDataset(enc_train, dec_known_train, dec_start_train_scaled, static_train, y_train_scaled)
    test_ds = Seq2SeqDataset(enc_test, dec_known_test, dec_start_test_scaled, static_test, y_test_scaled)
    val_ds = Seq2SeqDataset(enc_val, dec_known_val, dec_start_val_scaled, static_val, y_val_scaled)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = FullTFT(
        obs_size=enc_obs.shape[-1],
        known_size=dec_known.shape[-1],
        static_size=static_vecs.shape[-1],
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        horizon=cfg.horizon,
        quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)),
    ).to(cfg.device)

    def quantile_loss(preds: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
        # preds: (B, H, Q) target: (B, H)
        qs = torch.tensor(quantiles, device=preds.device).view(1, 1, -1)
        t = target.unsqueeze(-1)
        errors = t - preds
        loss = torch.maximum(qs * errors, (qs - 1.0) * errors)
        return loss.mean()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(2, cfg.patience // 2), min_lr=1e-6
    ) if cfg.lr_scheduler else None

    best_val = float("inf")
    best_state = None
    no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            enc_b, dec_k, dec_s, stc, yb = batch
            enc_b = enc_b.to(cfg.device)
            dec_k = dec_k.to(cfg.device)
            dec_s = dec_s.to(cfg.device)
            stc = stc.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad()
            pred = model(enc_b, dec_k, dec_s, stc)  # (B, H, Q)
            loss = quantile_loss(pred, yb, model.quantiles)
            loss.backward()
            if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            opt.step()
            train_loss += loss.item() * enc_b.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(float(train_loss))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                enc_b, dec_k, dec_s, stc, yb = batch
                enc_b = enc_b.to(cfg.device)
                dec_k = dec_k.to(cfg.device)
                dec_s = dec_s.to(cfg.device)
                stc = stc.to(cfg.device)
                yb = yb.to(cfg.device)
                pred = model(enc_b, dec_k, dec_s, stc)
                loss = quantile_loss(pred, yb, model.quantiles)
                val_loss += loss.item() * enc_b.size(0)
        val_loss /= len(val_loader.dataset) if len(val_loader.dataset) > 0 else val_loss
        val_losses.append(float(val_loss))
        if scheduler is not None:
            scheduler.step(val_loss)
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on train and test sets: use median quantile (closest to 0.5)
    model.eval()
    train_eval_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    val_eval_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_eval_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    qs = getattr(cfg, "quantiles", (0.1, 0.5, 0.9))
    med_idx = int(min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5)))

    def _collect_preds(loader: DataLoader, base_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds_seq_local: list[np.ndarray] = []
        trues_seq_local: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                enc_b, dec_k, dec_s, stc, yb = batch
                pred = model(enc_b.to(cfg.device), dec_k.to(cfg.device), dec_s.to(cfg.device), stc.to(cfg.device))  # (B,H,Q)
                preds_seq_local.append(pred.cpu().numpy())
                trues_seq_local.append(yb.numpy())
        if preds_seq_local:
            y_pred_q_local = np.concatenate(preds_seq_local, axis=0)
            y_true_seq_local = np.concatenate(trues_seq_local, axis=0)
        else:
            return np.zeros((0, cfg.horizon)), np.zeros((0, cfg.horizon))

        y_pred_med_local = y_pred_q_local[:, :, med_idx]
        # Inverse target scaling
        y_pred_med_local = (y_pred_med_local * y_std) + y_mean
        y_true_seq_local = (y_true_seq_local * y_std) + y_mean

        # Undo residualization for reporting if enabled
        if cfg.target == "next_close" and cfg.residualize:
            y_pred_med_local = y_pred_med_local + base_slice[:, None]
            y_true_seq_local = y_true_seq_local + base_slice[:, None]
        return y_true_seq_local, y_pred_med_local

    y_true_seq_train, y_pred_seq_train = _collect_preds(train_eval_loader, base_train)
    _y_true_seq_val, _y_pred_seq_val = _collect_preds(val_eval_loader, base_val if (val_count > 0) else base_test)
    y_true_seq, y_pred_med = _collect_preds(test_eval_loader, base_test)

    if y_pred_med.size == 0:
        raise RuntimeError("No predictions produced; check dataset sizes.")

    # Metrics per step and last step (test set)
    metrics_steps = {}
    for step in range(cfg.horizon):
        m = compute_metrics(y_true_seq[:, step], y_pred_med[:, step])
        metrics_steps[f"step_{step+1}"] = m
    y_pred_last = y_pred_med[:, -1]
    y_true_last = y_true_seq[:, -1]
    metrics_train = {}
    eps_dir = 0.001
    if y_pred_seq_train.size:
        y_pred_train_last = y_pred_seq_train[:, -1]
        y_true_train_last = y_true_seq_train[:, -1]
        metrics_train = compute_metrics(y_true_train_last, y_pred_train_last)
        try:
            dirm_train = compute_directional_metrics(
                y_true_train_last,
                y_pred_train_last,
                base=base_train if cfg.target == "next_close" else None,
                target_kind=cfg.target,
                eps=eps_dir,
            )
            metrics_train.update(dirm_train)
        except Exception:
            pass
    metrics = compute_metrics(y_true_last, y_pred_last)
    try:
        med_price = float(np.nanmedian(y_true_last)) if np.isfinite(np.nanmedian(y_true_last)) else float("nan")
        if med_price and np.isfinite(med_price) and med_price != 0:
            metrics["rmse_pct_price"] = 100.0 * metrics["rmse"] / med_price
            if metrics_train:
                metrics_train["rmse_pct_price"] = 100.0 * metrics_train["rmse"] / med_price
    except Exception:
        pass
    try:
        if metrics_train and metrics_train.get("rmse"):
            metrics["rmse_ratio"] = float(metrics["rmse"]) / float(metrics_train["rmse"])
    except Exception:
        pass
    try:
        dirm = compute_directional_metrics(
            y_true_last,
            y_pred_last,
            base=base_test if cfg.target == "next_close" else None,
            target_kind=cfg.target,
            eps=eps_dir,
        )
        metrics.update(dirm)
    except Exception:
        pass
    try:
        tr_rmse = metrics_train.get("rmse") if metrics_train else None
        te_rmse = metrics.get("rmse")
        if tr_rmse is not None and te_rmse is not None:
            metrics["overfit_gap_rmse"] = float(te_rmse) - float(tr_rmse)
    except Exception:
        pass
    try:
        f1_gap = compute_f1_stability(metrics_train, metrics) if metrics_train else None
        if f1_gap is not None:
            metrics["f1_stability"] = f1_gap
    except Exception:
        pass
    for sup_key in ("support_up", "support_down", "support_total"):
        metrics.pop(sup_key, None)
        if metrics_train:
            metrics_train.pop(sup_key, None)

    # Direction slices by horizon (e.g., 1d/5d/14d) and rolling monthly F1
    f1_by_horizon = compute_f1_by_horizon(
        y_true_seq,
        y_pred_med,
        base=base_test if cfg.target == "next_close" else None,
        steps=(1, 5, 14),
        target_kind=cfg.target,
        eps=eps_dir,
    )
    rolling_f1 = compute_rolling_f1(
        target_dates_test,
        y_true_last,
        y_pred_last,
        base=base_test if cfg.target == "next_close" else None,
        target_kind=cfg.target,
        eps=eps_dir,
    )

    diagnostics = build_diagnostics(
        y_true_last=y_true_last,
        y_pred_last=y_pred_last,
        model_rmse=metrics.get("rmse"),
        base_values=base_test,
        anchor_dates=anchor_test,
        df=df,
        train_losses=train_losses,
        val_losses=val_losses,
        target_kind=cfg.target,
    )

    future_point = None
    actual_target = None
    try:
        model.eval()
        enc_future = obs_scaler.transform(enc_obs[-1:])
        dec_known_future = dec_known[-1:] if dec_known.shape[1] else np.zeros((1, cfg.horizon, 0), dtype=np.float32)
        if known_scaler is not None:
            dec_known_future = known_scaler.transform(dec_known_future)
        dec_start_future = np.zeros_like(dec_start[-1:])
        static_future = static_vecs[-1:] if static_vecs.shape[1] else np.zeros((1, 0), dtype=np.float32)
        if static_scaler is not None:
            static_future = static_scaler.transform(static_future)
        with torch.no_grad():
            pred_future_full = model(
                torch.from_numpy(enc_future).to(cfg.device),
                torch.from_numpy(dec_known_future).to(cfg.device),
                torch.from_numpy(dec_start_future).to(cfg.device),
                torch.from_numpy(static_future).to(cfg.device),
            )  # (1, H, Q)
        pred_future_med = pred_future_full[..., med_idx].detach().cpu().numpy()
        y_future_scaled = float(pred_future_med.reshape(1, -1)[0, -1])
        y_future_eff = (y_future_scaled * y_std) + y_mean
        if cfg.target == "next_close" and cfg.residualize:
            y_future_eff = y_future_eff + float(base_last[-1])

        last_date = df.index[-1]
        future_date = last_date + _pd.tseries.offsets.BDay(cfg.horizon)
        try:
            df_after = get_ohlcv(
                ticker_yt=cfg.ticker,
                start=str(getattr(last_date, "date", lambda: last_date)()),
                end=None,
                auto_adjust=True,
            )
            df_after = standardize_ohlcv_columns(df_after)
            next_trading = df_after.index[df_after.index > last_date]
            if len(next_trading) >= cfg.horizon:
                future_date = next_trading[cfg.horizon - 1]
            elif len(next_trading) > 0:
                future_date = next_trading[-1] + _pd.tseries.offsets.BDay(cfg.horizon - len(next_trading))
        except Exception:
            pass

        future_point = {
            "date": str(getattr(future_date, "date", lambda: future_date)()),
            "value": float(y_future_eff),
        }

        try:
            tgt_str = str(getattr(future_date, "date", lambda: future_date)())
            df_target = get_ohlcv(ticker_yt=cfg.ticker, start=cfg.start, end=tgt_str, auto_adjust=True)
            df_target = standardize_ohlcv_columns(df_target)
            if future_date in df_target.index:
                actual_target = {"date": tgt_str, "value": float(df_target["Close"].loc[future_date])}
        except Exception:
            actual_target = None
    except Exception as ex:
        future_point_error = f"{type(ex).__name__}: {ex}"

    def _to_date_str(val):
        try:
            return str(getattr(val, "date", lambda: val)())
        except Exception:
            return str(val)
    test_dates_str = [_to_date_str(d) for d in target_dates_test]
    val_dates = [_to_date_str(d) for d in anchor_val] if anchor_val is not None else []
    split_meta = {
        "train_count": int(len(anchor_train)),
        "val_count": int(len(val_dates)),
        "test_count": int(len(anchor_test)),
        "train_start": _to_date_str(anchor_train[0]) if anchor_train else None,
        "train_end": _to_date_str(anchor_train[-1]) if anchor_train else None,
        "val_start": val_dates[0] if val_dates else None,
        "val_end": val_dates[-1] if val_dates else None,
        "test_start": _to_date_str(anchor_test[0]) if anchor_test else None,
        "test_end": _to_date_str(anchor_test[-1]) if anchor_test else None,
        "cutoff": _to_date_str(cfg.split_date) if cfg.split_date else None,
        "split_rule": split_rule,
        "requested_split_mode": cfg.split_mode,
        "requested_train_ratio": float(cfg.train_ratio),
        "effective_train_ratio": float(split_idx / num_samples),
        "separation_ok": bool(len(anchor_train) == 0 or len(anchor_test) == 0 or anchor_train[-1] <= anchor_test[0]),
        "data_quality": data_quality_report,
    }

    return {
        "run_dir": None,
        "metrics": metrics,
        "metrics_steps": metrics_steps,
        "metrics_test": metrics,
        "metrics_train": metrics_train if metrics_train else None,
        "f1_by_horizon": f1_by_horizon,
        "rolling_f1": rolling_f1,
        "feature_cols": observed_cols + known_future_cols + static_cols,
        "feature_cols_tech": observed_cols,
        "feature_cols_fund": static_cols,
        "X_shape": list(enc_obs.shape),
        "y_true": y_true_last,
        "y_pred": y_pred_last,
        "test_dates": test_dates_str,
        "target": cfg.target,
        "residualized": False,
        "residual_base": "Close",
        "future_point": future_point,
        "future_point_error": future_point_error,
        "actual_target": actual_target,
        "effective_window": int(cfg.window_size),
        "used_fundamentals": bool(static_cols),
        "split_info": split_meta,
        "diagnostics": diagnostics,
        "model_type": "tft",
        "model_name": "TFT Full",
    }
def build_seq2seq_tensors(
    feat_df: pd.DataFrame,
    raw_close: pd.Series,
    observed_cols: List[str],
    known_cols: List[str],
    static_cols: List[str],
    target_col: str,
    horizon: int,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List, List]:
    """Build encoder/decoder tensors for seq2seq models."""
    values_obs = feat_df[observed_cols].values.astype(np.float32)
    values_known = feat_df[known_cols].values.astype(np.float32) if known_cols else np.zeros((len(feat_df), 0), dtype=np.float32)
    values_static = feat_df[static_cols].values.astype(np.float32) if static_cols else np.zeros((len(feat_df), 0), dtype=np.float32)
    targets = feat_df[target_col].values.astype(np.float32)

    samples_enc, samples_known, samples_decstart, samples_static, samples_y, samples_base, anchor_dates, target_dates = [], [], [], [], [], [], [], []
    total = len(feat_df)
    H = int(max(1, horizon))
    W = int(max(1, window_size))
    idx = feat_df.index
    raw_close_aligned = raw_close.reindex(feat_df.index)

    for end_idx in range(W, total - H + 1):
        # past window ends at end_idx-1, forecast from end_idx to end_idx+H-1
        enc_obs = values_obs[end_idx - W:end_idx]
        dec_known = values_known[end_idx:end_idx + H] if known_cols else np.zeros((H, 0), dtype=np.float32)
        static_vec = values_static[end_idx - 1] if static_cols else np.zeros((0,), dtype=np.float32)
        y_seq = targets[end_idx:end_idx + H]
        # decoder start tokens: prepend 0, drop last
        dec_start = np.zeros_like(y_seq)
        dec_start[1:] = y_seq[:-1]
        base_last = float(raw_close_aligned.iloc[end_idx - 1])

        samples_enc.append(enc_obs)
        samples_known.append(dec_known)
        samples_decstart.append(dec_start[:, None])  # (H,1)
        samples_static.append(static_vec)
        samples_y.append(y_seq)
        samples_base.append(base_last)
        anchor_dates.append(idx[end_idx - 1])
        target_dates.append(idx[end_idx + H - 1])

    return (
        np.stack(samples_enc, axis=0),
        np.stack(samples_known, axis=0),
        np.stack(samples_decstart, axis=0),
        np.stack(samples_static, axis=0),
        np.stack(samples_y, axis=0),
        np.array(samples_base, dtype=np.float32),
        anchor_dates,
        target_dates,
    )


def run_training(cfg: TrainConfig) -> dict:
    """Run end-to-end training and return results for UI usage."""
    # Route to dedicated implementations
    mtype = getattr(cfg, "model_type", "transformer").lower()
    if mtype == "transformer":
        return run_training_transformer_seq2seq(cfg)
    if mtype == "tft":
        return run_training_tft_full(cfg)

    seed_everything(42)

    df = get_ohlcv(ticker_yt=cfg.ticker, start=cfg.start, end=cfg.end, auto_adjust=True)
    df = standardize_ohlcv_columns(df)

    # Known-future event countdown features (earnings, ex-div, futures expiry)
    events_df = get_event_countdowns(cfg.ticker, df.index)

    # Compute technical features first and keep a base copy (used for fallback)
    feat_df_base = compute_features(
        df, target=cfg.target, horizon=cfg.horizon, include_extra=True, events_df=events_df
    )
    feat_df = feat_df_base.copy()

    # Join Yahoo Finance fundamentals aligned to trading index (optional)
    good_f_cols = []
    if cfg.use_fundamentals:
        try:
            fund_df = get_fundamentals_timeseries(cfg.ticker, align_index=df.index)
            import pandas as _pd
            if isinstance(fund_df, _pd.DataFrame) and not fund_df.empty:
                # Avoid column name collisions (e.g., event-countdown features
                # that are already present in feat_df).
                overlap = [c for c in fund_df.columns if c in feat_df.columns]
                if overlap:
                    fund_df = fund_df.drop(columns=overlap)
                feat_df = feat_df.join(fund_df, how="left")
                # Consider only fundamental columns with at least minimal coverage
                fcols = [c for c in feat_df.columns if isinstance(c, str) and c.startswith("f_")]
                good_f_cols = [c for c in fcols if feat_df[c].notna().sum() >= 4]
                drop_f_cols = [c for c in fcols if c not in good_f_cols]
                if drop_f_cols:
                    feat_df = feat_df.drop(columns=drop_f_cols)
        except Exception:
            pass

    # Model-aware feature selection.
    # For TFT runs, use a curated feature subset that emphasises
    # known-future calendar features + core technical indicators
    # + static fundamentals. For transformer runs, fall back to
    # the broader default feature set.
    if getattr(cfg, "model_type", "transformer").lower() == "tft":
        feature_cols = get_tft_feature_columns(feat_df)
    else:
        feature_cols = get_default_feature_columns(feat_df)

    # Split into technical vs fundamentals
    fund_cols = [c for c in feature_cols if isinstance(c, str) and c.startswith('f_')]
    ta_cols = [c for c in feature_cols if c not in fund_cols]

    # Drop rows only where TA/target are missing (fundamentals are ffilled later)
    feat_df = feat_df.dropna(subset=ta_cols + ["target"])  # keep earlier rows trimmed safely

    # Decide effective window with a minimum sample requirement
    eff_window_req = int(cfg.window_size)
    rows_avail = len(feat_df)
    eff_window = eff_window_req if rows_avail > eff_window_req else max(5, rows_avail - 1)
    num_samples_est = max(0, rows_avail - eff_window + 1)
    # Debug guard: ensure we never call sliding-window builder with too-long windows.
    # This print goes to stdout (Flask logs / CLI) and is harmless for the UI.
    print(f"[run_training] rows_avail={rows_avail} eff_window_req={eff_window_req} eff_window={eff_window} num_samples_est={num_samples_est}")

    used_fundamentals = any(isinstance(c, str) and c.startswith('f_') for c in feature_cols)
    # If too few samples after adding fundamentals, fallback to technical-only (unless required)
    if num_samples_est < int(cfg.min_samples_with_fundamentals) and used_fundamentals and not cfg.require_fundamentals:
        feat_df = feat_df_base.copy()
        feature_cols = get_default_feature_columns(feat_df)
        feat_df = feat_df.dropna(subset=feature_cols + ["target"])  # should already be clean
        rows_avail = len(feat_df)
        eff_window = eff_window_req if rows_avail > eff_window_req else max(5, rows_avail - 1)
        num_samples_est = max(0, rows_avail - eff_window + 1)
        used_fundamentals = False

    # Choose base column for residualization
    base_col = "Close" if cfg.residual_base.lower() == "close" else ("ma20" if "ma20" in feat_df.columns else "Close")
    # Build TA windows and base labels (single-step target from compute_features)
    X_ta, y_single, base = make_sliding_windows_with_base(
        feat_df, ta_cols, target_col="target", base_col=base_col, window_size=eff_window
    )
    # Build FUND anchors aligned to each sample's end index (ffill then take last row of each window)
    if fund_cols:
        fdf = feat_df[fund_cols].ffill()
        fvals = fdf.values.astype(np.float32)
        end_idxs = np.arange(eff_window - 1, eff_window - 1 + X_ta.shape[0])
        X_fund = fvals[end_idxs]
        X_fund = np.nan_to_num(X_fund, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    else:
        X_fund = np.zeros((X_ta.shape[0], 0), dtype=np.float32)

    # Consistent time-ordered split for X, y, and base (avoid leakage)
    # Build per-sample target dates by advancing positions on the RAW df index
    # so that the last label for h=1 maps to the next actual trading day
    # (even when the last day in feat_df lacks a target row).
    import pandas as _pd
    num_samples = X_ta.shape[0]
    anchor_dates = feat_df.index[eff_window - 1 : eff_window - 1 + num_samples]
    # Map anchor dates to positions in the raw df index and shift by horizon
    anchor_pos = _pd.Index(df.index).get_indexer(anchor_dates)
    horizon = int(max(1, cfg.horizon))
    target_pos = anchor_pos + horizon
    # No clipping: compute_features drops last h rows, so these are valid
    target_dates = _pd.Index(df.index)[target_pos]

    # Multi-step labels for TFT: future Close prices for each horizon step.
    model_type = getattr(cfg, "model_type", "transformer").lower()
    if model_type == "tft":
        closes = df["Close"].values.astype(np.float32)
        future_offsets = np.arange(1, horizon + 1, dtype=int)
        future_pos = anchor_pos[:, None] + future_offsets[None, :]
        if np.any(future_pos > closes.shape[0] - 1):
            raise RuntimeError("Not enough history to build TFT multi-step targets; try a smaller horizon.")
        y_all = closes[future_pos]
    else:
        y_all = y_single

    # Determine split index based on explicit date or configured mode
    cut_date = None
    split_rule = "ratio_samples"
    if cfg.split_date:
        try:
            cut_date = _pd.to_datetime(cfg.split_date)
        except Exception:
            # Fallback: treat as string for searchsorted
            cut_date = cfg.split_date
        # Train on target_dates <= cut_date; test on > cut_date
        split_idx = int(_pd.Index(target_dates).searchsorted(cut_date, side="right"))
        split_rule = f"explicit_date<={cut_date}"
    elif cfg.split_mode == "last_n_days" and cfg.test_days and cfg.test_days > 0:
        cut_date = target_dates.max() - _pd.tseries.offsets.BDay(int(cfg.test_days))
        split_idx = int(_pd.Index(target_dates).searchsorted(cut_date, side="right"))
        split_rule = f"last_{int(cfg.test_days)}_bdays"
    elif cfg.split_mode == "ratio_time":
        start = target_dates.min()
        end = target_dates.max()
        frac = float(cfg.train_ratio)
        if frac <= 0 or frac >= 1:
            frac = 0.8
        cut_date = start + (end - start) * frac
        split_idx = int(_pd.Index(target_dates).searchsorted(cut_date, side="right"))
        split_rule = f"ratio_time_{frac:.2f}"
    else:  # ratio_samples
        split_idx = int(num_samples * float(cfg.train_ratio))
        split_rule = f"ratio_samples_{float(cfg.train_ratio):.2f}"
    # Ensure we have a meaningful test set size
    min_test = int(max(1, cfg.min_test_samples))
    if num_samples <= min_test + 1:
        # fall back to 80/20 when dataset too small for requested minimum
        split_idx = max(1, int(num_samples * 0.8))
    else:
        split_idx = max(1, min(split_idx, num_samples - min_test))

    # Capture split metadata for UI transparency
    train_dates = target_dates[:split_idx]
    test_dates_idx = target_dates[split_idx:]
    def _to_date_str(val):
        try:
            return str(getattr(val, "date", lambda: val)())
        except Exception:
            return str(val)
    split_meta = {
        "train_count": int(len(train_dates)),
        "test_count": int(len(test_dates_idx)),
        "train_start": _to_date_str(train_dates.min()) if len(train_dates) else None,
        "train_end": _to_date_str(train_dates.max()) if len(train_dates) else None,
        "test_start": _to_date_str(test_dates_idx.min()) if len(test_dates_idx) else None,
        "test_end": _to_date_str(test_dates_idx.max()) if len(test_dates_idx) else None,
        "cutoff": _to_date_str(cut_date) if cut_date is not None else None,
        "split_rule": split_rule,
        "separation_ok": bool(len(train_dates) == 0 or len(test_dates_idx) == 0 or train_dates.max() <= test_dates_idx.min()),
    }

    # Apply split
    X_train, y_train = X_ta[:split_idx], y_all[:split_idx]
    X_test, y_test = X_ta[split_idx:], y_all[split_idx:]
    F_train, F_test = X_fund[:split_idx], X_fund[split_idx:]
    base_train, base_test = base[:split_idx], base[split_idx:]

    x_scaler = SequenceStandardScaler().fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    f_scaler = None
    if F_train.shape[1] > 0:
        f_scaler = VectorStandardScaler().fit(F_train)
        F_train = f_scaler.transform(F_train)
        F_test = f_scaler.transform(F_test)

    # Optionally residualize for price targets
    if cfg.target == "next_close" and cfg.residualize:
        if y_train.ndim == 2:
            y_train_eff = y_train - base_train[:, None]
            y_test_eff = y_test - base_test[:, None]
        else:
            y_train_eff = y_train - base_train
            y_test_eff = y_test - base_test
    else:
        y_train_eff = y_train
        y_test_eff = y_test

    # Optional target scaling (force on for multi-day horizons for stability)
    do_scale = bool(cfg.scale_target or (cfg.horizon and int(cfg.horizon) > 1))
    if do_scale:
        y_mean = float(y_train_eff.mean())
        y_std = float(y_train_eff.std() if y_train_eff.std() != 0 else 1.0)
        y_train_scaled = (y_train_eff - y_mean) / y_std
        y_test_scaled = (y_test_eff - y_mean) / y_std
        y_scaler = {"mean": y_mean, "std": y_std}
    else:
        y_train_scaled = y_train_eff
        y_test_scaled = y_test_eff
        y_scaler = None

    train_ds = SeqFundDataset(X_train, F_train, y_train_scaled)
    test_ds = SeqFundDataset(X_test, F_test, y_test_scaled)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model_type = getattr(cfg, "model_type", "transformer").lower()
    if model_type == "tft":
        tft_layers = max(1, min(2, int(getattr(cfg, "num_layers", 1))))
        model = TemporalFusionRegressor(
            ta_input_size=X_ta.shape[-1],
            fund_input_size=X_fund.shape[1],
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=tft_layers,
            dropout=cfg.dropout,
            fund_hidden=64,
            horizon=horizon,
            quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)),
        )
    else:
        model = SeqFundRegressor(
            ta_input_size=X_ta.shape[-1],
            fund_input_size=X_fund.shape[1],
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            fund_hidden=64,
        )

    model, info = train_loop(model, (train_loader, val_loader), cfg)

    # Evaluate on train and test splits for overfit checks
    device = cfg.device
    model.eval()
    future_point_error = None

    def _eval_loader(loader: DataLoader, base_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds_local = []
        trues_local = []
        with torch.no_grad():
            for batch in loader:
                xb_seq, xb_fund, yb = batch
                xb_seq = xb_seq.to(device)
                xb_fund = xb_fund.to(device)
                if model_type == "tft":
                    pred_full = model(xb_seq, xb_fund)  # (B, H, Q)
                    qs = list(getattr(cfg, "quantiles", (0.1, 0.5, 0.9)))
                    med_idx = int(min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5)))
                    pred_med = pred_full[..., med_idx]  # (B, H)
                    pred_last = pred_med[:, -1]  # (B,)
                    y_last = yb[:, -1]
                    preds_local.append(pred_last.cpu().numpy())
                    trues_local.append(y_last.numpy())
                else:
                    pred = model(xb_seq, xb_fund)
                    preds_local.append(pred.cpu().numpy())
                    trues_local.append(yb.numpy())

        y_pred_scaled = np.concatenate(preds_local, axis=0)
        y_true_scaled = np.concatenate(trues_local, axis=0)

        # Inverse scale
        if y_scaler is not None:
            y_pred_eff = (y_pred_scaled * y_scaler["std"]) + y_scaler["mean"]
            y_true_eff = (y_true_scaled * y_scaler["std"]) + y_scaler["mean"]
        else:
            y_pred_eff = y_pred_scaled
            y_true_eff = y_true_scaled

        # If residualized, reconstruct predicted/true next_close for metrics/plot
        if cfg.target == "next_close" and cfg.residualize:
            y_pred_final = y_pred_eff + base_slice
            y_true_final = y_true_eff + base_slice
        else:
            y_pred_final = y_pred_eff
            y_true_final = y_true_eff

        return y_true_final, y_pred_final

    train_eval_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    y_true_train, y_pred_train = _eval_loader(train_eval_loader, base_train)
    y_true_test, y_pred_test = _eval_loader(val_loader, base_test)
    y_true, y_pred = y_true_test, y_pred_test

    # Build target dates per-sample aligned to the trading index
    plot_dates = target_dates[split_idx:]

    # Metrics on train and test splits
    metrics_train = compute_metrics(y_true_train, y_pred_train)
    metrics = compute_metrics(y_true, y_pred)
    # Normalised diagnostics
    try:
        med_price = float(np.nanmedian(y_true)) if np.isfinite(np.nanmedian(y_true)) else float("nan")
        if med_price and np.isfinite(med_price) and med_price != 0:
            metrics["rmse_pct_price"] = 100.0 * metrics["rmse"] / med_price
            metrics_train["rmse_pct_price"] = 100.0 * metrics_train["rmse"] / med_price
    except Exception:
        pass
    try:
        if metrics_train.get("rmse"):
            metrics["rmse_ratio"] = float(metrics["rmse"]) / float(metrics_train["rmse"])
    except Exception:
        pass
    # Add direction/F1 metrics (up vs not-up). For next_close, compare deltas vs base.
    try:
        eps_dir = None
    except Exception:
        eps_dir = None
    try:
        dirm_train = compute_directional_metrics(
            y_true_train,
            y_pred_train,
            base=base_train if (cfg.target == "next_close") else None,
            target_kind=cfg.target,
            eps=eps_dir,
        )
        metrics_train.update(dirm_train)
    except Exception:
        pass
    try:
        dirm = compute_directional_metrics(
            y_true,
            y_pred,
            base=base_test if (cfg.target == "next_close") else None,
            target_kind=cfg.target,
            eps=eps_dir,
        )
        metrics.update(dirm)
    except Exception:
        pass
    try:
        tr_rmse = metrics_train.get("rmse")
        te_rmse = metrics.get("rmse")
        if tr_rmse is not None and te_rmse is not None:
            metrics["overfit_gap_rmse"] = float(te_rmse) - float(tr_rmse)
    except Exception:
        pass

    walkforward = None
    try:
        def _run_walkforward_checks() -> Optional[Dict[str, Any]]:
            folds = int(getattr(cfg, "walkforward_splits", 0) or 0)
            if folds <= 0:
                return None
            min_test_local = max(3, int(cfg.min_test_samples))
            total = X_ta.shape[0]
            # ensure enough samples for at least one fold
            max_folds_possible = max(0, (total - min_test_local) // min_test_local)
            folds = min(folds, max_folds_possible)
            if folds <= 0:
                return None

            # Choose split indices spaced toward the end for realistic forward validation
            step = max(min_test_local, (total - min_test_local) // (folds + 1))
            split_indices = []
            cur = total - (folds * step)
            for _ in range(folds):
                if cur > min_test_local and cur < total - min_test_local:
                    split_indices.append(cur)
                cur += step
            if not split_indices:
                return None

            def _train_eval_fold(split_idx: int) -> Optional[Dict[str, Any]]:
                X_train_f, y_train_f = X_ta[:split_idx], y_all[:split_idx]
                X_test_f, y_test_f = X_ta[split_idx:], y_all[split_idx:]
                F_train_f, F_test_f = X_fund[:split_idx], X_fund[split_idx:]
                base_train_f, base_test_f = base[:split_idx], base[split_idx:]

                if len(X_test_f) < min_test_local or len(X_train_f) < 2:
                    return None

                x_scaler_f = SequenceStandardScaler().fit(X_train_f)
                X_train_s = x_scaler_f.transform(X_train_f)
                X_test_s = x_scaler_f.transform(X_test_f)
                f_scaler_f = None
                if F_train_f.shape[1] > 0:
                    f_scaler_f = VectorStandardScaler().fit(F_train_f)
                    F_train_s = f_scaler_f.transform(F_train_f)
                    F_test_s = f_scaler_f.transform(F_test_f)
                else:
                    F_train_s = F_train_f
                    F_test_s = F_test_f

                if cfg.target == "next_close" and cfg.residualize:
                    y_train_eff = y_train_f - base_train_f
                    y_test_eff = y_test_f - base_test_f
                else:
                    y_train_eff = y_train_f
                    y_test_eff = y_test_f

                do_scale = bool(cfg.scale_target or (cfg.horizon and int(cfg.horizon) > 1))
                if do_scale:
                    y_mean_f = float(y_train_eff.mean())
                    y_std_f = float(y_train_eff.std() if y_train_eff.std() != 0 else 1.0)
                    y_train_scaled = (y_train_eff - y_mean_f) / y_std_f
                    y_test_scaled = (y_test_eff - y_mean_f) / y_std_f
                    y_scaler_f = {"mean": y_mean_f, "std": y_std_f}
                else:
                    y_train_scaled = y_train_eff
                    y_test_scaled = y_test_eff
                    y_scaler_f = None

                train_ds_f = SeqFundDataset(X_train_s, F_train_s, y_train_scaled)
                test_ds_f = SeqFundDataset(X_test_s, F_test_s, y_test_scaled)
                train_loader_f = DataLoader(train_ds_f, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
                val_loader_f = DataLoader(test_ds_f, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

                model_type_f = getattr(cfg, "model_type", "transformer").lower()
                if model_type_f == "tft":
                    tft_layers = max(1, min(2, int(getattr(cfg, "num_layers", 1))))
                    model_f = TemporalFusionRegressor(
                        ta_input_size=X_ta.shape[-1],
                        fund_input_size=X_fund.shape[1],
                        d_model=cfg.d_model,
                        nhead=cfg.nhead,
                        num_layers=tft_layers,
                        dropout=cfg.dropout,
                        fund_hidden=64,
                        horizon=horizon,
                        quantiles=getattr(cfg, "quantiles", (0.1, 0.5, 0.9)),
                    )
                else:
                    model_f = SeqFundRegressor(
                        ta_input_size=X_ta.shape[-1],
                        fund_input_size=X_fund.shape[1],
                        d_model=cfg.d_model,
                        nhead=cfg.nhead,
                        num_layers=cfg.num_layers,
                        dim_feedforward=cfg.dim_feedforward,
                        dropout=cfg.dropout,
                        fund_hidden=64,
                    )

                wf_epochs = int(
                    cfg.walkforward_epochs
                    if cfg.walkforward_epochs is not None
                    else max(3, min(cfg.epochs, max(3, cfg.epochs // 3)))
                )
                cfg_fold = replace(cfg, epochs=wf_epochs, patience=max(2, min(cfg.patience, wf_epochs)))

                model_f, _ = train_loop(model_f, (train_loader_f, val_loader_f), cfg_fold)

                def _eval_loader_f(loader: DataLoader, base_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                    preds_local = []
                    trues_local = []
                    device = cfg.device
                    model_f.eval()
                    with torch.no_grad():
                        for batch in loader:
                            xb_seq, xb_fund, yb = batch
                            xb_seq = xb_seq.to(device)
                            xb_fund = xb_fund.to(device)
                            pred = model_f(xb_seq, xb_fund)
                            preds_local.append(pred.cpu().numpy())
                            trues_local.append(yb.numpy())
                    y_pred_scaled = np.concatenate(preds_local, axis=0)
                    y_true_scaled = np.concatenate(trues_local, axis=0)
                    if y_scaler_f is not None:
                        y_pred_eff = (y_pred_scaled * y_scaler_f["std"]) + y_scaler_f["mean"]
                        y_true_eff = (y_true_scaled * y_scaler_f["std"]) + y_scaler_f["mean"]
                    else:
                        y_pred_eff = y_pred_scaled
                        y_true_eff = y_true_scaled
                    if cfg.target == "next_close" and cfg.residualize:
                        y_pred_final = y_pred_eff + base_slice
                        y_true_final = y_true_eff + base_slice
                    else:
                        y_pred_final = y_pred_eff
                        y_true_final = y_true_eff
                    return y_true_final, y_pred_final

                y_true_train_f, y_pred_train_f = _eval_loader_f(train_loader_f, base_train_f)
                y_true_test_f, y_pred_test_f = _eval_loader_f(val_loader_f, base_test_f)

                met_tr = compute_metrics(y_true_train_f, y_pred_train_f)
                met_te = compute_metrics(y_true_test_f, y_pred_test_f)
                try:
                    tr_rmse = met_tr.get("rmse")
                    te_rmse = met_te.get("rmse")
                    gap = (float(te_rmse) - float(tr_rmse)) if tr_rmse is not None and te_rmse is not None else None
                except Exception:
                    gap = None

                def _to_date(val):
                    try:
                        return str(getattr(val, "date", lambda: val)())
                    except Exception:
                        return str(val)

                fold_info = {
                    "split_idx": int(split_idx),
                    "train_count": int(len(X_train_f)),
                    "test_count": int(len(X_test_f)),
                    "train_end": _to_date(target_dates[:split_idx][-1]) if len(target_dates[:split_idx]) else None,
                    "test_start": _to_date(target_dates[split_idx]) if len(target_dates) > split_idx else None,
                    "test_end": _to_date(target_dates[-1]) if len(target_dates) else None,
                    "train_rmse": met_tr.get("rmse"),
                    "test_rmse": met_te.get("rmse"),
                    "gap_rmse": gap,
                }
                return fold_info

            results = []
            for split_idx in split_indices:
                info = _train_eval_fold(int(split_idx))
                if info:
                    results.append(info)

            if not results:
                return None
            gaps = [r["gap_rmse"] for r in results if r.get("gap_rmse") is not None]
            summary = {
                "avg_gap_rmse": float(np.mean(gaps)) if gaps else None,
                "max_gap_rmse": float(np.max(gaps)) if gaps else None,
                "folds": results,
            }
            return summary

        walkforward = _run_walkforward_checks()
    except Exception as ex:
        print(f"[run_training] walkforward check failed: {type(ex).__name__}: {ex}")
        walkforward = None

    # One-step (horizon) ahead forecast beyond the last available date.
    # This is intentionally forgiving: it always attempts to produce a
    # reasonable future point as long as there is at least one row of data.
    future_point = None
    actual_target = None
    try:
        # Use as many recent rows as possible (up to eff_window).
        win = min(eff_window, len(feat_df))
        last_slice = feat_df.tail(win)
        last_ta = last_slice[ta_cols].values.astype(np.float32)
        X_last = x_scaler.transform(last_ta[None, ...])
        if f_scaler is not None and X_fund.shape[1] > 0 and fund_cols:
            last_f = last_slice[fund_cols].ffill().iloc[-1:].values.astype(np.float32)
            last_f = np.nan_to_num(last_f, nan=0.0, posinf=0.0, neginf=0.0)
            F_last = f_scaler.transform(last_f)
        else:
            F_last = np.zeros((1, 0), dtype=np.float32)

        model.eval()
        with torch.no_grad():
            if model_type == "tft":
                pred_full = model(
                    torch.from_numpy(X_last).to(cfg.device),
                    torch.from_numpy(F_last).to(cfg.device),
                )  # (1, H, Q)
                qs = list(getattr(cfg, "quantiles", (0.1, 0.5, 0.9)))
                med_idx = int(min(range(len(qs)), key=lambda i: abs(qs[i] - 0.5)))
                pred_med = pred_full[..., med_idx]  # (1, H)
                y_last_scaled = float(pred_med[:, -1].cpu().numpy().reshape(-1)[0])
            else:
                y_last_scaled = model(
                    torch.from_numpy(X_last).to(cfg.device),
                    torch.from_numpy(F_last).to(cfg.device),
                ).cpu().numpy().reshape(-1)[0]
        # Inverse scale back to effective target space.
        if y_scaler is not None:
            y_last_eff = float(y_last_scaled * y_scaler["std"] + y_scaler["mean"])
        else:
            y_last_eff = float(y_last_scaled)

        # Choose future date: by default, last trading date + horizon business days,
        # refined by querying Yahoo for post-last-date trading days when possible.
        last_date = df.index[-1]
        future_date = last_date + _pd.tseries.offsets.BDay(cfg.horizon)
        try:
            df_after = get_ohlcv(
                ticker_yt=cfg.ticker,
                start=str(getattr(last_date, "date", lambda: last_date)()),
                end=None,
                auto_adjust=True,
            )
            df_after = standardize_ohlcv_columns(df_after)
            next_trading = df_after.index[df_after.index > last_date]
            if len(next_trading) >= cfg.horizon:
                future_date = next_trading[cfg.horizon - 1]
            elif len(next_trading) > 0:
                future_date = next_trading[-1] + _pd.tseries.offsets.BDay(
                    cfg.horizon - len(next_trading)
                )
        except Exception:
            pass

        # Reconstruct absolute price based on current baseline.
        if cfg.target == "next_close":
            base_last_col = (
                "ma20"
                if (cfg.residual_base.lower() == "ma20" and "ma20" in last_slice.columns)
                else "Close"
            )
            base_last = float(last_slice[base_last_col].iloc[-1])
            future_value = y_last_eff + (base_last if cfg.residualize else 0.0)
        else:
            last_close = float(last_slice["Close"].iloc[-1])
            future_value = last_close * (1.0 + y_last_eff)

        future_point = {
            "date": str(getattr(future_date, "date", lambda: future_date)()),
            "value": float(future_value),
        }

        # Try to fetch actual close at the exact target date (End + h).
        try:
            tgt_str = str(getattr(future_date, "date", lambda: future_date)())
            df_target = get_ohlcv(
                ticker_yt=cfg.ticker, start=cfg.start, end=tgt_str, auto_adjust=True
            )
            df_target = standardize_ohlcv_columns(df_target)
            if future_date in df_target.index:
                actual_target = {
                    "date": tgt_str,
                    "value": float(df_target["Close"].loc[future_date]),
                }
        except Exception:
            actual_target = None
    except Exception as ex:
        # Log but do not fail training if future forecast cannot be built.
        err_msg = f"{type(ex).__name__}: {ex}"
        print(f"[run_training] future_point failed: {err_msg}")
        future_point = None
        future_point_error = err_msg

    # Only show actual_target when exact End+h exists; no fallback to latest

    # Finalize string dates for plotting (future-aligned)
    test_dates_str = [str(d.date()) if hasattr(d, 'date') else str(d) for d in plot_dates]

    # Persist artifacts only when requested
    run_dir = None
    if getattr(cfg, "persist", False):
        import time as _time
        prefix = "tft_ts" if model_type == "tft" else "transformer_ts"
        run_dir = os.path.join("runs", f"{prefix}_{_time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
        import pickle as _pickle
        with open(os.path.join(run_dir, "x_scaler.pkl"), "wb") as f:
            _pickle.dump(x_scaler.state_dict(), f)
        if 'f_scaler' in locals() and f_scaler is not None:
            with open(os.path.join(run_dir, "f_scaler.pkl"), "wb") as f:
                _pickle.dump(f_scaler.state_dict(), f)
        if y_scaler is not None:
            import json as _json
            with open(os.path.join(run_dir, "y_scaler.json"), "w", encoding="utf-8") as f:
                _json.dump(y_scaler, f)

        from dataclasses import asdict as _asdict
        import json as _json
        cfg_to_save = _asdict(cfg)
        cfg_to_save["feature_cols"] = feature_cols
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            _json.dump(cfg_to_save, f, indent=2)

        np.save(os.path.join(run_dir, "y_true.npy"), y_true)
        np.save(os.path.join(run_dir, "y_pred.npy"), y_pred)

        # Persist dates and optional future/actual points for reliable re-loading later
        try:
            with open(os.path.join(run_dir, "test_dates.json"), "w", encoding="utf-8") as f:
                _json.dump(test_dates_str, f)
        except Exception:
            pass
        try:
            if future_point is not None:
                with open(os.path.join(run_dir, "future_point.json"), "w", encoding="utf-8") as f:
                    _json.dump(future_point, f)
            if future_point_error is not None:
                with open(os.path.join(run_dir, "future_point_error.json"), "w", encoding="utf-8") as f:
                    _json.dump({"error": future_point_error}, f)
            if actual_target is not None:
                with open(os.path.join(run_dir, "actual_target.json"), "w", encoding="utf-8") as f:
                    _json.dump(actual_target, f)
        except Exception:
            pass
        # Save split + train-metric metadata for transparency when reloading
        try:
            import json as _json
            with open(os.path.join(run_dir, "metrics_train.json"), "w", encoding="utf-8") as f:
                _json.dump(metrics_train, f, indent=2)
            with open(os.path.join(run_dir, "split_info.json"), "w", encoding="utf-8") as f:
                _json.dump(split_meta, f, indent=2)
        except Exception:
            pass

    # Split features into tech vs fund for UI listing convenience
    feature_cols_tech = [c for c in feature_cols if not (isinstance(c, str) and c.startswith("f_"))]
    feature_cols_fund = [c for c in feature_cols if isinstance(c, str) and c.startswith("f_")]

    return {
        "run_dir": run_dir,
        "metrics": metrics,
        "metrics_test": metrics,
        "metrics_train": metrics_train,
        "feature_cols": feature_cols,
        "feature_cols_tech": feature_cols_tech,
        "feature_cols_fund": feature_cols_fund,
        "X_shape": list(X_ta.shape),
        "y_true": y_true,
        "y_pred": y_pred,
        "test_dates": test_dates_str,
        "target": cfg.target,
        "residualized": bool(cfg.target == "next_close" and cfg.residualize),
        "residual_base": base_col,
        "future_point": future_point,
        "future_point_error": future_point_error,
        "actual_target": actual_target,
        "effective_window": int(eff_window),
        "used_fundamentals": bool(used_fundamentals),
        "split_info": split_meta,
        "walkforward": walkforward,
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + eps
    r2 = float(1 - ss_res / ss_tot)
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def summarize_learning_curves(train_losses: Sequence[float], val_losses: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Lightweight trend summary for train/val losses."""
    try:
        tl = [float(x) for x in train_losses or []]
        vl = [float(x) for x in val_losses or []]
        summary: Dict[str, Any] = {}
        if tl:
            summary["train"] = tl
            summary["train_declining"] = bool(tl[-1] <= tl[0])
            if tl[0] != 0:
                summary["train_drop_pct"] = float(100.0 * (tl[0] - tl[-1]) / max(tl[0], 1e-8))
        if vl:
            summary["val"] = vl
            v_min = float(np.nanmin(vl))
            best_epoch = int(np.nanargmin(vl) + 1)
            summary["best_val_epoch"] = best_epoch
            summary["best_val_loss"] = v_min
            summary["val_declining"] = bool(vl[-1] <= vl[0])
            if vl[0] != 0:
                summary["val_drop_pct"] = float(100.0 * (vl[0] - vl[-1]) / max(vl[0], 1e-8))
            if np.isfinite(v_min) and v_min != 0:
                summary["val_last_over_min_pct"] = float(100.0 * (vl[-1] - v_min) / abs(v_min))
            early_turn = best_epoch <= max(2, len(vl) // 3) and vl[-1] > v_min * 1.05
            summary["val_early_rise"] = bool(early_turn)
        if summary:
            summary["epochs_ran"] = int(max(len(tl), len(vl)))
        return summary or None
    except Exception:
        return None


def compute_residual_acf_stats(residuals: np.ndarray, max_lag: int = 20) -> Optional[Dict[str, Any]]:
    """Compute simple ACF of residuals to flag structure (vs white noise)."""
    try:
        res = np.asarray(residuals, dtype=float)
        if res.size < 3:
            return None
        res = res - np.nanmean(res)
        acfs: list[Dict[str, float]] = []
        for lag in range(1, max_lag + 1):
            if lag >= res.shape[0]:
                break
            a = res[:-lag]
            b = res[lag:]
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < 2:
                continue
            corr = np.corrcoef(a[mask], b[mask])[0, 1]
            if np.isfinite(corr):
                acfs.append({"lag": int(lag), "acf": float(corr)})
        if not acfs:
            return None
        max_entry = max(acfs, key=lambda v: abs(v["acf"]))
        return {
            "lags": acfs,
            "max_abs": float(abs(max_entry["acf"])),
            "max_at_lag": int(max_entry["lag"]),
            "structured": bool(abs(max_entry["acf"]) > 0.2),
        }
    except Exception:
        return None


def build_baseline_diagnostics(
    y_true_last: np.ndarray,
    base_values: np.ndarray,
    anchor_dates: Sequence,
    df: pd.DataFrame,
    model_rmse: Optional[float],
    target_kind: str = "next_close",
) -> Optional[Dict[str, Any]]:
    """Compare against simple baselines (persistence, MA20)."""
    try:
        yt = np.asarray(y_true_last, dtype=float)
        if yt.size == 0:
            return None
        baselines: Dict[str, Dict[str, float]] = {}
        if target_kind == "next_close":
            baseline_pred = np.asarray(base_values, dtype=float) if base_values is not None else None
        else:
            baseline_pred = np.zeros_like(yt)
        if baseline_pred is not None and getattr(baseline_pred, "shape", (0,))[0] == yt.shape[0]:
            baselines["persistence"] = compute_metrics(yt, baseline_pred)
        try:
            ma20_series = df["Close"].rolling(20).mean()
            import pandas as _pd

            ma20_aligned = ma20_series.reindex(_pd.Index(anchor_dates))
            ma20_aligned = ma20_aligned.fillna(method="ffill").fillna(method="bfill")
            ma20_pred = ma20_aligned.to_numpy(dtype=float)
            if ma20_pred.shape[0] == yt.shape[0]:
                baselines["ma20"] = compute_metrics(yt, ma20_pred)
        except Exception:
            pass
        if not baselines:
            return None
        gains: Dict[str, float] = {}
        for name, b in baselines.items():
            b_rmse = b.get("rmse")
            if b_rmse and np.isfinite(b_rmse) and b_rmse > 0 and model_rmse is not None:
                gains[name] = float(100.0 * (float(b_rmse) - float(model_rmse)) / float(b_rmse))
        return {"baseline": baselines, "rmse_gain": gains}
    except Exception:
        return None


def build_diagnostics(
    y_true_last: np.ndarray,
    y_pred_last: np.ndarray,
    model_rmse: Optional[float],
    base_values: np.ndarray,
    anchor_dates: Sequence,
    df: pd.DataFrame,
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    target_kind: str = "next_close",
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    base_diag = build_baseline_diagnostics(
        y_true_last=y_true_last,
        base_values=base_values,
        anchor_dates=anchor_dates,
        df=df,
        model_rmse=model_rmse,
        target_kind=target_kind,
    )
    if base_diag:
        diagnostics.update(base_diag)
    lc = summarize_learning_curves(train_losses, val_losses)
    if lc:
        diagnostics["learning_curves"] = lc
    try:
        residuals = np.asarray(y_true_last, dtype=float) - np.asarray(y_pred_last, dtype=float)
        acf = compute_residual_acf_stats(residuals)
        if acf:
            diagnostics["residual_acf"] = acf
    except Exception:
        pass
    return diagnostics


def compute_directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    base: Optional[np.ndarray] = None,
    target_kind: str = "next_close",
    eps: Optional[float] = None,
) -> Dict[str, float]:
    """Binary direction metrics for both up and down classes.

    For target_kind == 'next_close', direction is computed from deltas
    relative to the last observed base (e.g., Close at t). For
    'next_return_pct', direction is computed directly on returns.

    If eps is None, it is auto-chosen from the 25th percentile of absolute
    true deltas with a small floor to avoid treating noise as signal.
    Values whose absolute change is within eps are ignored (neither up nor down).
    """
    try:
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)

        if target_kind == "next_close" and base is not None:
            b = np.asarray(base, dtype=float)
            d_true = yt - b
            d_pred = yp - b
        else:
            d_true = yt
            d_pred = yp

        eps_local = eps
        if eps_local is None:
            try:
                eps_local = float(np.nanquantile(np.abs(d_true), 0.25))
            except Exception:
                eps_local = 0.0
            if not np.isfinite(eps_local):
                eps_local = 0.0
            # Use a relative floor to avoid tiny eps when trend dominates
            med_abs = np.nanmedian(np.abs(d_true))
            floor = float(0.005 * med_abs) if np.isfinite(med_abs) else 0.0
            eps_local = max(eps_local, floor, 1e-6)

        # Ignore samples that are effectively flat in both true and pred
        mask = (np.abs(d_true) > eps_local) | (np.abs(d_pred) > eps_local)
        if not np.any(mask):
            return {
                "direction_acc": 0.0,
                "precision_up": 0.0,
                "recall_up": 0.0,
                "f1_up": 0.0,
                "precision_down": 0.0,
                "recall_down": 0.0,
                "f1_down": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "support_up": 0.0,
                "support_down": 0.0,
                "support_total": 0.0,
            }
        d_true_m = d_true[mask]
        d_pred_m = d_pred[mask]

        true_up = (d_true_m > eps_local).astype(int)
        pred_up = (d_pred_m > eps_local).astype(int)

        tp = float(np.sum((pred_up == 1) & (true_up == 1)))
        tn = float(np.sum((pred_up == 0) & (true_up == 0)))
        fp = float(np.sum((pred_up == 1) & (true_up == 0)))
        fn = float(np.sum((pred_up == 0) & (true_up == 1)))
        total = tp + tn + fp + fn
        acc = float((tp + tn) / total) if total > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        # Treat "down" as the positive class by symmetry
        tp_down = tn
        fp_down = fn
        fn_down = fp
        prec_down = float(tp_down / (tp_down + fp_down)) if (tp_down + fp_down) > 0 else 0.0
        rec_down = float(tp_down / (tp_down + fn_down)) if (tp_down + fn_down) > 0 else 0.0
        f1_down = (
            float(2 * prec_down * rec_down / (prec_down + rec_down))
            if (prec_down + rec_down) > 0
            else 0.0
        )
        support_up = float(np.sum(true_up == 1))
        support_down = float(np.sum(true_up == 0))
        support_total = support_up + support_down
        macro_f1 = float((f1 + f1_down) / 2.0)
        weighted_f1 = (
            float((f1 * support_up + f1_down * support_down) / support_total)
            if support_total > 0
            else 0.0
        )
        return {
            "direction_acc": acc,
            "precision_up": prec,
            "recall_up": rec,
            "f1_up": f1,
            "precision_down": prec_down,
            "recall_down": rec_down,
            "f1_down": f1_down,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "support_up": support_up,
            "support_down": support_down,
            "support_total": support_total,
        }
    except Exception:
        return {
            "direction_acc": 0.0,
            "precision_up": 0.0,
            "recall_up": 0.0,
            "f1_up": 0.0,
            "precision_down": 0.0,
            "recall_down": 0.0,
            "f1_down": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "support_up": 0.0,
            "support_down": 0.0,
            "support_total": 0.0,
        }


def compute_f1_by_horizon(
    y_true_seq: np.ndarray,
    y_pred_seq: np.ndarray,
    steps: Sequence[int],
    target_kind: str = "next_close",
    eps: Optional[float] = None,
    base: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Directional F1 slices for specific forecast horizons (1d/5d/14d)."""
    summary: Dict[str, Dict[str, float]] = {}
    try:
        if y_true_seq.ndim != 2 or y_pred_seq.ndim != 2:
            return summary
        H = y_true_seq.shape[1]
        for step in steps:
            idx = step - 1
            if idx < 0 or idx >= H:
                continue
            base_step = base if base is None else np.asarray(base, dtype=float)
            dirm = compute_directional_metrics(
                y_true_seq[:, idx],
                y_pred_seq[:, idx],
                base=base_step,
                target_kind=target_kind,
                eps=eps,
            )
            summary[f"h{step}"] = {
                "f1_up": dirm.get("f1_up", 0.0),
                "f1_down": dirm.get("f1_down", 0.0),
                "macro_f1": dirm.get("macro_f1", 0.0),
                "weighted_f1": dirm.get("weighted_f1", 0.0),
                "precision_up": dirm.get("precision_up", 0.0),
                "recall_up": dirm.get("recall_up", 0.0),
                "precision_down": dirm.get("precision_down", 0.0),
                "recall_down": dirm.get("recall_down", 0.0),
                "support": dirm.get("support_total", 0.0),
            }
    except Exception:
        return summary
    return summary


def compute_f1_stability(metrics_train: Dict[str, float], metrics_test: Dict[str, float]) -> Optional[float]:
    """Absolute train-test F1 gap (macro preferred, falls back to F1_up)."""
    try:
        f_tr = metrics_train.get("macro_f1")
        f_te = metrics_test.get("macro_f1")
        if f_tr is None or f_te is None:
            f_tr = metrics_train.get("f1_up")
            f_te = metrics_test.get("f1_up")
        if f_tr is None or f_te is None:
            return None
        return float(abs(float(f_tr) - float(f_te)))
    except Exception:
        return None


def compute_rolling_f1(
    dates: Sequence,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_kind: str = "next_close",
    eps: Optional[float] = None,
    base: Optional[np.ndarray] = None,
) -> list[Dict[str, Any]]:
    """Monthly rolling F1 to spot walk-forward degradation."""
    try:
        if len(dates) == 0 or len(y_true) == 0 or len(y_pred) == 0:
            return []
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(list(dates)),
                "y_true": y_true,
                "y_pred": y_pred,
                "base": base if base is not None else np.zeros_like(y_true),
            }
        )
        df = df.dropna(subset=["y_true", "y_pred"])
        if df.empty:
            return []
        df["month"] = df["date"].dt.to_period("M").astype(str)
        rows = []
        for month, grp in df.groupby("month"):
            base_local = grp["base"].values if base is not None else None
            dirm = compute_directional_metrics(
                grp["y_true"].values,
                grp["y_pred"].values,
                base=base_local,
                target_kind=target_kind,
                eps=eps,
            )
            rows.append(
                {
                    "month": month,
                    "f1_up": dirm.get("f1_up", 0.0),
                    "f1_down": dirm.get("f1_down", 0.0),
                    "macro_f1": dirm.get("macro_f1", 0.0),
                    "weighted_f1": dirm.get("weighted_f1", 0.0),
                    "support": int(dirm.get("support_total", len(grp))),
                }
            )
        rows.sort(key=lambda r: r.get("month", ""))
        return rows
    except Exception:
        return []


def train_loop(
    model: nn.Module,
    loaders: Tuple[DataLoader, DataLoader],
    cfg: TrainConfig,
) -> Tuple[nn.Module, Dict[str, float]]:
    train_loader, val_loader = loaders
    device = cfg.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    model_type = getattr(cfg, "model_type", "transformer").lower()
    if model_type == "tft":
        # Full TFT-style training with multi-quantile loss
        criterion = QuantileLoss(getattr(cfg, "quantiles", (0.1, 0.5, 0.9)))
    else:
        if cfg.loss.lower() == "huber":
            # PyTorch SmoothL1Loss implements the Huber loss (beta=delta)
            criterion = nn.SmoothL1Loss(beta=cfg.huber_delta)
        elif cfg.loss.lower() == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(2, cfg.patience // 2), min_lr=1e-6
    ) if cfg.lr_scheduler else None

    best_val = float("inf")
    best_state = None
    patience = cfg.patience
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb_seq, xb_fund, yb = batch
                xb_seq = xb_seq.to(device)
                xb_fund = xb_fund.to(device)
                yb = yb.to(device)
                pred = model(xb_seq, xb_fund)
                bs = xb_seq.size(0)
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                bs = xb.size(0)
            opt.zero_grad()
            loss = criterion(pred, yb)
            loss.backward()
            if cfg.clip_grad_norm and cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            opt.step()
            train_loss += loss.item() * bs
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    xb_seq, xb_fund, yb = batch
                    xb_seq = xb_seq.to(device)
                    xb_fund = xb_fund.to(device)
                    yb = yb.to(device)
                    pred = model(xb_seq, xb_fund)
                    bs = xb_seq.size(0)
                else:
                    xb, yb = batch
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    bs = xb.size(0)
                loss = criterion(pred, yb)
                val_loss += loss.item() * bs
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"best_val_mse": best_val}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer/TFT for TA prediction")
    parser.add_argument("--ticker", type=str, default="2330.TW")
    parser.add_argument("--start", type=str, default="2010-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--target", type=str, default="next_close", choices=["next_close", "next_return_pct"])
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scale_target", action="store_true")
    parser.add_argument("--no_residualize", action="store_true")
    parser.add_argument("--residual_base", type=str, default="close", choices=["close", "ma20"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "huber", "mae"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9", help="Quantiles for TFT (comma-separated)")
    parser.add_argument("--clip_grad", type=float, default=0.0)
    parser.add_argument("--no_scheduler", action="store_true")
    # auto window removed; window is suggested from horizon
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--split_mode", type=str, default="ratio_time", choices=["ratio_samples", "ratio_time", "last_n_days"])
    parser.add_argument("--test_days", type=int, default=0)
    parser.add_argument("--split_date", type=str, default=None, help="Cutoff date YYYY-MM-DD; train <= date, test > date")
    parser.add_argument("--min_test", type=int, default=30)
    parser.add_argument("--use_fundamentals", action="store_true")
    parser.add_argument("--require_fundamentals", action="store_true")
    parser.add_argument("--min_samples_fund", type=int, default=30)
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "tft"])
    parser.add_argument("--walk_splits", type=int, default=0, help="Number of walk-forward splits for overfit checks")
    parser.add_argument("--walk_epochs", type=int, default=None, help="Epochs per walk-forward split (optional override)")
    args = parser.parse_args()

    cfg = TrainConfig(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        target=args.target,
        horizon=args.horizon,
        window_size=args.window,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        patience=args.patience,
        scale_target=bool(args.scale_target),
        residualize=not bool(args.no_residualize),
        residual_base=args.residual_base,
        loss=args.loss,
        huber_delta=args.huber_delta,
        quantiles=tuple(float(x) for x in str(args.quantiles).split(",")),
        clip_grad_norm=args.clip_grad,
        lr_scheduler=not bool(args.no_scheduler),
        train_ratio=args.train_ratio,
        split_mode=args.split_mode,
        test_days=args.test_days,
        split_date=args.split_date,
        min_test_samples=args.min_test,
        use_fundamentals=bool(args.use_fundamentals),
        require_fundamentals=bool(args.require_fundamentals),
        min_samples_with_fundamentals=int(args.min_samples_fund),
        model_type=args.model_type,
        walkforward_splits=int(args.walk_splits),
        walkforward_epochs=args.walk_epochs,
    )
    # Use the unified training pipeline which saves artifacts and returns metrics
    result = run_training(cfg)
    print("Test metrics:")
    for k, v in result.get("metrics", {}).items():
        print(f"  {k}: {v:.6f}")
    print(f"Artifacts saved to {result.get('run_dir','(unknown)')}")


if __name__ == "__main__":
    main()


