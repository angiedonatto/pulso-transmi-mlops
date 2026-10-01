"""Candidate K: regime-robust deep forecaster (research module, not wired into Predict).

One global network for the 12 stations and the 4 horizons of an hourly cycle:

* inputs (all causal, <= data_cutoff): the last `lookback` 15-min observations of the station, the
  7-day same-slot median profile, the 4-week same-weekday baseline and yesterday's value over the
  lookback window *and* the target slots, Bogota calendar features and a station embedding;
* every demand channel is divided by the station's trailing 7-day mean (stable per-sample scale),
  so an L1 loss in that space approximates the leaderboard's per-station WAPE;
* `gated` architecture = a seasonal expert (profile-driven MLP with a linear skip) and a local
  dynamics expert (RevIN-normalized recent window -> MLP), mixed by a per-horizon gate that only sees
  regime evidence (recent-vs-profile log ratios and volatility). The gate value is the regime trace;
* training uses synthetic regime augmentation: for a random share of samples the recent window and
  the targets are replaced by a generic synthetic regime (level shift, oscillation with a random
  period between 1 h and 16 h, time-warped profile, ramp) that starts a random number of slots
  before the cutoff, while the seasonal references stay untouched -- exactly what a real regime
  change looks like to the model. No regime parameter is fitted to the evaluation data;
* a deep ensemble (several seeds) is aggregated with the median.

Requires PyTorch (`torch`), which is not a production dependency; the experiment lives in
`scripts/candidate_k_experiment.py`.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


SLOT = pd.Timedelta(minutes=15)
DAY_STEPS = 96
WEEK_STEPS = 7 * DAY_STEPS
BOGOTA_OFFSET = pd.Timedelta(hours=5)  # America/Bogota is UTC-5 all year
DEMAND_CHANNELS = ("y", "profile", "baseline", "yesterday")


@dataclass(frozen=True)
class CandidateKConfig:
    architecture: str = "gated"  # "gated" | "tide"
    lookback: int = 48
    horizons: int = 4
    profile_days: int = 7
    baseline_weeks: int = 4
    scale_window: int = WEEK_STEPS
    hidden: int = 128
    blocks: int = 2
    dropout: float = 0.1
    station_embedding: int = 4
    epochs: int = 10
    batch_size: int = 1024
    lr: float = 2e-3
    weight_decay: float = 1e-4
    augmentation_prob: float = 0.5
    min_onset: int = 4
    period_slots: tuple[float, float] = (4.0, 64.0)
    level_factor: tuple[float, float] = (0.2, 5.0)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Panel:
    """Demand panel on a regular 15-min UTC grid plus its causal seasonal references."""

    index: pd.DatetimeIndex
    stations: list[str]
    demand: np.ndarray  # raw, NaN where missing (T, N)
    filled: np.ndarray  # forward-filled demand, causal (T, N)
    profile: np.ndarray  # median of the same slot over the previous `profile_days` days
    baseline: np.ndarray  # mean of the same weekday-slot over the previous `baseline_weeks` weeks
    yesterday: np.ndarray
    scale: np.ndarray  # trailing mean demand (T, N), >= 1
    slot_of_day: np.ndarray  # Bogota slot of day (T,)
    day_of_week: np.ndarray  # Bogota day of week (T,)
    extra: dict[str, Any] = field(default_factory=dict)

    def position(self, timestamp: Any) -> int:
        return int(self.index.get_loc(pd.Timestamp(timestamp).tz_convert("UTC")))


def build_panel(observations: pd.DataFrame, config: CandidateKConfig, *, until: Any | None = None) -> Panel:
    """Wide causal panel. `until` truncates the observations (used to prove causality in tests)."""
    frame = observations[["station_id", "observed_at", "demand"]].copy()
    frame["station_id"] = frame["station_id"].astype(str).str.zfill(5)
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    if until is not None:
        frame = frame[frame["observed_at"] <= pd.Timestamp(until).tz_convert("UTC")]
    wide = (
        frame.drop_duplicates(["station_id", "observed_at"], keep="last")
        .pivot(index="observed_at", columns="station_id", values="demand")
        .sort_index()
        .astype(float)
    )
    # grid extends `horizons` slots past the last observation so the latest cutoff has target slots
    index = pd.date_range(wide.index.min(), wide.index.max() + config.horizons * SLOT, freq=SLOT)
    wide = wide.reindex(index)
    demand = wide.to_numpy()
    filled = wide.ffill().to_numpy()

    def lagged(steps: int) -> np.ndarray:
        out = np.full_like(filled, np.nan)
        out[steps:] = filled[:-steps]
        return out

    with warnings.catch_warnings():  # the first days have no lagged history (all-NaN slices)
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmedian(np.stack([lagged(DAY_STEPS * k) for k in range(1, config.profile_days + 1)]), axis=0)
        baseline = np.nanmean(np.stack([lagged(WEEK_STEPS * k) for k in range(1, config.baseline_weeks + 1)]), axis=0)
    baseline = np.where(np.isfinite(baseline), baseline, profile)
    scale = pd.DataFrame(filled).rolling(config.scale_window, min_periods=DAY_STEPS).mean().to_numpy()
    scale = np.clip(scale, 1.0, None)
    local = index - BOGOTA_OFFSET
    return Panel(
        index=index,
        stations=list(wide.columns),
        demand=demand,
        filled=filled,
        profile=profile,
        baseline=baseline,
        yesterday=lagged(DAY_STEPS),
        scale=scale,
        slot_of_day=(local.hour * 4 + local.minute // 15).to_numpy(),
        day_of_week=local.dayofweek.to_numpy(),
    )


@dataclass
class Samples:
    """Model-ready samples, one per (origin, station)."""

    origin: np.ndarray  # panel position of the data cutoff (S,)
    station: np.ndarray  # station position (S,)
    past: np.ndarray  # scaled recent observations (S, L)
    seasonal: np.ndarray  # scaled profile/baseline/yesterday over L + H positions (S, 3, L + H)
    calendar: np.ndarray  # (S, F) target-slot calendar features
    day_of_week: np.ndarray  # (S,) Bogota weekday of the first target
    scale: np.ndarray  # (S,)
    target: np.ndarray  # scaled targets, NaN when not observed (S, H)

    def subset(self, mask: np.ndarray) -> "Samples":
        return Samples(**{name: value[mask] for name, value in vars(self).items()})

    def __len__(self) -> int:
        return len(self.origin)


def build_samples(panel: Panel, config: CandidateKConfig, origins: np.ndarray | None = None) -> Samples:
    """Samples for the given cutoff positions (default: every position with complete features).

    Every input at origin i only reads panel rows <= i (profile/baseline/yesterday at the target
    slots read rows <= i + H - 96 < i); targets read rows i + 1 .. i + H.
    """
    L, H = config.lookback, config.horizons
    T, N = panel.filled.shape
    if origins is None:
        origins = np.arange(L - 1, T - H)
    origins = np.asarray(origins, dtype=int)
    origins = origins[(origins >= L - 1) & (origins + H < T)]
    offsets_past = np.arange(-L + 1, 1)
    offsets_all = np.arange(-L + 1, H + 1)
    rows_past = origins[:, None] + offsets_past[None, :]
    rows_all = origins[:, None] + offsets_all[None, :]
    rows_target = origins[:, None] + np.arange(1, H + 1)[None, :]

    scale = panel.scale[origins]  # (O, N)
    past = panel.filled[rows_past].transpose(0, 2, 1) / scale[:, :, None]  # (O, N, L)
    seasonal = np.stack(
        [getattr(panel, name)[rows_all].transpose(0, 2, 1) / scale[:, :, None] for name in ("profile", "baseline", "yesterday")],
        axis=2,
    )  # (O, N, 3, L + H)
    target = panel.demand[rows_target].transpose(0, 2, 1) / scale[:, :, None]  # (O, N, H)

    first_target = origins + 1
    slot = panel.slot_of_day[first_target]
    dow = panel.day_of_week[first_target]
    angle = 2 * np.pi * slot / DAY_STEPS
    calendar = np.stack([np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle), (dow >= 5).astype(float)], axis=1)

    O = len(origins)
    samples = Samples(
        origin=np.repeat(origins, N),
        station=np.tile(np.arange(N), O),
        past=past.reshape(O * N, L).astype(np.float32),
        seasonal=seasonal.reshape(O * N, 3, L + H).astype(np.float32),
        calendar=np.repeat(calendar, N, axis=0).astype(np.float32),
        day_of_week=np.repeat(dow, N).astype(np.int64),
        scale=scale.reshape(O * N).astype(np.float32),
        target=target.reshape(O * N, H).astype(np.float32),
    )
    finite = (
        np.isfinite(samples.past).all(axis=1)
        & np.isfinite(samples.seasonal).all(axis=(1, 2))
        & np.isfinite(samples.scale)
    )
    return samples.subset(finite)


# --------------------------------------------------------------------------------------
# Synthetic regime augmentation
# --------------------------------------------------------------------------------------


def augment(past: torch.Tensor, seasonal: torch.Tensor, target: torch.Tensor, config: CandidateKConfig, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the tail of the recent window and the targets with a generic synthetic regime.

    The regime starts `onset` slots before the cutoff (random) and continues through the targets.
    Seasonal references are left unchanged, as they would be at the start of a real regime change.
    """
    B, L = past.shape
    H = target.shape[1]
    chosen = torch.rand(B, generator=generator) < config.augmentation_prob
    if not chosen.any():
        return past, target
    idx = chosen.nonzero(as_tuple=True)[0]
    n = len(idx)
    series = torch.cat([past[idx], target[idx]], dim=1)  # (n, L + H)
    profile = seasonal[idx, 0]  # (n, L + H)
    positions = torch.arange(L + H, dtype=torch.float32)[None, :]
    onset = torch.randint(config.min_onset, L + 1, (n,), generator=generator)
    start = (L - onset).float()[:, None]
    active = positions >= start
    rel = (positions - start).clamp(min=0)

    def uniform(lo: float, hi: float, size: int = n) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(size, generator=generator)

    def log_uniform(lo: float, hi: float) -> torch.Tensor:
        return torch.exp(uniform(math.log(lo), math.log(hi)))

    kind = torch.multinomial(torch.tensor([0.25, 0.40, 0.20, 0.15]), n, replacement=True, generator=generator)
    level = log_uniform(*config.level_factor)[:, None]
    level_ref = profile.mean(dim=1, keepdim=True).clamp(min=0.05)
    synthetic = series.clone()
    # 0: level shift
    shift = series * level
    # 1: oscillation with a random period, phase and depth around a random level
    period = log_uniform(*config.period_slots)[:, None]
    phase = uniform(0.0, 2 * math.pi)[:, None]
    depth = uniform(0.3, 1.0)[:, None]
    osc = level_ref * log_uniform(0.3, 3.0)[:, None] * (1 + depth * torch.sin(2 * math.pi * rel / period + phase)).clamp(min=0)
    # 2: time-warped profile (the daily shape moves by up to +-8 h) times a level
    lag = torch.randint(-32, 33, (n,), generator=generator)
    gather = (positions.long() - lag[:, None]).clamp(0, L + H - 1)
    warp = torch.gather(profile, 1, gather) * level
    # 3: ramp
    rate = uniform(-0.08, 0.08)[:, None]
    ramp = series * torch.exp(rate * rel)
    for code, values in enumerate((shift, osc, warp, ramp)):
        rows = kind == code
        synthetic[rows] = values[rows]
    noise = torch.exp(torch.randn(n, L + H, generator=generator) * uniform(0.0, 0.15)[:, None])
    blend = uniform(0.5, 1.0)[:, None]
    mixed = torch.where(active, (1 - blend) * series + blend * synthetic * noise, series).clamp(min=0)
    # targets missing in the original sample stay missing
    mixed_target = torch.where(torch.isfinite(target[idx]), mixed[:, L:], target[idx])
    past = past.clone()
    target = target.clone()
    past[idx] = mixed[:, :L]
    target[idx] = mixed_target
    return past, target


def synthetic_series(n: int, length: int, generator: torch.Generator) -> torch.Tensor:
    """Generic non-negative series in the spirit of KernelSynth (no real data involved).

    Each series mixes up to three sinusoids with random periods (45 min .. 32 h) and harmonic
    distortion, a random trend, an optional level shift and an optional regime switch (the history
    before the switch comes from an independent process), with AR(1) multiplicative noise.
    """

    def uniform(lo: float, hi: float, *shape: int) -> torch.Tensor:
        return lo + (hi - lo) * torch.rand(*(shape or (n,)), generator=generator)

    def log_uniform(lo: float, hi: float, *shape: int) -> torch.Tensor:
        return torch.exp(uniform(math.log(lo), math.log(hi), *shape))

    t = torch.arange(length, dtype=torch.float32)[None, :]

    def process() -> torch.Tensor:
        level = log_uniform(0.05, 5.0)[:, None]
        signal = torch.ones(n, length)
        components = torch.multinomial(torch.tensor([0.1, 0.5, 0.3, 0.1]), n, replacement=True, generator=generator)
        for k in range(3):
            on = (components > k).float()[:, None]
            period = log_uniform(3.0, 128.0)[:, None]
            phase = uniform(0.0, 2 * math.pi)[:, None]
            amplitude = uniform(0.0, 1.0)[:, None] / (k + 1)
            wave = torch.sin(2 * math.pi * t / period + phase)
            wave = wave + uniform(-0.4, 0.4)[:, None] * torch.sin(4 * math.pi * t / period + 2 * phase)  # non-sinusoidal shapes
            signal = signal + on * amplitude * wave
        trend = (torch.rand(n, generator=generator) < 0.5).float()[:, None] * uniform(-0.02, 0.02)[:, None]
        signal = signal * torch.exp(trend * (t - length))
        shift_at = torch.randint(0, length, (n, 1), generator=generator)
        shifted = (torch.rand(n, generator=generator) < 0.3)[:, None] & (t >= shift_at)
        signal = torch.where(shifted, signal * log_uniform(0.3, 3.0)[:, None], signal)
        return level * signal.clamp(min=0)

    series = process()
    other = process()
    switch_at = torch.randint(length // 4, length - 6, (n, 1), generator=generator)
    switched = (torch.rand(n, generator=generator) < 0.3)[:, None] & (t < switch_at)
    series = torch.where(switched, other, series)
    rho = uniform(0.0, 0.9)[:, None]
    eps = torch.randn(n, length, generator=generator) * uniform(0.0, 0.25)[:, None]
    noise = torch.zeros(n, length)
    for k in range(length):
        noise[:, k] = eps[:, k] + (rho[:, 0] * noise[:, k - 1] if k else 0)
    return (series * torch.exp(noise)).clamp(min=0)


class LocalForecaster(nn.Module):
    """RevIN + residual MLP: recent window -> next `horizons` values (no seasonal inputs)."""

    def __init__(self, lookback: int, horizons: int, config: CandidateKConfig) -> None:
        super().__init__()
        self.lookback = lookback
        self.net = ResidualMLP(lookback, horizons, config)

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        past = past[:, -self.lookback :]
        mu = past.mean(dim=1, keepdim=True)
        sd = past.std(dim=1, keepdim=True) + 1e-3 + 0.05 * mu
        return (mu + sd * self.net((past - mu) / sd)).clamp(min=0)


def pretrain_local_forecaster(config: CandidateKConfig, *, lookback: int = 48, epochs: int = 40, per_epoch: int = 200_000, seed: int = 0, log: Any = None) -> LocalForecaster:
    """Train the local expert on synthetic series only (zero-shot on real data, no leakage)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    H = config.horizons
    model = LocalForecaster(lookback, H, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps = epochs * math.ceil(per_epoch / config.batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, total_steps=steps, pct_start=0.1)
    model.train()
    for epoch in range(epochs):
        series = synthetic_series(per_epoch, lookback + H, generator)
        past, target = series[:, :lookback], series[:, lookback:]
        scale = past.mean(dim=1, keepdim=True) + 0.05
        total = 0.0
        for start in range(0, per_epoch, config.batch_size):
            sl = slice(start, start + config.batch_size)
            loss = ((model(past[sl]) - target[sl]).abs() / scale[sl]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += float(loss) * len(past[sl])
        if log is not None:
            log(f"local pretrain epoch {epoch + 1}/{epochs} loss={total / per_epoch:.4f}")
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.inner = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.inner(self.norm(x))


class ResidualMLP(nn.Module):
    """TiDE-style dense encoder/decoder with a linear skip connection."""

    def __init__(self, d_in: int, d_out: int, config: CandidateKConfig) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, config.hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(config.hidden, config.dropout) for _ in range(config.blocks)])
        self.head = nn.Sequential(nn.LayerNorm(config.hidden), nn.Linear(config.hidden, d_out))
        self.skip = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.proj(x))) + self.skip(x)


def regime_features(past: torch.Tensor, seasonal: torch.Tensor, lookback: int) -> torch.Tensor:
    """Evidence about the current regime, recomputed from the (possibly augmented) inputs."""
    profile_past = seasonal[:, 0, :lookback]
    baseline_past = seasonal[:, 1, :lookback]
    feats = []
    for window in (1, 4, 16, lookback):
        y = past[:, -window:].sum(dim=1)
        feats.append(torch.log((y + 0.05) / (profile_past[:, -window:].sum(dim=1) + 0.05)))
        feats.append(torch.log((y + 0.05) / (baseline_past[:, -window:].sum(dim=1) + 0.05)))
    diffs = past[:, 1:] - past[:, :-1]
    feats.append(diffs[:, -16:].abs().mean(dim=1))
    feats.append(diffs[:, -4:].mean(dim=1))
    feats.append(past[:, -16:].std(dim=1))
    feats.append((past - profile_past).abs()[:, -16:].mean(dim=1))
    return torch.stack(feats, dim=1)


class CandidateKNet(nn.Module):
    def __init__(self, config: CandidateKConfig, n_stations: int) -> None:
        super().__init__()
        self.config = config
        L, H = config.lookback, config.horizons
        self.station = nn.Embedding(n_stations, config.station_embedding)
        self.weekday = nn.Embedding(7, 3)
        context = config.station_embedding + 3 + 5
        n_regime = 12
        d_full = L + 3 * (L + H) + context + n_regime
        if config.architecture == "tide":
            self.full = ResidualMLP(d_full, H, config)
        elif config.architecture == "gated":
            self.seasonal_expert = ResidualMLP(d_full, H, config)
            self.local_expert = ResidualMLP(L + context, H, config)
            self.gate = nn.Sequential(
                nn.Linear(n_regime + context, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, H)
            )
        else:
            raise ValueError(f"unknown architecture {config.architecture!r}")

    def forward(self, past, seasonal, calendar, station, day_of_week, *, return_parts: bool = False):
        L = self.config.lookback
        context = torch.cat([self.station(station), self.weekday(day_of_week), calendar], dim=1)
        regime = regime_features(past, seasonal, L)
        full = torch.cat([past, seasonal.flatten(1), context, regime], dim=1)
        if self.config.architecture == "tide":
            out = self.full(full)
            return (out, {}) if return_parts else out
        seasonal_out = self.seasonal_expert(full)
        # RevIN: the local expert sees the recent window normalized by its own mean/std
        mu = past.mean(dim=1, keepdim=True)
        sd = past.std(dim=1, keepdim=True) + 0.05
        local_out = mu + sd * self.local_expert(torch.cat([(past - mu) / sd, context], dim=1))
        gate = torch.sigmoid(self.gate(torch.cat([regime, context], dim=1)))
        out = gate * seasonal_out + (1 - gate) * local_out
        if return_parts:
            return out, {"gate": gate, "seasonal": seasonal_out, "local": local_out}
        return out


# --------------------------------------------------------------------------------------
# Training and inference
# --------------------------------------------------------------------------------------


def _tensors(samples: Samples) -> dict[str, torch.Tensor]:
    return {
        "past": torch.from_numpy(samples.past),
        "seasonal": torch.from_numpy(samples.seasonal),
        "calendar": torch.from_numpy(samples.calendar),
        "station": torch.from_numpy(samples.station.astype(np.int64)),
        "day_of_week": torch.from_numpy(samples.day_of_week),
        "target": torch.from_numpy(samples.target),
        "origin": torch.from_numpy(samples.origin.astype(np.int64)),
        "scale": torch.from_numpy(samples.scale),
    }


def train_model(samples: Samples, config: CandidateKConfig, n_stations: int, seed: int) -> CandidateKNet:
    """Train one ensemble member on fully observed samples (L1 on the per-station scaled target)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    keep = np.isfinite(samples.target).all(axis=1)
    data = _tensors(samples.subset(keep))
    n = len(data["target"])
    model = CandidateKNet(config, n_stations)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    steps = config.epochs * math.ceil(n / config.batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, total_steps=steps, pct_start=0.1)
    model.train()
    for _ in range(config.epochs):
        # fresh synthetic regimes every epoch, drawn once for the whole training set
        past_all, target_all = data["past"], data["target"]
        if config.augmentation_prob > 0:
            past_all, target_all = augment(past_all, data["seasonal"], target_all, config, generator)
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, config.batch_size):
            batch = order[start : start + config.batch_size]
            past, target = past_all[batch], target_all[batch]
            pred = model(past, data["seasonal"][batch], data["calendar"][batch], data["station"][batch], data["day_of_week"][batch])
            loss = (pred - target).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
    model.eval()
    return model


def augment_synthetic(past: torch.Tensor, seasonal: torch.Tensor, target: torch.Tensor, prob: float, generator: torch.Generator, *, min_onset: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the tail (and targets) by a generic synthetic regime from `synthetic_series`.

    The regime starts `onset` slots before the cutoff and its level is set relative to the
    station's profile, so it can sit anywhere from 0.3x to 3x the usual demand.
    """
    B, L = past.shape
    H = target.shape[1]
    chosen = torch.rand(B, generator=generator) < prob
    if not chosen.any():
        return past, target
    idx = chosen.nonzero(as_tuple=True)[0]
    n = len(idx)
    regime = synthetic_series(n, L + H, generator)
    level_ref = seasonal[idx, 0].mean(dim=1, keepdim=True).clamp(min=0.05)
    level = level_ref * torch.exp(math.log(0.3) + (math.log(3.0) - math.log(0.3)) * torch.rand(n, 1, generator=generator))
    regime = regime / (regime[:, :L].mean(dim=1, keepdim=True) + 1e-3) * level
    onset = torch.randint(min_onset, L + 1, (n,), generator=generator)
    active = torch.arange(L + H)[None, :] >= (L - onset)[:, None]
    series = torch.cat([past[idx], target[idx]], dim=1)
    mixed = torch.where(active, regime, series)
    past, target = past.clone(), target.clone()
    past[idx] = mixed[:, :L]
    target[idx] = torch.where(torch.isfinite(target[idx]), mixed[:, L:], target[idx])
    return past, target


def local_backtest(local: "LocalForecaster", past: torch.Tensor, seasonal: torch.Tensor, offsets: tuple[int, ...] = (4, 8, 12, 16)) -> torch.Tensor:
    """How the local expert and the plain profile would have done on the last slots (causal).

    For each offset o the local expert forecasts past[-o:-o+4] from the window that ends o slots
    before the cutoff; errors are normalized by the recent level. Returns (B, 3 * len(offsets)):
    local error, profile error, log ratio.
    """
    L = past.shape[1]
    H = 4
    scale = past[:, -16:].mean(dim=1) + 0.05
    local_err, profile_err = [], []
    for o in offsets:
        window = past[:, : L - o]
        actual = past[:, L - o : L - o + H]
        forecast = local(window)[:, : actual.shape[1]]
        profile = seasonal[:, 0, L - o : L - o + actual.shape[1]]
        local_err.append((forecast - actual).abs().mean(dim=1) / scale)
        profile_err.append((profile - actual).abs().mean(dim=1) / scale)
    local_err = torch.stack(local_err, dim=1)
    profile_err = torch.stack(profile_err, dim=1)
    return torch.cat([local_err, profile_err, torch.log((local_err + 0.02) / (profile_err + 0.02))], dim=1)


REALIZED_LAGS = (4, 8)  # the seasonal expert's own forecasts made 1 h and 2 h before the cutoff


class StagedK(nn.Module):
    """Seasonal expert (real data) + frozen synthetic-pretrained local expert + learned gate.

    With `realized_error=True` the gate also sees how the seasonal expert's own forecasts issued
    1 h and 2 h earlier did against the observations that have arrived since (all <= cutoff).
    """

    def __init__(self, config: CandidateKConfig, n_stations: int, seasonal: CandidateKNet, local: LocalForecaster, *, realized_error: bool = False) -> None:
        super().__init__()
        self.config = config
        self.seasonal_expert = seasonal
        self.local_expert = local
        self.uses_realized_error = realized_error
        self.seasonal_lookup: torch.Tensor | None = None  # (positions, stations, H), demand units
        self.station = nn.Embedding(n_stations, config.station_embedding)
        n_in = 12 + 12 + config.station_embedding + 5 + (2 * len(REALIZED_LAGS) if realized_error else 0)
        self.gate = nn.Sequential(nn.Linear(n_in, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, config.horizons))
        for module in (self.seasonal_expert, self.local_expert):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def build_lookup(self, samples: "Samples", n_stations: int) -> None:
        """Seasonal-expert forecast for every (origin, station); each one only uses data <= its origin."""
        data = _tensors(samples)
        out = []
        for start in range(0, len(samples), 8192):
            sl = slice(start, start + 8192)
            out.append(self.seasonal_expert(data["past"][sl], data["seasonal"][sl], data["calendar"][sl], data["station"][sl], data["day_of_week"][sl]))
        values = torch.cat(out) * data["scale"][:, None]
        lookup = torch.full((int(samples.origin.max()) + 1, n_stations, self.config.horizons), float("nan"))
        lookup[data["origin"], data["station"]] = values
        self.seasonal_lookup = lookup

    def realized_error(self, past: torch.Tensor, origin: torch.Tensor, station: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        L, H = past.shape[1], self.config.horizons
        level = past[:, -16:].mean(dim=1) + 0.05
        errors = []
        for lag in REALIZED_LAGS:
            lagged = origin - lag
            valid = lagged >= 0
            forecast = self.seasonal_lookup[lagged.clamp(min=0), station] / scale[:, None]  # scaled like this sample
            actual = past[:, L - lag : L - lag + H]
            err = (forecast - actual).abs().mean(dim=1) / level
            errors.append(torch.where(valid & torch.isfinite(err), err, torch.full_like(err, 0.5)))
        errors = torch.stack(errors, dim=1)
        local = local_backtest(self.local_expert, past, past.new_zeros(past.shape[0], 3, L + H), offsets=REALIZED_LAGS)[:, : len(REALIZED_LAGS)]
        return torch.cat([errors, torch.log((local + 0.02) / (errors + 0.02))], dim=1)

    def experts(self, past, seasonal, calendar, station, day_of_week, origin=None, scale=None):
        with torch.no_grad():
            s = self.seasonal_expert(past, seasonal, calendar, station, day_of_week)
            l = self.local_expert(past)
            parts = [regime_features(past, seasonal, self.config.lookback), local_backtest(self.local_expert, past, seasonal)]
            if self.uses_realized_error:
                parts.append(self.realized_error(past, origin, station, scale))
            evidence = torch.cat(parts, dim=1)
        return s, l, evidence

    def mix(self, s, l, evidence, calendar, station):
        gate = torch.sigmoid(self.gate(torch.cat([evidence, self.station(station), calendar], dim=1)))
        return gate * s + (1 - gate) * l, gate

    def forward(self, past, seasonal, calendar, station, day_of_week, *, origin=None, scale=None, return_parts: bool = False):
        s, l, evidence = self.experts(past, seasonal, calendar, station, day_of_week, origin, scale)
        out, gate = self.mix(s, l, evidence, calendar, station)
        if return_parts:
            return out, {"gate": gate, "seasonal": s, "local": l}
        return out


def train_staged(
    samples: Samples,
    config: CandidateKConfig,
    n_stations: int,
    seed: int,
    local: LocalForecaster,
    *,
    gate_epochs: int = 6,
    synthetic_prob: float = 0.35,
    realized_error: bool = False,
    lookup_samples: Samples | None = None,
) -> StagedK:
    """Stage 1: seasonal expert on real data. Stage 2: gate on real + synthetic regimes, experts frozen.

    `lookup_samples` (every origin that may be needed later, training and evaluation) feeds the
    realized-error lookup; each lookup entry is a causal forecast made at its own origin.
    """
    seasonal = train_model(samples, replace(config, architecture="tide", augmentation_prob=0.0), n_stations, seed)
    model = StagedK(config, n_stations, seasonal, local, realized_error=realized_error)
    if realized_error:
        model.build_lookup(lookup_samples if lookup_samples is not None else samples, n_stations)
    torch.manual_seed(seed + 1000)
    generator = torch.Generator().manual_seed(seed + 1000)
    keep = np.isfinite(samples.target).all(axis=1)
    data = _tensors(samples.subset(keep))
    n = len(data["target"])
    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=3e-3, weight_decay=1e-4)
    model.gate.train()
    for _ in range(gate_epochs):
        past, target = augment_synthetic(data["past"], data["seasonal"], data["target"], synthetic_prob, generator)
        s, l, evidence = model.experts(past, data["seasonal"], data["calendar"], data["station"], data["day_of_week"], data["origin"], data["scale"])
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, config.batch_size):
            b = order[start : start + config.batch_size]
            out, _ = model.mix(s[b], l[b], evidence[b], data["calendar"][b], data["station"][b])
            loss = (out - target[b]).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict(models: list[nn.Module], samples: Samples) -> dict[str, np.ndarray]:
    """Ensemble median in demand units, plus the mean gate (regime trace) when available."""
    data = _tensors(samples)
    members, gates = [], []
    for model in models:
        extra = {"origin": data["origin"], "scale": data["scale"]} if getattr(model, "uses_realized_error", False) else {}
        out, parts = model(data["past"], data["seasonal"], data["calendar"], data["station"], data["day_of_week"], return_parts=True, **extra)
        members.append(out.clamp(min=0).numpy())
        if "gate" in parts:
            gates.append(parts["gate"].numpy())
    stacked = np.stack(members)  # (M, S, H)
    scale = samples.scale[:, None]
    result = {
        "prediction": np.median(stacked, axis=0) * scale,
        "member_spread": (stacked.max(axis=0) - stacked.min(axis=0)) * scale,
    }
    if gates:
        result["gate"] = np.mean(np.stack(gates), axis=0)
    return result


def prediction_frame(panel: Panel, samples: Samples, output: dict[str, np.ndarray], config: CandidateKConfig) -> pd.DataFrame:
    """Long frame (one row per station x target) aligned with the cycle target convention."""
    rows = []
    for h in range(config.horizons):
        frame = pd.DataFrame(
            {
                "station_id": [panel.stations[s] for s in samples.station],
                "data_cutoff": panel.index[samples.origin],
                "target_at": panel.index[samples.origin + h + 1],
                "horizon_minutes": 15 * (h + 1),
                "k_prediction": output["prediction"][:, h],
                "k_member_spread": output["member_spread"][:, h],
            }
        )
        if "gate" in output:
            frame["k_seasonal_weight"] = output["gate"][:, h]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)
