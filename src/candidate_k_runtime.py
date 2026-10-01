"""Frozen Candidate K v2.1 runtime (PyTorch): daily refit + causal inference for one cycle.

Reproduces the walk-forward procedure of scripts/candidate_k_experiment.py for `k_staged_re`:
the refit for a cycle is the latest 05:00Z at or before its data_cutoff; each of the 3 seeds trains
`candidate_k_deep.train_staged` (realized-error gate, synthetic-pretrained local expert from the
artifact) on samples whose targets are observed by the refit time, one origin every 30 min; the
cycle forecast is the median over seeds. Trained refits are cached on disk (keyed by the frozen spec
and the refit time) so a refit is trained once and reused by every later cycle of the same day.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

import candidate_k_deep as ck
import candidate_k_shadow as cks


class FrozenKPredictor:
    def __init__(
        self,
        metadata: dict[str, Any],
        artifact_dir: str | Path,
        cache_dir: str | Path | None = None,
        *,
        config_overrides: dict[str, Any] | None = None,
        seeds: tuple[int, ...] | None = None,
        allow_unfrozen: bool = False,
        log: Callable[[str], None] | None = None,
    ) -> None:
        procedure = metadata["procedure"]
        frozen = _jsonable(ck.CandidateKConfig(architecture=procedure["architecture"]).to_dict())
        if metadata.get("config") != frozen:
            raise cks.CandidateKShadowError("candidate_k_deep.CandidateKConfig differs from the frozen artifact config; refusing to run")
        if (config_overrides or seeds is not None) and not allow_unfrozen:
            raise cks.CandidateKShadowError("config overrides / custom seeds are for tests only (allow_unfrozen=True)")
        self.config = replace(ck.CandidateKConfig(architecture=procedure["architecture"]), **(config_overrides or {}))
        self.seeds = tuple(seeds) if seeds is not None else tuple(procedure["seeds"])
        self.gate_epochs = int(procedure["gate_epochs"])
        self.synthetic_prob = float(procedure["synthetic_prob"])
        self.local = ck.LocalForecaster(int(procedure["local_lookback"]), self.config.horizons, ck.CandidateKConfig())
        self.local.load_state_dict(torch.load(Path(artifact_dir) / cks.LOCAL_WEIGHTS))
        self.local.eval()
        self.cache_dir = Path(cache_dir) if cache_dir else None
        spec = {"files": metadata["files"], "procedure": procedure, "config": _jsonable(self.config.to_dict()), "seeds": list(self.seeds)}
        self.spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        self._memory: dict[str, tuple[list[ck.StagedK], dict[str, Any]]] = {}
        self._log = log or (lambda message: None)

    # ------------------------------------------------------------------ inference

    def predict(self, observations: pd.DataFrame, data_cutoff: Any, cycle_targets: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        cutoff = cks._utc(data_cutoff)
        obs = cks._normalize_observations(observations)
        obs = obs[obs["observed_at"] <= cutoff]  # defense in depth: nothing after the cutoff
        if obs.empty or obs["observed_at"].max() < cutoff:
            raise cks.CandidateKShadowError(f"observations do not reach the data_cutoff {cks._ts(cutoff)}")
        refit = cks.refit_time(cutoff)
        models, info = self._models_for(obs, refit)
        torch.set_num_threads(1)
        panel = ck.build_panel(obs, self.config)
        if panel.stations != info["stations"]:
            raise cks.CandidateKShadowError(f"stations changed since the refit: {panel.stations} vs {info['stations']}")
        pos = panel.position(cutoff)
        samples = ck.build_samples(panel, self.config)
        lookup = samples.subset(samples.origin <= pos)
        evaluation = samples.subset(samples.origin == pos)
        if len(evaluation) != len(panel.stations):
            raise cks.CandidateKShadowError("not every station has complete inputs at the data_cutoff")
        data = ck._tensors(evaluation)
        members, seasonal, gates = [], [], []
        with torch.no_grad():
            for model in models:
                model.build_lookup(lookup, len(panel.stations))
                out, parts = model(data["past"], data["seasonal"], data["calendar"], data["station"], data["day_of_week"],
                                   origin=data["origin"], scale=data["scale"], return_parts=True)
                members.append(out.clamp(min=0).numpy())
                seasonal.append(parts["seasonal"].clamp(min=0).numpy())
                gates.append(parts["gate"].numpy())
        scale = evaluation.scale[:, None]
        stacked = np.stack(members)
        prediction = np.median(stacked, axis=0) * scale
        seasonal_prediction = np.median(np.stack(seasonal), axis=0) * scale
        gate = np.mean(np.stack(gates), axis=0)
        spread = (stacked.max(axis=0) - stacked.min(axis=0)) * scale
        rows = []
        for h in range(self.config.horizons):
            rows.append(pd.DataFrame({
                "station_id": [panel.stations[s] for s in evaluation.station],
                "target_at": panel.index[evaluation.origin + h + 1],
                "horizon_minutes": 15 * (h + 1),
                "k_prediction": prediction[:, h],
                "k_seasonal_prediction": seasonal_prediction[:, h],
                "k_seasonal_weight": gate[:, h],
                "k_member_spread": spread[:, h],
            }))
        frame = pd.concat(rows, ignore_index=True).sort_values(["station_id", "horizon_minutes"]).reset_index(drop=True)
        return frame, {key: value for key, value in info.items() if key != "stations"} | {"stations": len(info["stations"])}

    # ------------------------------------------------------------------ refits

    def _models_for(self, obs: pd.DataFrame, refit: pd.Timestamp) -> tuple[list[ck.StagedK], dict[str, Any]]:
        key = cks._ts(refit)
        if key in self._memory:
            models, info = self._memory[key]
            return models, {**info, "trained": False, "cache": "memory"}
        train_obs = obs[obs["observed_at"] <= refit]
        data_sha = _frame_sha(train_obs)
        loaded = self._load(refit)
        if loaded is not None:
            models, manifest = loaded
            info = {"refit_at": key, "trained": False, "cache": "disk", "train_rows": manifest["train_rows"], "stations": manifest["stations"],
                    "train_data_sha256": manifest["train_data_sha256"], "current_data_sha256": data_sha, "weights_sha256": manifest["weights_sha256"]}
        else:
            models, info = self._train(train_obs, refit, data_sha)
        self._memory[key] = (models, info)
        return models, info

    def _train(self, train_obs: pd.DataFrame, refit: pd.Timestamp, data_sha: str) -> tuple[list[ck.StagedK], dict[str, Any]]:
        started = time.perf_counter()
        torch.set_num_threads(1)
        panel = ck.build_panel(train_obs, self.config)
        if refit not in panel.index:
            raise cks.CandidateKShadowError(f"no observations at the refit time {cks._ts(refit)}")
        samples = ck.build_samples(panel, self.config)
        refit_pos = panel.position(refit)
        mask = (samples.origin + self.config.horizons <= refit_pos) & cks.training_origin_mask(panel.index[samples.origin])
        train = samples.subset(mask)
        lookup = samples.subset(samples.origin <= refit_pos)
        models = []
        for seed in self.seeds:
            self._log(f"training refit {cks._ts(refit)} seed {seed} on {len(train)} samples")
            models.append(ck.train_staged(train, self.config, len(panel.stations), seed, self.local, gate_epochs=self.gate_epochs,
                                          synthetic_prob=self.synthetic_prob, realized_error=True, lookup_samples=lookup))
        weights = [_state_sha(model) for model in models]
        info = {"refit_at": cks._ts(refit), "trained": True, "cache": "trained", "train_rows": int(len(train)), "stations": list(panel.stations),
                "train_data_sha256": data_sha, "current_data_sha256": data_sha, "weights_sha256": weights,
                "train_seconds": round(time.perf_counter() - started, 1)}
        self._save(refit, models, info)
        return models, info

    def _refit_dir(self, refit: pd.Timestamp) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self.spec_hash[:16] / f"{refit:%Y%m%dT%H%MZ}"

    def _save(self, refit: pd.Timestamp, models: list[ck.StagedK], info: dict[str, Any]) -> None:
        directory = self._refit_dir(refit)
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        for seed, model in zip(self.seeds, models):
            torch.save(model.state_dict(), directory / f"seed{seed}.pt")
        manifest = {"spec_hash": self.spec_hash, "refit_at": info["refit_at"], "seeds": list(self.seeds), "stations": info["stations"],
                    "train_rows": info["train_rows"], "train_data_sha256": info["train_data_sha256"], "weights_sha256": info["weights_sha256"],
                    "torch_version": torch.__version__, "created_at": pd.Timestamp.now(tz="UTC").isoformat()}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def _load(self, refit: pd.Timestamp) -> tuple[list[ck.StagedK], dict[str, Any]] | None:
        directory = self._refit_dir(refit)
        if directory is None or not (directory / "manifest.json").exists():
            return None
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("spec_hash") != self.spec_hash or manifest.get("seeds") != list(self.seeds) or manifest.get("refit_at") != cks._ts(refit):
            return None
        models = []
        for seed in self.seeds:
            model = self._empty_model(len(manifest["stations"]))
            model.load_state_dict(torch.load(directory / f"seed{seed}.pt"))
            model.eval()
            models.append(model)
        if [_state_sha(model) for model in models] != manifest["weights_sha256"]:
            return None
        return models, manifest

    def _empty_model(self, n_stations: int) -> ck.StagedK:
        seasonal = ck.CandidateKNet(replace(self.config, architecture="tide", augmentation_prob=0.0), n_stations)
        return ck.StagedK(self.config, n_stations, seasonal, self.local, realized_error=True)


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _frame_sha(frame: pd.DataFrame) -> str:
    canonical = frame[["station_id", "observed_at", "demand"]].sort_values(["observed_at", "station_id"])
    return hashlib.sha256(canonical.to_csv(index=False).encode()).hexdigest()


def _state_sha(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    for name, tensor in sorted(model.state_dict().items()):
        buffer.write(name.encode())
        buffer.write(tensor.detach().cpu().numpy().tobytes())
    return hashlib.sha256(buffer.getvalue()).hexdigest()
