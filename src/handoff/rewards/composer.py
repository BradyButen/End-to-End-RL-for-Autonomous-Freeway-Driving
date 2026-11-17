"""Config-driven reward composer (spec §9). Weighted sum of named terms; every term always computed."""

from __future__ import annotations

from handoff.rewards.terms import ALL_TERMS, TERM_NAMES, RewardTerm
from handoff.sim.state import EgoState, ObjectList


class RewardComposer:
    def __init__(self, terms: list[RewardTerm], weights: dict[str, float], terminal_mode: str = "add",
                 terminal_priority: list[str] | None = None):
        self.terms = terms
        self.names = [t.name for t in terms]
        unknown = set(weights) - set(self.names)
        if unknown:
            raise ValueError(f"Unknown reward terms in weights: {sorted(unknown)}")
        self.weights = {n: float(weights.get(n, 0.0)) for n in self.names}
        if terminal_mode not in ("add", "replace"):
            raise ValueError(terminal_mode)
        self.terminal_mode = terminal_mode
        self.terminal_names = [t.name for t in terms if t.terminal]
        self.terminal_priority = terminal_priority or self.terminal_names

    @classmethod
    def from_config(cls, reward_cfg: dict, ctx: dict) -> RewardComposer:
        terms = [T(ctx) for T in ALL_TERMS]
        return cls(terms, reward_cfg["weights"], reward_cfg.get("terminal_mode", "add"),
                   reward_cfg.get("terminal_priority"))

    def reset(self) -> None:
        for t in self.terms:
            t.reset()

    def compute(self, prev: EgoState, curr: EgoState, objects: ObjectList, info: dict,
                decision: dict | None = None) -> tuple[float, dict[str, float]]:
        """Returns (weighted total, raw values of all terms)."""
        raw = {t.name: float(t.compute(prev, curr, objects, info, decision)) for t in self.terms}
        return self.total(raw), raw

    def total(self, raw: dict[str, float]) -> float:
        if self.terminal_mode == "replace":
            for name in self.terminal_priority:
                if raw.get(name, 0.0) != 0.0:
                    return self.weights[name] * raw[name]
        return float(sum(self.weights[n] * raw[n] for n in self.names))

    def weighted(self, raw: dict[str, float]) -> dict[str, float]:
        """Per-term weighted contributions consistent with ``total`` (for logging)."""
        if self.terminal_mode == "replace":
            for name in self.terminal_priority:
                if raw.get(name, 0.0) != 0.0:
                    return {n: (self.weights[n] * raw[n] if n == name else 0.0) for n in self.names}
        return {n: self.weights[n] * raw[n] for n in self.names}


__all__ = ["RewardComposer", "TERM_NAMES"]
