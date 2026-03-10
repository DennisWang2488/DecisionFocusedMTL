"""Learning-rate and alpha schedule helpers used across active methods."""

import math
from typing import Dict


def alpha_value(t: int, schedule_cfg: Dict[str, float]) -> float:
    sch_type = schedule_cfg.get("type", "sigmoid_decay")
    if sch_type == "constant":
        return float(schedule_cfg.get("value", 0.0))
    if sch_type == "sigmoid_decay":
        alpha_max = float(schedule_cfg.get("alpha_max", 1.0))
        alpha_min = float(schedule_cfg.get("alpha_min", 0.0))
        midpoint = float(schedule_cfg.get("midpoint", 100.0))
        temperature = float(schedule_cfg.get("temperature", 20.0))
        scaled = (float(t) - midpoint) / max(temperature, 1e-8)
        return alpha_min + (alpha_max - alpha_min) / (1.0 + math.exp(scaled))
    if sch_type == "paper_decay":
        c = float(schedule_cfg.get("c", 100.0))
        kappa = float(schedule_cfg.get("kappa", 1.0))
        temperature = float(schedule_cfg.get("temperature", 1.0))
        scaled = (float(t) - c) / max(temperature, 1e-8)
        base = 1.0 + math.exp(scaled)
        return float(base ** (-kappa))
    if sch_type == "poly_decay":
        alpha_max = float(schedule_cfg.get("alpha_max", 1.0))
        alpha_min = float(schedule_cfg.get("alpha_min", 0.0))
        power = float(schedule_cfg.get("power", 1.0))
        horizon = float(schedule_cfg.get("horizon", 200.0))
        ratio = max(0.0, 1.0 - min(float(t) / max(horizon, 1.0), 1.0))
        return alpha_min + (alpha_max - alpha_min) * (ratio**power)
    if sch_type == "inv_sqrt":
        alpha0 = float(schedule_cfg.get("alpha0", 1.0))
        alpha_min = float(schedule_cfg.get("alpha_min", 0.0))
        return max(alpha_min, alpha0 / math.sqrt(float(t) + 1.0))
    raise ValueError(f"Unsupported alpha schedule type: {sch_type}")


def lr_value(t: int, lr: float, lr_decay: float) -> float:
    return lr / (1.0 + lr_decay * float(t))
