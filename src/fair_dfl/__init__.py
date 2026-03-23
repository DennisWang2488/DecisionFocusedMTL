"""Fair multi-objective decision-focused learning (fair_dfl)."""

__all__ = ["run_experiment"]


def __getattr__(name: str):
    if name == "run_experiment":
        from .runner import run_experiment

        return run_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
