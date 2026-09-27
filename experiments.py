import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit import fit_parameters
from model import validate_data, validate_parameters


def convert_json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot save {type(value).__name__} as JSON")


def rmse(errors: np.ndarray) -> float:
    if not isinstance(errors, np.ndarray) or errors.ndim != 1 or errors.size == 0:
        raise ValueError("errors must be a nonempty one-dimensional NumPy array")
    if not np.all(np.isfinite(errors)):
        raise ValueError("errors cannot contain NaN or infinity")
    return float(np.sqrt(np.mean(errors**2)))


def add_noise(
    current: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    if not isinstance(current, np.ndarray) or current.ndim != 1 or current.size == 0:
        raise ValueError("current must be a nonempty one-dimensional NumPy array")
    if not np.all(np.isfinite(current)) or not np.isfinite(sigma) or sigma < 0:
        raise ValueError("current must be finite and sigma cannot be negative")
    return current + rng.normal(0.0, sigma, size=current.size)


def monte_carlo(
    voltage: np.ndarray,
    current: np.ndarray,
    vt: float,
    theta0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    scales: np.ndarray,
    sigma: float,
    repeats: int = 100,
    seed: int = 42,
    ns: int = 1,
    root_method: str = "hybrid",
    fit_method: str = "lm",
) -> dict:
    validate_data(voltage, current)
    validate_parameters(theta0, bounds)
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    generator = np.random.default_rng(seed)
    trials = []
    for _ in range(repeats):
        noisy_current = add_noise(current, sigma, generator)
        trial = fit_parameters(
            voltage,
            noisy_current,
            vt,
            theta0,
            bounds,
            scales,
            ns,
            root_method,
            fit_method,
        )
        trials.append(trial)
    return {"seed": seed, "sigma": sigma, "trials": trials}


def summarize(result: dict) -> dict:
    trials = result.get("trials", [])
    if not trials:
        raise ValueError("result must contain at least one trial")
    successful = [trial for trial in trials if trial["converged"]]
    failures = len(trials) - len(successful)
    if not successful:
        return {
            "successes": 0,
            "failures": failures,
            "failure_rate": 1.0,
            "mean_rmse": None,
            "parameter_mean": None,
            "parameter_std": None,
        }

    parameters = np.array([trial["theta"] for trial in successful])
    errors = [rmse(trial["residuals"]) for trial in successful]
    parameter_std = None
    if len(successful) > 1:
        parameter_std = np.std(parameters, axis=0, ddof=1)
    return {
        "successes": len(successful),
        "failures": failures,
        "failure_rate": failures / len(trials),
        "mean_rmse": float(np.mean(errors)),
        "parameter_mean": np.mean(parameters, axis=0),
        "parameter_std": parameter_std,
    }


def save_results(result: dict, path: str | Path, metadata: dict) -> None:
    """Save the experiment, its summary, and dataset information as JSON."""
    output = {"metadata": metadata, "result": result, "summary": summarize(result)}

    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, allow_nan=False, default=convert_json_value)


def plot_fit(
    voltage: np.ndarray,
    measured_current: np.ndarray,
    calculated_current: np.ndarray,
    path: str | Path,
) -> None:
    validate_data(voltage, measured_current)
    validate_data(voltage, calculated_current)
    figure, axes = plt.subplots()
    axes.scatter(voltage, measured_current, label="Measured", color="black")
    order = np.argsort(voltage)
    axes.plot(voltage[order], calculated_current[order], label="Calculated")
    axes.set_xlabel("Voltage (V)")
    axes.set_ylabel("Current (A)")
    axes.set_title("PV current-voltage curve")
    axes.legend()
    axes.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
