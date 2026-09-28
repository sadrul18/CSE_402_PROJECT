"""Command-line runner for PV fitting, comparisons, and noise studies."""

import argparse
from pathlib import Path

import numpy as np

from current import solve_currents
from experiments import (
    PARAMETER_NAMES,
    compare_current_solvers,
    compare_fit_methods,
    condition_diagnostics,
    plot_convergence,
    plot_fit,
    plot_noise_summary,
    rmse,
    run_noise_study,
    save_results,
    save_table,
)
from fit import fit_parameters
from model import initial_parameters, load_data, thermal_voltage, validate_data, parameter_names


def run_baseline(
    voltage: np.ndarray,
    current: np.ndarray,
    vt: float,
    theta0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    scales: np.ndarray,
    ns: int = 1,
    root_method: str = "hybrid",
    fit_method: str = "lm",
    max_iter: int = 100,
) -> dict:
    """Run one fit and return its details and final RMSE."""
    validate_data(voltage, current)
    fit = fit_parameters(
        voltage,
        current,
        vt,
        theta0,
        bounds,
        scales,
        ns,
        root_method,
        fit_method,
        max_iter=max_iter,
    )
    final_rmse = rmse(fit["residuals"]) if fit["residuals"] is not None else None
    return {"fit": fit, "rmse": final_rmse}


def build_parser() -> argparse.ArgumentParser:
    """Build the documented command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="baseline",
        choices=("baseline", "compare", "noise", "all"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/rtc_france.csv"))
    parser.add_argument("--temperature", type=float, default=306.15)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument(
        "--root-method", choices=("newton", "bisection", "hybrid"), default="hybrid"
    )
    parser.add_argument(
        "--fit-method", choices=("gauss_newton", "lm"), default="lm"
    )
    parser.add_argument("--model", choices=("sdm", "ddm"), default="sdm")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="outer iteration limit (SDM: 100; DDM: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--noise-levels", nargs="+", type=float, default=(0.1, 0.5, 1.0, 2.0)
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/latest"))
    return parser


def experiment_metadata(args: argparse.Namespace) -> dict:
    """Return settings included in every saved experiment artifact."""
    return {
        "model": args.model,
        "max_iter": args.max_iter,
        "dataset": str(args.dataset),
        "temperature_k": args.temperature,
        "cells_in_series": args.cells,
        "seed": args.seed,
        "repeats": args.repeats,
        "noise_levels_percent": args.noise_levels,
        "root_method": args.root_method,
        "fit_method": args.fit_method,
    }


def print_baseline(result: dict) -> None:
    """Print the compact baseline result used by the original runner."""
    fit = result["fit"]
    print("Converged:", fit["converged"])
    print("Reason:", fit["reason"])
    print("Parameters", parameter_names(fit["theta"]), ":", fit["theta"])
    print("Iterations:", fit["iterations"])
    print("Residual evaluations:", fit["evaluations"])
    print("PV equation evaluations:", fit["function_evaluations"])
    print("RMSE (A):", result["rmse"])


def fit_comparison_rows(comparisons: list[dict]) -> list[dict]:
    """Flatten fit comparisons for CSV output."""
    rows = []
    for comparison in comparisons:
        row = {
            key: comparison[key]
            for key in (
                "root_method",
                "fit_method",
                "converged",
                "reason",
                "runtime_seconds",
                "iterations",
                "residual_evaluations",
                "function_evaluations",
                "rmse",
            )
        }
        row.update(
            {
                name: comparison["theta"][index]
                for index, name in enumerate(parameter_names(comparison["theta"]))
            }
        )
        rows.append(row)
    return rows


def current_comparison_rows(comparisons: list[dict]) -> list[dict]:
    """Remove current arrays from current-solver CSV rows."""
    return [
        {
            key: comparison[key]
            for key in (
                "configuration",
                "method",
                "warm_start",
                "converged",
                "failures",
                "runtime_seconds",
                "iterations",
                "function_evaluations",
            )
        }
        for comparison in comparisons
    ]


def noise_summary_rows(study: dict) -> list[dict]:
    """Flatten noise-level summaries for CSV output."""
    rows = []
    for item in study["studies"]:
        summary = item["summary"]
        parameter_mean = summary["parameter_mean"]
        parameter_std = summary["parameter_std"]
        parameter_cv = summary["parameter_cv_percent"]
        relative_error = summary["parameter_mean_absolute_relative_error_percent"]
        row = {
            "noise_percent": item["noise_percent"],
            "successes": summary["successes"],
            "failures": summary["failures"],
            "failure_rate": summary["failure_rate"],
            "mean_rmse": summary["mean_rmse"],
            "std_rmse": summary["std_rmse"],
            "mean_runtime_seconds": summary["mean_runtime_seconds"],
            "mean_iterations": summary["mean_iterations"],
            "mean_residual_evaluations": summary["mean_residual_evaluations"],
            "mean_function_evaluations": summary["mean_function_evaluations"],
        }
        for index, name in enumerate(study.get("parameter_names", PARAMETER_NAMES)):
            row[f"{name}_mean"] = None if parameter_mean is None else parameter_mean[index]
            row[f"{name}_std"] = None if parameter_std is None else parameter_std[index]
            row[f"{name}_cv_percent"] = None if parameter_cv is None else parameter_cv[index]
            row[f"{name}_mean_relative_error_percent"] = (
                None if relative_error is None else relative_error[index]
            )
        rows.append(row)
    return rows


def save_baseline_artifacts(
    result: dict,
    voltage: np.ndarray,
    current: np.ndarray,
    vt: float,
    ns: int,
    root_method: str,
    output_dir: Path,
    metadata: dict,
) -> None:
    """Save baseline JSON and measured-versus-fitted curve."""
    save_results(result, output_dir / "baseline.json", metadata)
    if result["fit"]["residuals"] is None:
        return
    calculated = solve_currents(
        voltage, result["fit"]["theta"], vt, ns, root_method
    )["current"]
    plot_fit(voltage, current, calculated, output_dir / "fit.png")


def run_comparisons(
    voltage: np.ndarray,
    current: np.ndarray,
    vt: float,
    theta0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    scales: np.ndarray,
    baseline: dict,
    args: argparse.Namespace,
    metadata: dict,
) -> dict:
    """Run, save, and plot all deterministic method comparisons."""
    fit_comparisons = compare_fit_methods(
        voltage, current, vt, theta0, bounds, scales, args.cells, max_iter=args.max_iter
    )
    current_comparisons = compare_current_solvers(
        voltage, baseline["fit"]["theta"], vt, args.cells
    )
    conditioning = condition_diagnostics(
        baseline["fit"]["theta"],
        voltage,
        current,
        vt,
        bounds,
        scales,
        args.cells,
        args.root_method,
    )
    result = {
        "fit_comparisons": fit_comparisons,
        "current_comparisons": current_comparisons,
        "condition_diagnostics": conditioning,
    }
    save_results(result, args.output_dir / "method_comparison.json", metadata)
    save_table(
        fit_comparison_rows(fit_comparisons),
        args.output_dir / "fit_method_comparison.csv",
    )
    save_table(
        current_comparison_rows(current_comparisons),
        args.output_dir / "current_solver_comparison.csv",
    )
    plot_convergence(fit_comparisons, args.output_dir / "convergence.png")
    return result


def run_noise(
    voltage: np.ndarray,
    current: np.ndarray,
    vt: float,
    theta0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    scales: np.ndarray,
    baseline: dict,
    args: argparse.Namespace,
    metadata: dict,
) -> dict:
    """Run, save, and plot the configured Monte Carlo protocol."""

    def show_progress(noise_percent: float, state: str) -> None:
        print(f"Noise {noise_percent:g}%: {state}", flush=True)

    study = run_noise_study(
        voltage,
        current,
        vt,
        theta0,
        bounds,
        scales,
        baseline["fit"]["theta"],
        tuple(args.noise_levels),
        args.repeats,
        args.seed,
        args.cells,
        args.root_method,
        args.fit_method,
        show_progress,
        max_iter=args.max_iter,
    )
    save_results(study, args.output_dir / "noise_study.json", metadata)
    save_table(noise_summary_rows(study), args.output_dir / "noise_summary.csv")
    plot_noise_summary(study, args.output_dir / "noise_summary.png")
    return study


def main(argv: list[str] | None = None) -> int:
    """Run the selected reproducible project workflow."""
    args = build_parser().parse_args(argv)
    if args.max_iter is None:
        args.max_iter = 500 if args.model == "ddm" else 100
    voltage, current = load_data(args.dataset)
    vt = thermal_voltage(args.temperature)
    theta0, bounds, scales = initial_parameters(voltage, current, vt, ns=args.cells, model=args.model)
    metadata = experiment_metadata(args)
    baseline = run_baseline(
        voltage,
        current,
        vt,
        theta0,
        bounds,
        scales,
        args.cells,
        args.root_method,
        args.fit_method,
        max_iter=args.max_iter,
    )
    print_baseline(baseline)
    if not baseline["fit"]["converged"]:
        return 1
    if args.command == "baseline":
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_baseline_artifacts(
        baseline,
        voltage,
        current,
        vt,
        args.cells,
        args.root_method,
        args.output_dir,
        metadata,
    )
    if args.command in ("compare", "all"):
        run_comparisons(
            voltage, current, vt, theta0, bounds, scales, baseline, args, metadata
        )
        print("Saved method comparisons to", args.output_dir)
    if args.command in ("noise", "all"):
        run_noise(
            voltage, current, vt, theta0, bounds, scales, baseline, args, metadata
        )
        print("Saved noise study to", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())