"""Tests for the complete PV parameter-estimation pipeline."""

import json
from pathlib import Path

import numpy as np
import pytest

import current as current_module
import numerics as numerics_module
import run as run_module
from current import solve_currents, solve_root
from experiments import (
    add_noise,
    add_relative_noise,
    compare_current_solvers,
    compare_fit_methods,
    condition_diagnostics,
    monte_carlo,
    plot_convergence,
    plot_fit,
    plot_noise_summary,
    rmse,
    save_results,
    save_table,
    summarize,
)
from fit import fit_parameters
from model import (
    current_derivative,
    equation,
    initial_parameters,
    load_data,
    thermal_voltage,
    validate_data,
    validate_parameters,
)
from numerics import EvaluationCounter, jacobian, residuals, solve_linear


def sample_problem() -> tuple:
    """Return a small exact I-V curve used by several tests."""
    voltage = np.linspace(0.0, 0.55, 12)
    theta = np.array([0.76079, 0.31069e-6, 0.03655, 52.88991, 1.47727])
    vt = thermal_voltage(306.15)
    current = solve_currents(voltage, theta, vt)["current"]
    lower = np.array([0.0, 1e-12, 0.0, 1.0, 1.0])
    upper = np.array([1.0, 1e-6, 0.5, 100.0, 2.0])
    scales = np.array([1.0, 1e-6, 0.1, 50.0, 1.0])
    return voltage, current, theta, vt, (lower, upper), scales


def analytic_current_jacobian(
    voltage: np.ndarray, theta: np.ndarray, vt: float, ns: int = 1
) -> np.ndarray:
    """Differentiate the implicit single-diode equation with respect to theta."""
    current = solve_currents(voltage, theta, vt, ns)["current"]
    _, i0, rs, rsh, ideality = theta
    table = np.empty((voltage.size, theta.size))

    for row, (point_voltage, point_current) in enumerate(zip(voltage, current)):
        diode_voltage = point_voltage + point_current * rs
        exponent = diode_voltage / (ideality * ns * vt)
        exponential = np.exp(exponent)
        equation_current_slope = (
            -i0 * exponential * rs / (ideality * ns * vt) - rs / rsh - 1.0
        )
        equation_parameter_slopes = np.array(
            [
                1.0,
                -np.expm1(exponent),
                -i0 * exponential * point_current / (ideality * ns * vt)
                - point_current / rsh,
                diode_voltage / rsh**2,
                i0
                * exponential
                * diode_voltage
                / (ideality**2 * ns * vt),
            ]
        )
        table[row] = -equation_parameter_slopes / equation_current_slope

    return table


def test_load_and_validate_data(tmp_path: Path) -> None:
    """The CSV loader separates the two named columns and rejects bad data."""
    csv_path = tmp_path / "small.csv"
    csv_path.write_text("voltage_v,current_a\n0.0,0.8\n0.5,0.2\n")
    voltage, current = load_data(csv_path)
    assert np.array_equal(voltage, [0.0, 0.5])
    assert np.array_equal(current, [0.8, 0.2])
    with pytest.raises(ValueError):
        validate_data(np.array([0.0]), np.array([0.8, 0.2]))


def test_csv_header_and_single_row_validation(tmp_path: Path) -> None:
    """The loader accepts one data row and rejects malformed CSV structure."""
    single = tmp_path / "single.csv"
    single.write_text("voltage_v,current_a\n0.25,0.7\n", encoding="utf-8")
    voltage, current = load_data(single)
    assert np.array_equal(voltage, [0.25])
    assert np.array_equal(current, [0.7])

    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text("voltage,current\n0.25,0.7\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV header"):
        load_data(bad_header)

    bad_row = tmp_path / "bad_row.csv"
    bad_row.write_text("voltage_v,current_a\n0.25\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2"):
        load_data(bad_row)


def test_parameter_bounds_must_be_physical() -> None:
    """Bounds cannot allow values outside the physical parameter domain."""
    _, _, theta, _, bounds, _ = sample_problem()
    lower, upper = (value.copy() for value in bounds)
    lower[1] = 0.0
    with pytest.raises(ValueError, match="physically impossible"):
        validate_parameters(theta, (lower, upper))


def test_model_equation_and_derivative() -> None:
    """A solved current balances the equation and the derivative matches a small difference."""
    voltage, current, theta, vt, _, _ = sample_problem()
    assert abs(equation(float(current[4]), float(voltage[4]), theta, vt)) < 1e-8
    step = 1e-6
    numerical_slope = (
        equation(float(current[4] + step), float(voltage[4]), theta, vt)
        - equation(float(current[4] - step), float(voltage[4]), theta, vt)
    ) / (2 * step)
    assert current_derivative(
        float(current[4]), float(voltage[4]), theta, vt
    ) == pytest.approx(numerical_slope, rel=1e-6)


@pytest.mark.parametrize(
    ("current", "voltage", "vt", "ns"),
    [
        (np.inf, 0.2, 0.026, 1),
        (0.7, np.nan, 0.026, 1),
        (0.7, 0.2, 0.0, 1),
        (0.7, 0.2, 0.026, 0),
    ],
)
def test_current_derivative_validates_inputs(
    current: float, voltage: float, vt: float, ns: int
) -> None:
    """The derivative rejects the same invalid direct inputs as the equation."""
    _, _, theta, _, _, _ = sample_problem()
    with pytest.raises(ValueError):
        current_derivative(current, voltage, theta, vt, ns)


def test_starting_parameters() -> None:
    """Starting values have five entries and lie inside their limits."""
    voltage, current, _, vt, _, _ = sample_problem()
    theta0, bounds, scales = initial_parameters(voltage, current, vt)
    assert theta0.shape == bounds[0].shape == bounds[1].shape == scales.shape == (5,)
    assert np.all(theta0 >= bounds[0]) and np.all(theta0 <= bounds[1])


def test_all_root_methods() -> None:
    """Newton, bisection, and hybrid all solve the same PV current."""
    voltage, _, theta, vt, _, _ = sample_problem()
    answers = [
        solve_root(float(voltage[5]), theta, vt, initial_guess=0.7, method="newton"),
        solve_root(
            float(voltage[5]), theta, vt, bracket=(-0.5, 1.5), method="bisection"
        ),
        solve_root(
            float(voltage[5]),
            theta,
            vt,
            initial_guess=0.7,
            bracket=(-0.5, 1.5),
            method="hybrid",
        ),
    ]
    assert all(answer["converged"] for answer in answers)
    assert (
        max(answer["root"] for answer in answers)
        - min(answer["root"] for answer in answers)
        < 1e-7
    )


def test_root_solver_failure_diagnostics() -> None:
    """Endpoint, bad-bracket, iteration-limit, and overflow paths are explicit."""
    _, _, theta, vt, _, _ = sample_problem()
    endpoint_theta = np.array([0.0, 1e-7, 0.1, 50.0, 1.5])
    endpoint = solve_root(
        0.0, endpoint_theta, vt, bracket=(0.0, 1.0), method="bisection"
    )
    bad_bracket = solve_root(
        0.25, theta, vt, bracket=(1.0, 2.0), method="bisection"
    )
    limited = solve_root(
        0.25,
        theta,
        vt,
        bracket=(-0.5, 1.5),
        method="bisection",
        tol=1e-15,
        max_iter=1,
    )
    overflow = solve_root(
        100.0, theta, vt, bracket=(0.0, 1.0), method="bisection"
    )

    assert endpoint["converged"] and endpoint["reason"] == "lower endpoint is the root"
    assert endpoint["iterations"] == 0 and endpoint["function_evaluations"] == 2
    assert not bad_bracket["converged"]
    assert bad_bracket["reason"] == "bracket does not contain a sign change"
    assert bad_bracket["function_evaluations"] == 2
    assert not limited["converged"] and limited["reason"] == "maximum iterations reached"
    assert limited["iterations"] == 1 and limited["function_evaluations"] == 3
    assert not overflow["converged"] and overflow["reason"] == "PV equation overflowed"
    assert overflow["function_evaluations"] == 1


def test_hybrid_rejects_a_worsening_newton_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-bracket Newton proposal falls back when its residual gets larger."""
    _, _, theta, vt, _, _ = sample_problem()
    evaluated_currents = []

    def fake_equation(current: float, *args: object) -> float:
        evaluated_currents.append(current)
        return 0.75 - current

    monkeypatch.setattr(current_module, "equation", fake_equation)
    monkeypatch.setattr(current_module, "current_derivative", lambda *args: 1.0)
    result = solve_root(
        0.25,
        theta,
        vt,
        initial_guess=0.7,
        bracket=(0.0, 1.0),
        method="hybrid",
        tol=1e-12,
        max_iter=1,
    )

    assert not result["converged"]
    assert evaluated_currents == pytest.approx([0.0, 1.0, 0.7, 0.65, 0.5])
    assert result["function_evaluations"] == 5


def test_narrow_bracket_result_is_accurate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bracket-width convergence returns a current inside the requested tolerance."""
    _, _, theta, vt, _, _ = sample_problem()
    expected_root = 0.25
    monkeypatch.setattr(
        current_module,
        "equation",
        lambda current, *args: 1000.0 * (expected_root - current),
    )
    result = solve_root(
        0.25,
        theta,
        vt,
        bracket=(expected_root - 1.1e-8, expected_root + 1.4e-8),
        method="bisection",
        tol=1e-8,
        max_iter=10,
    )

    assert result["converged"] and result["reason"] == "bracket became small"
    assert abs(result["root"] - expected_root) <= 1e-8


@pytest.mark.parametrize("method", ["newton", "bisection", "hybrid"])
def test_warm_start_preserves_curve_and_reduces_work(method: str) -> None:
    """Warm starts keep the curve while using no more equation calls."""
    voltage, _, theta, vt, _, _ = sample_problem()
    warm = solve_currents(voltage, theta, vt, method=method, warm_start=True)
    cold = solve_currents(voltage, theta, vt, method=method, warm_start=False)

    assert all(root["converged"] for root in warm["roots"])
    assert all(root["converged"] for root in cold["roots"])
    assert np.allclose(warm["current"], cold["current"], atol=2e-8, rtol=0.0)
    assert warm["function_evaluations"] <= cold["function_evaluations"]


def test_failed_curve_point_is_nan() -> None:
    """A numerical point failure stays aligned with a NaN curve entry."""
    _, _, theta, vt, _, _ = sample_problem()
    result = solve_currents(np.array([100.0]), theta, vt, method="newton")

    assert np.isnan(result["current"][0])
    assert not result["roots"][0]["converged"]
    assert result["roots"][0]["root"] is None
    assert result["function_evaluations"] == 1


def test_current_curve_keeps_input_order() -> None:
    """Calculated currents line up with unsorted and repeated voltages."""
    _, _, theta, vt, _, _ = sample_problem()
    voltage = np.array([0.4, 0.0, 0.4])
    result = solve_currents(voltage, theta, vt)
    assert np.all(np.isfinite(result["current"]))
    assert result["current"][0] == pytest.approx(result["current"][2])
    assert len(result["roots"]) == 3


def test_residuals_and_jacobian() -> None:
    """Exact data has tiny errors and the Jacobian has 12 rows by 5 columns."""
    voltage, current, theta, vt, bounds, scales = sample_problem()
    errors = residuals(theta, voltage, current, vt)
    table = jacobian(theta, voltage, current, vt, bounds, scales)
    assert np.max(np.abs(errors)) < 1e-8
    assert table.shape == (12, 5)
    assert np.all(np.isfinite(table))


def test_jacobian_matches_implicit_derivative() -> None:
    """Central finite differences match an independent implicit derivative."""
    voltage, current, theta, vt, bounds, scales = sample_problem()
    originals = tuple(
        value.copy() for value in (voltage, current, theta, *bounds, scales)
    )
    expected = analytic_current_jacobian(voltage, theta, vt)
    actual = jacobian(theta, voltage, current, vt, bounds, scales)
    column_scales = np.max(np.abs(expected), axis=0)

    normalized_error = np.abs(actual - expected) / column_scales
    assert np.max(normalized_error) < 5e-7
    for value, original in zip((voltage, current, theta, *bounds, scales), originals):
        assert np.array_equal(value, original)


@pytest.mark.parametrize("bound_side", ["lower", "upper"])
def test_jacobian_at_parameter_bounds(bound_side: str) -> None:
    """One-sided finite differences remain accurate at either parameter bound."""
    voltage, current, theta, vt, bounds, scales = sample_problem()
    lower, upper = (bound.copy() for bound in bounds)
    if bound_side == "lower":
        lower = theta.copy()
    else:
        upper = theta.copy()
    expected = analytic_current_jacobian(voltage, theta, vt)
    actual = jacobian(theta, voltage, current, vt, (lower, upper), scales)
    column_scales = np.max(np.abs(expected), axis=0)

    normalized_error = np.abs(actual - expected) / column_scales
    assert np.max(normalized_error) < 5e-6


def test_jacobian_rejects_unrepresentable_step() -> None:
    """A step below floating-point resolution fails before evaluating residuals."""
    voltage, current, theta, vt, bounds, scales = sample_problem()
    counter = EvaluationCounter()

    with pytest.raises(ValueError, match="too small to perturb parameter"):
        jacobian(
            theta,
            voltage,
            current,
            vt,
            bounds,
            scales,
            relative_step=1e-18,
            counter=counter,
        )

    assert counter.residuals == 0


def test_residual_evaluation_counts() -> None:
    """The fit count includes its initial residual and all Jacobian residuals."""
    voltage, current, theta, vt, bounds, scales = sample_problem()
    counter = EvaluationCounter()
    jacobian(theta, voltage, current, vt, bounds, scales, counter=counter)
    assert counter.residuals == 11

    result = fit_parameters(
        voltage, current, vt, theta, bounds, scales, max_iter=1
    )
    assert result["converged"]
    assert result["reason"] == "gradient became small"
    assert result["evaluations"] == 12


def test_failed_residual_attempt_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed current curve still consumes one residual evaluation."""
    voltage, current, theta, vt, bounds, scales = sample_problem()
    failed_root = {
        "root": None,
        "converged": False,
        "reason": "forced failure",
        "iterations": 0,
        "function_evaluations": 1,
    }
    monkeypatch.setattr(
        numerics_module,
        "solve_currents",
        lambda *args, **kwargs: {
            "current": np.full(voltage.size, np.nan),
            "roots": [failed_root] * voltage.size,
            "function_evaluations": voltage.size,
        },
    )

    result = fit_parameters(voltage, current, vt, theta, bounds, scales)

    assert not result["converged"]
    assert result["evaluations"] == 1


def test_curve_function_evaluation_counts() -> None:
    """A curve point includes bracket-search calls in its equation-call total."""
    voltage, _, theta, vt, _, _ = sample_problem()
    point = np.array([voltage[5]])
    curve = solve_currents(point, theta, vt, method="bisection")
    direct = solve_root(
        float(point[0]),
        theta,
        vt,
        initial_guess=float(theta[0]),
        bracket=(float(theta[0]) - 1.0, float(theta[0]) + 1.0),
        method="bisection",
    )

    assert curve["roots"][0]["function_evaluations"] == (
        direct["function_evaluations"] + 2
    )
    assert curve["function_evaluations"] == curve["roots"][0][
        "function_evaluations"
    ]


def test_failed_bracket_search_counts_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exhausted bracket searches retain all 30 pairs of equation calls."""
    _, _, theta, vt, _, _ = sample_problem()
    monkeypatch.setattr(current_module, "equation", lambda *args: 1.0)

    result = solve_currents(np.array([0.25]), theta, vt, method="hybrid")

    assert not result["roots"][0]["converged"]
    assert result["roots"][0]["function_evaluations"] == 60
    assert result["function_evaluations"] == 60


def test_gaussian_elimination() -> None:
    """Row swapping solves a system whose first diagonal entry is zero."""
    matrix = np.array([[0.0, 2.0], [1.0, 3.0]])
    rhs = np.array([4.0, 7.0])
    assert np.allclose(solve_linear(matrix, rhs), [1.0, 2.0])
    with pytest.raises(RuntimeError):
        solve_linear(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="pivot_rtol"):
        solve_linear(matrix, rhs, pivot_rtol=0.0)
    with pytest.raises(ValueError, match="pivot_rtol"):
        solve_linear(matrix, rhs, pivot_rtol=np.inf)


def test_parameter_fit_reduces_error() -> None:
    """LM improves a nearby starting guess without changing that input array."""
    voltage, current, _, vt, bounds, scales = sample_problem()
    theta0 = np.array([0.75, 0.4e-6, 0.05, 50.0, 1.5])
    original = theta0.copy()
    result = fit_parameters(voltage, current, vt, theta0, bounds, scales, max_iter=30)
    assert result["converged"]
    assert result["history"][-1] < result["history"][0]
    assert result["function_evaluations"] > result["evaluations"]
    assert np.array_equal(theta0, original)


def test_gauss_newton_fits_rtc_data() -> None:
    """Scaled Gauss-Newton fits the measured RTC curve without a singular solve."""
    data_path = Path(__file__).parent / "data" / "rtc_france.csv"
    voltage, current = load_data(data_path)
    vt = thermal_voltage(306.15)
    theta0, bounds, scales = initial_parameters(voltage, current, vt)

    result = fit_parameters(
        voltage,
        current,
        vt,
        theta0,
        bounds,
        scales,
        method="gauss_newton",
        max_iter=100,
    )

    assert result["converged"], result["reason"]
    assert rmse(result["residuals"]) < 2e-3
    assert result["history"][-1] < result["history"][0]


def test_rtc_fit_from_multiple_initial_guesses() -> None:
    """Both optimizers reach the same measured-data minimum from varied starts."""
    data_path = Path(__file__).parent / "data" / "rtc_france.csv"
    voltage, current = load_data(data_path)
    vt = thermal_voltage(306.15)
    theta0, bounds, scales = initial_parameters(voltage, current, vt)
    starts = {
        "default": theta0,
        "near": np.array([0.75, 4e-7, 0.05, 40.0, 1.4]),
        "far": np.array([0.90, 9e-7, 0.30, 10.0, 1.9]),
        "alternate": np.array([0.70, 1e-7, 0.01, 90.0, 1.1]),
    }
    outcomes = {}
    failures = []

    for start_name, start in starts.items():
        for method in ("gauss_newton", "lm"):
            result = fit_parameters(
                voltage,
                current,
                vt,
                start,
                bounds,
                scales,
                method=method,
            )
            outcomes[(start_name, method)] = result
            if not result["converged"]:
                failures.append((start_name, method, result["reason"]))

    assert not failures, failures
    reference = outcomes[("default", "lm")]["theta"]
    for result in outcomes.values():
        assert rmse(result["residuals"]) < 7.6e-4
        assert np.max(np.abs(result["theta"] - reference) / scales) < 1e-5

    lm_history = outcomes[("default", "lm")]["history"]
    relative_improvements = [
        (previous - current_mse) / previous
        for previous, current_mse in zip(lm_history, lm_history[1:])
    ]
    assert all(improvement > 1e-8 for improvement in relative_improvements[:-1])
    assert relative_improvements[-1] <= 1e-8


def test_parameter_step_stopping_rule() -> None:
    """A fit stops when its accepted scaled parameter change is below tolerance."""
    voltage, current, _, vt, bounds, scales = sample_problem()
    theta0 = np.array([0.7608, 0.31e-6, 0.0366, 52.9, 1.477])
    tolerance = 1e-3
    result = fit_parameters(
        voltage,
        current,
        vt,
        theta0,
        bounds,
        scales,
        method="gauss_newton",
        tol=tolerance,
    )

    assert result["converged"]
    assert result["reason"] == "parameter change became small"
    assert result["iterations"] == 1
    assert np.max(np.abs(result["theta"] - theta0) / scales) <= tolerance


def test_error_and_noise_helpers() -> None:
    """RMSE and seeded noise produce known, repeatable results."""
    assert rmse(np.array([-3.0, 4.0])) == pytest.approx(np.sqrt(12.5))
    current = np.array([1.0, 2.0, 3.0])
    first = add_noise(current, 0.1, np.random.default_rng(42))
    second = add_noise(current, 0.1, np.random.default_rng(42))
    assert np.array_equal(first, second)
    assert np.array_equal(current, [1.0, 2.0, 3.0])

    relative = add_relative_noise(current, 1.0, np.random.default_rng(42))
    assert relative[0] - current[0] == pytest.approx((first[0] - current[0]) / 10)
    zero_current = np.array([0.0, 1.0])
    relative_zero = add_relative_noise(
        zero_current, 2.0, np.random.default_rng(3)
    )
    assert relative_zero[0] == 0.0


def test_monte_carlo_is_repeatable() -> None:
    """The same seed reproduces fitted parameters and residuals."""
    voltage, current, theta, vt, bounds, scales = sample_problem()
    first = monte_carlo(
        voltage,
        current,
        vt,
        theta,
        bounds,
        scales,
        noise_percent=0.1,
        repeats=2,
        seed=7,
        reference_theta=theta,
    )
    second = monte_carlo(
        voltage,
        current,
        vt,
        theta,
        bounds,
        scales,
        noise_percent=0.1,
        repeats=2,
        seed=7,
        reference_theta=theta,
    )

    assert first["noise_percent"] == 0.1
    for first_trial, second_trial in zip(first["trials"], second["trials"]):
        assert first_trial["converged"] == second_trial["converged"]
        assert np.array_equal(first_trial["theta"], second_trial["theta"])
        assert np.array_equal(first_trial["residuals"], second_trial["residuals"])

    legacy = monte_carlo(
        voltage,
        current,
        vt,
        theta,
        bounds,
        scales,
        sigma=0.0,
        repeats=1,
    )
    assert legacy["sigma"] == 0.0 and legacy["noise_percent"] is None


def test_summarize_retains_failures_and_handles_small_samples() -> None:
    """Summaries cover mixed, one-success, and zero-success trial sets."""
    theta = np.array([0.76, 3e-7, 0.04, 53.0, 1.48])
    success = {
        "converged": True,
        "reason": "found",
        "theta": theta,
        "residuals": np.array([0.001, -0.001]),
        "runtime_seconds": 0.1,
        "iterations": 5,
        "evaluations": 60,
        "function_evaluations": 1200,
    }
    failure = {"converged": False, "reason": "forced failure"}
    mixed = summarize(
        {"trials": [success, failure], "reference_theta": theta.copy()}
    )
    assert mixed["successes"] == 1 and mixed["failures"] == 1
    assert mixed["failure_rate"] == 0.5
    assert mixed["failure_reasons"] == ["forced failure"]
    assert np.array_equal(mixed["parameter_std"], np.zeros(5))
    assert np.array_equal(
        mixed["parameter_mean_absolute_relative_error_percent"], np.zeros(5)
    )

    failed = summarize({"trials": [failure]})
    assert failed["successes"] == 0 and failed["failure_rate"] == 1.0
    assert failed["mean_rmse"] is None and failed["parameter_mean"] is None
    failed_rows = run_module.noise_summary_rows(
        {"studies": [{"noise_percent": 2.0, "summary": failed}]}
    )
    assert failed_rows[0]["Iph_mean"] is None


def test_experiment_exports_and_plots(tmp_path: Path) -> None:
    """Experiment JSON, CSV, and plots create strict nonempty outputs."""
    voltage, current, theta, vt, bounds, scales = sample_problem()
    result = monte_carlo(
        voltage,
        current,
        vt,
        theta,
        bounds,
        scales,
        noise_percent=0.1,
        repeats=1,
        seed=2,
        reference_theta=theta,
    )
    json_path = tmp_path / "nested" / "result.json"
    save_results(result, json_path, {"temperature_k": np.float64(306.15)})
    saved_text = json_path.read_text(encoding="utf-8")
    saved = json.loads(saved_text)
    assert "NaN" not in saved_text
    assert saved["summary"]["successes"] == 1

    csv_path = tmp_path / "nested" / "table.csv"
    save_table([{"method": "lm", "rmse": 0.001}], csv_path)
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "method,rmse"

    fit_comparison = compare_fit_methods(
        voltage,
        current,
        vt,
        theta,
        bounds,
        scales,
        root_methods=("hybrid",),
        fit_methods=("lm",),
    )
    current_comparison = compare_current_solvers(voltage, theta, vt)
    assert len(fit_comparison) == 1 and fit_comparison[0]["converged"]
    assert len(current_comparison) == 4

    convergence_path = tmp_path / "plots" / "convergence.png"
    plot_convergence(fit_comparison, convergence_path)
    study = {
        "studies": [
            {"noise_percent": 0.1, "summary": summarize(result)},
            {"noise_percent": 0.5, "summary": summarize(result)},
        ]
    }
    noise_path = tmp_path / "plots" / "noise.png"
    plot_noise_summary(study, noise_path)
    assert convergence_path.stat().st_size > 0
    assert noise_path.stat().st_size > 0


def test_condition_diagnostics_are_finite() -> None:
    """Scaled Jacobian condition diagnostics are finite and internally consistent."""
    voltage, current, theta, vt, bounds, scales = sample_problem()
    result = condition_diagnostics(theta, voltage, current, vt, bounds, scales)
    assert np.all(result["scaled_singular_values"] > 0)
    assert np.isfinite(result["scaled_jacobian_condition"])
    assert result["scaled_normal_condition"] == pytest.approx(
        result["scaled_jacobian_condition"] ** 2
    )


def test_failed_baseline_returns_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command-line runner signals a failed baseline through its exit status."""
    failed_fit = {
        "theta": np.ones(5),
        "residuals": None,
        "converged": False,
        "reason": "forced failure",
        "iterations": 0,
        "evaluations": 1,
        "function_evaluations": 1,
        "history": [],
    }
    monkeypatch.setattr(
        run_module,
        "run_baseline",
        lambda *args, **kwargs: {"fit": failed_fit, "rmse": None},
    )
    data_path = Path(__file__).parent / "data" / "rtc_france.csv"
    assert run_module.main(["baseline", "--dataset", str(data_path)]) == 1


def test_plot(tmp_path: Path) -> None:
    """The plot helper writes a nonempty image file."""
    voltage = np.array([0.0, 0.5])
    current = np.array([0.8, 0.2])
    path = tmp_path / "curve.png"
    plot_fit(voltage, current, current, path)
    assert path.exists() and path.stat().st_size > 0