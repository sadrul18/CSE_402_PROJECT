"""Independent numerical checks for double-diode fitting and paper comparisons."""
from pathlib import Path

import numpy as np
import pytest

from current import solve_currents, solve_root
from ddm_study import average_rmse_history, paper_comparison, published_parameters
from experiments import monte_carlo, summarize
from fit import fit_parameters
from model import current_derivative, equation, initial_parameters, load_data, thermal_voltage, validate_parameters
from numerics import jacobian


def problem():
    voltage, measured = load_data(Path(__file__).parent / "data/rtc_france.csv")
    vt = thermal_voltage(306.15)
    theta, bounds, scales = initial_parameters(voltage, measured, vt, model="ddm")
    return voltage, measured, vt, theta, bounds, scales


def test_ddm_equation_and_derivative():
    _, _, vt, theta, _, _ = problem()
    i, v = 0.4, 0.5
    u = v + i * theta[2]
    expected = theta[0] - theta[1] * np.expm1(u / (theta[4] * vt))
    expected -= theta[5] * np.expm1(u / (theta[6] * vt)) + u / theta[3] + i
    assert equation(i, v, theta, vt) == pytest.approx(expected)
    h = 1e-6
    numerical = (equation(i+h, v, theta, vt) - equation(i-h, v, theta, vt)) / (2*h)
    assert current_derivative(i, v, theta, vt) == pytest.approx(numerical, rel=1e-8)
    assert current_derivative(i, v, theta, vt) < 0


def test_ddm_reduces_to_sdm_for_equal_ideality():
    _, _, vt, theta, _, _ = problem()
    theta[6] = theta[4]
    sdm = theta[:5].copy()
    sdm[1] += theta[5]
    for ns in (1, 36):
        for voltage in (-0.2, 0.0, 0.55):
            assert equation(0.2, voltage, theta, vt, ns) == pytest.approx(
                equation(0.2, voltage, sdm, vt, ns), abs=1e-12)
            assert current_derivative(0.2, voltage, theta, vt, ns) == pytest.approx(
                current_derivative(0.2, voltage, sdm, vt, ns))


@pytest.mark.parametrize("method", ("newton", "bisection", "hybrid"))
def test_ddm_roots_and_negative_current(method):
    voltage, _, vt, theta, _, _ = problem()
    voltage = voltage[::-1].copy()
    answer = solve_currents(voltage, theta, vt, method=method)
    assert all(root["converged"] for root in answer["roots"])
    assert answer["current"][0] < 0
    assert max(abs(equation(i, v, theta, vt)) for v, i in zip(voltage, answer["current"])) < 1e-7
    assert answer["function_evaluations"] == sum(r["function_evaluations"] for r in answer["roots"])


def test_ddm_implicit_jacobian():
    voltage, measured, vt, theta, bounds, scales = problem()
    numerical = jacobian(theta, voltage, measured, vt, bounds, scales)
    currents = solve_currents(voltage, theta, vt)["current"]
    u = voltage + currents * theta[2]
    e1, e2 = np.exp(u/(theta[4]*vt)), np.exp(u/(theta[6]*vt))
    conductance = theta[1]*e1/(theta[4]*vt) + theta[5]*e2/(theta[6]*vt) + 1/theta[3]
    partials = np.column_stack((
        np.ones_like(u), -(e1-1), -currents*conductance, u/theta[3]**2,
        theta[1]*e1*u/(theta[4]**2*vt), -(e2-1),
        theta[5]*e2*u/(theta[6]**2*vt),
    ))
    analytic = partials / (1 + theta[2]*conductance)[:, None]
    np.testing.assert_allclose(numerical * scales, analytic * scales, rtol=2e-4, atol=1e-7)


@pytest.mark.parametrize("method", ("gauss_newton", "lm"))
def test_ddm_rtc_fit_regression(method):
    voltage, measured, vt, theta, bounds, scales = problem()
    original = theta.copy()
    result = fit_parameters(voltage, measured, vt, theta, bounds, scales, method=method, max_iter=500)
    assert result["converged"], result["reason"]
    assert np.sqrt(np.mean(result["residuals"]**2)) < 0.00068
    assert result["theta"][6] == pytest.approx(5.0, abs=1e-7)
    assert np.all(np.diff(result["history"]) <= 0)
    np.testing.assert_array_equal(theta, original)
    validate_parameters(result["theta"], bounds)


def test_ddm_failures_are_explicit():
    voltage, measured, vt, theta, bounds, scales = problem()
    result = fit_parameters(voltage, measured, vt, theta, bounds, scales, max_iter=1)
    assert not result["converged"]
    assert result["reason"] == "maximum iterations reached"
    assert result["residuals"] is not None
    bad = theta.copy()
    bad[5] = 0
    with pytest.raises(ValueError):
        validate_parameters(bad)
    with pytest.raises(ValueError):
        validate_parameters(theta, (bounds[0][:5], bounds[1][:5]))
    with pytest.raises(ValueError):
        initial_parameters(voltage, measured, vt, model="unknown")


def test_ddm_exact_data_fit_and_noise_summary():
    voltage, _, vt, theta, bounds, scales = problem()
    exact = solve_currents(voltage, theta, vt)["current"]
    result = monte_carlo(voltage, exact, vt, theta, bounds, scales,
                         noise_percent=0, repeats=2, seed=42, reference_theta=theta)
    assert len(result["parameter_names"]) == 7
    stats = summarize(result)
    assert stats["successes"] == 2
    assert stats["parameter_mean"].shape == (7,)
    assert stats["mean_rmse"] < 1e-8


def test_mean_history_is_mean_rmse_and_retains_failures():
    trials = [
        {"history": [16.0, 4.0], "converged": True},
        {"history": [4.0], "converged": False},
        {"history": [], "converged": False},
    ]
    stats = average_rmse_history(trials, 3)
    np.testing.assert_array_equal(stats["mean_rmse"], [3, 2, 2, 2])
    assert stats["valid_runs"] == 2 and stats["invalid_runs"] == 1
    assert average_rmse_history([{"history": []}], 3)["mean_rmse"] is None


def test_paper_reference_units_and_error_identities():
    voltage, measured, vt, _, _, _ = problem()
    parameters = published_parameters()
    assert len(parameters) == 10
    assert parameters[0]["theta"][5] == pytest.approx(121.55410e-6)
    curves = paper_comparison(voltage, measured, vt)
    for row in curves:
        np.testing.assert_allclose(row["power"], voltage * row["current"])
        np.testing.assert_allclose(row["absolute_power_error"], np.abs(voltage) * row["absolute_current_error"])
        assert row["recomputed_rmse"] < 0.002


def test_newton_accepts_verified_tiny_final_step():
    vt = thermal_voltage(306.15)
    theta = np.array([2.0, 1e-6, 0.0, 50.0, 1.5, 1e-4, 3.0])
    # Rs=0 and V=0 give f(I)=2-I. The step is smaller than the relative
    # step threshold, but the initial residual exceeds tolerance.
    answer = solve_root(0.0, theta, vt, initial_guess=2.0+1.5e-8,
                        method="newton", tol=1e-8)
    assert answer["converged"]
    assert answer["iterations"] == 1
    assert answer["function_evaluations"] == 2
    assert abs(equation(answer["root"], 0.0, theta, vt)) <= 1e-8