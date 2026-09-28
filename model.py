"""SDM [Iph,I0,Rs,Rsh,n]; DDM [Iph,I01,Rs,Rsh,n1,I02,n2].

Resistances are effective device resistances; ns scales thermal voltage.
"""

import csv
from pathlib import Path

import numpy as np

SDM_PARAMETER_NAMES = ("Iph", "I0", "Rs", "Rsh", "n")
DDM_PARAMETER_NAMES = ("Iph", "I01", "Rs", "Rsh", "n1", "I02", "n2")


def parameter_names(theta: np.ndarray) -> tuple[str, ...]:
    """Return labels for a validated SDM or DDM parameter vector."""
    validate_parameters(theta)
    return SDM_PARAMETER_NAMES if theta.size == 5 else DDM_PARAMETER_NAMES


def validate_data(voltage: np.ndarray, current: np.ndarray) -> None:
    """Check that voltage and current are matching, usable one-dimensional arrays."""
    if not isinstance(voltage, np.ndarray) or not isinstance(current, np.ndarray):
        raise TypeError("voltage and current must be NumPy arrays")
    if voltage.ndim != 1 or current.ndim != 1:
        raise ValueError("voltage and current must be one-dimensional")
    if voltage.size == 0 or current.size == 0:
        raise ValueError("voltage and current cannot be empty")
    if voltage.size != current.size:
        raise ValueError("voltage and current must have the same length")
    if not np.all(np.isfinite(voltage)) or not np.all(np.isfinite(current)):
        raise ValueError("voltage and current cannot contain NaN or infinity")


def validate_parameters(
    theta: np.ndarray, bounds: tuple[np.ndarray, np.ndarray] | None = None
) -> None:
    """Check an SDM or DDM vector and optional matching physical limits."""
    if not isinstance(theta, np.ndarray) or theta.shape not in ((5,), (7,)):
        raise ValueError("theta must be a NumPy array containing five or seven values")
    if not np.all(np.isfinite(theta)):
        raise ValueError("theta cannot contain NaN or infinity")
    iph, i0, rs, rsh, ideality = theta[:5]
    if iph < 0 or i0 <= 0 or rs < 0 or rsh <= 0 or ideality <= 0:
        raise ValueError("theta contains a physically impossible value")
    if theta.size == 7 and (theta[5] <= 0 or theta[6] <= 0):
        raise ValueError("second diode parameters must be positive")
    if bounds is None:
        return
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError("bounds must be (lower, upper)")
    lower, upper = bounds
    if not isinstance(lower, np.ndarray) or not isinstance(upper, np.ndarray):
        raise TypeError("lower and upper bounds must be NumPy arrays")
    if lower.shape != theta.shape or upper.shape != theta.shape:
        raise ValueError("lower and upper bounds must match theta")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("bounds cannot contain NaN or infinity")
    if np.any(lower >= upper):
        raise ValueError("every lower bound must be less than its upper bound")
    if (
        lower[0] < 0
        or lower[1] <= 0
        or lower[2] < 0
        or lower[3] <= 0
        or lower[4] <= 0
        or (theta.size == 7 and (lower[5] <= 0 or lower[6] <= 0))
    ):
        raise ValueError("bounds include physically impossible parameter values")
    if np.any(theta < lower) or np.any(theta > upper):
        raise ValueError("theta lies outside the supplied bounds")


def equation(
    current: float, voltage: float, theta: np.ndarray, vt: float, ns: int = 1
) -> float:
    """Return the leftover error after putting a current into the PV equation.

    A result near zero means the current fits this voltage and parameter set.
    """
    validate_parameters(theta)
    if not np.isfinite(current) or not np.isfinite(voltage):
        raise ValueError("current and voltage must be finite")
    if not np.isfinite(vt) or vt <= 0 or not isinstance(ns, int) or ns < 1:
        raise ValueError("vt must be positive and ns must be a positive integer")
    iph, i0, rs, rsh, ideality = theta[:5]
    diode_voltage = voltage + current * rs
    exponent = diode_voltage / (ideality * ns * vt)
    with np.errstate(over="raise", invalid="raise"):
        try:
            answer = iph - i0 * np.expm1(exponent) - diode_voltage / rsh - current
            if theta.size == 7:
                answer -= theta[5] * np.expm1(diode_voltage / (theta[6] * ns * vt))
        except FloatingPointError as error:
            raise RuntimeError("PV equation overflowed") from error
    return float(answer)


def current_derivative(
    current: float, voltage: float, theta: np.ndarray, vt: float, ns: int = 1
) -> float:
    """Return the slope that Newton's method needs for its next current guess."""
    validate_parameters(theta)
    if not np.isfinite(current) or not np.isfinite(voltage):
        raise ValueError("current and voltage must be finite")
    if not np.isfinite(vt) or vt <= 0 or not isinstance(ns, int) or ns < 1:
        raise ValueError("vt must be positive and ns must be a positive integer")
    _, i0, rs, rsh, ideality = theta[:5]
    exponent = (voltage + current * rs) / (ideality * ns * vt)
    with np.errstate(over="raise", invalid="raise"):
        try:
            slope = -i0 * np.exp(exponent) * rs / (ideality * ns * vt) - rs / rsh - 1
            if theta.size == 7:
                slope -= theta[5] * np.exp((voltage + current * rs) / (theta[6] * ns * vt)) * rs / (theta[6] * ns * vt)
        except FloatingPointError as error:
            raise RuntimeError("PV derivative overflowed") from error
    return float(slope)


def thermal_voltage(temperature_k: float) -> float:
    """Calculate thermal voltage vt from temperature in kelvin."""
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    boltzmann_constant = 1.380649e-23
    electron_charge = 1.602176634e-19
    return boltzmann_constant * temperature_k / electron_charge


def load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load voltage_v and current_a from the CSV file named by path."""
    expected_header = ["voltage_v", "current_a"]
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("CSV cannot be empty") from error
        if header != expected_header:
            raise ValueError("CSV header must be voltage_v,current_a")

        rows = []
        for row_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"CSV row {row_number} must contain two values")
            try:
                rows.append((float(row[0]), float(row[1])))
            except ValueError as error:
                raise ValueError(f"CSV row {row_number} must be numeric") from error
    if not rows:
        raise ValueError("CSV must contain at least one data row")

    data = np.asarray(rows, dtype=np.float64)
    voltage = data[:, 0]
    current = data[:, 1]
    validate_data(voltage, current)
    return voltage, current


def initial_parameters(
    voltage: np.ndarray, current: np.ndarray, vt: float, ns: int = 1,
    model: str = "sdm",
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Return a reasonable starting point, limits, and step sizes for fitting."""
    validate_data(voltage, current)
    if not np.isfinite(vt) or vt <= 0:
        raise ValueError("vt must be finite and greater than zero")
    if not isinstance(ns, int) or ns < 1:
        raise ValueError("ns must be a positive integer")
    if model not in ("sdm", "ddm"):
        raise ValueError("model must be sdm or ddm")
    if ns != 1 or max(current) > 1:
        raise ValueError("initial_parameters supplies RTC bounds only (ns=1, I<=1 A)")
    if model == "ddm":
        # Qin et al. (2024), Table 1; saturation currents converted from uA.
        theta0 = np.array([max(current), 3e-7, 0.05, 60.0, 1.5, 1e-4, 3.0])
        lower = np.array([0.0, 1e-15, 0.0, 0.001, 0.5, 1e-15, 1.0])
        upper = np.array([1.0, 1e-3, 0.5, 100.0, 5.0, 1e-3, 5.0])
        scales = np.array([1.0, 1e-6, 0.1, 50.0, 1.0, 1e-4, 1.0])
        validate_parameters(theta0, (lower, upper))
        return theta0, (lower, upper), scales
    # These limits come from the French RTC single-cell dataset used here.
    theta0 = np.array([max(current), 5e-7, 0.1, 50.0, 1.5], dtype=np.float64)
    lower = np.array([0.0, 1e-12, 0.0, 1.0, 1.0], dtype=np.float64)
    upper = np.array([1.0, 1e-6, 0.5, 100.0, 2.0], dtype=np.float64)
    scales = np.array([1.0, 1e-6, 0.1, 50.0, 1.0], dtype=np.float64)
    bounds = (lower, upper)
    validate_parameters(theta0, bounds)
    return theta0, bounds, scales
