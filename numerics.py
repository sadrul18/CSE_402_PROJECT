"""Residuals, numerical derivatives, and Gaussian elimination."""

from dataclasses import dataclass

import numpy as np

from current import solve_currents
from model import validate_data, validate_parameters


@dataclass
class EvaluationCounter:
    """Count attempted full residual-vector calculations during one fit."""

    residuals: int = 0
    function_evaluations: int = 0


def residuals(
    theta: np.ndarray,
    voltage: np.ndarray,
    measured_current: np.ndarray,
    vt: float,
    ns: int = 1,
    root_method: str = "hybrid",
    counter: EvaluationCounter | None = None,
) -> np.ndarray:
    """Return calculated current minus measured current at every voltage."""
    if counter is not None:
        counter.residuals += 1
    validate_data(voltage, measured_current)
    validate_parameters(theta)
    result = solve_currents(voltage, theta, vt, ns, root_method)
    if counter is not None:
        counter.function_evaluations += result["function_evaluations"]
    for index, root in enumerate(result["roots"]):
        if not root["converged"]:
            raise RuntimeError(f"current solve failed at row {index}: {root['reason']}")
    errors = result["current"] - measured_current
    if not np.all(np.isfinite(errors)):
        raise RuntimeError("residual calculation produced NaN or infinity")
    return errors


def jacobian(
    theta: np.ndarray,
    voltage: np.ndarray,
    measured_current: np.ndarray,
    vt: float,
    bounds: tuple[np.ndarray, np.ndarray],
    scales: np.ndarray,
    ns: int = 1,
    root_method: str = "hybrid",
    relative_step: float = 1e-6,
    counter: EvaluationCounter | None = None,
) -> np.ndarray:
    """Show how every residual changes when each PV parameter changes slightly.

    Rows represent measured points. Columns follow the SDM/DDM parameter order.
    """
    validate_data(voltage, measured_current)
    validate_parameters(theta, bounds)
    lower, upper = bounds
    if not isinstance(scales, np.ndarray) or scales.shape != theta.shape:
        raise ValueError("scales must have the same shape as theta")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scales must contain positive finite values")
    if not np.isfinite(relative_step) or relative_step <= 0:
        raise ValueError("relative_step must be positive")

    parameter_steps = relative_step * np.maximum(np.abs(theta), scales)
    forward_values = np.minimum(theta + parameter_steps, upper)
    backward_values = np.maximum(theta - parameter_steps, lower)
    movable = (forward_values != theta) | (backward_values != theta)
    if not np.all(movable):
        column = int(np.flatnonzero(~movable)[0])
        raise ValueError(
            f"relative_step is too small to perturb parameter {column}"
        )

    base_errors = residuals(
        theta, voltage, measured_current, vt, ns, root_method, counter
    )
    table = np.empty((voltage.size, theta.size), dtype=np.float64)

    for column in range(theta.size):
        forward_theta = theta.copy()
        backward_theta = theta.copy()
        forward_theta[column] = forward_values[column]
        backward_theta[column] = backward_values[column]

        can_move_forward = forward_theta[column] != theta[column]
        can_move_backward = backward_theta[column] != theta[column]

        if can_move_forward and can_move_backward:
            forward_errors = residuals(
                forward_theta,
                voltage,
                measured_current,
                vt,
                ns,
                root_method,
                counter,
            )
            backward_errors = residuals(
                backward_theta,
                voltage,
                measured_current,
                vt,
                ns,
                root_method,
                counter,
            )
            distance = forward_theta[column] - backward_theta[column]
            table[:, column] = (forward_errors - backward_errors) / distance
        elif can_move_forward:
            forward_errors = residuals(
                forward_theta,
                voltage,
                measured_current,
                vt,
                ns,
                root_method,
                counter,
            )
            distance = forward_theta[column] - theta[column]
            table[:, column] = (forward_errors - base_errors) / distance
        elif can_move_backward:
            backward_errors = residuals(
                backward_theta,
                voltage,
                measured_current,
                vt,
                ns,
                root_method,
                counter,
            )
            distance = theta[column] - backward_theta[column]
            table[:, column] = (base_errors - backward_errors) / distance
        else:
            raise RuntimeError(f"cannot change parameter {column} inside its bounds")

    return table


def solve_linear(
    matrix: np.ndarray, rhs: np.ndarray, pivot_rtol: float = 1e-12
) -> np.ndarray:
    """Solve matrix @ solution = rhs with Gaussian elimination and row swapping."""
    if not isinstance(matrix, np.ndarray) or not isinstance(rhs, np.ndarray):
        raise TypeError("matrix and rhs must be NumPy arrays")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square and nonempty")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("rhs length must match the matrix size")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError("matrix and rhs cannot contain NaN or infinity")
    if not np.isfinite(pivot_rtol) or pivot_rtol <= 0:
        raise ValueError("pivot_rtol must be positive and finite")

    work = matrix.astype(np.float64, copy=True)
    answers = rhs.astype(np.float64, copy=True)
    size = work.shape[0]
    minimum_pivot = pivot_rtol * np.max(np.abs(work))

    # Turn the matrix into an upper triangular matrix.
    for column in range(size):
        pivot_row = column + int(np.argmax(np.abs(work[column:, column])))
        if abs(work[pivot_row, column]) <= minimum_pivot:
            raise RuntimeError("matrix is singular")
        if pivot_row != column:
            work[[column, pivot_row]] = work[[pivot_row, column]]
            answers[[column, pivot_row]] = answers[[pivot_row, column]]
        for row in range(column + 1, size):
            multiplier = work[row, column] / work[column, column]
            work[row, column:] -= multiplier * work[column, column:]
            answers[row] -= multiplier * answers[column]

    # Work upward from the last row to find every unknown.
    solution = np.empty(size)
    for row in range(size - 1, -1, -1):
        known = work[row, row + 1 :] @ solution[row + 1 :]
        solution[row] = (answers[row] - known) / work[row, row]
    return solution
