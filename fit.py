
import numpy as np

from model import validate_data, validate_parameters
from numerics import EvaluationCounter, jacobian, residuals, solve_linear


def fit_parameters(
    voltage: np.ndarray,
    measured_current: np.ndarray,
    vt: float,
    theta0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    scales: np.ndarray,
    ns: int = 1,
    root_method: str = "hybrid",
    method: str = "lm",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> dict:
   
    validate_data(voltage, measured_current)
    validate_parameters(theta0, bounds)
    if not isinstance(scales, np.ndarray) or scales.shape != theta0.shape:
        raise ValueError("scales must have the same shape as theta0")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("scales must contain positive finite values")
    if method not in ("gauss_newton", "lm"):
        raise ValueError("method must be gauss_newton or lm")
    if (
        not np.isfinite(tol)
        or tol <= 0
        or not isinstance(max_iter, int)
        or max_iter < 1
    ):
        raise ValueError("tol and max_iter must be positive")

    lower, upper = bounds
    theta = theta0.astype(np.float64, copy=True)
    history = []
    evaluation_counter = EvaluationCounter()
    damping = 1e-3

    try:
        current_errors = residuals(
            theta,
            voltage,
            measured_current,
            vt,
            ns,
            root_method,
            evaluation_counter,
        )
    except NotImplementedError:
        raise
    except RuntimeError as error:
        return {
            "theta": theta,
            "residuals": None,
            "converged": False,
            "reason": str(error),
            "iterations": 0,
            "evaluations": evaluation_counter.residuals,
            "function_evaluations": evaluation_counter.function_evaluations,
            "history": history,
        }

    current_mse = float(np.mean(current_errors**2))
    history.append(current_mse)

    for iteration in range(1, max_iter + 1):
        try:
            slopes = jacobian(
                theta,
                voltage,
                measured_current,
                vt,
                bounds,
                scales,
                ns,
                root_method,
                counter=evaluation_counter,
            )
        except NotImplementedError:
            raise
        except RuntimeError as error:
            return {
                "theta": theta,
                "residuals": current_errors,
                "converged": False,
                "reason": str(error),
                "iterations": iteration - 1,
                "evaluations": evaluation_counter.residuals,
                "function_evaluations": evaluation_counter.function_evaluations,
                "history": history,
            }

        # Solve in dimensionless parameter coordinates. Without this column
        # scaling, I0 and Rsh differ enough in magnitude to make J.T @ J look
        # singular even when the local fitting problem has full rank.
        scaled_slopes = slopes * scales
        gradient = scaled_slopes.T @ current_errors
        if np.max(np.abs(gradient)) <= tol:
            return {
                "theta": theta,
                "residuals": current_errors,
                "converged": True,
                "reason": "gradient became small",
                "iterations": iteration - 1,
                "evaluations": evaluation_counter.residuals,
                "function_evaluations": evaluation_counter.function_evaluations,
                "history": history,
            }

        normal_matrix = scaled_slopes.T @ scaled_slopes
        accepted = False
        trial_step = np.zeros(theta.size)

        # Gauss-Newton tries its full step, then shorter versions if needed.
        if method == "gauss_newton":
            try:
                scaled_full_step = solve_linear(normal_matrix, -gradient)
            except RuntimeError as error:
                return {
                    "theta": theta,
                    "residuals": current_errors,
                    "converged": False,
                    "reason": str(error),
                    "iterations": iteration - 1,
                    "evaluations": evaluation_counter.residuals,
                    "function_evaluations": evaluation_counter.function_evaluations,
                    "history": history,
                }

            bound_tolerance = np.finfo(np.float64).eps * np.maximum(
                1.0, np.maximum(np.abs(lower), np.abs(upper))
            )
            at_lower = theta <= lower + bound_tolerance
            at_upper = theta >= upper - bound_tolerance
            active = (at_lower & (scaled_full_step < 0)) | (
                at_upper & (scaled_full_step > 0)
            )
            if np.any(active):
                free = ~active
                if not np.any(free):
                    return {
                        "theta": theta,
                        "residuals": current_errors,
                        "converged": False,
                        "reason": "all parameters are blocked by their bounds",
                        "iterations": iteration - 1,
                        "evaluations": evaluation_counter.residuals,
                        "function_evaluations": evaluation_counter.function_evaluations,
                        "history": history,
                    }
                scaled_full_step[active] = 0.0
                try:
                    scaled_full_step[free] = solve_linear(
                        normal_matrix[np.ix_(free, free)], -gradient[free]
                    )
                except RuntimeError as error:
                    return {
                        "theta": theta,
                        "residuals": current_errors,
                        "converged": False,
                        "reason": str(error),
                        "iterations": iteration - 1,
                        "evaluations": evaluation_counter.residuals,
                        "function_evaluations": evaluation_counter.function_evaluations,
                        "history": history,
                    }

            parameter_step = scales * scaled_full_step
            fraction = 1.0
            for index, step in enumerate(parameter_step):
                if step > 0:
                    fraction = min(fraction, (upper[index] - theta[index]) / step)
                elif step < 0:
                    fraction = min(fraction, (lower[index] - theta[index]) / step)

            for _ in range(20):
                candidate = np.clip(
                    theta + fraction * parameter_step, lower, upper
                )
                try:
                    candidate_errors = residuals(
                        candidate,
                        voltage,
                        measured_current,
                        vt,
                        ns,
                        root_method,
                        evaluation_counter,
                    )
                except RuntimeError:
                    fraction /= 2
                    continue
                candidate_mse = float(np.mean(candidate_errors**2))
                if candidate_mse < current_mse:
                    trial_step = candidate - theta
                    accepted = True
                    break
                fraction /= 2

        # LM adds damping. More damping means a smaller, safer step.
        else:
            for _ in range(20):
                damped_matrix = normal_matrix + damping * np.eye(theta.size)
                try:
                    scaled_proposed_step = solve_linear(damped_matrix, -gradient)
                    # Re-solve on free coordinates when a bound blocks a step.
                    # Clipping alone can falsely stop DDM fits at n2's upper bound.
                    bound_tolerance = np.finfo(float).eps * np.maximum(
                        1.0, np.maximum(np.abs(lower), np.abs(upper))
                    )
                    active = ((theta <= lower + bound_tolerance) & (scaled_proposed_step < 0)) | (
                        (theta >= upper - bound_tolerance) & (scaled_proposed_step > 0)
                    )
                    if np.any(active):
                        free = ~active
                        scaled_proposed_step[active] = 0.0
                        if np.any(free):
                            scaled_proposed_step[free] = solve_linear(
                                damped_matrix[np.ix_(free, free)], -gradient[free]
                            )
                except RuntimeError:
                    damping *= 10
                    continue
                candidate = np.clip(
                    theta + scales * scaled_proposed_step, lower, upper
                )
                try:
                    candidate_errors = residuals(
                        candidate,
                        voltage,
                        measured_current,
                        vt,
                        ns,
                        root_method,
                        evaluation_counter,
                    )
                except RuntimeError:
                    damping *= 10
                    continue
                candidate_mse = float(np.mean(candidate_errors**2))
                if candidate_mse < current_mse:
                    trial_step = candidate - theta
                    damping = max(damping / 3, 1e-15)
                    accepted = True
                    break
                damping *= 10

        if not accepted:
            return {
                "theta": theta,
                "residuals": current_errors,
                "converged": False,
                "reason": "no improving step found",
                "iterations": iteration,
                "evaluations": evaluation_counter.residuals,
                "function_evaluations": evaluation_counter.function_evaluations,
                "history": history,
            }

        previous_mse = current_mse
        theta = candidate
        current_errors = candidate_errors
        current_mse = candidate_mse
        history.append(current_mse)

        small_step = np.max(np.abs(trial_step) / scales) <= tol
        relative_mse_scale = max(previous_mse, np.finfo(np.float64).tiny)
        small_improvement = previous_mse - current_mse <= tol * relative_mse_scale
        if small_step or small_improvement:
            reason = (
                "parameter change became small"
                if small_step
                else "error stopped improving"
            )
            return {
                "theta": theta,
                "residuals": current_errors,
                "converged": True,
                "reason": reason,
                "iterations": iteration,
                "evaluations": evaluation_counter.residuals,
                "function_evaluations": evaluation_counter.function_evaluations,
                "history": history,
            }

    return {
        "theta": theta,
        "residuals": current_errors,
        "converged": False,
        "reason": "maximum iterations reached",
        "iterations": max_iter,
        "evaluations": evaluation_counter.residuals,
        "function_evaluations": evaluation_counter.function_evaluations,
        "history": history,
    }
