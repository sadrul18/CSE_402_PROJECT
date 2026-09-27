"""Calculate solar-cell current with Newton, bisection, or a hybrid method."""

import numpy as np

from model import current_derivative, equation, validate_parameters


def solve_root(
    voltage: float,
    theta: np.ndarray,
    vt: float,
    ns: int = 1,
    initial_guess: float | None = None,
    bracket: tuple[float, float] | None = None,
    method: str = "hybrid",
    tol: float = 1e-8,
    max_iter: int = 100,
) -> dict:
    """Solve the PV equation for current at one voltage.

    The returned dictionary contains the current, success status, and work count.
    """
    validate_parameters(theta)
    if method not in ("newton", "bisection", "hybrid"):
        raise ValueError("method must be newton, bisection, or hybrid")
    if not np.isfinite(voltage) or not np.isfinite(vt) or vt <= 0:
        raise ValueError("voltage must be finite and vt must be positive")
    if not isinstance(ns, int) or ns < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("ns and tol must be positive")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")

    evaluations = 0

    # Newton repeatedly follows the local slope toward zero.
    if method == "newton":
        if initial_guess is None:
            raise ValueError("Newton needs an initial current guess")
        current = float(initial_guess)
        try:
            evaluations += 1
            error = equation(current, voltage, theta, vt, ns)
            for iteration in range(max_iter + 1):
                if abs(error) <= tol:
                    return {
                        "root": current,
                        "converged": True,
                        "reason": "found",
                        "iterations": iteration,
                        "function_evaluations": evaluations,
                    }
                if iteration == max_iter:
                    break
                slope = current_derivative(current, voltage, theta, vt, ns)
                if not np.isfinite(slope) or slope == 0:
                    return {
                        "root": None,
                        "converged": False,
                        "reason": "zero or invalid slope",
                        "iterations": iteration,
                        "function_evaluations": evaluations,
                    }
                next_current = current - error / slope
                if not np.isfinite(next_current):
                    return {
                        "root": None,
                        "converged": False,
                        "reason": "invalid Newton step",
                        "iterations": iteration,
                        "function_evaluations": evaluations,
                    }
                if abs(next_current - current) <= tol * max(1.0, abs(next_current)):
                    # A tiny step can finish a valid root. Check the new residual
                    # before classifying it as stagnation.
                    evaluations += 1
                    next_error = equation(next_current, voltage, theta, vt, ns)
                    if abs(next_error) <= tol:
                        return {
                            "root": next_current,
                            "converged": True,
                            "reason": "found",
                            "iterations": iteration + 1,
                            "function_evaluations": evaluations,
                        }
                    return {
                        "root": None,
                        "converged": False,
                        "reason": "step became tiny before finding a root",
                        "iterations": iteration + 1,
                        "function_evaluations": evaluations,
                    }
                current = next_current
                evaluations += 1
                error = equation(current, voltage, theta, vt, ns)
        except NotImplementedError:
            raise
        except (RuntimeError, ValueError, OverflowError) as error_message:
            return {
                "root": None,
                "converged": False,
                "reason": str(error_message),
                "iterations": 0,
                "function_evaluations": evaluations,
            }
        return {
            "root": None,
            "converged": False,
            "reason": "maximum iterations reached",
            "iterations": max_iter,
            "function_evaluations": evaluations,
        }

    # Bisection and hybrid both need a sign-changing interval.
    if not isinstance(bracket, tuple) or len(bracket) != 2:
        raise ValueError("bisection and hybrid need a bracket")
    low, high = float(bracket[0]), float(bracket[1])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("bracket must contain two ordered finite values")
    try:
        evaluations += 1
        low_error = equation(low, voltage, theta, vt, ns)
        evaluations += 1
        high_error = equation(high, voltage, theta, vt, ns)
    except NotImplementedError:
        raise
    except (RuntimeError, ValueError, OverflowError) as error_message:
        return {
            "root": None,
            "converged": False,
            "reason": str(error_message),
            "iterations": 0,
            "function_evaluations": evaluations,
        }
    if abs(low_error) <= tol:
        return {
            "root": low,
            "converged": True,
            "reason": "lower endpoint is the root",
            "iterations": 0,
            "function_evaluations": evaluations,
        }
    if abs(high_error) <= tol:
        return {
            "root": high,
            "converged": True,
            "reason": "upper endpoint is the root",
            "iterations": 0,
            "function_evaluations": evaluations,
        }
    if low_error * high_error > 0:
        return {
            "root": None,
            "converged": False,
            "reason": "bracket does not contain a sign change",
            "iterations": 0,
            "function_evaluations": evaluations,
        }

    current = (low + high) / 2 if initial_guess is None else float(initial_guess)
    if current < low or current > high:
        current = (low + high) / 2

    for iteration in range(1, max_iter + 1):
        # Bisection always uses the middle. Hybrid tries Newton first.
        midpoint = (low + high) / 2
        candidate = midpoint
        used_newton = False
        current_error = None
        if method == "hybrid":
            try:
                evaluations += 1
                current_error = equation(current, voltage, theta, vt, ns)
                slope = current_derivative(current, voltage, theta, vt, ns)
                newton_candidate = current - current_error / slope
                if np.isfinite(newton_candidate) and low < newton_candidate < high:
                    candidate = newton_candidate
                    used_newton = True
            except NotImplementedError:
                raise
            except (RuntimeError, ValueError, OverflowError, ZeroDivisionError):
                candidate = midpoint
        try:
            evaluations += 1
            candidate_error = equation(candidate, voltage, theta, vt, ns)
        except NotImplementedError:
            raise
        except (RuntimeError, ValueError, OverflowError) as error_message:
            return {
                "root": None,
                "converged": False,
                "reason": str(error_message),
                "iterations": iteration,
                "function_evaluations": evaluations,
            }
        if used_newton and abs(candidate_error) > abs(current_error):
            candidate = midpoint
            try:
                evaluations += 1
                candidate_error = equation(candidate, voltage, theta, vt, ns)
            except NotImplementedError:
                raise
            except (RuntimeError, ValueError, OverflowError) as error_message:
                return {
                    "root": None,
                    "converged": False,
                    "reason": str(error_message),
                    "iterations": iteration,
                    "function_evaluations": evaluations,
                }
        if abs(candidate_error) <= tol:
            return {
                "root": candidate,
                "converged": True,
                "reason": "found",
                "iterations": iteration,
                "function_evaluations": evaluations,
            }
        if low_error * candidate_error < 0:
            high = candidate
            high_error = candidate_error
        else:
            low = candidate
            low_error = candidate_error
        current = candidate
        if high - low <= tol * max(1.0, abs(low), abs(high)):
            return {
                "root": (low + high) / 2,
                "converged": True,
                "reason": "bracket became small",
                "iterations": iteration,
                "function_evaluations": evaluations,
            }
    return {
        "root": None,
        "converged": False,
        "reason": "maximum iterations reached",
        "iterations": max_iter,
        "function_evaluations": evaluations,
    }


def solve_currents(
    voltage: np.ndarray,
    theta: np.ndarray,
    vt: float,
    ns: int = 1,
    method: str = "hybrid",
    warm_start: bool = True,
) -> dict:
    """Solve the PV equation once for every voltage and return all currents."""
    if not isinstance(voltage, np.ndarray) or voltage.ndim != 1 or voltage.size == 0:
        raise ValueError("voltage must be a nonempty one-dimensional NumPy array")
    if not np.all(np.isfinite(voltage)):
        raise ValueError("voltage cannot contain NaN or infinity")
    validate_parameters(theta)
    if method not in ("newton", "bisection", "hybrid"):
        raise ValueError("method must be newton, bisection, or hybrid")

    currents = np.full(voltage.size, np.nan)
    details = [None] * voltage.size
    order = np.argsort(voltage, kind="stable")
    previous_current = None
    total_evaluations = 0

    for index in order:
        guess = (
            previous_current
            if warm_start and previous_current is not None
            else float(theta[0])
        )

        # Expand around the guess until the equation changes sign.
        bracket = None
        bracket_evaluations = 0
        width = max(1.0, abs(guess))
        if method != "newton":
            for _ in range(30):
                low = guess - width
                high = guess + width
                try:
                    bracket_evaluations += 1
                    low_error = equation(low, float(voltage[index]), theta, vt, ns)
                    bracket_evaluations += 1
                    high_error = equation(high, float(voltage[index]), theta, vt, ns)
                    if low_error * high_error <= 0:
                        bracket = (low, high)
                        break
                except RuntimeError:
                    pass
                width *= 2
            if bracket is None:
                details[index] = {
                    "root": None,
                    "converged": False,
                    "reason": "could not find a bracket",
                    "iterations": 0,
                    "function_evaluations": bracket_evaluations,
                }
                total_evaluations += bracket_evaluations
                continue

        answer = solve_root(
            float(voltage[index]), theta, vt, ns, guess, bracket, method
        )
        answer["function_evaluations"] += bracket_evaluations
        total_evaluations += answer["function_evaluations"]
        details[index] = answer
        if answer["converged"]:
            currents[index] = answer["root"]
            previous_current = answer["root"]

    return {
        "current": currents,
        "roots": details,
        "function_evaluations": total_evaluations,
    }
