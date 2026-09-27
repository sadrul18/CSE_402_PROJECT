# PV parameter estimation

A five-person Python Numerical Methods project that estimates the five parameters
of a solar cell from measured voltage and current. The core numerical pipeline is
implemented and the bundled French RTC dataset runs end to end.

## Current status

Verified on 2026-09-20, using implementation commit `0bff062`:

- **Working baseline:** `python run.py` fits the 26-point RTC dataset with the
  hybrid current solver and Levenberg–Marquardt (LM) optimizer.
- **Tests:** all 10 tests passed locally, with no skips. They cover the model,
  current solvers, residuals, Jacobian shape, linear solve, an LM fit, noise and plotting.
- **Implemented but needing more validation:** Gauss–Newton fitting, repeated
  noisy fits, experiment summaries and JSON export.
- **Remaining:** stronger numerical/failure tests, reliable work counters,
  reproducible method comparisons and noise-study results, and a final report.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the evidence, known limitations,
and remaining tasks by owner. A successful baseline is not proof that every
method is robust or that all five parameters have been accurately recovered.

## Files and ownership

| File | Owner | Responsibility |
| --- | --- | --- |
| `model.py` | Member 1 | PV equation/derivative, CSV loading, validation, initial parameters |
| `current.py` | Member 2 | Newton, bisection, hybrid solver and current calculation |
| `fit.py` | Member 3 | Gauss–Newton and Levenberg–Marquardt parameter fitting |
| `numerics.py` | Member 4 | Residuals, finite-difference Jacobian, pivoted linear solve |
| `experiments.py` | Member 5 | RMSE, seeded noise studies, summaries and plots |
| `run.py` | Member 1 | Connect the components and provide the entry point |
| `test_project.py` | Everyone | Maintain tests for each component and the full pipeline |

Measurements → solve currents → residuals → update parameters → repeat.
The **inner solver changes current** with parameters fixed.
The **outer optimizer changes parameters** and repeats the current calculations.

## Setup

Use Python 3.11 or newer, from the project folder:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -ra
python run.py
```

If the folder moved, recreate the virtual environment before activation.
Run the commands from the project root because the dataset path is relative to it.
On a machine without a display, use `MPLBACKEND=Agg python -m pytest -ra`.
GitHub Actions is configured to run this suite on pushes to `main` and pull requests.

The baseline prints convergence status, its stopping reason, the five parameters
and RMSE. The verified run reported `Converged: True`, reason
`error stopped improving`, and RMSE about **0.000906422 A**. These are observed
results for the current defaults, not a guaranteed accuracy target.

The runner currently prints one fit. It does not save plots, export JSON or run
noise experiments automatically; those helpers are in `experiments.py`.

## How data moves through the program

`run.py` loads voltage and measured current from the CSV, creates a starting
parameter array, and passes those values directly to `fit_parameters()`.

The fitter calls `residuals()`. Residuals calls `solve_currents()`, which solves
the PV equation at every voltage. The difference between each calculated current
and measured current is returned to the fitter. The Jacobian shows the fitter how
those errors respond to changes in the five parameters.

The functions use normal NumPy arrays and ordinary dictionaries. Dataset values
are passed explicitly; no callback wrappers or custom result types are used.

- Parameters always follow `[Iph, I0, Rs, Rsh, n]`, shape `(5,)`.
- Voltage/current arrays have matching shape `(N,)`, in volts/amperes.
  Preserve measurement order; reject empty, malformed or nonfinite data.
- `vt` is per-cell thermal voltage `k*T/q`; `ns` is series-cell count.
  The model denominator is `n*ns*vt`. Temperature is in kelvin.
- Physical domain: `Iph >= 0, I0 > 0, Rs >= 0, Rsh > 0, n > 0`.
  `bounds` is `(lower_array, upper_array)`; use finite, physically valid limits.
  Initial guesses and bounds are documented dataset-specific heuristics.
- Residuals are **predicted minus measured**, shape `(N,)`; Jacobian shape is `(N,5)`.
  Minimize mean squared residuals. RMSE is in amperes.
- Root results: `root, converged, reason, iterations, function_evaluations`.
  Failed solves return `root=None`, never the last unverified guess.
- Curve results: `current, roots`. Failed points are NaN with aligned root diagnostics.
  The residual function raises `RuntimeError` on any failed point; never drop rows.
- Fit results: `theta, residuals, converged, reason, iterations, evaluations, history`.
  Keep the best valid iterate; residuals may be None. History contains accepted MSEs.
  `evaluations` currently omits Jacobian residual calls and failed residual attempts;
  do not use it as a total work count until corrected.
- Input checks raise `ValueError` or `TypeError`; numerical evaluation/linear-solve
  failures raise `RuntimeError` or appear in result diagnostics. Validation gaps
  and failure-path tests are tracked in [PROJECT_STATUS.md](PROJECT_STATUS.md).
- Do not mutate caller arrays. Return failures explicitly; low RMSE alone does not prove
  that all five fitted parameters are physically identifiable.

M2 calls M1's equation. M4 calls M2's calculated currents. M3 calls M4's
residual, Jacobian, and linear-solver functions. M1's runner connects them, and
M5 repeats whole fits. The code passes the dataset and parameters directly.

## Next work

1. Strengthen correctness and failure tests, especially Gauss–Newton and optimizer stopping.
2. Fix diagnostic counts and input/runner limitations listed in the status file.
3. Run reproducible solver comparisons and noise studies, saving settings and results.
4. Prepare the figures, interpretation and final project report.

Update [PROJECT_STATUS.md](PROJECT_STATUS.md) with evidence when a task is completed.

## Data and results

The bundled [RTC CSV](data/rtc_france.csv) contains 26 measurements with columns
`voltage_v,current_a`. Its source, reuse terms and measurement conditions are
recorded in [data/README.md](data/README.md). The runner uses 306.15 K and one
series cell. Starting parameters and bounds are specific to this dataset.

Generated results belong in ignored `outputs/`; create that directory before
calling the plot or export helpers. The repository does not yet contain a curated
comparison/noise-study results package or a final report. The proposal and base
paper PDFs are reference documents. `demo.py` is a small Python scratch example,
not the project entry point.

## Contribute: fork → branch → pull request

Fork [the team repository](https://github.com/whoIsJihad/Numerical_Project), then:

```bash
git clone https://github.com/YOUR-USERNAME/Numerical_Project.git
cd Numerical_Project
git remote add upstream https://github.com/whoIsJihad/Numerical_Project.git
git fetch upstream
git switch -c member2/root-failure-tests upstream/main
# Complete one task from PROJECT_STATUS.md and its tests.
python -m pytest -ra
git add current.py test_project.py
git commit -m "Test root solver failure cases"
git push -u origin member2/root-failure-tests
```

Open a PR to the team's `main`. Include the method, tests, failure cases and remaining
work. Add component tests to `test_project.py` and update the status checklist.
The component author addresses review feedback. Interface changes require agreement.
Review before merging; merge commits preserve individual authorship. Member 1's work
also receives teammate review. Fetch upstream before starting each new task.
