# Project reading guide and detailed file walkthrough

## What this project does

This project estimates the electrical parameters of a photovoltaic device from
measured voltage-current data. It supports two related implicit models:

- the five-parameter single-diode model (SDM), with parameter order
  `[Iph, I0, Rs, Rsh, n]`;
- the seven-parameter double-diode model (DDM), with parameter order
  `[Iph, I01, Rs, Rsh, n1, I02, n2]`.

The current is not available by a simple direct formula in the implemented
workflow. For every voltage and every proposed parameter vector, the program
first solves an implicit nonlinear equation for current. An outer least-squares
method then changes the parameters and repeats those current calculations until
the calculated curve fits the measured data.

The most useful mental model is therefore:

```text
measured CSV
   -> validate data and choose an initial parameter vector
   -> solve the implicit current equation at every voltage
   -> subtract measured current to form a residual vector
   -> perturb parameters to form a Jacobian
   -> solve a scaled least-squares step
   -> accept or reject the parameter update
   -> repeat fits for method, initialization, and noise studies
   -> save tables and plots
```

## Recommended reading order

Read the project in the following order. This sequence introduces the problem
before the implementation, then follows the actual runtime dependency chain.

1. [`README.md`](README.md) - learn the scope, setup, commands, ownership, and
   top-level data flow.
2. [`PV_Parameter_Estimation_Proposal-2.pdf`](PV_Parameter_Estimation_Proposal-2.pdf)
   - understand the agreed project deliverables: SDM/DDM fitting, deterministic
   optimizers, current solvers, and robustness analysis.
3. [`base_paper.pdf`](base_paper.pdf) - read the scientific comparison source,
   especially the model definitions, parameter bounds, Table 8, Figures 13-14,
   Table 10, and Figure 17.
4. [`data/README.md`](data/README.md), then the two CSV files - establish where
   the measurements came from, their units, and which dataset the study uses.
5. [`model.py`](model.py) - learn the parameter conventions, equations,
   derivative, validation rules, thermal voltage, CSV loader, starting values,
   bounds, and scales.
6. [`current.py`](current.py) - see how current is calculated for one voltage
   and then for a complete curve.
7. [`numerics.py`](numerics.py) - follow the transformation from current curves
   to residuals and a finite-difference Jacobian, then inspect the custom linear
   solver.
8. [`fit.py`](fit.py) - understand how Gauss-Newton (GN) and
   Levenberg-Marquardt (LM) use those residuals and derivatives to update the
   parameters.
9. [`experiments.py`](experiments.py) - see how individual fits become method
   comparisons, Monte Carlo studies, condition diagnostics, CSV/JSON exports,
   and the original SDM plots.
10. [`run.py`](run.py) - follow the supported command-line workflows from data
    loading to saved output.
11. [`ddm_study.py`](ddm_study.py) - examine the complete SDM-versus-DDM study,
    paper-reference reconstruction, varied-start ensembles, paired noise
    experiment, and extended plots.
12. [`test_project.py`](test_project.py), followed by
    [`test_ddm.py`](test_ddm.py) - use the tests as executable specifications of
    success cases, failure cases, counts, bounds, and reproducibility.
13. [`REPORT.md`](REPORT.md), then [`DDM_REPORT.md`](DDM_REPORT.md) - read the
    results only after understanding how they were computed.
14. [`PLOT_ANALYSIS.md`](PLOT_ANALYSIS.md) - finish with the detailed visual and
    numerical interpretation of every curated figure.

`PROJECT_STATUS.md`, `todo.md`, the result tables, and the JSON files are best
used as audit material while reading steps 12-14 rather than as the first
introduction to the code.

## Core mathematical flow

At a measured voltage `V`, the SDM implementation searches for a current `I`
that makes this equation equal to zero:

```text
u = V + I*Rs
f(I) = Iph - I0*expm1(u/(n*Ns*Vt)) - u/Rsh - I
```

The DDM adds a second diode term:

```text
f(I) = Iph - I01*expm1(u/(n1*Ns*Vt))
             - I02*expm1(u/(n2*Ns*Vt)) - u/Rsh - I
```

Here, `Vt = k*T/q` is the per-cell thermal voltage, `Ns` is the series-cell
count, and `u` is the diode/shunt voltage after the series-resistance drop is
included. The inner root solver changes `I` while holding the parameter vector
fixed. The outer optimizer changes the parameter vector and calls the inner
solver again for all measured voltages.

For measurement `i`, the residual convention is:

```text
r_i(theta) = I_predicted(V_i, theta) - I_measured_i
```

The fitter minimizes the squared residuals. The displayed RMSE is
`sqrt(mean(r**2))`. Because the parameters have very different magnitudes - for
example, a saturation current can be near `1e-7 A` while `Rsh` is tens of ohms -
the optimizer operates in scaled parameter coordinates.

## Detailed source-file walkthrough

### `model.py` - physical model, validation, and initialization

This is the foundation of the numerical stack. It deliberately supports both
models without introducing separate class hierarchies.

- `SDM_PARAMETER_NAMES` and `DDM_PARAMETER_NAMES` define the public parameter
  order. Every table, plot label, bound array, scale array, and fitted vector
  depends on this order.
- `parameter_names(theta)` validates a five- or seven-element vector and returns
  the matching labels. This keeps exports and plots model-aware.
- `validate_data(voltage, current)` requires finite, nonempty, one-dimensional
  NumPy arrays of equal length. Rows are never silently dropped.
- `validate_parameters(theta, bounds=None)` enforces vector length, finite
  values, physical signs, matching finite bounds, ordered lower/upper limits,
  and inclusion of the vector inside its bounds. DDM additionally requires
  positive `I02` and `n2`.
- `equation(current, voltage, theta, vt, ns=1)` evaluates the implicit SDM or
  DDM current balance. It uses `numpy.expm1`, which is more accurate than
  `exp(x)-1` when `x` is small, and converts exponential overflow into an
  explicit numerical failure.
- `current_derivative(...)` supplies `df/dI` to Newton's method. In DDM it adds
  the second diode's derivative term.
- `thermal_voltage(temperature_k)` calculates `kT/q` using exact SI constants
  and rejects nonphysical temperatures.
- `load_data(path)` accepts only the exact header `voltage_v,current_a`, reports
  malformed rows with their row number, converts to `float64`, and then applies
  the common data validation.
- `initial_parameters(...)` returns three coupled objects: the initial vector,
  `(lower, upper)` bounds, and scaling values. Its heuristics are specifically
  limited to the one-cell, at-most-one-ampere RTC case. DDM bounds follow the
  base paper's wider physically motivated ranges; SDM keeps the original
  project ranges.

When debugging an incorrect result, verify the parameter order and units here
before inspecting the optimizer. A swapped `Rsh` and `n`, or microamperes that
were not converted to amperes, can produce misleading but numerically finite
curves.

### `current.py` - implicit current solution

`solve_root(...)` solves the equation at one voltage using one of three methods.
It always returns a dictionary containing the root or `None`, convergence flag,
reason, iteration count, and scalar PV-equation evaluation count.

- **Newton** starts from a supplied current guess, uses the analytic derivative,
  and is fast when the guess and local slope are good. It verifies the residual
  at a tiny proposed final step before declaring success or stagnation.
- **Bisection** requires a sign-changing interval. It is conservative and does
  not need the derivative, but it usually needs many more equation calls.
- **Hybrid** keeps a valid bracket, tries a Newton proposal inside it, and falls
  back to the midpoint if the proposal is invalid or worsens the residual. This
  trades some speed for safeguards.

No failed method returns its last guess as if it were a verified solution.
Failure reasons distinguish invalid slopes, invalid steps, a missing sign
change, overflow, stagnation, and the iteration limit.

`solve_currents(...)` applies the scalar solver to a voltage array. It sorts the
voltages internally so the preceding current can warm-start the next point, but
stores every result back in the original row order. For bisection and hybrid it
expands an interval around the guess until a sign change appears or 30 expansion
attempts fail. A failed row remains aligned with the input: the current is NaN
and the corresponding root diagnostic explains why.

The distinction between `iterations` and `function_evaluations` matters.
Bracket search calls and rejected/failing equation calls are included in the
latter. Derivative calls are not counted as PV-equation calls.

### `numerics.py` - residuals, Jacobian, and linear algebra

`EvaluationCounter` tracks attempted full residual-vector calculations and the
underlying scalar equation calls during a fit.

`residuals(...)` calculates a complete current curve, verifies that every inner
solve converged, and returns predicted minus measured current. It raises an
error on any failed row rather than fitting a smaller, easier subset.

`jacobian(...)` approximates how every residual changes with every parameter.
For parameter `j`, the nominal perturbation is:

```text
h_j = relative_step * max(abs(theta_j), scale_j)
```

Central differences are used when movement is possible in both directions.
At a bound, the function uses a forward or backward difference. It fails if a
floating-point perturbation is too small to change a parameter. The output has
one row per measurement and five or seven columns according to the model.

`solve_linear(matrix, rhs, pivot_rtol=1e-12)` implements Gaussian elimination
with partial pivoting and back substitution. It works on copies, swaps in the
largest available pivot, and rejects a numerically singular system. The project
uses this explicit implementation to demonstrate the numerical method instead
of hiding the step inside `numpy.linalg.solve`.

### `fit.py` - outer parameter estimation

`fit_parameters(...)` is the only public fitting function. It validates all
inputs, copies the starting vector, evaluates the initial residual, and keeps a
history of accepted mean squared errors.

At each outer iteration it:

1. constructs a finite-difference Jacobian;
2. multiplies its columns by the parameter scales;
3. computes the scaled gradient and normal matrix;
4. obtains a GN or damped LM step;
5. respects active physical bounds;
6. evaluates candidate residuals;
7. accepts only an improving candidate;
8. stops on a small gradient, scaled parameter change, or relative MSE
   improvement, or reports a specific failure.

GN solves the scaled normal equations and uses a feasible fraction plus a
backtracking line search. If a bound blocks the proposed direction, the active
coordinate is removed and the reduced system is solved on free coordinates.

LM adds `damping * I` to the normal matrix. Failed or non-improving proposals
increase damping by a factor of ten; an accepted proposal decreases it by a
factor of three. LM also re-solves on free coordinates when a bound blocks a
step. This is important for the DDM result, where `n2` reaches its upper bound.

The returned dictionary includes the best parameter vector, its residuals when
available, convergence flag and reason, accepted iteration count, attempted
residual evaluations, scalar equation evaluations, and accepted MSE history.
A low retained RMSE can coexist with `converged=False`; callers must inspect
both fields.

### `experiments.py` - repeatable studies and the original plot layer

This module turns individual fits into auditable experiments.

- `rmse(errors)` computes the curve error used throughout the reports.
- `add_noise(...)` adds absolute Gaussian noise; `add_relative_noise(...)`
  uses pointwise standard deviation `abs(I_i)*percentage/100`.
- `monte_carlo(...)` uses a local seeded NumPy generator, creates a fresh noisy
  current array for every trial, times the fit, and retains successes and
  failures.
- `summarize(result)` reports success/failure counts, failure reasons, mean and
  standard deviation of RMSE, work and runtime means, parameter mean/standard
  deviation/coefficient of variation, and error relative to an optional clean
  reference vector. Parameter statistics condition on converged fits.
- `condition_diagnostics(...)` computes singular values and condition numbers
  for the scaled Jacobian and scaled normal matrix. These are local sensitivity
  indicators, not proof of global identifiability.
- `compare_fit_methods(...)` runs all requested combinations of outer
  optimizer and inner root solver with shared settings.
- `compare_current_solvers(...)` compares cold Newton against warm Newton,
  warm bisection, and warm hybrid on the same fitted curve.
- `run_noise_study(...)` runs the documented four-level Monte Carlo protocol,
  advancing the seed once per noise level.
- `save_results(...)` creates strict JSON with metadata and disallows NaN;
  `save_table(...)` creates consistent flat CSV files.
- `plot_fit`, `plot_convergence`, and `plot_noise_summary` generate the original
  SDM figure set.

### `run.py` - supported command-line entry point

The CLI supports `baseline`, `compare`, `noise`, and `all`, with an SDM or DDM
model selection. The default outer limit is 100 iterations for SDM and 500 for
DDM. A baseline fit is always run first; if it does not converge, the process
returns status 1 and does not present later work as a successful workflow.

- `run_baseline` executes one configured fit and adds final RMSE.
- `build_parser` defines all public command-line options.
- row-building helpers flatten nested fit and noise results for CSV.
- `save_baseline_artifacts` writes JSON and a fitted I-V plot.
- `run_comparisons` runs six inner/outer combinations, current-solver work
  comparisons, and condition diagnostics.
- `run_noise` executes and saves the configured Monte Carlo study.
- `main` connects data loading, thermal voltage, initialization, fitting,
  printing, output directory creation, and command dispatch.

Use this file to understand what an ordinary project user can reproduce. Use
`ddm_study.py` for the larger report-specific experiment.

### `ddm_study.py` - complete report experiment

This script is intentionally separate from the ordinary CLI because it runs a
much larger, paper-oriented study: 12 clean baselines, 360 varied-start fits,
and 800 paired noisy-data fits under the documented full settings.

- `ALGORITHMS`, `PAPER_APPROX`, `PAPER_NR`, and `PAPER_MEAN_RMSE` preserve the
  base paper's five algorithm labels and rounded Table 8/Table 10 values.
- `published_parameters()` converts saturation currents from microamperes to
  amperes and places values in the project's DDM order.
- `paper_comparison(...)` treats the paper's approximation method as a
  measured-current substitution on the equation's right-hand side, while NR
  references are genuine implicit solves. These are references, not project
  optimizers.
- `fit_trial(...)` is a process-safe worker that retains all diagnostics.
- `ensemble(...)` may run jobs in multiple processes but restores deterministic
  original trial ordering.
- `average_rmse_history(...)` converts each accepted MSE history to RMSE,
  carries a stopped run's incumbent forward to a common budget, and averages
  per-run RMSE without dropping finite failed runs.
- `create_plots(...)` generates paper-reference characteristics/errors, project
  DDM characteristics/errors, average convergence, paired noise comparison,
  and parameter-variation figures. It also saves numeric plot data.
- `rounding_diagnostic(...)` changes one current value only in memory to explain
  the difference between the repository CSV and a common RTC benchmark. It
  never edits the source dataset or enters the main experiments.
- `main(...)` records dataset hash, environment versions, model settings,
  timing protocol, and convergence protocol; runs every experiment; saves raw
  results; creates curated-compatible tables and figures; and prints progress.

Do not mistake the five paper algorithm labels for implemented algorithms.
AEO, GBO, GNDO, BO, and RTH appear only as published reference parameter sets
and terminal statistics. This project implements GN and LM.

## Entry points and tests

### `test_project.py`

This is the broad executable specification. Its tests cover strict CSV loading,
physical validation, model balance and derivatives, starting parameters, all
root methods, root failure reasons, hybrid fallback, warm starts, order
preservation, residual and Jacobian correctness, bounds, evaluation counts,
Gaussian elimination, optimizer regression and stopping, noise reproducibility,
failure-aware summaries, exports, plotting, conditioning, and CLI failure
status.

The helper `analytic_current_jacobian` is important: it independently derives
the implicit current sensitivity, so the finite-difference Jacobian is not
merely checked against another copy of the same numerical code.

### `test_ddm.py`

This focused extension verifies the seven-parameter equation and derivative,
reduction to SDM when the diode ideality factors coincide, all root methods at
negative current, an independent implicit DDM Jacobian, clean RTC regression
fits, explicit failure retention, exact-data fitting, seven-parameter noise
summaries, correct mean-RMSE history arithmetic, paper-reference units and
error identities, and Newton's verified tiny final step.

### `demo.py`

This file is empty and is not an entry point. It can be removed in a future
cleanup or reserved for a minimal teaching example, but no reader should infer
project behavior from it.

## Data and reference files

### `data/rtc_france.csv`

The main experiment input contains 26 measured French RTC solar-cell points
with columns `voltage_v,current_a`. The order, signs, and published precision
are preserved. Negative voltage and negative current values are valid measured
points, not parsing errors. All main reported SDM/DDM results use this unchanged
file.

At `V=0.5833 V`, it contains `I=-0.1233 A`. Other common RTC tables contain
`-0.1230 A`. That single difference materially shifts a sub-milliampere RMSE,
which is why the project saves a separate sensitivity diagnostic.

### `data/photowatt_pwp201.csv`

This file contains 25 Photowatt PWP 201 module measurements, also using the
standard two-column header. It is preserved for the wider proposal context but
is not fitted in the current RTC-only study. `initial_parameters` explicitly
rejects unsupported module-scale conditions, so selecting this CSV is not a
drop-in replacement without new model/bound work.

### `data/README.md`

This provenance record states the article, supplemental table, temperature,
irradiance, series-cell count, preprocessing, reuse license, the one-point RTC
qualification, and the Photowatt file's current status. Read it whenever a
reported RMSE is compared with external literature.

### `PV_Parameter_Estimation_Proposal-2.pdf`

The revised proposal is the scope contract. It motivates deterministic
Gauss-Newton/LM extraction, the three numerical current solvers, convergence
comparison, noise sensitivity, and the SDM/DDM extension. It should be used to
judge whether the submitted work satisfies the course project, not as a source
of generated numerical results.

### `base_paper.pdf`

Qin et al. (2024) is the scientific reference. The most relevant material is:

- the SDM/DDM equations and simulation-current methods;
- physically motivated DDM bounds;
- Table 8's French RTC DDM parameters;
- Figure 13's measured/simulated I-V and P-V curves;
- Figure 14's absolute current and power errors;
- Table 10's statistics from 30 DDM runs; and
- Figure 17's convergence curves.

The paper combines five metaheuristics with approximation and Newton-Raphson
current calculations. This project instead combines two deterministic local
optimizers with Newton, bisection, and hybrid current solvers. The terms are not
interchangeable.

## Documentation files

### `README.md`

The operational manual: status, ownership, setup, commands, interfaces, data
flow, output rules, contribution workflow, and extension summary. Use it to run
the software.

### `REPORT.md`

The original SDM report documents the mathematical objective, root solvers,
experimental settings, clean fit, six-method comparison, original 100-iteration
noise study, conditioning, verification, limitations, and reference.

### `DDM_REPORT.md`

The extension report documents the seven-parameter implementation, common
500-iteration comparison, clean performance, 30-start sensitivity, Figures
13/14/17 comparisons, paired 800-fit noise experiment, conditioning, data
qualification, and reproduction steps. It is the primary narrative source for
the DDM study.

### `PROJECT_STATUS.md`

An acceptance-evidence checklist. It connects each responsibility to tests,
commands, or reviewed artifacts and separates the original SDM delivery from
the later DDM extension.

### `todo.md`

A short completion checklist for the DDM request. It is historical project
tracking, not a replacement for the reports or test suite.

### `COMMIT_DISTRIBUTION.md`

The collaboration guide that assigns primary file ownership and a dependency-
safe commit order.

### `PLOT_ANALYSIS.md`

The detailed figure companion. It explains axes, construction, comparison with
the base paper, numerical observations, limitations, and the source table for
every curated plot.

## Curated SDM artifacts

The `report/results/` files are machine-readable support for `REPORT.md`.

- `baseline.json` stores metadata plus the representative SDM LM/hybrid fit,
  including parameters, residuals, convergence reason, work counts, and
  history.
- `fit_method_comparison.csv` contains the six combinations of GN/LM and
  Newton/bisection/hybrid, including convergence, runtime, work, RMSE, and
  fitted parameters.
- `current_solver_comparison.csv` isolates cold/warm current-solver work at a
  fixed fitted vector.
- `method_comparison.json` preserves the same comparison in nested form plus
  full histories and condition diagnostics.
- `noise_summary.csv` gives the original SDM four-level Monte Carlo summaries
  and parameter statistics.

The `report/figures/` files visualize those results:

- `fit.png` is the measured-versus-fitted SDM I-V curve;
- `convergence.png` shows accepted clean-fit MSE histories for six method
  combinations;
- `noise_summary.png` shows original SDM fit error, failure rate, and parameter
  coefficient of variation against noise.

## Curated DDM-study artifacts

The `report/ddm/` directory is the audit package for `DDM_REPORT.md`.

- `README.md` states that the directory contains selected full-study outputs.
- `baselines.json` contains full diagnostic dictionaries for 12 clean fits.
- `baseline_comparison.csv` is the flat clean-fit comparison used in tables.
- `initial_value_summary.csv` summarizes 30 starts for every one of the 12
  model/optimizer/root-solver combinations.
- `average_rmse_iterations.csv` contains every plotted mean-RMSE point through
  the common 500-iteration budget, along with valid/invalid run counts.
- `fitted_curves.csv` records measured and predicted I/P values plus absolute
  errors for each clean fit.
- `paper_reference.json` preserves rounded Table 8 parameter sets and Table 10
  terminal means.
- `paper_reference_curves.csv` contains all per-voltage values used to recreate
  paper-reference curves and errors.
- `paper_rmse_comparison.csv` places reported and recomputed RMSE side by side.
- `noise_comparison.csv` gives SDM/DDM success, error, and iteration summaries
  for the paired noise study.
- `noise_summary.json` adds full five- or seven-parameter statistics at every
  noise level.
- `diagnostics.json` stores scaled condition diagnostics and current-solver
  work comparisons for both models.
- `rounding_diagnostic.json` stores the explicitly labeled one-point
  sensitivity experiment.
- `ddm_characteristics.png` overlays all project DDM fits and representative
  paper references on I-V/P-V axes.
- `ddm_absolute_errors.png` shows their absolute current and power errors.
- `figure13_reference_curves.png` evaluates all ten rounded Table 8 reference
  parameter sets on the unchanged repository data.
- `figure14_reference_errors.png` shows the corresponding absolute errors.
- `average_rmse_iterations.png` compares 30-start mean convergence for SDM and
  DDM with paper terminal reference lines.
- `sdm_ddm_noise.png` compares paired-noise fit error, stopping failures, and
  error against the original measurements.
- `noise_parameter_variation.png` compares parameter coefficients of variation
  for the five- and seven-parameter models.

These files are curated copies. The complete trial-level files remain in an
ignored `outputs/` directory because they are larger and reproducible.

## Repository support files

### `requirements.txt`

Lists the runtime/test dependencies needed for a clean environment. Installation
instructions are in `README.md`.

### `.github/workflows/tests.yml`

Runs the test suite on repository events so a change is checked outside an
individual member's machine.

### `.gitignore`

Excludes generated environments, caches, and raw output directories. Curated
report artifacts remain tracked deliberately.

### `data/.gitkeep`

Historically preserves the data directory when it would otherwise be empty. It
has no runtime behavior now that CSV files are present.

## How to trace one baseline execution

For `python run.py baseline --model ddm`, follow this call chain:

1. `run.main` parses the command.
2. `model.load_data` reads the RTC CSV.
3. `model.thermal_voltage(306.15)` calculates `Vt`.
4. `model.initial_parameters(..., model="ddm")` returns seven values, bounds,
   and scales.
5. `run.run_baseline` calls `fit.fit_parameters`.
6. The fitter calls `numerics.residuals` for its initial and candidate vectors.
7. `residuals` calls `current.solve_currents`.
8. `solve_currents` calls `solve_root` 26 times per curve.
9. Later fitter iterations call `numerics.jacobian`, which repeats the residual
   calculation for parameter perturbations.
10. `numerics.solve_linear` solves the scaled GN/LM system.
11. The fitter accepts an improvement, updates its history, and repeats.
12. `run.print_baseline` prints convergence, parameters, work, and RMSE.

This nested structure explains why one outer iteration can require many full
curve evaluations and tens of thousands of scalar equation evaluations.

## Interpretation rules to keep in mind

- A visually excellent I-V curve does not prove that every physical parameter
  is uniquely identified.
- `converged=False` does not mean that the retained curve is unusable; it means
  the stopping contract was not satisfied. Always report the flag and reason.
- Bisection's reliable scalar bracketing does not automatically make it the best
  inner method for a finite-difference outer optimizer; small solver-level
  changes can affect the numerical Jacobian.
- Runtime values are machine-dependent. Compare only timings produced under the
  same protocol, and do not compare sequential baseline times with parallel
  ensemble wall times.
- Paper approximation, paper NR, project Newton, project bisection, and project
  hybrid describe current calculation, not the outer parameter-search method.
- Paper AEO/GBO/GNDO/BO/RTH are metaheuristic parameter searches. Project GN/LM
  are deterministic local least-squares methods.
- The extended 500-iteration paired study and original 100-iteration SDM study
  answer related but not identical questions; use the result set associated
  with the plot being discussed.
- The one-point diagnostic explains provenance sensitivity. It is not permission
  to overwrite the supplied measurements.
