# Temporal Perturbation Test Suite

This project investigates how temporal clinical risk prediction models behave when the observation process changes. Rather than evaluating only predictive performance on a fixed test set, it applies controlled perturbations to the same underlying patient trajectories and measures how risk estimates change when records are made sparser, truncated to prefixes, or temporally stretched.

The current experiments are centered on PhysioNet 2012 in-hospital mortality prediction and include GRU-D, latent ODE, transformer, state-space, and landmarking models, alongside clinical baseline scores such as qSOFA, SIRS, and SOFA. The goal is to characterize robustness to sparsity, missingness, observation timing, and partial history.

## Use Cases

This repository can be used to:
- train multiple temporal risk prediction models on the same irregular-time dataset
- measure how learned-model predictions change when observations are thinned
- compare prediction instability across repeated random thinnings of the same patient record
- evaluate how predictions evolve as progressively longer prefixes of a patient trajectory become available
- test sensitivity to changes in timestamp spacing while keeping observed values fixed
- evaluate learned models alongside clinical baseline scores such as qSOFA, SIRS, and SOFA
- measure when baseline scores lose coverage or become unstable under sparse observations

## Quickstart

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate temporal-robustness
```

Verify the setup:

```bash
python scripts/smoke_test.py
```

Train checkpoints:

```bash
python training.py --fast
```

Run the full pipeline:

```bash
python run_experiment.py --fast --run_subdir seed42_run1
```

Train checkpoints with MLflow tracking:

```bash
python training.py --fast --track --experiment_name temporal_robustness
```

Run learned-model perturbation tests:

```bash
python scripts/run_perturbation_test.py --models grud,transformer,state_space
```

Run baseline stress tests:

```bash
python scripts/run_stress_tests.py --baselines qsofa,sirs,sofa
```

Use an isolated run root so checkpoints, perturbation outputs, stress-test
outputs, and provenance files all live under the same subdirectory inside
`results/`:

```bash
python training.py --fast --run_subdir seed42_run1
python scripts/run_perturbation_test.py --models grud,transformer,state_space --run_subdir seed42_run1
python scripts/run_stress_tests.py --baselines qsofa,sirs,sofa --run_subdir seed42_run1
```

Typical workflow:

```bash
python scripts/smoke_test.py
python training.py --fast --track --experiment_name temporal_robustness
python scripts/run_perturbation_test.py --models grud,transformer,state_space --track --experiment_name temporal_robustness
python scripts/run_stress_tests.py --baselines qsofa,sirs,sofa --track --experiment_name temporal_robustness
```

## Data

The repository is currently configured for PhysioNet 2012 in-hospital mortality prediction.

If the PhysioNet 2012 files are not already available locally, they are downloaded automatically on first use.

## Code Structure

- `run_experiment.py`: run training, perturbation tests, and stress tests sequentially
- `training.py`: train models and baseline checkpoints
- `experiment_registry.py`: dataset, task, model, and baseline registries
- `tracking.py`: shared MLflow tracking and provenance logging utilities
- `datasets/`: dataset code including shared sequence utilities and the PhysioNet 2012 loader
- `models/`: learned model implementations such as GRU-D, latent ODE, transformer, state-space, and landmarking
- `baselines/`: baseline score implementations
- `diagnostics/`: perturbation-test and stress-test analysis code
- `scripts/`: runnable entry-point scripts including smoke test, perturbation test, and stress tests
- `data/`: local dataset files and automatic downloads
- `results/`: generated summaries, figures, and JSON outputs

## Outputs

- `results/`: base output root; without `--run_subdir`, checkpoints and analysis outputs are written directly here
- `results/<run_subdir>/`: isolated run root when `--run_subdir` is used; contains checkpoints, provenance, and analysis outputs for one run
- `results/<run_subdir>/perturbation_test/` or `results/perturbation_test/`: learned-model perturbation outputs including JSON summaries, Markdown summaries, and generated figures for thinning sensitivity, repeated thinning instability, sparsity response, prefix volatility, and timestamp sensitivity
- `results/<run_subdir>/stress_test/` or `results/stress_test/`: baseline stress-test outputs including JSON summaries, Markdown summaries, and generated figures for coverage decay, component availability, covered-subset thinning sensitivity, repeated-thinning availability, prefix availability, and covered-subset performance under sparsity
- `mlruns/`: MLflow tracking history, when enabled, including tracked runs, parameters, provenance metadata, logged artifacts, and output snapshots

## Extensibility

This codebase is not limited to the current PhysioNet 2012 mortality setup. The same workflow can be adapted to other irregular-time datasets, other prediction tasks, additional learned model families, and additional clinical baseline scores.

In practice, extending the setup means adding dataset/task-specific loading and labeling logic, or adding new model and baseline implementations, while keeping the same top-level workflow:

1. train checkpoints with `training.py`
2. run perturbation analyses with `scripts/run_perturbation_test.py`
3. run baseline stress tests with `scripts/run_stress_tests.py`

These extension points are organized through dataset, task, model, and baseline registries in `experiment_registry.py`.

## Models

### Learned models

`grud`
- GRU-D recurrent model for irregularly observed multivariate time series with learned decay on inputs and hidden state.
- Uncertainty is estimated from prediction variance across repeated dropout-enabled forward passes.

`latent_ode`
- Latent ODE model with an ODE-RNN encoder and a neural ODE decoder for continuous-time latent trajectories.
- Uncertainty is estimated by sampling from the latent posterior.

`transformer`
- Time-aware transformer encoder over observation tokens built from values, masks, absolute times, and local time gaps.
- Uncertainty is estimated from repeated dropout-enabled forward passes.

`state_space`
- Probabilistic state-space risk model with latent Gaussian state propagation and observation-time filtering updates.
- Uncertainty comes directly from the model’s latent state variance.

`landmarking`
- Discrete-time landmarking baseline with separate logistic-regression ensembles at fixed landmark times.
- Uncertainty is estimated from variation across bootstrap-fitted landmark models.

### Guideline baselines

`qsofa`
- Bedside score used to flag possible sepsis-related organ dysfunction or deterioration risk.
- Computed from respiratory rate, systolic blood pressure, and GCS.
- A value is produced only when all three components are available at the same observation step.

`sirs`
- Rule-based inflammatory response score intended to capture systemic inflammatory physiology.
- Computed from temperature, heart rate, respiratory rate, and white blood cell count.
- A value is produced only when all four components are available at the same observation step.

`sofa`
- Organ dysfunction score intended to summarize multi-system severity in critical illness.
- Computed from platelet count, bilirubin, creatinine, GCS, MAP, and PaO2/FiO2.
- Partial component scoring is allowed, so it remains available more often under sparse observations.

Raw baseline scores are converted to mortality probabilities. If enough training data is available, this mapping is learned with a logistic-regression calibrator using score-derived features. Otherwise, probability is assigned with a simple heuristic based on the score value and overall event prevalence in the training data.

## Test Definitions

### Perturbation tests

These tests compare predictions for counterfactual versions of the same patient history.

`thinning_sensitivity`
- Measures how much predictions move when the same patient record is thinned to different observation-retention levels.
- Compares predictions on the original record with predictions on retrospectively thinned versions of the same record.
- Example: keep 100%, 75%, 50%, 25%, or 10% of observations from the same trajectory and compare the resulting risks.

`repeated_thinning_instability`
- Measures how sensitive predictions are to which specific observations are retained, not just to the overall retention rate.
- Repeats thinning with different random seeds at the same retention level and summarizes variability across repeated perturbed versions.
- Example: create ten different 25%-retention versions of the same patient record and measure how much the predictions vary across them.

`sparsity_response`
- Measures how predictive performance changes as observation density decreases.
- Recomputes performance metrics such as AUC, Brier score, log loss, and ECE across multiple retention levels.
- Example: evaluate the same trained model on test sets thinned to 100%, 50%, and 20% retention and compare AUC and calibration.

`prefix_volatility`
- Measures how predictions change as progressively longer prefixes of the same trajectory become available.
- Evaluates all observation prefixes of length at least two and summarizes how much the resulting prediction sequence moves.
- Example: score the first 2 observations, then the first 3, then the first 4, and track how much the predicted risk changes each time.

`timestamp_stretching`
- Measures sensitivity to elapsed-time structure while keeping observed values fixed.
- Rescales timestamps by multiplicative factors without changing values or masks, then compares the resulting predictions.
- Example: keep the same measurements but double every time gap and compare the new prediction to the original one.

### Stress tests

These tests target guideline-score availability and robustness under sparse observations.

`coverage_decay`
- Measures how often a baseline score can still be produced as observations are removed.
- Tracks patient-level score coverage, number of score points, and final component availability across retention levels.
- Example: after thinning records to 10% retention, count how many patients still have enough observed components for qSOFA, SIRS, or SOFA to be computed.

`component_availability`
- Measures which clinical score components remain observed under thinning.
- Reports component-level availability profiles rather than only final score availability.
- Example: track how often respiratory rate or bilirubin remains observed after thinning, even before asking whether the full score can be computed.

`thinning_sensitivity_covered`
- Measures score and probability drift among patients who remain scorable after thinning.
- Restricts comparison to covered patient pairs and summarizes both calibrated-probability drift and raw-score drift.
- Example: among patients with a valid SOFA score before and after thinning, compare how much the raw score and calibrated probability changed.

`repeated_thinning_availability`
- Measures whether score availability itself is unstable across different random thinnings of the same record.
- Repeats thinning with different seeds and summarizes how often each patient switches between scorable and unscorable states.
- Example: if one 10%-retention version of a patient yields a SIRS score and another does not, that contributes to availability instability.

`prefix_availability_volatility`
- Measures whether a baseline becomes available early in a trajectory and whether its prefix-wise values remain stable.
- Evaluates growing prefixes and summarizes coverage fraction, first available prefix, total variation, and large jumps.
- Example: recompute SOFA after each additional observation prefix and record when it first becomes available and how much it moves afterward.

`sparsity_performance_covered`
- Measures predictive performance only on the subset of patients for whom a baseline score is actually available.
- Recomputes metrics such as AUC, Brier score, log loss, and ECE on the covered subset at each retention level.
- Example: at 25% retention, compute AUC only over the patients for whom the baseline score can still be emitted.

## MLflow Tracking

MLflow tracking is optional. When enabled, it records run parameters, dataset and path metadata, Git provenance, checkpoint locations, and generated result artifacts. This provides a local history of training runs, perturbation analyses, and stress-test outputs in `mlruns/`.

## Citation

If you use this repository, cite the underlying models and dataset as appropriate:

```text
Che et al. (2018). Recurrent Neural Networks for Multivariate Time Series
with Missing Values. Scientific Reports.

Rubanova et al. (2019). Latent ODEs for Irregularly-Sampled Time Series.
NeurIPS 2019.

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation 101(23).
```
