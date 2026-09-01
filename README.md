# Synthetic Theoretical Verification

Synthetic experiments for validating distribution-shift bounds and active-learning critics.

## Structure

- **level1/** — GMM covariate shift (moment / IPM / LD-FD-GC proxies)
- **level2/** — Interleaved spirals density-gap experiment (Jacobian collapse & temperature softening)

## Setup

```bash
pip install -r level1/requirements.txt
pip install -r level2/requirements.txt
```

## Run

```bash
# Level 1
cd level1 && python3 run_active_learning.py

# Level 2
cd level2 && python3 run_experiment.py
```

See each level's `markdown/` folder for experiment specifications.
