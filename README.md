# Measurement Layouts

``Measurement layouts’’ are a modelling recipe for capability-oriented AI evaluation. They formalise how task-instance features interact with latent system capabilities to shape observed performance, letting you infer capability profiles from heterogeneous test batteries and predict generalisation to new task instances.

This repository collects minimal, end-to-end examples (mostly notebooks) and small utilities for building, fitting, and validating measurement layouts on synthetic and real tasks.

## What’s inside

- `basic_measurement_layout/` — a minimal, didactic layout showing the full pipeline on toy data (define features → specify layout → fit → diagnose → predict).
- `object_permanence_synth/` — synthetic object-permanence tasks demonstrating how to extend Measurement Layouts to more complex tasks. 
- `object_permanence_voudouris2024/` — An example with real agents on object-permanence experiments using Animal-AI environments.
- `analysis/measurement-layouts/` — additional analysis notebooks (model diagnostics, posterior predictive checks, ablations).
- `LICENSE` — MIT.

# Citing

If you use this repository or the underlying framework, please cite:
@article{Burden2023BayesianTriangulation,
  title   = {Inferring Capabilities from Task Performance with Bayesian Triangulation},
  author  = {Burden, John and Voudouris, Konstantinos and Burnell, Robert and Rutar, David and Cheke, Lucy and Hern{\'a}ndez-Orallo, Jos{\'e}},
  journal = {arXiv:2309.11975},
  year    = {2023}
}
