# Agentic end-to-end supervisor runs: FEMNIST, 3 seeds

Generated 2026-11-22 13:54 UTC by `scripts/paper_campaign.py`.

Each seed is one independent run of the autonomous supervisor on the real
8-Jetson cluster. The on-device SLM (Qwen3.5 4B on the master Jetson) chooses
every improvement step from the allowed tools; fixed rules only start the
baseline, compress a model that met the target and stop the run. Accuracy is
the coordinator's global evaluation of every worker's model on the official
test split (mean over the 8 workers); decisions use the global validation split.

## Goal

| Setting | Value |
|---|---|
| dataset | femnist |
| target_accuracy | 0.99 |
| max_experiments | 16 |
| time_budget_minutes | 720 |
| dataset_split | dirichlet_a05 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 66.73% ± 0.86 |
| Best model test accuracy | 82.73% ± 1.00 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 82.73% ± 1.00 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 67.30% | Measurement | mesh | 133,150 | 83.90% ± 0.00% | n/a | 6 / 1 / 0 | 2.00 h |
| 42 | completed | 67.37% | Measurement | mesh | 112,830 | 82.84% ± 0.00% | n/a | 6 / 1 / 0 | 2.19 h |
| 2026 | completed | 65.52% | Measurement | mesh | 73,150 | 81.45% ± 0.00% | n/a | 6 / 1 / 0 | 2.50 h |

### Seed 17 (run `b971a806-6b7d-4c96-ae2f-2f6cf7125e5a`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 67.54% | 67.30% ± 3.77% | 73.46% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 133,150 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 133,150 | 74.70% | 74.52% ± 2.68% | 79.85% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 133,150 | 83.06% | 83.06% ± 0.00% | 82.58% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 133,150 | 83.06% | 83.29% ± 0.00% | 82.66% |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 133,150 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 133,150 | 83.28% | 83.40% ± 0.00% | 82.85% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 133,150 | 83.83% | 83.90% ± 0.00% | 83.73% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs32 | 133,150 | 81.16% | 80.76% ± 0.73% | 84.23% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 67.54% | 60.83% | 67.30% | 60.32% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop35 (133150 params) | 74.70% | 67.86% | 74.52% | 68.00% | +7.16 | +7.16 | yes |
| Topology sweep | topology mesh | 83.06% | 83.06% | 83.06% | 83.06% | +8.36 | +15.52 | yes |
| Aggregation sweep | aggregation uniform | 83.06% | 83.06% | 83.29% | 83.29% | +0.00 | +15.52 | yes |
| HPO | lr 0.001, batch 32 | 83.28% | 83.28% | 83.40% | 83.40% | +0.22 | +15.74 | yes |
| Synchronization sweep | sync sync | 83.83% | 83.83% | 83.90% | 83.90% | +0.55 | +16.29 | yes |
| APG learned topology | learned (APG) topology | 81.16% | 80.25% | 80.76% | 79.61% | -2.67 | +13.62 | – |

Agent decisions:

1. run_nas chosen by slm: Search a better small CNN on every worker's own data and validation split to improve baseline accuracy from 0.675 to target 0.99.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID femnist data after NAS completion.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy before final training rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy before final experiments
1. learn_topology chosen by slm: Improve topology learning on non-IID femnist data to boost validation accuracy toward 0.99 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 83.83%, test 83.90%, installed.

### Seed 42 (run `bc9199de-4187-4599-b2fd-2c018d5e800a`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 67.44% | 67.37% ± 3.58% | 77.21% |
| NAS | completed |  | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 71.03% | 70.61% ± 4.97% | 80.20% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 82.41% | 82.48% ± 0.00% | 82.30% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 82.71% | 83.14% ± 0.00% | 82.80% |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 112,830 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 112,830 | 83.40% | 82.84% ± 0.00% | 83.01% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 112,830 | 82.61% | 82.58% ± 0.00% | 82.48% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs32 | 112,830 | 79.90% | 80.56% ± 1.65% | 82.32% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 67.44% | 59.94% | 67.37% | 60.02% | – | – | – |
| NAS | architecture nas-d2-w64x64-k5-conv-h64-drop35 (112830 params) | 71.03% | 59.29% | 70.61% | 59.04% | +3.59 | +3.59 | yes |
| Topology sweep | topology mesh | 82.41% | 82.41% | 82.48% | 82.48% | +11.39 | +14.98 | yes |
| Aggregation sweep | aggregation trimmed_mean | 82.71% | 82.71% | 83.14% | 83.14% | +0.30 | +15.28 | yes |
| HPO | lr 0.001, batch 32 | 83.40% | 83.40% | 82.84% | 82.84% | +0.68 | +15.96 | yes |
| Synchronization sweep | sync sync | 82.61% | 82.61% | 82.58% | 82.58% | -0.78 | +15.18 | no (reverted) |
| APG learned topology | learned (APG) topology | 79.90% | 76.66% | 80.56% | 76.91% | -2.71 | +12.46 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline accuracy (0.674) is far from target (0.99); search for a better CNN architecture on worker data to improve performance before tuning hyperparameters or optimizing topology.
1. optimize_topology chosen by slm: Improve topology selection for non-IID femnist data to boost validation accuracy toward 0.99 target.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy before final training.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy before final experiments
1. learn_topology chosen by slm: Improve topology learning on non-IID femnist data to boost validation accuracy toward 0.99 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 83.40%, test 82.84%, installed.

### Seed 2026 (run `ef20b0b9-fa84-42a2-b1b0-a8a93d97f241`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 65.78% | 65.52% ± 2.31% | 74.97% |
| NAS | completed |  | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 71.38% | 70.85% ± 1.55% | 78.55% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 70.71% | 70.48% ± 1.36% | 77.80% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 70.91% | 70.91% ± 1.81% | 78.37% |
| HPO | completed |  | hpo-adam-lr3e-3-bs64 | 73,150 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-3-bs64 | 73,150 | 70.32% | 70.03% ± 1.95% | 77.28% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 82.10% | 81.45% ± 0.00% | 80.61% |
| APG learned topology | completed | learned | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 77.72% | 77.40% ± 1.84% | 82.87% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 65.78% | 59.94% | 65.52% | 60.37% | – | – | – |
| NAS | architecture nas-d3-w16x32x64-k5-conv-h64-drop0 (73150 params) | 71.38% | 68.44% | 70.85% | 67.88% | +5.60 | +5.60 | yes |
| Aggregation sweep | aggregation metropolis | 70.71% | 68.66% | 70.48% | 68.05% | -0.67 | +4.93 | no (reverted) |
| Synchronization sweep | sync sync | 70.91% | 67.44% | 70.91% | 67.89% | +0.21 | +5.14 | yes |
| HPO | lr 0.003, batch 64 | 70.32% | 67.38% | 70.03% | 66.21% | -0.59 | +4.55 | no (reverted) |
| Topology sweep | topology mesh | 82.10% | 82.10% | 81.45% | 81.45% | +11.77 | +16.32 | yes |
| APG learned topology | learned (APG) topology | 77.72% | 74.53% | 77.40% | 74.09% | -4.38 | +11.94 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small with only 0.657 accuracy; need to search for a better architecture on worker data before further optimization.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to find best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) with current directed_ring topology to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy on non-IID femnist data with 11 experiments remaining and 655 minutes budget.
1. run_hpo chosen by slm: Tune learning rate and batch size on each worker's data to improve validation accuracy from 0.71 to target 0.99, with 633 minutes remaining.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.71 to target 0.99 with remaining 616 minutes and 7 experiments; current directed_ring topology underperforms potential peer-to-peer configurations on non-IID femnist data.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, as peers learn slowly from each other on non-IID femnist data.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 82.10%, test 81.45%, installed.
