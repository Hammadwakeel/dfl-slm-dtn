# Agentic end-to-end supervisor runs: FEMNIST, 3 seeds

Generated 2026-11-28 15:50 UTC by `scripts/paper_campaign.py`.

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
| target_accuracy | 0.995 |
| max_experiments | 7 |
| time_budget_minutes | 720 |
| dataset_split | dirichlet_a05 |
| rounds | 15 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 69.00% ± 0.82 |
| Best model test accuracy | 79.30% ± 4.61 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 80.84% ± 2.49 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | completed | 67.88% | Measurement | mesh | 73,150 | 81.56% ± 0.00% | n/a | 3 / 1 / 0 | 1.27 h |
| 17 | completed | 69.81% | Measurement | mesh | 133,150 | 83.47% ± 0.00% | n/a | 3 / 1 / 0 | 1.15 h |
| 42 | completed | 69.32% | Measurement | directed_ring | 112,830 | 72.87% ± 3.66% | n/a | 3 / 1 / 0 | 1.14 h |

### Seed 2026 (run `062c1425-6e19-4f0c-9dfa-c9dee07ca7d8`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 68.49% | 67.88% ± 2.51% | 76.50% |
| NAS | completed |  | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 72.23% | 72.12% ± 0.83% | 79.46% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 82.16% | 80.72% ± 0.00% | 80.63% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 82.60% | 81.56% ± 0.00% | 81.19% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 68.49% | 64.33% | 67.88% | 63.17% | – | – | – |
| NAS | architecture nas-d3-w16x32x64-k5-conv-h64-drop0 (73150 params) | 72.23% | 70.31% | 72.12% | 70.53% | +3.74 | +3.74 | yes |
| Topology sweep | topology mesh | 82.16% | 82.16% | 80.72% | 80.72% | +9.93 | +13.68 | yes |
| Aggregation sweep | aggregation trimmed_mean | 82.60% | 82.60% | 81.56% | 81.56% | +0.43 | +14.11 | yes |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small with only 0.6849 validation accuracy; search for a better architecture on worker data to improve performance before further optimization.
1. optimize_topology chosen by slm: Improve topology selection for non-IID femnist data to boost validation accuracy toward 0.995 target
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 82.60%, test 81.56%, installed.

### Seed 17 (run `3dca9aba-8cf0-4efa-a15a-250eea9009f8`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 70.02% | 69.81% ± 3.24% | 76.97% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop0 | 133,150 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop0 | 133,150 | 75.74% | 75.88% ± 1.94% | 80.37% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop0 | 133,150 | 82.98% | 82.76% ± 0.00% | 82.24% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop0 | 133,150 | 83.80% | 83.47% ± 0.00% | 83.21% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 70.02% | 65.34% | 69.81% | 64.76% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop0 (133150 params) | 75.74% | 71.55% | 75.88% | 72.18% | +5.72 | +5.72 | yes |
| Topology sweep | topology mesh | 82.98% | 82.98% | 82.76% | 82.76% | +7.24 | +12.96 | yes |
| Aggregation sweep | aggregation uniform | 83.80% | 83.80% | 83.47% | 83.47% | +0.82 | +13.78 | yes |

Agent decisions:

1. run_nas chosen by slm: Search for a better CNN architecture on worker data to improve validation accuracy from 0.7002 toward target 0.995.
1. optimize_topology chosen by slm: Improve topology selection for non-IID femnist data to boost validation accuracy toward 0.995 target
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.995 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 83.80%, test 83.47%, installed.

### Seed 42 (run `2b116436-3f3e-4b15-90a3-2908b164e7e0`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 69.54% | 69.32% ± 3.73% | 78.65% |
| NAS | completed |  | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 72.85% | 72.78% ± 4.04% | 81.47% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 73.20% | 72.87% ± 3.66% | 81.59% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 72.98% | 72.86% ± 4.02% | 81.56% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 69.54% | 60.58% | 69.32% | 60.87% | – | – | – |
| NAS | architecture nas-d2-w64x64-k5-conv-h64-drop35 (112830 params) | 72.85% | 63.69% | 72.78% | 63.83% | +3.30 | +3.30 | yes |
| Aggregation sweep | aggregation uniform | 73.20% | 65.03% | 72.87% | 65.10% | +0.35 | +3.65 | yes |
| Synchronization sweep | sync semi_sync | 72.98% | 65.28% | 72.86% | 65.09% | -0.22 | +3.44 | yes |

Agent decisions:

1. run_nas chosen by slm: The baseline architecture is underperforming (69.5% accuracy) and NAS can find a better small CNN tailored to the non-IID femnist data before further optimization steps.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to find best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) with current directed_ring topology to improve validation accuracy toward 0.995 target.
1. optimize_sync chosen by slm: Workers may be waiting for slow peers; optimize synchronization mode to improve validation accuracy toward 0.995 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker2), global val 77.70%, test 77.50%, installed.
