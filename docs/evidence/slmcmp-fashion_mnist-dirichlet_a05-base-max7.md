# Agentic end-to-end supervisor runs: FASHION_MNIST, 3 seeds

Generated 2026-11-28 12:15 UTC by `scripts/paper_campaign.py`.

Each seed is one independent run of the autonomous supervisor on the real
8-Jetson cluster. The on-device SLM (Qwen3.5 4B on the master Jetson) chooses
every improvement step from the allowed tools; fixed rules only start the
baseline, compress a model that met the target and stop the run. Accuracy is
the coordinator's global evaluation of every worker's model on the official
test split (mean over the 8 workers); decisions use the global validation split.

## Goal

| Setting | Value |
|---|---|
| dataset | fashion_mnist |
| target_accuracy | 0.995 |
| max_experiments | 7 |
| time_budget_minutes | 480 |
| dataset_split | dirichlet_a05 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 78.01% ± 2.82 |
| Best model test accuracy | 89.16% ± 1.77 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 89.16% ± 1.77 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | completed | 79.53% | Measurement | mesh | 28,554 | 89.69% ± 0.00% | n/a | 3 / 1 / 0 | 1.11 h |
| 17 | completed | 80.44% | Measurement | mesh | 129,770 | 91.02% ± 0.00% | n/a | 3 / 1 / 0 | 0.85 h |
| 42 | completed | 74.05% | Measurement | mesh | 28,810 | 86.78% ± 0.00% | n/a | 3 / 1 / 0 | 0.86 h |

### Seed 2026 (run `b6eef2c1-4248-465e-b4cc-6b458b9ff114`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 79.87% | 79.53% ± 2.98% | 85.60% |
| NAS | completed |  | nas-d3-w16x32x64-k3-conv-h64-drop0 | 28,554 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x64-k3-conv-h64-drop0 | 28,554 | 82.99% | 82.67% ± 1.49% | 88.59% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k3-conv-h64-drop0 | 28,554 | 90.22% | 89.59% ± 0.00% | 89.80% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k3-conv-h64-drop0 | 28,554 | 90.37% | 89.69% ± 0.00% | 90.25% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 79.87% | 75.30% | 79.53% | 75.13% | – | – | – |
| NAS | architecture nas-d3-w16x32x64-k3-conv-h64-drop0 (28554 params) | 82.99% | 80.07% | 82.67% | 80.41% | +3.12 | +3.12 | yes |
| Topology sweep | topology mesh | 90.22% | 90.22% | 89.59% | 89.59% | +7.22 | +10.34 | yes |
| Aggregation sweep | aggregation trimmed_mean | 90.37% | 90.37% | 89.69% | 89.69% | +0.15 | +10.49 | yes |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved only 79.9% validation accuracy on non-IID fashion_mnist data, which is far from the 99.5% target. A search for a better small CNN architecture on each worker's local data should improve performance before proceeding to optimization steps.
1. optimize_topology chosen by slm: Improve topology selection for non-IID fashion_mnist data to boost validation accuracy toward 0.995 target.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.995 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 90.37%, test 89.69%, installed.

### Seed 17 (run `274d355f-57e4-4832-b270-918510ba8f1b`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 81.21% | 80.44% ± 3.56% | 84.43% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 86.09% | 85.43% ± 1.49% | 88.66% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 91.58% | 90.62% ± 0.00% | 91.25% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 92.02% | 91.02% ± 0.00% | 91.51% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 81.21% | 76.25% | 80.44% | 75.24% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop35 (129770 params) | 86.09% | 83.27% | 85.43% | 82.70% | +4.88 | +4.88 | yes |
| Topology sweep | topology mesh | 91.58% | 91.58% | 90.62% | 90.62% | +5.49 | +10.38 | yes |
| Aggregation sweep | aggregation fedavg | 92.02% | 92.02% | 91.02% | 91.02% | +0.43 | +10.81 | yes |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved only 0.812 validation accuracy, which is far from the target of 0.995. Searching for a better CNN architecture on each worker's data is the most logical next step to improve model performance before optimizing other hyperparameters or topology.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.86 on non-IID fashion_mnist data with 8 workers before using remaining 4 experiments and 456 minutes.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.995 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 92.02%, test 91.02%, installed.

### Seed 42 (run `1269b16c-c31d-4d12-bbfb-d976b6f25492`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 74.52% | 74.05% ± 5.51% | 83.20% |
| NAS | completed |  | seed-baseline | 28,810 |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 28,810 | 75.11% | 75.09% ± 4.60% | 85.98% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 28,810 | 87.52% | 86.78% ± 0.00% | 87.88% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 28,810 | 87.25% | 87.37% ± 0.00% | 87.43% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 74.52% | 67.15% | 74.05% | 66.02% | – | – | – |
| NAS | architecture seed-baseline (28810 params) | 75.11% | 70.28% | 75.09% | 69.34% | +0.59 | +0.59 | yes |
| Topology sweep | topology mesh | 87.52% | 87.52% | 86.78% | 86.78% | +12.40 | +12.99 | yes |
| Aggregation sweep | aggregation metropolis | 87.25% | 87.25% | 87.37% | 87.37% | -0.27 | +12.73 | yes |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small (0.745229 accuracy) to reach 0.995 target; NAS will find a better architecture on worker data.
1. optimize_topology chosen by slm: Improve topology selection for non-IID fashion_mnist data to boost validation accuracy toward 0.995 target.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward before proceeding.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 87.52%, test 86.78%, installed.
