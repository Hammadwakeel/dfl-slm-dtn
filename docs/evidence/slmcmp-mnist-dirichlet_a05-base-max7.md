# Agentic end-to-end supervisor runs: MNIST, 3 seeds

Generated 2026-11-28 09:25 UTC by `scripts/paper_campaign.py`.

Each seed is one independent run of the autonomous supervisor on the real
8-Jetson cluster. The on-device SLM (Qwen3.5 4B on the master Jetson) chooses
every improvement step from the allowed tools; fixed rules only start the
baseline, compress a model that met the target and stop the run. Accuracy is
the coordinator's global evaluation of every worker's model on the official
test split (mean over the 8 workers); decisions use the global validation split.

## Goal

| Setting | Value |
|---|---|
| dataset | mnist |
| target_accuracy | 0.995 |
| max_experiments | 7 |
| time_budget_minutes | 360 |
| dataset_split | dirichlet_a05 |
| rounds | 5 |
| local_epochs | 3 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 93.74% ± 2.44 |
| Best model test accuracy | 97.09% ± 1.46 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 98.42% ± 0.42 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | completed | 95.57% | Measurement | directed_ring | 22,682 | 96.78% ± 0.88% | n/a | 3 / 1 / 0 | 0.73 h |
| 17 | completed | 90.29% | Measurement | mesh | 129,770 | 99.02% ± 0.00% | n/a | 3 / 1 / 0 | 0.81 h |
| 42 | completed | 95.35% | Measurement | directed_ring | 32,794 | 95.48% ± 2.27% | n/a | 3 / 1 / 0 | 0.95 h |

### Seed 2026 (run `97e495b7-2b69-4d30-89b1-35bccf15e1b2`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 94.94% | 95.57% ± 1.78% | 97.22% |
| NAS | completed |  | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 95.82% | 96.54% ± 1.17% | 98.03% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 96.18% | 96.78% ± 0.88% | 98.09% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 95.91% | 96.51% ± 1.37% | 97.97% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 94.94% | 91.22% | 95.57% | 92.32% | – | – | – |
| NAS | architecture nas-d3-w16x16x32-k5-conv-h64-drop20 (22682 params) | 95.82% | 93.67% | 96.54% | 94.33% | +0.88 | +0.88 | yes |
| Aggregation sweep | aggregation trimmed_mean | 96.18% | 94.50% | 96.78% | 95.27% | +0.36 | +1.24 | yes |
| Synchronization sweep | sync semi_sync | 95.91% | 92.92% | 96.51% | 93.80% | -0.26 | +0.98 | yes |

Agent decisions:

1. run_nas chosen by slm: Baseline model is small and may be too simple to reach 0.995 accuracy on non-IID data; NAS will find a better architecture for each worker's local data.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID Dirichlet data; need to find best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) with current directed_ring topology to improve validation accuracy toward 0.995 target.
1. optimize_sync chosen by slm: Select best synchronization mode to minimize waiting on slow peers and improve validation accuracy toward 0.995 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker2), global val 97.57%, test 98.11%, installed.

### Seed 17 (run `f1f717ba-183a-44d2-87af-afc594bbca41`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 90.09% | 90.29% ± 5.00% | 96.39% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 | 94.76% | 95.06% ± 2.38% | 98.59% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 | 98.87% | 99.02% ± 0.00% | 98.90% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 | 98.68% | 98.93% ± 0.00% | 98.89% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 90.09% | 81.75% | 90.29% | 81.94% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop0 (129770 params) | 94.76% | 91.23% | 95.06% | 91.44% | +4.67 | +4.67 | yes |
| Topology sweep | topology mesh | 98.87% | 98.87% | 99.02% | 99.02% | +4.10 | +8.78 | yes |
| Aggregation sweep | aggregation trimmed_mean | 98.68% | 98.68% | 98.93% | 98.93% | -0.18 | +8.59 | yes |

Agent decisions:

1. run_nas chosen by slm: Baseline model achieved 0.9009 accuracy; need to reach 0.995 with 6 experiments left. Search for a better CNN architecture on worker data to improve performance potential.
1. optimize_topology chosen by slm: Improve topology selection for non-IID Dirichlet data to boost validation accuracy toward 0.995 target
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID Dirichlet data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 98.87%, test 99.02%, installed.

### Seed 42 (run `575459a2-9907-4c04-946e-ceb3649bede2`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 95.11% | 95.35% ± 2.23% | 96.22% |
| NAS | completed |  | seed-baseline | 32,794 |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 94.61% | 94.97% ± 3.05% | 95.89% |
| Topology sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 95.32% | 95.48% ± 2.27% | 96.07% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 94.89% | 95.21% ± 2.30% | 96.25% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 95.11% | 90.37% | 95.35% | 90.51% | – | – | – |
| NAS | architecture seed-baseline (32794 params) | 94.61% | 88.11% | 94.97% | 88.92% | -0.50 | -0.50 | yes |
| Topology sweep | topology directed_ring | 95.32% | 91.48% | 95.48% | 91.58% | +0.71 | +0.22 | yes |
| Aggregation sweep | aggregation fedavg | 94.89% | 89.40% | 95.21% | 90.54% | -0.43 | -0.21 | yes |

Agent decisions:

1. run_nas chosen by slm: Baseline model achieved 0.951 accuracy; need to reach 0.995 with 6 experiments left. Search for a better CNN architecture on worker data to improve performance potential.
1. optimize_topology chosen by slm: Improve topology selection for non-IID Dirichlet data to boost validation accuracy toward 0.995 target.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID Dirichlet data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker7), global val 98.02%, test 98.14%, installed.
