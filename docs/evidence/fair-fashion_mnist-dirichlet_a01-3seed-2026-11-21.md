# Agentic end-to-end supervisor runs: FASHION_MNIST, 3 seeds

Generated 2026-11-21 15:00 UTC by `scripts/paper_campaign.py`.

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
| target_accuracy | 0.99 |
| max_experiments | 16 |
| time_budget_minutes | 720 |
| dataset_split | dirichlet_a01 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 41.85% ± 2.60 |
| Best model test accuracy | 74.32% ± 9.81 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 74.32% ± 9.81 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 42.03% | Measurement | mesh | 129,770 | 85.62% ± 0.00% | n/a | 6 / 1 / 0 | 1.83 h |
| 42 | completed | 44.94% | Measurement | mesh | 109,450 | 61.71% ± 0.00% | n/a | 6 / 1 / 0 | 1.90 h |
| 2026 | completed | 38.58% | Measurement | mesh | 51,754 | 75.64% ± 0.00% | n/a | 6 / 1 / 0 | 1.81 h |

### Seed 17 (run `2184fee2-c451-43ab-b3c6-baf0f610a1c2`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 42.24% | 42.03% ± 6.81% | 71.94% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 53.00% | 52.73% ± 6.30% | 75.38% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 86.43% | 85.62% ± 0.00% | 87.40% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 82.35% | 81.65% ± 0.00% | 82.87% |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 129,770 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 129,770 | 84.92% | 84.13% ± 0.00% | 85.71% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 129,770 | 84.13% | 83.95% ± 0.00% | 85.47% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs32 | 129,770 | 59.58% | 59.05% ± 8.95% | 82.13% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 42.24% | 31.17% | 42.03% | 31.08% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop35 (129770 params) | 53.00% | 40.62% | 52.73% | 40.28% | +10.75 | +10.75 | yes |
| Topology sweep | topology mesh | 86.43% | 86.43% | 85.62% | 85.62% | +33.44 | +44.19 | yes |
| Aggregation sweep | aggregation fedavg | 82.35% | 82.35% | 81.65% | 81.65% | -4.08 | +40.11 | yes |
| HPO | lr 0.001, batch 32 | 84.92% | 84.92% | 84.13% | 84.13% | +2.57 | +42.67 | yes |
| Synchronization sweep | sync sync | 84.13% | 84.13% | 83.95% | 83.95% | -0.78 | +41.89 | yes |
| APG learned topology | learned (APG) topology | 59.58% | 41.68% | 59.05% | 41.32% | -24.55 | +17.34 | – |

Agent decisions:

1. run_nas chosen by slm: Current baseline accuracy (0.42) is far from target (0.99). Need to search for a better CNN architecture on worker data before optimizing other hyperparameters or topology.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID fashion_mnist data after NAS completion.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on best NAS architecture to improve validation accuracy before final experiments.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: The current mesh topology achieved only 0.864 validation accuracy, far from the 0.99 target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers is the most promising next step to improve convergence and performance, especially gi
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 86.43%, test 85.62%, installed.

### Seed 42 (run `46d4b307-9f60-42c8-afef-3d7af367b75f`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 45.32% | 44.94% ± 6.98% | 46.54% |
| NAS | completed |  | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 46.72% | 46.47% ± 8.91% | 48.30% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 50.46% | 50.04% ± 10.31% | 52.53% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 40.86% | 40.84% ± 7.75% | 72.81% |
| HPO | completed |  | hpo-adam-lr3e-3-bs64 | 109,450 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-3-bs64 | 109,450 | 42.96% | 42.90% ± 11.15% | 57.87% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 61.95% | 61.71% ± 0.00% | 52.73% |
| APG learned topology | completed | learned | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 57.29% | 57.11% ± 2.82% | 55.05% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 45.32% | 39.38% | 44.94% | 38.60% | – | – | – |
| NAS | architecture nas-d2-w64x64-k5-conv-h64-drop35 (109450 params) | 46.72% | 31.97% | 46.47% | 31.93% | +1.40 | +1.40 | yes |
| Aggregation sweep | aggregation fedavg | 50.46% | 35.67% | 50.04% | 35.63% | +3.74 | +5.14 | yes |
| Synchronization sweep | sync async | 40.86% | 33.10% | 40.84% | 33.06% | -9.59 | -4.45 | no (reverted) |
| HPO | lr 0.003, batch 64 | 42.96% | 25.10% | 42.90% | 24.86% | +2.10 | -2.36 | no (reverted) |
| Topology sweep | topology mesh | 61.95% | 61.95% | 61.71% | 61.71% | +18.99 | +16.63 | yes |
| APG learned topology | learned (APG) topology | 57.29% | 54.07% | 57.11% | 54.33% | -4.66 | +11.98 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small with only 45% validation accuracy; search for a better architecture on worker data to improve performance before further optimization.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to find best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) with current directed_ring topology to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Workers are waiting for slow peers; optimize synchronization mode to improve convergence speed and validation accuracy.
1. run_hpo chosen by slm: Tune learning rate and batch size on each worker's own data to improve validation accuracy from 0.504563 toward target 0.99, using remaining 649 minutes budget.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.5045 by optimizing peer-to-peer topology for non-IID fashion_mnist data with 8 workers and 640 minutes remaining.
1. learn_topology chosen by slm: The current mesh topology achieved 0.6195 accuracy, but the target is 0.99. Learning a new peer-to-peer topology from measured losses and waiting times on real workers may improve convergence and final accuracy, especially given the remaining budget of 5 experiments and
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 61.95%, test 61.71%, installed.

### Seed 2026 (run `5bf77132-a377-4996-8a38-5913b4bf84ba`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 38.61% | 38.58% ± 14.66% | 67.84% |
| NAS | completed |  | nas-d3-w16x64x64-k3-conv-h64-drop0 | 51,754 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k3-conv-h64-drop0 | 51,754 | 51.11% | 50.95% ± 15.91% | 80.65% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k3-conv-h64-drop0 | 51,754 | 70.93% | 70.66% ± 0.00% | 70.46% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k3-conv-h64-drop0 | 51,754 | 71.07% | 70.73% ± 0.00% | 70.34% |
| HPO | completed |  | hpo-adam-lr3e-4-bs64 | 51,754 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs64 | 51,754 | 75.22% | 75.64% ± 0.00% | 75.74% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs64 | 51,754 | 75.07% | 74.72% ± 0.00% | 75.05% |
| APG learned topology | completed | learned | hpo-adam-lr3e-4-bs64 | 51,754 | 64.98% | 65.37% ± 9.65% | 88.54% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 38.61% | 14.05% | 38.58% | 14.25% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k3-conv-h64-drop0 (51754 params) | 51.11% | 29.95% | 50.95% | 29.73% | +12.50 | +12.50 | yes |
| Topology sweep | topology mesh | 70.93% | 70.93% | 70.66% | 70.66% | +19.82 | +32.32 | yes |
| Aggregation sweep | aggregation fedavg | 71.07% | 71.07% | 70.73% | 70.73% | +0.13 | +32.45 | yes |
| HPO | lr 0.0003, batch 64 | 75.22% | 75.22% | 75.64% | 75.64% | +4.15 | +36.60 | yes |
| Synchronization sweep | sync semi_sync | 75.07% | 75.07% | 74.72% | 74.72% | -0.15 | +36.45 | yes |
| APG learned topology | learned (APG) topology | 64.98% | 41.57% | 65.37% | 41.94% | -10.08 | +26.37 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small with only 0.386 validation accuracy; need to search for a better architecture on worker data before tuning hyperparameters or optimizing topology.
1. optimize_topology chosen by slm: Improve peer-to-peer communication efficiency on non-IID fashion_mnist data to reach 99% validation accuracy within remaining budget.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy from 0.71 toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy before final experiments.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, which is useful for non-IID data like fashion_mnist where peers learn slowly.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 75.22%, test 75.64%, installed.
