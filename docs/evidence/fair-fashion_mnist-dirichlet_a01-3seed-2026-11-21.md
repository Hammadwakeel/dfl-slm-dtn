# Agentic end-to-end supervisor runs: FASHION_MNIST, 3 seeds

Generated 2026-11-23 04:14 UTC by `scripts/paper_campaign.py`.

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
| Baseline test accuracy | 42.22% ± 2.67 |
| Best model test accuracy | 73.25% ± 8.61 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 73.25% ± 8.61 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 43.13% | Measurement | mesh | 129,770 | 82.40% ± 0.00% | n/a | 6 / 1 / 0 | 3.79 h |
| 42 | completed | 44.94% | Measurement | mesh | 109,450 | 61.71% ± 0.00% | n/a | 6 / 1 / 0 | 1.90 h |
| 2026 | completed | 38.58% | Measurement | mesh | 51,754 | 75.64% ± 0.00% | n/a | 6 / 1 / 0 | 1.81 h |

### Seed 17 (run `98432cc6-4ce3-450a-8e00-28dd7f8e5afa`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 43.29% | 43.13% ± 9.36% | 75.80% |
| NAS | failed |  |  |  |  |  |  |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 53.71% | 53.32% ± 7.78% | 76.74% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 81.55% | 80.67% ± 0.00% | 82.00% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 81.57% | 80.65% ± 0.00% | 82.63% |
| HPO | completed |  | hpo-adam-lr3e-3-bs32 | 129,770 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-3-bs32 | 129,770 | 77.03% | 75.45% ± 0.00% | 76.20% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 82.83% | 82.40% ± 0.00% | 84.49% |
| APG learned topology | completed | learned | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 59.56% | 59.07% ± 9.53% | 83.95% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 43.29% | 28.95% | 43.13% | 28.95% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop35 (129770 params) | 53.71% | 40.02% | 53.32% | 39.85% | +10.42 | +10.42 | yes |
| Topology sweep | topology mesh | 81.55% | 81.55% | 80.67% | 80.67% | +27.84 | +38.26 | yes |
| Aggregation sweep | aggregation fedavg | 81.57% | 81.57% | 80.65% | 80.65% | +0.02 | +38.28 | yes |
| HPO | lr 0.003, batch 32 | 77.03% | 77.03% | 75.45% | 75.45% | -4.53 | +33.75 | no (reverted) |
| Synchronization sweep | sync sync | 82.83% | 82.83% | 82.40% | 82.40% | +5.80 | +39.55 | yes |
| APG learned topology | learned (APG) topology | 59.56% | 44.33% | 59.07% | 43.71% | -23.27 | +16.28 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small with only 0.43 accuracy; need to search for a better architecture on worker data before further optimization.
1. optimize_topology chosen by slm: Current directed_ring topology yields only 53.7% accuracy; need to find a better topology for non-IID fashion_mnist data with 12 experiments remaining and 652 minutes budget.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: The current mesh topology yields only 0.828 validation accuracy, far from the 0.99 target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers may improve convergence and final accuracy within the remaining 514 minutes budget.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 82.83%, test 82.40%, installed.

Errors recorded during the run:

- run_nas failed: TrainingError: Worker request failed: worker6: worker6 returned HTTP 409 for /v1/nas/search: {"error":"Isolated dataset NAS search exited with code 1: TimeoutError: Dataset NAS candidate nas-d3-w16x64x64-k5-conv-h0-drop35 exceeded the time budget (RSS 1535.8 MiB)"}; worker5: worker5 

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
