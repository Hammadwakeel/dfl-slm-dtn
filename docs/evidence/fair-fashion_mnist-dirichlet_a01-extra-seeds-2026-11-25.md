# Agentic end-to-end supervisor runs: FASHION_MNIST, 2 seeds

Generated 2026-11-25 11:01 UTC by `scripts/paper_campaign.py`.

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
| Baseline test accuracy | 60.03% ± 1.08 |
| Best model test accuracy | 60.03% ± 1.08 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 71.17% ± 0.40 |
| Target met (seeds) | 0 / 2 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 7 | completed | 58.95% | Baseline | directed_ring | 28,810 | 58.95% ± 7.10% | n/a | 6 / 1 / 0 | 1.59 h |
| 99 | completed | 61.10% | Baseline | directed_ring | 28,810 | 61.10% ± 4.51% | n/a | 6 / 1 / 0 | 1.56 h |

### Seed 7 (run `84e487ed-94e9-433c-be70-4838ba569f38`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 59.40% | 58.95% ± 7.10% | 70.74% |
| NAS | completed |  | nas-d2-w32x64-k3-conv-h64-drop20 | 24,010 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w32x64-k3-conv-h64-drop20 | 24,010 | 45.62% | 45.28% ± 10.35% | 56.26% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | baseline | 28,810 | 57.10% | 56.69% ± 0.00% | 49.52% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | baseline | 28,810 | 51.92% | 51.44% ± 6.08% | 62.18% |
| HPO | completed |  | hpo-adam-lr3e-4-bs64 | 28,810 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-4-bs64 | 28,810 | 50.54% | 49.77% ± 4.55% | 68.53% |
| APG learned topology | completed | learned | baseline | 28,810 | 56.65% | 56.28% ± 8.63% | 64.35% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | baseline | 28,810 | 48.89% | 48.47% ± 6.95% | 59.15% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 59.40% | 51.38% | 58.95% | 51.34% | – | – | – |
| NAS | architecture nas-d2-w32x64-k3-conv-h64-drop20 (24010 params) | 45.62% | 32.45% | 45.28% | 32.23% | -13.78 | -13.78 | no (reverted) |
| Topology sweep | topology mesh | 57.10% | 57.10% | 56.69% | 56.69% | +11.48 | -2.30 | no (reverted) |
| Aggregation sweep | aggregation metropolis | 51.92% | 43.13% | 51.44% | 42.14% | -5.18 | -7.48 | no (reverted) |
| HPO | lr 0.0003, batch 64 | 50.54% | 44.87% | 49.77% | 44.50% | -1.38 | -8.86 | no (reverted) |
| APG learned topology | learned (APG) topology | 56.65% | 42.07% | 56.28% | 42.39% | +6.11 | -2.75 | – |
| Synchronization sweep | sync semi_sync | 48.89% | 37.87% | 48.47% | 37.58% | -7.75 | -10.51 | no (reverted) |

Agent decisions:

1. run_nas chosen by slm: Current baseline accuracy (0.594) is far from target (0.99). Need to search for a better CNN architecture on worker data before optimizing other hyperparameters.
1. optimize_topology chosen by slm: Improve topology selection after NAS to better handle non-IID data on fashion_mnist with 13 experiments remaining and 691 minutes budget.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to find best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy from 0.594 toward target 0.99, after NAS and topology optimization are complete.
1. learn_topology chosen by slm: Improve topology selection based on measured peer-model losses and waiting times to enhance convergence for fashion_mnist dataset.
1. optimize_sync chosen by slm: Select the best synchronization mode (sync, semi_sync, or async) to maximize quality-gated reward given workers are ready and time budget is large.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Baseline (worker1), global val 71.08%, test 70.77%, installed.

### Seed 99 (run `39006d6d-aa43-42e1-b807-c581ae29ddb5`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 61.71% | 61.10% ± 4.51% | 78.91% |
| NAS | completed |  | nas-d2-w32x64-k5-conv-h0-drop20 | 53,130 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w32x64-k5-conv-h0-drop20 | 53,130 | 55.33% | 55.03% ± 5.32% | 74.69% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | baseline | 28,810 | 57.93% | 57.37% ± 0.00% | 61.74% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | baseline | 28,810 | 51.35% | 51.01% ± 6.19% | 58.15% |
| HPO | completed |  | hpo-adam-lr1e-3-bs64 | 28,810 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr1e-3-bs64 | 28,810 | 45.14% | 45.28% ± 6.62% | 56.07% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | baseline | 28,810 | 43.92% | 43.75% ± 13.70% | 84.68% |
| APG learned topology | completed | learned | baseline | 28,810 | 54.43% | 54.40% ± 7.17% | 73.02% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 61.71% | 57.13% | 61.10% | 56.77% | – | – | – |
| NAS | architecture nas-d2-w32x64-k5-conv-h0-drop20 (53130 params) | 55.33% | 47.35% | 55.03% | 47.05% | -6.38 | -6.38 | no (reverted) |
| Topology sweep | topology mesh | 57.93% | 57.93% | 57.37% | 57.37% | +2.60 | -3.78 | no (reverted) |
| Aggregation sweep | aggregation uniform | 51.35% | 39.23% | 51.01% | 38.82% | -6.58 | -10.35 | no (reverted) |
| HPO | lr 0.001, batch 64 | 45.14% | 34.13% | 45.28% | 34.17% | -6.22 | -16.57 | no (reverted) |
| Synchronization sweep | sync async | 43.92% | 21.02% | 43.75% | 20.74% | -1.21 | -17.78 | no (reverted) |
| APG learned topology | learned (APG) topology | 54.43% | 41.40% | 54.40% | 41.64% | +10.50 | -7.28 | – |

Agent decisions:

1. run_nas chosen by slm: Current baseline accuracy (61.7%) is far from target (99%). Need to search for a better CNN architecture on worker data before optimizing other hyperparameters.
1. optimize_topology chosen by slm: Improve topology selection to enhance learning on non-IID fashion_mnist data after NAS found a candidate architecture.
1. optimize_aggregation chosen by slm: Workers' models disagree on non-IID fashion_mnist data; need to find best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on workers' own data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given workers are waiting for slow peers in directed_ring topology.
1. learn_topology chosen by slm: Improve topology learning on non-IID fashion_mnist data to boost validation accuracy toward 0.99 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Baseline (worker6), global val 72.70%, test 71.58%, installed.
