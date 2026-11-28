# Agentic end-to-end supervisor runs: CIFAR10, 3 seeds

Generated 2026-11-26 22:42 UTC by `scripts/paper_campaign.py`.

Each seed is one independent run of the autonomous supervisor on the real
8-Jetson cluster. The on-device SLM (Qwen3.5 4B on the master Jetson) chooses
every improvement step from the allowed tools; fixed rules only start the
baseline, compress a model that met the target and stop the run. Accuracy is
the coordinator's global evaluation of every worker's model on the official
test split (mean over the 8 workers); decisions use the global validation split.

## Goal

| Setting | Value |
|---|---|
| dataset | cifar10 |
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
| Baseline test accuracy | 35.36% ± 6.78 |
| Best model test accuracy | 61.42% ± 3.96 |
| Pruned model test accuracy | 65.44% |
| Model installed on the workers at the end (test) | 61.42% ± 3.96 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 42.38% | Pruning + fine-tune | mesh | 161,094 | 65.44% ± 0.00% | 65.44% (161,094) | 7 / 1 / 0 | 6.01 h |
| 42 | completed | 26.19% | Measurement | mesh | 95,434 | 56.03% ± 0.00% | n/a | 6 / 1 / 0 | 2.04 h |
| 2026 | completed | 37.50% | Measurement | mesh | 324,394 | 62.79% ± 0.00% | n/a | 6 / 1 / 0 | 4.09 h |

### Seed 17 (run `b782f92c-602c-4b4c-9f8b-0ba81cd5e38b`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 42.46% | 42.38% ± 13.37% | 52.00% |
| NAS | completed |  | seed-baseline | 324,394 |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 324,394 | 42.24% | 42.21% ± 13.10% | 52.13% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 324,394 | 60.96% | 60.48% ± 0.00% | 59.28% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 324,394 | 60.78% | 61.04% ± 0.00% | 58.12% |
| HPO | completed |  | hpo-adam-lr1e-3-bs8 | 324,394 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs8 | 324,394 | 60.76% | 60.12% ± 0.00% | 59.79% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs8 | 324,394 | 42.38% | 42.56% ± 5.77% | 58.24% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs8 | 324,394 | 59.46% | 59.68% ± 0.00% | 57.38% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr1e-3-bs8-pruned30 | 161,094 | 65.78% | 65.44% ± 0.00% | 64.52% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 42.46% | 30.60% | 42.38% | 30.68% | – | – | – |
| NAS | architecture seed-baseline (324394 params) | 42.24% | 27.72% | 42.21% | 28.13% | -0.22 | -0.22 | yes |
| Topology sweep | topology mesh | 60.96% | 60.96% | 60.48% | 60.48% | +18.72 | +18.50 | yes |
| Aggregation sweep | aggregation fedavg | 60.78% | 60.78% | 61.04% | 61.04% | -0.18 | +18.32 | yes |
| HPO | lr 0.001, batch 8 | 60.76% | 60.76% | 60.12% | 60.12% | -0.02 | +18.30 | yes |
| APG learned topology | learned (APG) topology | 42.38% | 35.28% | 42.56% | 35.44% | -18.38 | -0.08 | – |
| Synchronization sweep | sync sync | 59.46% | 59.46% | 59.68% | 59.68% | +17.08 | +17.00 | no (reverted) |
| Pruning + fine-tune | 30% channels pruned (161094 params) | 65.78% | 65.78% | 65.44% | 65.44% | +6.32 | +23.32 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small with only 42% validation accuracy; search for a better CNN architecture on each worker's data to improve performance before tuning hyperparameters or optimizing topology.
1. optimize_topology chosen by slm: NAS found a new architecture, need to find an optimal topology for it on non-IID CIFAR-10 data before training.
1. optimize_aggregation chosen by slm: Workers' models may disagree on CIFAR-10 non-IID data; select best aggregation rule to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best architecture on every worker's own data to improve validation accuracy toward 0.99 target.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, as peers may learn slowly from each other on CIFAR10 data.
1. optimize_sync chosen by slm: Select the best synchronization mode to maximize validation accuracy within the remaining 449 minutes, as the current sync mode is null and workers are ready.
1. prune_model chosen by slm: Current accuracy is 0.61, far from target 0.99. Pruning and fine-tuning on all workers can improve efficiency while potentially boosting accuracy by removing redundant channels.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 65.78% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 65.78%, test 65.44%, installed.

### Seed 42 (run `aaf7fe07-ae7b-49d7-8808-c34d5171961f`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 26.24% | 26.19% ± 11.20% | 26.14% |
| NAS | completed |  | nas-d3-w32x64x128-k3-conv-h0-drop35 | 95,434 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w32x64x128-k3-conv-h0-drop35 | 95,434 | 35.82% | 35.73% ± 6.02% | 57.76% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w32x64x128-k3-conv-h0-drop35 | 95,434 | 35.69% | 35.51% ± 6.05% | 53.02% |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 95,434 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr1e-3-bs32 | 95,434 | 35.01% | 34.91% ± 6.24% | 57.65% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w32x64x128-k3-conv-h0-drop35 | 95,434 | 55.74% | 56.03% ± 0.00% | 51.64% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w32x64x128-k3-conv-h0-drop35 | 95,434 | 55.30% | 54.10% ± 0.00% | 52.84% |
| APG learned topology | completed | learned | nas-d3-w32x64x128-k3-conv-h0-drop35 | 95,434 | 35.40% | 35.03% ± 5.84% | 52.12% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 26.24% | 10.00% | 26.19% | 10.00% | – | – | – |
| NAS | architecture nas-d3-w32x64x128-k3-conv-h0-drop35 (95434 params) | 35.82% | 25.86% | 35.73% | 26.31% | +9.58 | +9.58 | yes |
| Aggregation sweep | aggregation fedavg | 35.69% | 26.44% | 35.51% | 26.93% | -0.13 | +9.45 | yes |
| HPO | lr 0.001, batch 32 | 35.01% | 27.22% | 34.91% | 26.92% | -0.68 | +8.77 | no (reverted) |
| Topology sweep | topology mesh | 55.74% | 55.74% | 56.03% | 56.03% | +20.73 | +29.50 | yes |
| Synchronization sweep | sync semi_sync | 55.30% | 55.30% | 54.10% | 54.10% | -0.44 | +29.06 | yes |
| APG learned topology | learned (APG) topology | 35.40% | 25.90% | 35.03% | 25.91% | -19.90 | +9.16 | – |

Agent decisions:

1. run_nas chosen by slm: Search a better small CNN on every worker's own data and validation split since baseline accuracy is very low (0.26) and we have ample time to improve architecture before reaching target 0.99.
1. optimize_aggregation chosen by slm: Workers' models disagree on non-IID CIFAR10 data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) with current directed_ring topology to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on each worker's data to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.3581 to target 0.99 with 9 experiments and 637 minutes remaining on CIFAR-10 non-IID data.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: The current mesh topology yields only 55.74% validation accuracy, far from the 99% target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers may improve convergence and performance, especially given the non-IID nature of CIFAR-1
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 55.74%, test 56.03%, installed.

### Seed 2026 (run `e2a4d9a5-32d2-4fc3-8629-66a52a6a4294`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 37.34% | 37.50% ± 6.99% | 50.41% |
| NAS | completed |  | nas-d3-w32x32x128-k3-conv-h64-drop35 | 56,810 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w32x32x128-k3-conv-h64-drop35 | 56,810 | 34.61% | 34.65% ± 5.56% | 51.87% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | baseline | 324,394 | 56.74% | 56.54% ± 0.00% | 52.38% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | baseline | 324,394 | 58.22% | 58.02% ± 0.00% | 56.63% |
| HPO | completed |  | hpo-adam-lr3e-4-bs8 | 324,394 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs8 | 324,394 | 61.68% | 61.63% ± 0.00% | 61.06% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs8 | 324,394 | 62.94% | 62.79% ± 0.00% | 62.84% |
| APG learned topology | completed | learned | hpo-adam-lr3e-4-bs8 | 324,394 | 40.39% | 40.51% ± 3.38% | 80.69% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 37.34% | 26.02% | 37.50% | 26.45% | – | – | – |
| NAS | architecture nas-d3-w32x32x128-k3-conv-h64-drop35 (56810 params) | 34.61% | 24.28% | 34.65% | 24.24% | -2.74 | -2.74 | no (reverted) |
| Topology sweep | topology mesh | 56.74% | 56.74% | 56.54% | 56.54% | +22.13 | +19.39 | yes |
| Aggregation sweep | aggregation uniform | 58.22% | 58.22% | 58.02% | 58.02% | +1.48 | +20.88 | yes |
| HPO | lr 0.0003, batch 8 | 61.68% | 61.68% | 61.63% | 61.63% | +3.46 | +24.34 | yes |
| Synchronization sweep | sync semi_sync | 62.94% | 62.94% | 62.79% | 62.79% | +1.26 | +25.59 | yes |
| APG learned topology | learned (APG) topology | 40.39% | 33.44% | 40.51% | 33.53% | -22.55 | +3.04 | – |

Agent decisions:

1. run_nas chosen by slm: Search for a better CNN architecture since the baseline validation accuracy (0.37) is far from the target (0.99) and NAS can find a more suitable model for this dataset.
1. optimize_topology chosen by slm: NAS architecture found, need to find optimal topology for non-IID CIFAR10 data before training
1. optimize_aggregation chosen by slm: Workers' models may disagree on CIFAR-10 non-IID data; select best aggregation rule to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best architecture on every worker's own data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given slow peers in mesh topology.
1. learn_topology chosen by slm: The current mesh topology achieved only 0.6294 validation accuracy, far from the 0.99 target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers is the most promising next step to improve convergence and performance, especially g
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 62.94%, test 62.79%, installed.
