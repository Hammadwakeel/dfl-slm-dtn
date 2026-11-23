# Agentic end-to-end supervisor runs: FEMNIST, 3 seeds

Generated 2026-11-23 00:25 UTC by `scripts/paper_campaign.py`.

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
| dataset_split | dirichlet_a09 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 71.78% ± 1.70 |
| Best model test accuracy | 83.62% ± 0.92 |
| Pruned model test accuracy | 83.36% ± 0.34 |
| Model installed on the workers at the end (test) | 83.05% ± 0.52 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 74.11% | Measurement | mesh | 133,150 | 84.68% ± 0.00% | 83.70% (66,632) | 7 / 1 / 0 | 3.18 h |
| 42 | completed | 71.08% | Measurement | mesh | 112,830 | 83.75% ± 0.00% | 83.02% (57,122) | 7 / 1 / 0 | 3.70 h |
| 2026 | completed | 70.14% | Measurement | mesh | 55,134 | 82.43% ± 0.00% | n/a | 6 / 1 / 0 | 3.62 h |

### Seed 17 (run `62f03aeb-4cab-419c-9cd1-7cb53d09181a`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 74.34% | 74.11% ± 1.38% | 78.37% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop0 | 133,150 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop0 | 133,150 | 77.15% | 77.01% ± 4.27% | 79.73% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop0 | 133,150 | 79.20% | 79.19% ± 1.10% | 81.97% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop0 | 133,150 | 79.38% | 79.73% ± 0.51% | 82.25% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop0 | 133,150 | 84.28% | 84.77% ± 0.00% | 84.32% |
| HPO | completed |  | hpo-adam-lr3e-3-bs32 | 133,150 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-3-bs32 | 133,150 | 84.43% | 84.68% ± 0.00% | 84.53% |
| APG learned topology | completed | learned | hpo-adam-lr3e-3-bs32 | 133,150 | 83.47% | 83.40% ± 0.52% | 84.35% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr3e-3-bs32-pruned30 | 66,632 | 83.46% | 83.70% ± 0.00% | 83.39% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 74.34% | 71.16% | 74.11% | 71.45% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop0 (133150 params) | 77.15% | 66.38% | 77.01% | 65.85% | +2.81 | +2.81 | yes |
| Aggregation sweep | aggregation uniform | 79.20% | 77.15% | 79.19% | 76.79% | +2.04 | +4.85 | yes |
| Synchronization sweep | sync sync | 79.38% | 78.18% | 79.73% | 79.09% | +0.19 | +5.04 | yes |
| Topology sweep | topology mesh | 84.28% | 84.28% | 84.77% | 84.77% | +4.90 | +9.94 | yes |
| HPO | lr 0.003, batch 32 | 84.43% | 84.43% | 84.68% | 84.68% | +0.15 | +10.09 | yes |
| APG learned topology | learned (APG) topology | 83.47% | 82.18% | 83.40% | 82.57% | -0.96 | +9.13 | – |
| Pruning + fine-tune | 30% channels pruned (66632 params) | 83.46% | 83.46% | 83.70% | 83.70% | -0.01 | +9.12 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small with only 0.743 validation accuracy; search for a better CNN architecture on each worker's data to improve performance before further optimization.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; select best aggregation rule to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Workers may be waiting for slow peers; optimize synchronization mode to improve convergence speed and accuracy.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.79 to target 0.99 on femnist dataset with 8 workers and 616 minutes remaining.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on worker data to improve validation accuracy toward 0.99 target.
1. learn_topology chosen by slm: The current mesh topology achieved 0.844 accuracy, but we have 5 experiments left and 561 minutes to reach 0.99 accuracy. Learning a new peer-to-peer topology from measured losses may improve convergence on this non-IID femnist dataset where peers learn slowly.
1. prune_model chosen by slm: The model has achieved 0.844 validation accuracy, which is good enough to proceed with pruning to make it smaller and faster on Jetson devices while fine-tuning on all workers.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 83.46% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 83.46%, test 83.70%, installed.

### Seed 42 (run `38ae1cd8-2075-4244-b1c6-043bf0afe591`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 71.28% | 71.08% ± 3.51% | 75.55% |
| NAS | failed |  |  |  |  |  |  |
| NAS | completed |  | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 76.91% | 77.00% ± 1.79% | 80.76% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 83.46% | 83.73% ± 0.00% | 83.30% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 84.13% | 83.75% ± 0.00% | 83.50% |
| HPO | completed |  | hpo-adam-lr3e-3-bs32 | 112,830 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-3-bs32 | 112,830 | 83.31% | 83.33% ± 0.00% | 83.42% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 83.70% | 83.66% ± 0.00% | 83.58% |
| APG learned topology | completed | learned | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 83.12% | 82.67% ± 0.24% | 83.65% |
| Pruning + fine-tune | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35-pruned30 | 57,122 | 83.41% | 83.02% ± 0.00% | 83.38% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 71.28% | 64.68% | 71.08% | 64.73% | – | – | – |
| NAS | architecture nas-d2-w64x64-k5-conv-h64-drop35 (112830 params) | 76.91% | 73.75% | 77.00% | 73.62% | +5.63 | +5.63 | yes |
| Topology sweep | topology mesh | 83.46% | 83.46% | 83.73% | 83.73% | +6.55 | +12.18 | yes |
| Aggregation sweep | aggregation trimmed_mean | 84.13% | 84.13% | 83.75% | 83.75% | +0.67 | +12.85 | yes |
| HPO | lr 0.003, batch 32 | 83.31% | 83.31% | 83.33% | 83.33% | -0.82 | +12.03 | no (reverted) |
| Synchronization sweep | sync sync | 83.70% | 83.70% | 83.66% | 83.66% | +0.38 | +12.41 | yes |
| APG learned topology | learned (APG) topology | 83.12% | 82.23% | 82.67% | 82.22% | -0.58 | +11.83 | – |
| Pruning + fine-tune | 30% channels pruned (57122 params) | 83.41% | 83.41% | 83.02% | 83.02% | +0.30 | +12.13 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small (0.712 accuracy), need to search for a better architecture on worker data before further optimization.
1. optimize_topology chosen by slm: Current directed_ring topology yields 76.9% accuracy; need to improve toward 99% target with remaining 12 experiments and 658 minutes.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's data to improve validation accuracy before final training rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given remaining budget and current mesh topology with trimmed mean aggregation.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, as peers learn slowly from each other on non-IID femnist data.
1. prune_model chosen by slm: The model has achieved 84.13% validation accuracy, which is good enough to proceed with pruning to make it smaller and faster for Jetson devices while maintaining performance.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 83.41% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 83.41%, test 83.02%, installed.

Errors recorded during the run:

- run_nas failed: TrainingError: Worker request failed: worker3: worker3 returned HTTP 409 for /v1/nas/search: {"error":"Isolated dataset NAS search exited with code 1: TimeoutError: Dataset NAS candidate nas-d2-w16x32-k3-conv-h64-drop20-attn exceeded the time budget (RSS 1689.8 MiB)"}; worker4: worke

### Seed 2026 (run `dedf638f-c914-4986-abdf-e9ecab2dee3c`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 70.36% | 70.14% ± 3.82% | 75.38% |
| NAS | failed |  |  |  |  |  |  |
| NAS | completed |  | nas-d3-w16x64x64-k3-conv-h64-drop0 | 55,134 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k3-conv-h64-drop0 | 55,134 | 75.58% | 75.22% ± 1.30% | 79.21% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k3-conv-h64-drop0 | 55,134 | 75.29% | 74.98% ± 1.06% | 79.60% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k3-conv-h64-drop0 | 55,134 | 75.63% | 75.57% ± 0.79% | 79.87% |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 55,134 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr1e-3-bs32 | 55,134 | 75.50% | 75.15% ± 1.32% | 79.16% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 55,134 | 82.78% | 82.43% ± 0.00% | 81.70% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs32 | 55,134 | 80.21% | 79.83% ± 1.12% | 81.96% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 70.36% | 60.39% | 70.14% | 60.76% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k3-conv-h64-drop0 (55134 params) | 75.58% | 74.23% | 75.22% | 73.32% | +5.22 | +5.22 | yes |
| Aggregation sweep | aggregation fedavg | 75.29% | 74.35% | 74.98% | 73.51% | -0.29 | +4.93 | yes |
| Synchronization sweep | sync sync | 75.63% | 74.05% | 75.57% | 74.38% | +0.34 | +5.27 | yes |
| HPO | lr 0.001, batch 32 | 75.50% | 74.23% | 75.15% | 73.63% | -0.13 | +5.14 | yes |
| Topology sweep | topology mesh | 82.78% | 82.78% | 82.43% | 82.43% | +7.28 | +12.42 | yes |
| APG learned topology | learned (APG) topology | 80.21% | 77.26% | 79.83% | 77.33% | -2.57 | +9.85 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline architecture is underperforming (70.36% accuracy) and NAS can find a better small CNN tailored to the femnist dataset's non-IID characteristics.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) with current directed_ring topology to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to minimize waiting on slow peers and improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on each worker's own data to improve validation accuracy from 0.756 toward target 0.99, with 8 experiments remaining and 588 minutes left.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.756 by optimizing peer-to-peer topology on non-IID femnist data with 8 workers and 565 minutes remaining.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from real worker peer-model losses and waiting times to optimize aggregation and synchronization for femnist dataset.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 82.78%, test 82.43%, installed.

Errors recorded during the run:

- run_nas failed: TrainingError: Worker request failed: worker6: worker6 returned HTTP 409 for /v1/nas/search: {"error":"Isolated dataset NAS search exited with code 1: TimeoutError: Dataset NAS candidate nas-d2-w32x32-k5-conv-h64-drop35 exceeded the time budget (RSS 1375.0 MiB)"}; worker8: worker8 re
