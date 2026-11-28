# Agentic end-to-end supervisor runs: CIFAR10, 3 seeds

Generated 2026-11-27 15:27 UTC by `scripts/paper_campaign.py`.

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
| dataset_split | dirichlet_a09 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 68.90% ± 2.57 |
| Best model test accuracy | 82.44% ± 0.98 |
| Pruned model test accuracy | 81.54% |
| Model installed on the workers at the end (test) | 82.44% ± 0.98 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 71.31% | Measurement | mesh | 324,394 | 81.98% ± 0.00% | n/a | 6 / 1 / 0 | 3.28 h |
| 42 | completed | 65.34% | Measurement | mesh | 297,706 | 83.81% ± 0.00% | n/a | 6 / 1 / 0 | 2.39 h |
| 2026 | completed | 70.05% | Pruning + fine-tune | mesh | 97,700 | 81.54% ± 0.00% | 81.54% (97,700) | 7 / 1 / 0 | 2.09 h |

### Seed 17 (run `bd61af77-e6ab-4e5f-ba84-0a3ced48f92c`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 70.96% | 71.31% ± 6.99% | 74.25% |
| NAS | completed |  | seed-baseline | 324,394 |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 324,394 | 69.62% | 69.22% ± 8.86% | 73.17% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | baseline | 324,394 | 82.84% | 81.98% ± 0.00% | 83.37% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | baseline | 324,394 | 82.00% | 82.05% ± 0.00% | 82.80% |
| HPO | completed |  | hpo-adam-lr3e-4-bs8 | 324,394 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs8 | 324,394 | 81.68% | 81.47% ± 0.00% | 82.54% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | baseline | 324,394 | 81.50% | 81.41% ± 0.00% | 82.43% |
| APG learned topology | completed | learned | baseline | 324,394 | 79.36% | 79.24% ± 0.94% | 82.91% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 70.96% | 53.48% | 71.31% | 53.47% | – | – | – |
| NAS | architecture seed-baseline (324394 params) | 69.62% | 46.70% | 69.22% | 46.68% | -1.34 | -1.34 | no (reverted) |
| Topology sweep | topology mesh | 82.84% | 82.84% | 81.98% | 81.98% | +13.22 | +11.88 | yes |
| Aggregation sweep | aggregation fedavg | 82.00% | 82.00% | 82.05% | 82.05% | -0.84 | +11.04 | no (reverted) |
| HPO | lr 0.0003, batch 8 | 81.68% | 81.68% | 81.47% | 81.47% | -0.32 | +10.72 | no (reverted) |
| Synchronization sweep | sync semi_sync | 81.50% | 81.50% | 81.41% | 81.41% | -0.18 | +10.54 | no (reverted) |
| APG learned topology | learned (APG) topology | 79.36% | 76.70% | 79.24% | 77.05% | -2.13 | +8.40 | – |

Agent decisions:

1. run_nas chosen by slm: Search for a better CNN architecture to improve validation accuracy from 0.7096 toward target 0.99, as current baseline is underperforming and NAS can find optimized models on worker data.
1. optimize_topology chosen by slm: Improve topology selection to enhance learning on non-IID CIFAR10 data after NAS found a better architecture.
1. optimize_aggregation chosen by slm: Workers' models may disagree on CIFAR-10 non-IID data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given remaining budget and current mesh topology performance.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, which is useful for non-IID data like CIFAR10 where peers learn slowly.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 82.84%, test 81.98%, installed.

### Seed 42 (run `85153dfd-e6a7-4578-a2e1-31d68643df50`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 65.40% | 65.34% ± 13.57% | 71.90% |
| NAS | completed |  | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 | 71.41% | 70.93% ± 5.54% | 77.87% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 | 83.82% | 84.04% ± 0.00% | 82.40% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 | 83.90% | 83.40% ± 0.00% | 82.43% |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 297,706 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 297,706 | 84.78% | 83.81% ± 0.00% | 82.27% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 297,706 | 83.28% | 83.38% ± 0.00% | 81.66% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs32 | 297,706 | 81.91% | 82.09% ± 2.23% | 83.61% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 65.40% | 33.02% | 65.34% | 33.20% | – | – | – |
| NAS | architecture nas-d3-w32x64x128-k3-conv-h64-drop0-c2 (297706 params) | 71.41% | 62.26% | 70.93% | 61.98% | +6.01 | +6.01 | yes |
| Topology sweep | topology mesh | 83.82% | 83.82% | 84.04% | 84.04% | +12.41 | +18.42 | yes |
| Aggregation sweep | aggregation fedavg | 83.90% | 83.90% | 83.40% | 83.40% | +0.08 | +18.50 | yes |
| HPO | lr 0.001, batch 32 | 84.78% | 84.78% | 83.81% | 83.81% | +0.88 | +19.38 | yes |
| Synchronization sweep | sync sync | 83.28% | 83.28% | 83.38% | 83.38% | -1.50 | +17.88 | no (reverted) |
| APG learned topology | learned (APG) topology | 81.91% | 77.52% | 82.09% | 77.33% | -1.37 | +16.51 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small (accuracy 0.654) to reach the target of 0.99; search for a better architecture on worker data.
1. optimize_topology chosen by slm: Improve topology selection for non-IID CIFAR10 data after NAS to boost validation accuracy toward 0.99 target.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID CIFAR10 data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given remaining budget and current mesh topology.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, which is useful for non-IID data like CIFAR10 when peers learn slowly.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 84.78%, test 83.81%, installed.

### Seed 2026 (run `e5d72ac7-da24-4242-ba93-49e7d5dea686`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 70.12% | 70.05% ± 3.85% | 74.40% |
| NAS | completed |  | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 71.39% | 70.98% ± 3.08% | 76.43% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 80.48% | 79.96% ± 0.00% | 80.39% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 78.66% | 78.96% ± 0.00% | 80.13% |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 197,290 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs32 | 197,290 | 76.56% | 75.75% ± 0.00% | 76.17% |
| APG learned topology | completed | learned | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 78.49% | 78.06% ± 0.93% | 81.19% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 79.82% | 79.79% ± 0.00% | 79.87% |
| Pruning + fine-tune | completed | mesh | nas-d2-w32x128-k3-conv-h0-drop35-c2-pruned30 | 97,700 | 82.22% | 81.54% ± 0.00% | 81.49% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 70.12% | 62.28% | 70.05% | 62.24% | – | – | – |
| NAS | architecture nas-d2-w32x128-k3-conv-h0-drop35-c2 (197290 params) | 71.39% | 64.36% | 70.98% | 63.46% | +1.27 | +1.27 | yes |
| Topology sweep | topology mesh | 80.48% | 80.48% | 79.96% | 79.96% | +9.08 | +10.36 | yes |
| Aggregation sweep | aggregation metropolis | 78.66% | 78.66% | 78.96% | 78.96% | -1.82 | +8.54 | no (reverted) |
| HPO | lr 0.0003, batch 32 | 76.56% | 76.56% | 75.75% | 75.75% | -2.10 | +6.44 | no (reverted) |
| APG learned topology | learned (APG) topology | 78.49% | 76.74% | 78.06% | 76.51% | +1.93 | +8.37 | – |
| Synchronization sweep | sync semi_sync | 79.82% | 79.82% | 79.79% | 79.79% | +1.33 | +9.70 | no (reverted) |
| Pruning + fine-tune | 30% channels pruned (97700 params) | 82.22% | 82.22% | 81.54% | 81.54% | +2.40 | +12.10 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small (70% accuracy) to reach 99% target; search for a better CNN architecture on worker data.
1. optimize_topology chosen by slm: Improve topology selection for non-IID CIFAR-10 data after NAS to boost validation accuracy toward 0.99 target.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID CIFAR10 data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's data to improve validation accuracy before final training.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers to better handle non-IID data where peers learn slowly.
1. optimize_sync chosen by slm: Select the best synchronization mode (sync, semi_sync, or async) to maximize quality-gated reward given workers are ready and time budget is large.
1. prune_model chosen by slm: Current accuracy is 0.8048, well below the 0.99 target, so pruning and fine-tuning is premature; need more optimization first.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 82.22% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 82.22%, test 81.54%, installed.
