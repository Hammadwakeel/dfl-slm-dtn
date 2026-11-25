# Agentic end-to-end supervisor runs: FASHION_MNIST, 2 seeds

Generated 2026-11-25 15:16 UTC by `scripts/paper_campaign.py`.

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
| Baseline test accuracy | 47.33% ± 0.93 |
| Best model test accuracy | 69.11% ± 4.98 |
| Pruned model test accuracy | 64.14% |
| Model installed on the workers at the end (test) | 73.99% ± 0.10 |
| Target met (seeds) | 0 / 2 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 7 | completed | 48.26% | Measurement | mesh | 34,154 | 74.09% ± 0.00% | n/a | 6 / 1 / 0 | 2.18 h |
| 99 | completed | 46.40% | Pruning + fine-tune | learned | 8,403 | 64.14% ± 7.35% | 64.14% (8,403) | 7 / 1 / 0 | 1.87 h |

### Seed 7 (run `9315ff67-8a9c-4b96-950e-454863bea5f8`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 48.73% | 48.26% ± 7.54% | 59.75% |
| NAS | completed |  | nas-d2-w32x64-k3-conv-h64-drop20-attn | 34,154 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w32x64-k3-conv-h64-drop20-attn | 34,154 | 49.84% | 49.47% ± 7.38% | 69.77% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w32x64-k3-conv-h64-drop20-attn | 34,154 | 70.03% | 69.36% ± 0.00% | 64.90% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w32x64-k3-conv-h64-drop20-attn | 34,154 | 73.75% | 72.82% ± 0.00% | 69.91% |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 34,154 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs32 | 34,154 | 74.40% | 74.22% ± 0.00% | 73.95% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs32 | 34,154 | 74.83% | 74.09% ± 0.00% | 72.53% |
| APG learned topology | completed | learned | hpo-adam-lr3e-4-bs32 | 34,154 | 65.49% | 64.64% ± 3.93% | 82.25% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 48.73% | 41.13% | 48.26% | 41.36% | – | – | – |
| NAS | architecture nas-d2-w32x64-k3-conv-h64-drop20-attn (34154 params) | 49.84% | 39.95% | 49.47% | 39.10% | +1.11 | +1.11 | yes |
| Topology sweep | topology mesh | 70.03% | 70.03% | 69.36% | 69.36% | +20.19 | +21.30 | yes |
| Aggregation sweep | aggregation trimmed_mean | 73.75% | 73.75% | 72.82% | 72.82% | +3.72 | +25.02 | yes |
| HPO | lr 0.0003, batch 32 | 74.40% | 74.40% | 74.22% | 74.22% | +0.65 | +25.67 | yes |
| Synchronization sweep | sync sync | 74.83% | 74.83% | 74.09% | 74.09% | +0.43 | +26.10 | yes |
| APG learned topology | learned (APG) topology | 65.49% | 59.70% | 64.64% | 58.44% | -9.34 | +16.76 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small with only 0.487 validation accuracy; search for a better architecture on worker data to improve performance before proceeding to optimization steps.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID fashion_mnist data after NAS found a baseline architecture.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on best architecture to improve validation accuracy before final training rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given remaining budget and current mesh topology.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, as current validation accuracy (0.748) is far from target (0.99).
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 74.83%, test 74.09%, installed.

### Seed 99 (run `f5b81042-dfb1-439e-8261-6515eac02728`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 46.49% | 46.40% ± 5.93% | 48.66% |
| NAS | completed |  | nas-d3-w16x32x32-k3-conv-h64-drop20 | 17,130 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x32-k3-conv-h64-drop20 | 17,130 | 52.64% | 52.34% ± 5.18% | 58.83% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x32-k3-conv-h64-drop20 | 17,130 | 53.06% | 53.06% ± 6.17% | 57.53% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x32-k3-conv-h64-drop20 | 17,130 | 52.61% | 52.45% ± 8.85% | 83.81% |
| HPO | completed |  | hpo-adam-lr3e-3-bs32 | 17,130 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-3-bs32 | 17,130 | 51.48% | 51.35% ± 9.20% | 81.28% |
| Topology sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x32-k3-conv-h64-drop20 | 17,130 | 52.83% | 52.64% ± 7.81% | 87.53% |
| APG learned topology | completed | learned | nas-d3-w16x32x32-k3-conv-h64-drop20 | 17,130 | 59.83% | 59.47% ± 5.85% | 83.84% |
| Pruning + fine-tune | completed | learned | nas-d3-w16x32x32-k3-conv-h64-drop20-pruned30 | 8,403 | 64.69% | 64.14% ± 7.35% | 88.30% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 46.49% | 31.50% | 46.40% | 31.93% | – | – | – |
| NAS | architecture nas-d3-w16x32x32-k3-conv-h64-drop20 (17130 params) | 52.64% | 44.87% | 52.34% | 44.40% | +6.15 | +6.15 | yes |
| Aggregation sweep | aggregation fedavg | 53.06% | 42.42% | 53.06% | 42.83% | +0.41 | +6.57 | yes |
| Synchronization sweep | sync async | 52.61% | 40.37% | 52.45% | 40.95% | -0.45 | +6.12 | yes |
| HPO | lr 0.003, batch 32 | 51.48% | 39.98% | 51.35% | 39.78% | -1.13 | +4.99 | no (reverted) |
| Topology sweep | topology directed_ring | 52.83% | 40.13% | 52.64% | 40.02% | +1.35 | +6.34 | yes |
| APG learned topology | learned (APG) topology | 59.83% | 45.05% | 59.47% | 45.05% | +7.00 | +13.34 | – |
| Pruning + fine-tune | 30% channels pruned (8403 params) | 64.69% | 51.05% | 64.14% | 51.27% | +4.86 | +18.20 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small with only 0.46 validation accuracy; search for a better architecture on worker data to improve performance before further optimization.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) with current directed_ring topology to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to minimize waiting on slow peers and improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on each worker's local data to improve validation accuracy before final topology selection.
1. optimize_topology chosen by slm: Improve validation accuracy on fashion_mnist with 7 experiments remaining and 644 minutes budget by selecting the best peer-to-peer topology from measured losses.
1. learn_topology chosen by slm: The current directed ring topology yields only 51-53% validation accuracy, far from the 99% target. Learning a new topology from measured peer-model losses and waiting times on real workers is the most promising next step to improve convergence and performance, especial
1. prune_model chosen by slm: The model has achieved 59.8% validation accuracy on Fashion-MNIST, which is good enough to proceed with pruning to make it smaller and faster for Jetson deployment while maintaining performance.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 64.69% (kept).

Model left on every worker: Pruning + fine-tune (worker3), global val 74.60%, test 73.89%, installed.
