# Agentic end-to-end supervisor runs: MNIST, 3 seeds

Generated 2026-11-25 23:44 UTC by `scripts/paper_campaign.py`.

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
| Baseline test accuracy | 56.33% ± 5.94 |
| Best model test accuracy | 96.83% ± 1.09 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 96.83% ± 1.09 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 49.05% | Measurement | mesh | 129,770 | 96.54% ± 0.00% | n/a | 6 / 1 / 0 | 1.60 h |
| 42 | completed | 56.35% | Measurement | mesh | 32,794 | 95.67% ± 0.00% | n/a | 6 / 1 / 0 | 3.78 h |
| 2026 | completed | 63.59% | Measurement | mesh | 22,682 | 98.29% ± 0.00% | n/a | 6 / 1 / 0 | 1.57 h |

### Seed 17 (run `df6f6038-f6cf-4a40-af47-2138010357aa`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 48.79% | 49.05% ± 4.84% | 67.59% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 56.33% | 56.11% ± 6.17% | 78.16% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 94.48% | 94.99% ± 0.00% | 94.93% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 96.77% | 96.54% ± 0.00% | 96.86% |
| HPO | completed |  | hpo-adam-lr1e-3-bs64 | 129,770 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs64 | 129,770 | 95.67% | 95.26% ± 0.00% | 95.41% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 96.45% | 96.33% ± 0.00% | 96.73% |
| APG learned topology | completed | learned | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 80.71% | 80.76% ± 8.01% | 98.01% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 48.79% | 42.44% | 49.05% | 42.72% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop35 (129770 params) | 56.33% | 48.74% | 56.11% | 48.45% | +7.54 | +7.54 | yes |
| Topology sweep | topology mesh | 94.48% | 94.48% | 94.99% | 94.99% | +38.15 | +45.69 | yes |
| Aggregation sweep | aggregation trimmed_mean | 96.77% | 96.77% | 96.54% | 96.54% | +2.28 | +47.97 | yes |
| HPO | lr 0.001, batch 64 | 95.67% | 95.67% | 95.26% | 95.26% | -1.10 | +46.87 | no (reverted) |
| Synchronization sweep | sync semi_sync | 96.45% | 96.45% | 96.33% | 96.33% | +0.78 | +47.66 | yes |
| APG learned topology | learned (APG) topology | 80.71% | 64.23% | 80.76% | 65.08% | -15.74 | +31.92 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small with only 48.8% accuracy; search for a better CNN architecture on worker data to improve performance before tuning hyperparameters.
1. optimize_topology chosen by slm: Improve peer-to-peer topology to enhance learning on non-IID MNIST data before aggregation optimization
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID MNIST data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy before final training.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: The current mesh topology achieved 0.967 accuracy, but we have 5 experiments left and 630 minutes to reach 0.99. Learning a new peer-to-peer topology from measured losses may improve communication efficiency on non-IID MNIST data, potentially boosting validation accurac
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 96.77%, test 96.54%, installed.

### Seed 42 (run `92d2301b-24a1-4e13-a24e-7420ecfa39a1`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 56.23% | 56.35% ± 5.87% | 79.27% |
| NAS | completed |  | seed-baseline | 32,794 |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 58.98% | 58.96% ± 3.97% | 79.51% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 32,794 | 95.30% | 95.67% ± 0.00% | 93.67% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 32,794 | 87.18% | 87.05% ± 0.00% | 93.46% |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 32,794 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs32 | 32,794 | 94.67% | 94.82% ± 0.00% | 94.27% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 32,794 | 94.98% | 95.43% ± 0.00% | 93.02% |
| APG learned topology | completed | learned | seed-baseline | 32,794 | 67.50% | 67.53% ± 5.91% | 81.21% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 56.23% | 41.69% | 56.35% | 42.04% | – | – | – |
| NAS | architecture seed-baseline (32794 params) | 58.98% | 49.19% | 58.96% | 49.76% | +2.75 | +2.75 | yes |
| Topology sweep | topology mesh | 95.30% | 95.30% | 95.67% | 95.67% | +36.32 | +39.07 | yes |
| Aggregation sweep | aggregation trimmed_mean | 87.18% | 87.18% | 87.05% | 87.05% | -8.12 | +30.96 | no (reverted) |
| HPO | lr 0.0003, batch 32 | 94.67% | 94.67% | 94.82% | 94.82% | +7.48 | +38.44 | no (reverted) |
| Synchronization sweep | sync sync | 94.98% | 94.98% | 95.43% | 95.43% | +0.32 | +38.76 | yes |
| APG learned topology | learned (APG) topology | 67.50% | 61.04% | 67.53% | 60.98% | -27.48 | +11.28 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small with only 0.56 accuracy; need to search for a better architecture on worker data before further optimization.
1. optimize_topology chosen by slm: Improve topology selection after NAS to boost validation accuracy toward 0.99 target on non-IID MNIST data.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID MNIST data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, which is useful for non-IID data when peers learn slowly.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 95.30%, test 95.67%, installed.

### Seed 2026 (run `65c1abb2-b37f-47e2-9d02-a5e9d110150c`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 63.19% | 63.59% ± 7.90% | 78.12% |
| NAS | completed |  | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 66.96% | 66.90% ± 10.37% | 83.29% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 97.68% | 98.29% ± 0.00% | 98.13% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 97.52% | 97.97% ± 0.00% | 98.13% |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 22,682 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 22,682 | 97.42% | 97.91% ± 0.00% | 97.96% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 22,682 | 97.13% | 97.83% ± 0.00% | 97.65% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs32 | 22,682 | 88.04% | 89.05% ± 3.55% | 97.43% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 63.19% | 50.51% | 63.59% | 50.37% | – | – | – |
| NAS | architecture nas-d3-w16x16x32-k5-conv-h64-drop20 (22682 params) | 66.96% | 57.06% | 66.90% | 56.27% | +3.77 | +3.77 | yes |
| Topology sweep | topology mesh | 97.68% | 97.68% | 98.29% | 98.29% | +30.72 | +34.49 | yes |
| Aggregation sweep | aggregation fedavg | 97.52% | 97.52% | 97.97% | 97.97% | -0.17 | +34.33 | yes |
| HPO | lr 0.001, batch 32 | 97.42% | 97.42% | 97.91% | 97.91% | -0.10 | +34.23 | yes |
| Synchronization sweep | sync semi_sync | 97.13% | 97.13% | 97.83% | 97.83% | -0.28 | +33.94 | no (reverted) |
| APG learned topology | learned (APG) topology | 88.04% | 82.20% | 89.05% | 83.57% | -9.09 | +24.85 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small (63.19% accuracy) to reach 99% target; search for a better CNN architecture on worker data.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID MNIST data after NAS found a baseline architecture.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID MNIST data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's data to improve validation accuracy before final training rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy before final experiments
1. learn_topology chosen by slm: The current mesh topology achieved 0.9768 accuracy, but we have 5 experiments left and need to reach 0.99. Learning a new peer-to-peer topology from measured losses may improve convergence on this non-IID MNIST dataset, potentially leading to better validation accuracy 
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 97.68%, test 98.29%, installed.
