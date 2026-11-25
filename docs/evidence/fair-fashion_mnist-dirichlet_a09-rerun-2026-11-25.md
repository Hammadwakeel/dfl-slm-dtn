# Agentic end-to-end supervisor runs: FASHION_MNIST, 1 seed

Generated 2026-11-25 16:47 UTC by `scripts/paper_campaign.py`.

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
| dataset_split | dirichlet_a09 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 83.24% |
| Best model test accuracy | 90.25% |
| Pruned model test accuracy | 89.42% |
| Model installed on the workers at the end (test) | 89.42% |
| Target met (seeds) | 0 / 1 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | completed | 83.24% | Measurement | mesh | 22,682 | 90.25% ± 0.00% | 89.42% (11,065) | 7 / 1 / 0 | 1.52 h |

### Seed 2026 (run `dc848546-2703-43cc-8cb0-81710de8cc46`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 83.87% | 83.24% ± 2.14% | 87.57% |
| NAS | completed |  | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 86.02% | 85.63% ± 1.31% | 89.81% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 90.40% | 90.16% ± 0.00% | 91.03% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 90.33% | 90.20% ± 0.00% | 91.34% |
| HPO | completed |  | hpo-adam-lr3e-3-bs64 | 22,682 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-3-bs64 | 22,682 | 90.17% | 90.02% ± 0.00% | 90.99% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-3-bs64 | 22,682 | 90.55% | 90.25% ± 0.00% | 91.10% |
| APG learned topology | completed | learned | hpo-adam-lr3e-3-bs64 | 22,682 | 88.72% | 88.13% ± 0.75% | 90.60% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr3e-3-bs64-pruned30 | 11,065 | 90.10% | 89.42% ± 0.00% | 90.40% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 83.87% | 80.85% | 83.24% | 79.89% | – | – | – |
| NAS | architecture nas-d3-w16x16x32-k5-conv-h64-drop20 (22682 params) | 86.02% | 83.60% | 85.63% | 82.98% | +2.15 | +2.15 | yes |
| Topology sweep | topology mesh | 90.40% | 90.40% | 90.16% | 90.16% | +4.38 | +6.53 | yes |
| Aggregation sweep | aggregation trimmed_mean | 90.33% | 90.33% | 90.20% | 90.20% | -0.07 | +6.46 | yes |
| HPO | lr 0.003, batch 64 | 90.17% | 90.17% | 90.02% | 90.02% | -0.17 | +6.29 | yes |
| Synchronization sweep | sync sync | 90.55% | 90.55% | 90.25% | 90.25% | +0.38 | +6.68 | yes |
| APG learned topology | learned (APG) topology | 88.72% | 87.27% | 88.13% | 86.95% | -1.82 | +4.85 | – |
| Pruning + fine-tune | 30% channels pruned (11065 params) | 90.10% | 90.10% | 89.42% | 89.42% | +1.38 | +6.23 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline architecture achieved only 0.838729 validation accuracy, which is far from the target of 0.99. Searching for a better small CNN on each worker's data and validation split is the most logical next step to improve model performance before optimizing other hyperparam
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID fashion_mnist data after NAS completion.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy before final training.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, as non-IID data may cause slow convergence.
1. prune_model chosen by slm: Accuracy is good enough (0.9055) to proceed with model pruning for speed on Jetsons, and we have ample time remaining.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 90.10% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 90.10%, test 89.42%, installed.
