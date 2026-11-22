# Agentic end-to-end supervisor runs: FASHION_MNIST, 3 seeds

Generated 2026-11-22 00:46 UTC by `scripts/paper_campaign.py`.

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
| Baseline test accuracy | 83.61% ± 0.41 |
| Best model test accuracy | 89.98% ± 0.74 |
| Pruned model test accuracy | 89.11% ± 0.97 |
| Model installed on the workers at the end (test) | 89.11% ± 0.97 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 84.19% | Measurement | mesh | 129,770 | 90.85% ± 0.00% | 90.41% (64,240) | 7 / 1 / 0 | 1.68 h |
| 42 | completed | 83.27% | Measurement | mesh | 28,810 | 89.05% ± 0.00% | 88.08% (14,493) | 7 / 1 / 0 | 1.65 h |
| 2026 | completed | 83.38% | Measurement | mesh | 22,682 | 90.04% ± 0.00% | 88.84% (11,065) | 7 / 1 / 0 | 1.69 h |

### Seed 17 (run `7be393d1-0bd5-4fd7-9fed-2c7b38e746d5`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 84.90% | 84.19% ± 1.66% | 87.06% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 | 88.34% | 87.43% ± 1.28% | 90.38% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 | 91.80% | 90.79% ± 0.00% | 91.67% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 | 91.53% | 90.31% ± 0.00% | 91.59% |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 129,770 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs32 | 129,770 | 91.23% | 90.53% ± 0.00% | 91.15% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 | 91.98% | 90.85% ± 0.00% | 91.53% |
| APG learned topology | completed | learned | nas-d3-w16x64x64-k5-conv-h0-drop0 | 129,770 | 90.82% | 89.88% ± 0.37% | 91.33% |
| Pruning + fine-tune | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop0-pruned30 | 64,240 | 91.47% | 90.41% ± 0.00% | 91.18% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 84.90% | 81.93% | 84.19% | 81.44% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop0 (129770 params) | 88.34% | 86.07% | 87.43% | 85.31% | +3.43 | +3.43 | yes |
| Topology sweep | topology mesh | 91.80% | 91.80% | 90.79% | 90.79% | +3.46 | +6.90 | yes |
| Aggregation sweep | aggregation metropolis | 91.53% | 91.53% | 90.31% | 90.31% | -0.27 | +6.63 | yes |
| HPO | lr 0.0003, batch 32 | 91.23% | 91.23% | 90.53% | 90.53% | -0.30 | +6.33 | no (reverted) |
| Synchronization sweep | sync semi_sync | 91.98% | 91.98% | 90.85% | 90.85% | +0.75 | +7.08 | yes |
| APG learned topology | learned (APG) topology | 90.82% | 90.53% | 89.88% | 89.22% | -1.16 | +5.92 | – |
| Pruning + fine-tune | 30% channels pruned (64240 params) | 91.47% | 91.47% | 90.41% | 90.41% | +0.65 | +6.56 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline architecture achieved only 0.849 validation accuracy, which is far from the 0.99 target. A better small CNN should be found on each worker's data to improve performance before further optimization steps.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID fashion_mnist data after NAS completion.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's data to improve validation accuracy before final training rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given remaining budget and current mesh topology with metropolis aggregation.
1. learn_topology chosen by slm: The current mesh topology achieved 0.9198 accuracy, but the target is 0.99. Learning a new topology from measured peer-model losses and waiting times on real workers may improve convergence and final accuracy, especially given the remaining budget of 5 experiments and 6
1. prune_model chosen by slm: The model has achieved 91.98% validation accuracy, which is good enough to proceed with pruning to make it smaller and faster for Jetson devices while fine-tuning on all workers.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 91.47% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 91.47%, test 90.41%, installed.

Errors recorded during the run:

- measure_step failed: Could not reach worker4 to check training status: <urlopen error timed out>

### Seed 42 (run `c412f174-dac9-4027-998c-914ec16d8696`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 83.34% | 83.27% ± 1.99% | 86.22% |
| NAS | completed |  | seed-baseline | 28,810 |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 28,810 | 83.62% | 83.45% ± 1.42% | 86.51% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 28,810 | 89.03% | 89.05% ± 0.00% | 89.79% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 28,810 | 88.27% | 87.90% ± 0.00% | 89.01% |
| HPO | completed |  | hpo-adam-lr1e-3-bs64 | 28,810 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs64 | 28,810 | 88.45% | 88.52% ± 0.00% | 89.24% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 28,810 | 88.68% | 88.20% ± 0.00% | 89.21% |
| APG learned topology | completed | learned | seed-baseline | 28,810 | 87.80% | 87.93% ± 0.85% | 88.59% |
| Pruning + fine-tune | completed | mesh | seed-baseline-pruned30 | 14,493 | 88.18% | 88.08% ± 0.00% | 88.57% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 83.34% | 80.47% | 83.27% | 80.31% | – | – | – |
| NAS | architecture seed-baseline (28810 params) | 83.62% | 81.83% | 83.45% | 81.74% | +0.28 | +0.28 | yes |
| Topology sweep | topology mesh | 89.03% | 89.03% | 89.05% | 89.05% | +5.41 | +5.69 | yes |
| Aggregation sweep | aggregation metropolis | 88.27% | 88.27% | 87.90% | 87.90% | -0.77 | +4.93 | no (reverted) |
| HPO | lr 0.001, batch 64 | 88.45% | 88.45% | 88.52% | 88.52% | +0.18 | +5.11 | no (reverted) |
| Synchronization sweep | sync semi_sync | 88.68% | 88.68% | 88.20% | 88.20% | +0.23 | +5.34 | yes |
| APG learned topology | learned (APG) topology | 87.80% | 86.28% | 87.93% | 86.10% | -0.89 | +4.46 | – |
| Pruning + fine-tune | 30% channels pruned (14493 params) | 88.18% | 88.18% | 88.08% | 88.08% | +0.39 | +4.84 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model achieved only 83% accuracy on fashion_mnist; need to search for a better small CNN architecture to improve validation accuracy toward 99%.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID fashion_mnist data after NAS completion.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. run_hpo chosen by slm: Tune learning rate and batch size on workers' own data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given remaining budget and current mesh topology performance.
1. learn_topology chosen by slm: The current mesh topology yields 0.89 accuracy, far from the 0.99 target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers may improve communication efficiency and model convergence for non-IID fashion_mnist data.
1. prune_model chosen by slm: Accuracy is good enough (0.89) to proceed with pruning for speed on Jetsons, and we have ample time remaining.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 88.18% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 88.18%, test 88.08%, installed.

### Seed 2026 (run `6c220b8e-49eb-448d-bbc7-70bad254f70f`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 83.47% | 83.38% ± 1.62% | 86.67% |
| NAS | completed |  | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 86.46% | 85.97% ± 0.89% | 89.81% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 90.45% | 90.04% ± 0.00% | 90.92% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 90.25% | 90.10% ± 0.00% | 91.00% |
| HPO | completed |  | hpo-adam-lr1e-3-bs64 | 22,682 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs64 | 22,682 | 89.98% | 89.59% ± 0.00% | 90.45% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs64 | 22,682 | 90.20% | 89.65% ± 0.00% | 90.29% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs64 | 22,682 | 89.44% | 88.44% ± 0.31% | 90.45% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr1e-3-bs64-pruned30 | 11,065 | 89.42% | 88.84% ± 0.00% | 89.66% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 83.47% | 80.85% | 83.38% | 80.84% | – | – | – |
| NAS | architecture nas-d3-w16x16x32-k5-conv-h64-drop20 (22682 params) | 86.46% | 84.82% | 85.97% | 84.42% | +2.99 | +2.99 | yes |
| Topology sweep | topology mesh | 90.45% | 90.45% | 90.04% | 90.04% | +3.99 | +6.97 | yes |
| Aggregation sweep | aggregation fedavg | 90.25% | 90.25% | 90.10% | 90.10% | -0.20 | +6.77 | yes |
| HPO | lr 0.001, batch 64 | 89.98% | 89.98% | 89.59% | 89.59% | -0.27 | +6.51 | yes |
| Synchronization sweep | sync semi_sync | 90.20% | 90.20% | 89.65% | 89.65% | +0.22 | +6.73 | yes |
| APG learned topology | learned (APG) topology | 89.44% | 88.75% | 88.44% | 87.86% | -0.76 | +5.96 | – |
| Pruning + fine-tune | 30% channels pruned (11065 params) | 89.42% | 89.42% | 88.84% | 88.84% | -0.02 | +5.94 | – |

Agent decisions:

1. run_nas chosen by slm: Current baseline accuracy (0.835) is far from target (0.99). Need to search for a better CNN architecture on worker data before optimizing other hyperparameters.
1. optimize_topology chosen by slm: Improve peer-to-peer communication efficiency on non-IID fashion_mnist data to reach 99% validation accuracy within remaining budget.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy before final training rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining 640 minutes budget.
1. learn_topology chosen by slm: The current mesh topology achieved only 0.9045 validation accuracy, far from the 0.99 target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers may improve convergence and final accuracy, especially given the non-IID nature of f
1. prune_model chosen by slm: The model has achieved 90.45% validation accuracy, which is good enough to proceed with pruning to make it smaller and faster for Jetson devices while fine-tuning on all workers.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 89.42% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 89.42%, test 88.84%, installed.
