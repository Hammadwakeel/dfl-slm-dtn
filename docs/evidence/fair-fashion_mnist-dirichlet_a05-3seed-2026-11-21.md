# Agentic end-to-end supervisor runs: FASHION_MNIST, 3 seeds

Generated 2026-11-21 19:43 UTC by `scripts/paper_campaign.py`.

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
| dataset_split | dirichlet_a05 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 78.41% ± 1.31 |
| Best model test accuracy | 89.67% ± 1.05 |
| Pruned model test accuracy | 89.56% |
| Model installed on the workers at the end (test) | 89.47% ± 1.00 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 79.69% | Measurement | mesh | 129,770 | 90.65% ± 0.00% | n/a | 6 / 1 / 0 | 1.34 h |
| 42 | completed | 76.62% | Measurement | mesh | 109,450 | 88.21% ± 0.00% | n/a | 6 / 1 / 0 | 1.48 h |
| 2026 | completed | 78.93% | Measurement | mesh | 69,770 | 90.14% ± 0.00% | 89.56% (33,995) | 7 / 1 / 0 | 1.88 h |

### Seed 17 (run `fd68e1db-bfa1-453c-bc0f-6bdeaa64b44b`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 80.67% | 79.69% ± 3.23% | 84.02% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 86.21% | 85.44% ± 1.36% | 88.47% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 91.73% | 90.64% ± 0.00% | 91.60% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 91.93% | 90.65% ± 0.00% | 91.30% |
| HPO | completed |  | hpo-adam-lr1e-3-bs64 | 129,770 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs64 | 129,770 | 91.52% | 90.69% ± 0.00% | 91.06% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs64 | 129,770 | 91.48% | 90.84% ± 0.00% | 91.12% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs64 | 129,770 | 89.30% | 88.64% ± 0.94% | 91.27% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 80.67% | 76.20% | 79.69% | 74.92% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop35 (129770 params) | 86.21% | 83.45% | 85.44% | 83.19% | +5.54 | +5.54 | yes |
| Topology sweep | topology mesh | 91.73% | 91.73% | 90.64% | 90.64% | +5.52 | +11.06 | yes |
| Aggregation sweep | aggregation fedavg | 91.93% | 91.93% | 90.65% | 90.65% | +0.20 | +11.26 | yes |
| HPO | lr 0.001, batch 64 | 91.52% | 91.52% | 90.69% | 90.69% | -0.42 | +10.84 | yes |
| Synchronization sweep | sync sync | 91.48% | 91.48% | 90.84% | 90.84% | -0.03 | +10.81 | yes |
| APG learned topology | learned (APG) topology | 89.30% | 86.87% | 88.64% | 86.51% | -2.18 | +8.63 | – |

Agent decisions:

1. run_nas chosen by slm: Current baseline accuracy (0.807) is far from target (0.99). Need to search for a better CNN architecture on worker data before optimizing other hyperparameters.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID fashion_mnist data after NAS completion.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy before final training rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy before final experiments
1. learn_topology chosen by slm: The current mesh topology achieved only 0.919 accuracy, far from the 0.99 target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers may find a more efficient communication structure to improve convergence speed and final accurac
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 91.93%, test 90.65%, installed.

### Seed 42 (run `8b17c522-fe7f-497d-8de7-e637e2023609`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 76.72% | 76.62% ± 5.69% | 85.67% |
| NAS | completed |  | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 78.62% | 78.26% ± 4.90% | 88.44% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 88.52% | 87.56% ± 0.00% | 88.78% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 88.53% | 88.21% ± 0.00% | 88.75% |
| HPO | completed |  | hpo-adam-lr3e-4-bs64 | 109,450 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs64 | 109,450 | 87.52% | 87.12% ± 0.00% | 88.10% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 86.65% | 86.50% ± 0.00% | 87.17% |
| APG learned topology | completed | learned | nas-d2-w64x64-k5-conv-h64-drop35 | 109,450 | 86.19% | 86.08% ± 1.66% | 91.45% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 76.72% | 66.68% | 76.62% | 65.69% | – | – | – |
| NAS | architecture nas-d2-w64x64-k5-conv-h64-drop35 (109450 params) | 78.62% | 69.18% | 78.26% | 68.82% | +1.91 | +1.91 | yes |
| Topology sweep | topology mesh | 88.52% | 88.52% | 87.56% | 87.56% | +9.89 | +11.80 | yes |
| Aggregation sweep | aggregation uniform | 88.53% | 88.53% | 88.21% | 88.21% | +0.02 | +11.82 | yes |
| HPO | lr 0.0003, batch 64 | 87.52% | 87.52% | 87.12% | 87.12% | -1.02 | +10.80 | no (reverted) |
| Synchronization sweep | sync sync | 86.65% | 86.65% | 86.50% | 86.50% | -0.87 | +9.93 | no (reverted) |
| APG learned topology | learned (APG) topology | 86.19% | 82.55% | 86.08% | 82.51% | -0.46 | +9.47 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small (76.7% accuracy) to reach 99% target; need better architecture on worker data
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID fashion_mnist data after NAS completion.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given slow peers in mesh topology.
1. learn_topology chosen by slm: The current mesh topology achieved 0.885 accuracy, but we have 5 experiments left and need to reach 0.99. Learning a new peer-to-peer topology from measured losses may improve communication efficiency on non-IID fashion_mnist data, potentially boosting final validation 
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 88.53%, test 88.21%, installed.

### Seed 2026 (run `a976a0c3-a48d-439c-9c4e-6e21cb6dae73`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 28,810 | 78.99% | 78.93% ± 3.97% | 86.07% |
| NAS | completed |  | nas-d3-w16x32x64-k5-conv-h64-drop0 | 69,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x64-k5-conv-h64-drop0 | 69,770 | 83.54% | 83.13% ± 1.45% | 90.38% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0 | 69,770 | 90.28% | 89.69% ± 0.00% | 90.25% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0 | 69,770 | 90.62% | 90.35% ± 0.00% | 90.40% |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 69,770 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs32 | 69,770 | 89.48% | 89.01% ± 0.00% | 89.70% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0 | 69,770 | 90.70% | 90.14% ± 0.00% | 90.78% |
| APG learned topology | completed | learned | nas-d3-w16x32x64-k5-conv-h64-drop0 | 69,770 | 89.03% | 88.31% ± 1.05% | 91.93% |
| Pruning + fine-tune | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0-pruned30 | 33,995 | 89.95% | 89.56% ± 0.00% | 90.13% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 78.99% | 71.37% | 78.93% | 71.01% | – | – | – |
| NAS | architecture nas-d3-w16x32x64-k5-conv-h64-drop0 (69770 params) | 83.54% | 80.87% | 83.13% | 80.57% | +4.55 | +4.55 | yes |
| Topology sweep | topology mesh | 90.28% | 90.28% | 89.69% | 89.69% | +6.74 | +11.29 | yes |
| Aggregation sweep | aggregation trimmed_mean | 90.62% | 90.62% | 90.35% | 90.35% | +0.33 | +11.62 | yes |
| HPO | lr 0.0003, batch 32 | 89.48% | 89.48% | 89.01% | 89.01% | -1.13 | +10.49 | no (reverted) |
| Synchronization sweep | sync sync | 90.70% | 90.70% | 90.14% | 90.14% | +1.22 | +11.71 | yes |
| APG learned topology | learned (APG) topology | 89.03% | 86.57% | 88.31% | 86.06% | -1.67 | +10.04 | – |
| Pruning + fine-tune | 30% channels pruned (33995 params) | 89.95% | 89.95% | 89.56% | 89.56% | +0.92 | +10.96 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved only 0.79 validation accuracy, which is far from the 0.99 target. Searching for a better small CNN architecture on each worker's data to improve performance.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID fashion_mnist data after NAS found a baseline architecture.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID fashion_mnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, as this is likely needed to reach 0.99 accuracy given the current best of 0.907.
1. prune_model chosen by slm: Model accuracy is good enough (0.907) to prune and fine-tune for speed on Jetsons, with 4 experiments and 614 minutes remaining.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 89.95% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 89.95%, test 89.56%, installed.
