# Agentic end-to-end supervisor runs: MNIST, 3 seeds

Generated 2026-11-26 10:32 UTC by `scripts/paper_campaign.py`.

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
| dataset_split | dirichlet_a09 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 98.09% ± 0.54 |
| Best model test accuracy | 98.89% ± 0.30 |
| Pruned model test accuracy | 99.09% ± 0.00 |
| Model installed on the workers at the end (test) | 98.90% ± 0.25 |
| Target met (seeds) | 1 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 97.33% | Pruning + fine-tune | learned | 64,240 | 99.10% ± 0.04% | 99.10% (64,240) | 7 / 1 / 0 | 1.60 h |
| 42 | completed | 98.49% | Measurement | directed_ring | 32,794 | 98.47% ± 0.09% | n/a | 6 / 1 / 0 | 1.96 h |
| 2026 | completed | 98.46% | Pruning + fine-tune | learned | 11,065 | 99.09% ± 0.05% | 99.09% (11,065) | 7 / 1 / 0 | 1.46 h |

### Seed 17 (run `e4607464-e68a-4f7f-b62a-a528af47bf6b`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 97.24% | 97.33% ± 1.56% | 98.60% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 98.06% | 98.08% ± 1.21% | 99.18% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 98.45% | 98.59% ± 0.46% | 99.26% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 98.56% | 98.63% ± 0.44% | 99.27% |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-4-bs32 | 129,770 | 98.39% | 98.44% ± 0.53% | 99.14% |
| Topology sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-4-bs32 | 129,770 | 98.43% | 98.48% ± 0.51% | 99.11% |
| APG learned topology | completed | learned | hpo-adam-lr3e-4-bs32 | 129,770 | 99.00% | 99.00% ± 0.04% | 99.16% |
| Pruning + fine-tune | completed | learned | hpo-adam-lr3e-4-bs32-pruned30 | 64,240 | 99.17% | 99.10% ± 0.04% | 99.31% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 97.24% | 94.30% | 97.33% | 94.47% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop35 (129770 params) | 98.06% | 95.67% | 98.08% | 95.62% | +0.83 | +0.83 | yes |
| Aggregation sweep | aggregation uniform | 98.45% | 97.33% | 98.59% | 97.75% | +0.38 | +1.21 | yes |
| Synchronization sweep | sync sync | 98.56% | 97.52% | 98.63% | 97.81% | +0.12 | +1.32 | yes |
| HPO | lr 0.0003, batch 32 | 98.39% | 97.33% | 98.44% | 97.60% | -0.18 | +1.15 | yes |
| Topology sweep | topology directed_ring | 98.43% | 97.28% | 98.48% | 97.66% | +0.05 | +1.19 | yes |
| APG learned topology | learned (APG) topology | 99.00% | 98.88% | 99.00% | 98.96% | +0.56 | +1.76 | – |
| Pruning + fine-tune | 30% channels pruned (64240 params) | 99.17% | 99.10% | 99.10% | 99.06% | +0.18 | +1.93 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved 0.972 accuracy, which is below the 0.99 target. NAS will search for a better small CNN architecture on each worker's data to improve performance.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID MNIST data; need to find best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) with current directed_ring topology to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Workers are waiting for slow peers; optimize synchronization mode to improve convergence speed and final accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on each worker's data to improve validation accuracy toward 0.99 target.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.9856 by optimizing peer-to-peer topology for non-IID MNIST data with remaining 7 experiments and 651 minutes budget.
1. learn_topology chosen by slm: Improve peer-to-peer communication efficiency on MNIST to reach 99% validation accuracy within remaining budget.
1. prune_model chosen by slm: Target accuracy 0.99 is nearly reached at 0.989957; pruning will reduce model size and improve inference speed on Jetsons while maintaining high accuracy, with ample time remaining.
1. finish_workflow chosen by rule: The measured result meets the goal.

Pruning attempts: 30% → 99.17% (kept).

Model left on every worker: Pruning + fine-tune (worker6), global val 99.23%, test 99.06%, installed.

### Seed 42 (run `63256fff-5348-4634-970b-b674fa63ef8d`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 98.24% | 98.49% ± 0.36% | 98.78% |
| NAS | completed |  | seed-baseline | 32,794 |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 98.34% | 98.51% ± 0.07% | 98.73% |
| Topology sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 98.36% | 98.47% ± 0.09% | 98.65% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 98.26% | 98.49% ± 0.15% | 98.61% |
| HPO | completed |  | hpo-adam-lr3e-4-bs64 | 32,794 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-4-bs64 | 32,794 | 97.91% | 97.98% ± 0.25% | 98.22% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-4-bs64 | 32,794 | 97.32% | 97.50% ± 0.30% | 98.02% |
| APG learned topology | completed | learned | hpo-adam-lr3e-4-bs64 | 32,794 | 98.11% | 98.04% ± 0.13% | 98.18% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 98.24% | 97.38% | 98.49% | 97.65% | – | – | – |
| NAS | architecture seed-baseline (32794 params) | 98.34% | 98.20% | 98.51% | 98.39% | +0.10 | +0.10 | yes |
| Topology sweep | topology directed_ring | 98.36% | 98.27% | 98.47% | 98.31% | +0.02 | +0.12 | yes |
| Aggregation sweep | aggregation metropolis | 98.26% | 98.02% | 98.49% | 98.31% | -0.09 | +0.03 | yes |
| HPO | lr 0.0003, batch 64 | 97.91% | 97.78% | 97.98% | 97.64% | -0.35 | -0.33 | yes |
| Synchronization sweep | sync async | 97.32% | 96.60% | 97.50% | 97.08% | -0.59 | -0.92 | no (reverted) |
| APG learned topology | learned (APG) topology | 98.11% | 97.88% | 98.04% | 97.87% | +0.79 | -0.13 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved 0.982 accuracy, which is close to but below the 0.99 target. NAS will search for a better small CNN architecture on each worker's data to potentially improve performance before further optimization steps.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID MNIST data after NAS completion.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID MNIST data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: Improve topology learning on non-IID MNIST data to boost validation accuracy toward 0.99 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker7), global val 98.43%, test 98.55%, installed.

### Seed 2026 (run `e57cd4d2-9ec6-4661-a0bd-b445d580700c`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 98.09% | 98.46% ± 0.33% | 98.77% |
| NAS | completed |  | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 98.14% | 98.50% ± 0.19% | 98.52% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 98.18% | 98.62% ± 0.12% | 98.69% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 98.20% | 98.54% ± 0.19% | 98.78% |
| HPO | completed |  | hpo-adam-lr1e-3-bs64 | 22,682 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr1e-3-bs64 | 22,682 | 98.09% | 98.35% ± 0.28% | 98.67% |
| Topology sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr1e-3-bs64 | 22,682 | 98.03% | 98.31% ± 0.25% | 98.56% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs64 | 22,682 | 98.71% | 98.95% ± 0.07% | 99.15% |
| Pruning + fine-tune | completed | learned | hpo-adam-lr1e-3-bs64-pruned30 | 11,065 | 98.74% | 99.09% ± 0.05% | 98.93% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 98.09% | 97.57% | 98.46% | 97.84% | – | – | – |
| NAS | architecture nas-d3-w16x16x32-k5-conv-h64-drop20 (22682 params) | 98.14% | 97.88% | 98.50% | 98.17% | +0.05 | +0.05 | yes |
| Aggregation sweep | aggregation fedavg | 98.18% | 97.90% | 98.62% | 98.44% | +0.04 | +0.09 | yes |
| Synchronization sweep | sync semi_sync | 98.20% | 97.90% | 98.54% | 98.24% | +0.02 | +0.12 | yes |
| HPO | lr 0.001, batch 64 | 98.09% | 97.45% | 98.35% | 98.04% | -0.11 | +0.00 | yes |
| Topology sweep | topology directed_ring | 98.03% | 97.63% | 98.31% | 97.90% | -0.06 | -0.05 | yes |
| APG learned topology | learned (APG) topology | 98.71% | 98.62% | 98.95% | 98.82% | +0.68 | +0.63 | – |
| Pruning + fine-tune | 30% channels pruned (11065 params) | 98.74% | 98.67% | 99.09% | 99.02% | +0.03 | +0.65 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved 0.98 accuracy, but the target is 0.99. NAS can find a better small CNN architecture on each worker's data to improve performance before tuning hyperparameters or optimizing topology.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID MNIST data; need to find best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) with current directed_ring topology to improve validation accuracy from 0.9814 toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to minimize waiting on slow peers and improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on each worker's local data to improve validation accuracy toward 0.99 target.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.982 to target 0.99 using remaining 7 experiments and 652 minutes on MNIST with non-IID data.
1. learn_topology chosen by slm: The current directed ring topology achieved 0.982 validation accuracy, but learning a new peer-to-peer topology from measured losses and waiting times on real workers may improve performance further before the experiments run out.
1. prune_model chosen by slm: The model has achieved 98.71% accuracy, close to the 99% target. Pruning will reduce model size and improve inference speed on Jetson devices while potentially maintaining or slightly improving accuracy through fine-tuning.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 98.74% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 98.83%, test 99.08%, installed.
