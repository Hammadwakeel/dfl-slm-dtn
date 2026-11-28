# Agentic end-to-end supervisor runs: MNIST, 3 seeds

Generated 2026-11-26 05:29 UTC by `scripts/paper_campaign.py`.

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
| dataset_split | dirichlet_a05 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 96.96% ± 0.80 |
| Best model test accuracy | 98.87% ± 0.38 |
| Pruned model test accuracy | 99.14% ± 0.08 |
| Model installed on the workers at the end (test) | 98.96% ± 0.26 |
| Target met (seeds) | 1 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 95.84% | Measurement | mesh | 129,770 | 99.20% ± 0.00% | 99.22% (64,240) | 2 / 2 / 0 | 1.05 h |
| 42 | completed | 97.69% | APG learned topology | learned | 32,794 | 98.34% ± 0.25% | n/a | 6 / 1 / 0 | 2.82 h |
| 2026 | completed | 97.35% | Pruning + fine-tune | learned | 11,065 | 99.06% ± 0.05% | 99.06% (11,065) | 7 / 1 / 0 | 1.87 h |

### Seed 17 (run `c814fabe-c7b4-400f-b6b8-36052a9475ca`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 95.68% | 95.84% ± 1.69% | 98.50% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 97.20% | 97.38% ± 1.32% | 99.00% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 | 99.25% | 99.20% ± 0.00% | 99.18% |
| Pruning + fine-tune | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35-pruned30 | 64,240 | 99.12% | 99.22% ± 0.00% | 99.20% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 95.68% | 92.43% | 95.84% | 92.57% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop35 (129770 params) | 97.20% | 94.72% | 97.38% | 94.80% | +1.51 | +1.51 | yes |
| Topology sweep | topology mesh | 99.25% | 99.25% | 99.20% | 99.20% | +2.05 | +3.57 | yes |
| Pruning + fine-tune | 30% channels pruned (64240 params) | 99.12% | 99.12% | 99.22% | 99.22% | -0.13 | +3.43 | – |

Agent decisions:

1. run_nas chosen by slm: Search for a better CNN architecture to improve validation accuracy from 0.956826 toward the target of 0.99, as the baseline model may be suboptimal.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.971953 toward target 0.99 using measured peer-model losses and waiting times on non-IID MNIST data.
1. prune_model chosen by rule: The target is met; compress the model for the Jetsons before finishing.
1. finish_workflow chosen by rule: The measured result meets the goal.

Pruning attempts: 30% → 99.12% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 99.12%, test 99.22%, installed.

### Seed 42 (run `64ddda00-8ef9-4b7e-86bb-3c72d24975e1`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 97.61% | 97.69% ± 0.68% | 98.19% |
| NAS | completed |  | seed-baseline | 32,794 |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 97.55% | 97.60% ± 0.75% | 98.00% |
| Topology sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 97.48% | 97.65% ± 0.69% | 98.34% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 32,794 | 97.47% | 97.57% ± 0.67% | 97.98% |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 32,794 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-4-bs32 | 32,794 | 97.19% | 97.25% ± 0.51% | 98.01% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-4-bs32 | 32,794 | 97.31% | 97.47% ± 0.48% | 97.92% |
| APG learned topology | completed | learned | hpo-adam-lr3e-4-bs32 | 32,794 | 98.42% | 98.34% ± 0.25% | 98.71% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 97.61% | 96.58% | 97.69% | 96.25% | – | – | – |
| NAS | architecture seed-baseline (32794 params) | 97.55% | 96.72% | 97.60% | 96.28% | -0.06 | -0.06 | yes |
| Topology sweep | topology directed_ring | 97.48% | 96.53% | 97.65% | 96.55% | -0.07 | -0.13 | yes |
| Aggregation sweep | aggregation fedavg | 97.47% | 96.22% | 97.57% | 96.45% | -0.02 | -0.14 | yes |
| HPO | lr 0.0003, batch 32 | 97.19% | 96.60% | 97.25% | 96.44% | -0.27 | -0.42 | yes |
| Synchronization sweep | sync semi_sync | 97.31% | 96.98% | 97.47% | 96.68% | +0.11 | -0.31 | yes |
| APG learned topology | learned (APG) topology | 98.42% | 98.18% | 98.34% | 97.93% | +1.11 | +0.81 | – |

Agent decisions:

1. run_nas chosen by slm: Search for a better CNN architecture to improve validation accuracy from 0.976 to target 0.99 before running hyperparameter optimization.
1. optimize_topology chosen by slm: Improve topology selection to enhance peer learning on non-IID MNIST data after NAS has identified a promising architecture.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID MNIST data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy before final experiments.
1. learn_topology chosen by slm: Improve topology learning on non-IID MNIST data to boost validation accuracy toward 0.99 target.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: APG learned topology (worker2), global val 98.58%, test 98.60%, installed.

### Seed 2026 (run `14428678-f78c-4d8c-8384-65762bb4ef7d`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 96.92% | 97.35% ± 0.51% | 98.30% |
| NAS | completed |  | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 97.30% | 97.59% ± 1.07% | 98.55% |
| Topology sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 97.34% | 97.78% ± 0.57% | 98.54% |
| Aggregation sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x16x32-k5-conv-h64-drop20 | 22,682 | 97.11% | 97.48% ± 0.81% | 98.66% |
| HPO | completed |  | hpo-adam-lr3e-3-bs32 | 22,682 |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-3-bs32 | 22,682 | 97.10% | 97.52% ± 0.97% | 98.49% |
| Synchronization sweep | completed | directed_ring |  |  |  |  |  |
| Measurement | completed | directed_ring | hpo-adam-lr3e-3-bs32 | 22,682 | 97.16% | 97.53% ± 0.38% | 98.49% |
| APG learned topology | completed | learned | hpo-adam-lr3e-3-bs32 | 22,682 | 98.40% | 98.88% ± 0.20% | 98.99% |
| Pruning + fine-tune | completed | learned | hpo-adam-lr3e-3-bs32-pruned30 | 11,065 | 98.70% | 99.06% ± 0.05% | 98.83% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 96.92% | 96.03% | 97.35% | 96.52% | – | – | – |
| NAS | architecture nas-d3-w16x16x32-k5-conv-h64-drop20 (22682 params) | 97.30% | 94.67% | 97.59% | 94.97% | +0.38 | +0.38 | yes |
| Topology sweep | topology directed_ring | 97.34% | 96.13% | 97.78% | 96.53% | +0.05 | +0.43 | yes |
| Aggregation sweep | aggregation fedavg | 97.11% | 95.55% | 97.48% | 95.74% | -0.23 | +0.20 | yes |
| HPO | lr 0.003, batch 32 | 97.10% | 94.80% | 97.52% | 95.37% | -0.01 | +0.19 | yes |
| Synchronization sweep | sync sync | 97.16% | 96.48% | 97.53% | 96.98% | +0.06 | +0.24 | yes |
| APG learned topology | learned (APG) topology | 98.40% | 98.03% | 98.88% | 98.40% | +1.24 | +1.48 | – |
| Pruning + fine-tune | 30% channels pruned (11065 params) | 98.70% | 98.50% | 99.06% | 98.96% | +0.30 | +1.78 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline architecture achieved 96.9% accuracy, which is below the 99% target. NAS will search for a better small CNN on each worker's data to potentially improve performance before further optimization steps.
1. optimize_topology chosen by slm: Improve peer-to-peer topology to enhance learning on non-IID MNIST data with 8 workers, aiming for 99% validation accuracy within remaining budget.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID MNIST data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: Improve topology learning from measured peer-model losses and waiting times to boost validation accuracy toward 0.99 target.
1. prune_model chosen by slm: Accuracy is good (98.4%), and pruning will make the model smaller and faster on Jetson devices, which is the stated goal for this phase.
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 98.70% (kept).

Model left on every worker: Pruning + fine-tune (worker5), global val 98.82%, test 99.06%, installed.
