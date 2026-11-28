# Agentic end-to-end supervisor runs: CIFAR10, 3 seeds

Generated 2026-11-27 07:39 UTC by `scripts/paper_campaign.py`.

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
| dataset_split | dirichlet_a05 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 59.57% ± 1.40 |
| Best model test accuracy | 80.11% ± 1.88 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 80.11% ± 1.88 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 58.62% | Measurement | mesh | 324,394 | 79.08% ± 0.00% | n/a | 6 / 1 / 0 | 4.48 h |
| 42 | completed | 61.55% | Measurement | mesh | 297,706 | 82.75% ± 0.00% | n/a | 6 / 1 / 0 | 2.63 h |
| 2026 | completed | 58.55% | Measurement | mesh | 197,290 | 78.50% ± 0.00% | n/a | 6 / 1 / 0 | 1.82 h |

### Seed 17 (run `f1ffd4da-bac9-4dc6-8bf0-6eba40a08907`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 58.44% | 58.62% ± 11.80% | 66.11% |
| NAS | completed |  | seed-baseline | 324,394 |  |  |  |
| Measurement | completed | directed_ring | seed-baseline | 324,394 | 58.96% | 58.93% ± 12.58% | 65.88% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 324,394 | 79.14% | 78.50% ± 0.00% | 80.18% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | seed-baseline | 324,394 | 78.02% | 77.80% ± 0.00% | 78.80% |
| HPO | completed |  | hpo-adam-lr3e-4-bs8 | 324,394 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs8 | 324,394 | 79.42% | 79.08% ± 0.00% | 80.35% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs8 | 324,394 | 79.28% | 79.37% ± 0.00% | 80.55% |
| APG learned topology | completed | learned | hpo-adam-lr3e-4-bs8 | 324,394 | 74.09% | 74.22% ± 1.96% | 81.11% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 58.44% | 34.90% | 58.62% | 35.21% | – | – | – |
| NAS | architecture seed-baseline (324394 params) | 58.96% | 35.92% | 58.93% | 35.65% | +0.52 | +0.52 | yes |
| Topology sweep | topology mesh | 79.14% | 79.14% | 78.50% | 78.50% | +20.18 | +20.70 | yes |
| Aggregation sweep | aggregation trimmed_mean | 78.02% | 78.02% | 77.80% | 77.80% | -1.12 | +19.58 | no (reverted) |
| HPO | lr 0.0003, batch 8 | 79.42% | 79.42% | 79.08% | 79.08% | +1.40 | +20.98 | yes |
| Synchronization sweep | sync semi_sync | 79.28% | 79.28% | 79.37% | 79.37% | -0.14 | +20.84 | yes |
| APG learned topology | learned (APG) topology | 74.09% | 71.68% | 74.22% | 71.73% | -5.19 | +15.64 | – |

Agent decisions:

1. run_nas chosen by slm: The baseline model is too small with only 58% validation accuracy; search for a better architecture on worker data to improve performance before further optimization.
1. optimize_topology chosen by slm: Improve peer-to-peer topology to enhance learning on non-IID CIFAR10 data after NAS found a baseline architecture.
1. optimize_aggregation chosen by slm: Select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) on mesh topology to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy toward 0.99 target.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy before final HPO rounds
1. learn_topology chosen by slm: The current mesh topology achieved only 0.7942 validation accuracy, which is far from the target of 0.99. Learning a new peer-to-peer topology from measured losses and waiting times on real workers may improve communication efficiency and model convergence for this non-
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 79.42%, test 79.08%, installed.

### Seed 42 (run `4c140094-affb-4fb9-b8d5-f3e3b561b424`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | failed | directed_ring | baseline | 324,394 |  |  |  |
| Baseline | completed | directed_ring | baseline | 324,394 | 62.19% | 61.55% ± 5.62% | 76.77% |
| NAS | completed |  | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 | 65.21% | 64.54% ± 5.09% | 78.83% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 | 82.84% | 82.63% ± 0.00% | 82.21% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 | 83.46% | 82.75% ± 0.00% | 82.73% |
| HPO | completed |  | hpo-adam-lr3e-3-bs32 | 297,706 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-3-bs32 | 297,706 | 82.44% | 81.42% ± 0.00% | 81.16% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 | 82.84% | 83.15% ± 0.00% | 81.85% |
| APG learned topology | completed | learned | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 | 297,706 | 78.50% | 78.42% ± 2.29% | 84.83% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 62.19% | 49.48% | 61.55% | 49.29% | – | – | – |
| NAS | architecture nas-d3-w32x64x128-k3-conv-h64-drop0-c2 (297706 params) | 65.21% | 54.68% | 64.54% | 55.07% | +3.02 | +3.02 | yes |
| Topology sweep | topology mesh | 82.84% | 82.84% | 82.63% | 82.63% | +17.63 | +20.65 | yes |
| Aggregation sweep | aggregation fedavg | 83.46% | 83.46% | 82.75% | 82.75% | +0.62 | +21.27 | yes |
| HPO | lr 0.003, batch 32 | 82.44% | 82.44% | 81.42% | 81.42% | -1.02 | +20.25 | no (reverted) |
| Synchronization sweep | sync sync | 82.84% | 82.84% | 83.15% | 83.15% | +0.40 | +20.65 | no (reverted) |
| APG learned topology | learned (APG) topology | 78.50% | 74.82% | 78.42% | 74.36% | -4.34 | +16.31 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small with only 0.62 accuracy; need to search for a better architecture on worker data before proceeding with optimization steps.
1. optimize_topology chosen by slm: Improve peer-to-peer communication efficiency on non-IID CIFAR10 data to boost validation accuracy toward 0.99 target.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID CIFAR10 data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size on each worker's local data to improve validation accuracy from 0.8346 toward target 0.99, using remaining 605 minutes budget.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining budget
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, as peers learn slowly from each other on CIFAR10 data.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 83.46%, test 82.75%, installed.

Errors recorded during the run:

- baseline_training failed: TrainingError: Worker request failed: worker1: Could not reach worker1 for /v1/config: timed out

### Seed 2026 (run `49efd009-5865-4496-9e13-01181cc3dc13`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 59.01% | 58.55% ± 4.53% | 66.45% |
| NAS | completed |  | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 62.77% | 62.31% ± 4.48% | 67.80% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 77.94% | 77.34% ± 0.00% | 77.02% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 79.44% | 78.50% ± 0.00% | 78.83% |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 197,290 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr3e-4-bs32 | 197,290 | 72.44% | 72.40% ± 0.00% | 72.79% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 78.20% | 77.52% ± 0.00% | 77.39% |
| APG learned topology | completed | learned | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 | 69.76% | 68.61% ± 2.59% | 75.88% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 59.01% | 50.24% | 58.55% | 49.76% | – | – | – |
| NAS | architecture nas-d2-w32x128-k3-conv-h0-drop35-c2 (197290 params) | 62.77% | 52.70% | 62.31% | 52.65% | +3.76 | +3.76 | yes |
| Topology sweep | topology mesh | 77.94% | 77.94% | 77.34% | 77.34% | +15.17 | +18.93 | yes |
| Aggregation sweep | aggregation trimmed_mean | 79.44% | 79.44% | 78.50% | 78.50% | +1.50 | +20.43 | yes |
| HPO | lr 0.0003, batch 32 | 72.44% | 72.44% | 72.40% | 72.40% | -7.00 | +13.43 | no (reverted) |
| Synchronization sweep | sync sync | 78.20% | 78.20% | 77.52% | 77.52% | +5.76 | +19.19 | no (reverted) |
| APG learned topology | learned (APG) topology | 69.76% | 65.20% | 68.61% | 63.96% | -8.44 | +10.75 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small with only 59% validation accuracy; need to search for a better architecture on worker data before further optimization.
1. optimize_topology chosen by slm: Improve peer-to-peer communication efficiency on CIFAR10 non-IID data to boost validation accuracy toward 0.99 target.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID CIFAR10 data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy from 0.7944 toward the 0.99 target, with 638 minutes remaining.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy within remaining 630 minutes and 7 experiments left.
1. learn_topology chosen by slm: The current mesh topology yields only 79% accuracy, far from the 99% target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers may improve communication efficiency and model convergence for this non-IID CIFAR10 dataset.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 79.44%, test 78.50%, installed.
