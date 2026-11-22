# Agentic end-to-end supervisor runs: FEMNIST, 3 seeds

Generated 2026-11-22 07:10 UTC by `scripts/paper_campaign.py`.

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
| dataset_split | dirichlet_a01 |
| rounds | 10 |
| local_epochs | 5 |
| measure_each_step | True |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 33.38% ± 3.58 |
| Best model test accuracy | 70.57% ± 3.66 |
| Pruned model test accuracy | n/a |
| Model installed on the workers at the end (test) | 70.57% ± 3.66 |
| Target met (seeds) | 0 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 28.39% | Measurement | mesh | 133,150 | 74.20% ± 0.00% | n/a | 6 / 1 / 0 | 1.99 h |
| 42 | completed | 35.18% | Measurement | mesh | 112,830 | 65.56% ± 0.00% | n/a | 6 / 1 / 0 | 2.21 h |
| 2026 | completed | 36.58% | Measurement | mesh | 73,150 | 71.97% ± 0.00% | n/a | 6 / 1 / 0 | 1.95 h |

### Seed 17 (run `fd67969e-17bd-443d-8922-77215b31529d`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 28.65% | 28.39% ± 5.68% | 28.37% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 133,150 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x64x64-k5-conv-h0-drop35 | 133,150 | 43.66% | 43.47% ± 4.42% | 60.36% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 133,150 | 70.56% | 70.80% ± 0.00% | 67.06% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x64x64-k5-conv-h0-drop35 | 133,150 | 73.30% | 73.23% ± 0.00% | 72.51% |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 133,150 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 133,150 | 72.95% | 72.97% ± 0.00% | 71.72% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs32 | 133,150 | 74.73% | 74.20% ± 0.00% | 73.45% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs32 | 133,150 | 56.86% | 56.74% ± 3.29% | 71.99% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 28.65% | 20.27% | 28.39% | 19.58% | – | – | – |
| NAS | architecture nas-d3-w16x64x64-k5-conv-h0-drop35 (133150 params) | 43.66% | 37.31% | 43.47% | 36.99% | +15.02 | +15.02 | yes |
| Topology sweep | topology mesh | 70.56% | 70.56% | 70.80% | 70.80% | +26.90 | +41.92 | yes |
| Aggregation sweep | aggregation uniform | 73.30% | 73.30% | 73.23% | 73.23% | +2.73 | +44.65 | yes |
| HPO | lr 0.001, batch 32 | 72.95% | 72.95% | 72.97% | 72.97% | -0.35 | +44.30 | yes |
| Synchronization sweep | sync semi_sync | 74.73% | 74.73% | 74.20% | 74.20% | +1.78 | +46.08 | yes |
| APG learned topology | learned (APG) topology | 56.86% | 51.84% | 56.74% | 51.52% | -17.87 | +28.22 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small with only 0.286 validation accuracy; need to search for a better architecture on worker data before further optimization.
1. optimize_topology chosen by slm: Improve topology selection to boost validation accuracy toward 0.99 target on non-IID femnist data after NAS completion.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on each worker's local data to improve validation accuracy before final training rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy before final experiments
1. learn_topology chosen by slm: The current mesh topology achieved only 0.747 accuracy, far from the 0.99 target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers is the most promising next step to improve performance, especially given the non-IID nature of f
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 74.73%, test 74.20%, installed.

### Seed 42 (run `146962c7-0ad3-4453-9862-3a136f1f6b19`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 35.20% | 35.18% ± 7.22% | 49.95% |
| NAS | completed |  | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 |  |  |  |
| Measurement | completed | directed_ring | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 36.55% | 36.49% ± 7.17% | 51.56% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 64.01% | 64.34% ± 0.00% | 59.43% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 65.33% | 65.56% ± 0.00% | 59.47% |
| HPO | completed |  | hpo-adam-lr1e-3-bs64 | 112,830 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs64 | 112,830 | 63.78% | 63.61% ± 0.00% | 58.64% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 64.43% | 64.80% ± 0.00% | 59.31% |
| APG learned topology | completed | learned | nas-d2-w64x64-k5-conv-h64-drop35 | 112,830 | 55.33% | 55.36% ± 2.07% | 61.78% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 35.20% | 25.65% | 35.18% | 24.69% | – | – | – |
| NAS | architecture nas-d2-w64x64-k5-conv-h64-drop35 (112830 params) | 36.55% | 28.04% | 36.49% | 27.15% | +1.35 | +1.35 | yes |
| Topology sweep | topology mesh | 64.01% | 64.01% | 64.34% | 64.34% | +27.46 | +28.81 | yes |
| Aggregation sweep | aggregation fedavg | 65.33% | 65.33% | 65.56% | 65.56% | +1.32 | +30.13 | yes |
| HPO | lr 0.001, batch 64 | 63.78% | 63.78% | 63.61% | 63.61% | -1.55 | +28.58 | no (reverted) |
| Synchronization sweep | sync semi_sync | 64.43% | 64.43% | 64.80% | 64.80% | +0.65 | +29.23 | no (reverted) |
| APG learned topology | learned (APG) topology | 55.33% | 52.68% | 55.36% | 52.47% | -9.10 | +20.13 | – |

Agent decisions:

1. run_nas chosen by slm: Search a better small CNN on every worker's own data and validation split since baseline accuracy is low (0.352) and NAS may find a more suitable architecture for this non-IID dataset.
1. optimize_topology chosen by slm: Improve peer-to-peer topology for non-IID femnist data to enhance learning efficiency before aggregation optimization.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward to improve validation accuracy toward 0.99 target.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on worker data to improve validation accuracy before final training rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given remaining budget and current mesh topology with fedavg aggregation.
1. learn_topology chosen by slm: Improve mesh topology performance by learning from measured peer-model losses and waiting times on real workers, as peers learn slowly on non-IID femnist data.
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 65.33%, test 65.56%, installed.

### Seed 2026 (run `98aa1112-12c7-4ebe-b209-536e148d6f7d`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 35,518 | 36.47% | 36.58% ± 4.65% | 49.68% |
| NAS | completed |  | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 |  |  |  |
| Measurement | completed | directed_ring | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 40.42% | 40.17% ± 3.03% | 58.15% |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 71.01% | 71.24% ± 0.00% | 70.24% |
| Aggregation sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 72.41% | 71.97% ± 0.00% | 73.40% |
| HPO | completed |  | hpo-adam-lr1e-3-bs64 | 73,150 |  |  |  |
| Measurement | completed | mesh | hpo-adam-lr1e-3-bs64 | 73,150 | 71.76% | 71.08% ± 0.00% | 72.37% |
| Synchronization sweep | completed | mesh |  |  |  |  |  |
| Measurement | completed | mesh | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 71.91% | 71.28% ± 0.00% | 72.58% |
| APG learned topology | completed | learned | nas-d3-w16x32x64-k5-conv-h64-drop0 | 73,150 | 54.22% | 53.84% ± 4.22% | 68.30% |

Step impact (every trained model uses the baseline's rounds x epochs):

| After | Config change | Val | Weakest (val) | Test | Weakest (test) | Δ val vs previous | Δ val vs baseline | Kept |
|---|---|---|---|---|---|---|---|---|
| Baseline | directed_ring, fedavg, semi_sync | 36.47% | 28.00% | 36.58% | 28.79% | – | – | – |
| NAS | architecture nas-d3-w16x32x64-k5-conv-h64-drop0 (73150 params) | 40.42% | 37.31% | 40.17% | 36.83% | +3.95 | +3.95 | yes |
| Topology sweep | topology mesh | 71.01% | 71.01% | 71.24% | 71.24% | +30.60 | +34.55 | yes |
| Aggregation sweep | aggregation uniform | 72.41% | 72.41% | 71.97% | 71.97% | +1.40 | +35.95 | yes |
| HPO | lr 0.001, batch 64 | 71.76% | 71.76% | 71.08% | 71.08% | -0.65 | +35.30 | no (reverted) |
| Synchronization sweep | sync sync | 71.91% | 71.91% | 71.28% | 71.28% | +0.15 | +35.45 | no (reverted) |
| APG learned topology | learned (APG) topology | 54.22% | 48.37% | 53.84% | 47.73% | -17.69 | +17.76 | – |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small with only 0.36 accuracy; need to search for a better architecture on worker data before further optimization.
1. optimize_topology chosen by slm: Improve peer-to-peer topology to enhance learning on non-IID femnist data with 13 experiments remaining and 686 minutes budget.
1. optimize_aggregation chosen by slm: Workers' models may disagree on non-IID femnist data; need to select best aggregation rule (fedavg, uniform, metropolis, trimmed_mean) by quality-gated reward.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy before final rounds.
1. optimize_sync chosen by slm: Select best synchronization mode to improve validation accuracy given remaining budget and current performance gap.
1. learn_topology chosen by slm: Current mesh topology yields only 0.71-0.72 validation accuracy, far from the 0.99 target. Learning a new peer-to-peer topology from measured losses and waiting times on real workers may improve communication efficiency and model performance before continuing with hyper
1. finish_workflow chosen by rule: No further step is available.

Model left on every worker: Measurement (worker1), global val 72.41%, test 71.97%, installed.
