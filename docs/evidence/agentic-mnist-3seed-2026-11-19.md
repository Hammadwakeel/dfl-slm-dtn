# Agentic end-to-end supervisor runs: MNIST, 3 seeds

Generated 2026-11-19 15:57 UTC by `scripts/paper_campaign.py`.

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
| target_accuracy | 0.95 |
| max_experiments | 8 |
| time_budget_minutes | 480 |
| dataset_split | dirichlet_a03 |
| rounds | 10 |
| local_epochs | 5 |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 86.44% ± 5.01 |
| Best model test accuracy | 98.51% ± 0.70 |
| Pruned model test accuracy | 99.02% |
| Target met (seeds) | 3 / 3 |

Notes:

- Seeds 42 and 2026 kept the installed baseline (NAS v2's federated check found no
  searched design better than it). That baseline is a hybrid CNN + transformer
  (patch embedding, transformer encoder), which channel pruning does not support, so
  the supervisor correctly did not offer pruning; the pruned-model row covers seed 17.
- Earlier attempts of these seeds that broke on infrastructure bugs (a coordinator
  restart, a stale worker registration, a 300 s HPO cap) were fixed in commits
  65c7485, 2061551 and 85e6195 and re-run from scratch; only clean runs are reported.

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 80.16% | Final training | mesh | 129,770 | 99.05% ± 0.00% | 99.02% (64,240) | 4 / 2 / 0 | 1.08 h |
| 42 | completed | 92.43% | APG learned topology | learned | 32,794 | 97.52% ± 0.56% | n/a | 4 / 1 / 0 | 0.97 h |
| 2026 | completed | 86.72% | Final training | mesh | 32,794 | 98.97% ± 0.00% | n/a | 4 / 1 / 0 | 1.16 h |

### Seed 17 (run `a3b71655-d7c4-4d2a-8973-354d28c7a276`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 80.21% | 80.16% ± 8.92% | 89.94% |
| NAS | completed |  | nas-d3-w16x64x64-k5-conv-h0-drop35 | 129,770 |  |  |  |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 129,770 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Final training | completed | mesh | hpo-adam-lr3e-4-bs32 | 129,770 | 99.22% | 99.05% ± 0.00% | 99.03% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr3e-4-bs32-pruned30 | 64,240 | 99.12% | 99.02% ± 0.00% | 99.00% |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved only 0.802 accuracy, which is far from the 0.95 target. Running NAS on worker data to find a better architecture is the most direct way to improve performance before considering topology changes or hyperparameter tuning.
1. run_hpo chosen by slm: Tune learning rate and batch size of the NAS architecture on every worker's data to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Search for a better topology to improve validation accuracy on MNIST dataset, as current best is 0.802113 and target is 0.95.
1. train_final chosen by slm: Retrain with best found configuration (hpo-adam-lr3e-4-bs32 on mesh topology) to maximize accuracy within remaining budget.
1. prune_model chosen by rule: The target is met; compress the model for the Jetsons before finishing.
1. finish_workflow chosen by rule: The measured result meets the goal.

Pruning attempts: 30% → 99.12% (kept).

### Seed 42 (run `87a54188-3852-4473-b35e-b3dcaaccc225`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 92.34% | 92.43% ± 5.52% | 97.91% |
| NAS | completed |  | seed-baseline | 32,794 |  |  |  |
| HPO | completed |  | hpo-adam-lr1e-3-bs64 | 32,794 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs64 | 32,794 | 97.51% | 97.52% ± 0.56% | 98.74% |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved 0.9234 validation accuracy, which is below the 0.95 target. Running NAS to find a better architecture on worker data is the most direct way to improve performance before considering topology changes or hyperparameter tuning.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best architecture (seed-baseline) on every worker's data to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Search for a better peer-to-peer topology to improve validation accuracy on MNIST dataset, as current best is 0.9234 with directed_ring topology.
1. learn_topology chosen by slm: The recent experiments used a mesh topology, but learned topologies on real workers may perform better, especially given the non-IID nature of MNIST data and the remaining budget.
1. finish_workflow chosen by rule: The measured result meets the goal.

### Seed 2026 (run `ed456368-6001-4341-9e13-6788fb062494`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 32,794 | 86.18% | 86.72% ± 6.65% | 95.50% |
| NAS | completed |  | seed-baseline | 32,794 |  |  |  |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 32,794 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Final training | completed | mesh | hpo-adam-lr1e-3-bs32 | 32,794 | 98.78% | 98.97% ± 0.00% | 99.07% |

Agent decisions:

1. run_nas chosen by slm: Search for a better CNN architecture since the baseline model achieved only 0.86 accuracy, far from the 0.95 target, and NAS is indicated as not yet performed.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best architecture (seed-baseline) on every worker's own data to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Best architecture found via HPO, need to find better topology for non-IID MNIST data with 8 workers ready.
1. train_final chosen by slm: Train final model with best found topology (mesh) and architecture to maximize validation accuracy within remaining budget.
1. finish_workflow chosen by rule: The measured result meets the goal.
