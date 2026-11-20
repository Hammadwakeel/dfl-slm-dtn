# Agentic end-to-end supervisor runs: CIFAR10, 1 seed

Generated 2026-11-20 07:23 UTC by `scripts/paper_campaign.py`.

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
| target_accuracy | 0.7 |
| max_experiments | 8 |
| time_budget_minutes | 900 |
| dataset_split | dirichlet_a03 |
| rounds | 10 |
| local_epochs | 5 |

## Summary over seeds

| Metric | Mean ± std over seeds |
|---|---|
| Baseline test accuracy | 51.80% |
| Best model test accuracy | 80.27% |
| Pruned model test accuracy | 78.70% |
| Model installed on the workers at the end (test) | 78.70% |
| Target met (seeds) | 1 / 1 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 51.80% | Final training | mesh | 297,706 | 80.27% ± 0.00% | 78.70% (147,560) | 4 / 2 / 0 | 1.26 h |

### Seed 17 (run `d4bf89d9-f9a9-4f83-837e-ec5e76603e8a`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 52.09% | 51.80% ± 6.68% | 71.65% |
| NAS | completed |  | nas-d3-w32x64x128-k3-conv-h64-drop20-c2 | 297,706 |  |  |  |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 297,706 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Final training | completed | mesh | hpo-adam-lr1e-3-bs32 | 297,706 | 80.14% | 80.27% ± 0.00% | 81.81% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr1e-3-bs32-pruned30 | 147,560 | 79.00% | 78.70% ± 0.00% | 79.99% |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved only 52% validation accuracy, which is below the 70% target. Running NAS on each worker's data to find a better architecture is the most direct way to improve performance before considering topology changes or hyperparameter tuning.
1. run_hpo chosen by slm: Tune learning rate and batch size on best architecture (NAS-d3-w32x64x128-k3-conv-h64-drop20-c2) with learned topology to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.52 by finding a better peer-to-peer topology for non-IID CIFAR-10 data, with 5 experiments and 859 minutes remaining.
1. train_final chosen by slm: Train final model with best found topology (mesh) and architecture to maximize accuracy within remaining budget.
1. prune_model chosen by rule: The target is met; compress the model for the Jetsons before finishing.
1. finish_workflow chosen by rule: The measured result meets the goal.

Pruning attempts: 30% → 79.00% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 79.00%, test 78.70%, installed.

## Notes

- Purpose: live check of commit 11c954c (best-model checkpoint, continuation
  training, final install) on the real cluster, with the same goal as the
  3-seed CIFAR-10 campaign (`agentic-cifar10-3seed-2026-11-20.md`).
- The target was met by the final training (80.14% global validation), so the
  rule pruned instead of continuing; `continue_training` was not needed and is
  covered by the backend tests. The accepted 30% prune (147,560 parameters,
  78.70% test) was saved as the run's checkpoint and installed on all eight
  workers when the run finished.
- This seed reached 80.27% test versus 65.74% in the campaign run of the same
  seed: the SLM picked HPO then the topology sweep (mesh) here, and NAS v2 found
  a wider 297,706-parameter CNN. Agent decisions are not deterministic across
  runs, so a seed repeats the setting, not the trajectory.
