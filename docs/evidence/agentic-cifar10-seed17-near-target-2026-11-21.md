# Agentic end-to-end supervisor runs: CIFAR10, 1 seed

Generated 2026-11-20 18:03 UTC by `scripts/paper_campaign.py`.

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
| Baseline test accuracy | 50.19% |
| Best model test accuracy | 79.55% |
| Pruned model test accuracy | 79.63% |
| Model installed on the workers at the end (test) | 79.63% |
| Target met (seeds) | 1 / 1 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 50.19% | Final training | mesh | 324,394 | 79.55% ± 0.00% | 79.63% (161,094) | 4 / 2 / 0 | 1.86 h |

### Seed 17 (run `5588000d-8a0a-4c04-b8b7-05ccbbaa077d`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 50.26% | 50.19% ± 5.88% | 71.22% |
| NAS | completed |  | seed-baseline | 324,394 |  |  |  |
| HPO | completed |  | hpo-adam-lr1e-3-bs8 | 324,394 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Final training | completed | mesh | hpo-adam-lr1e-3-bs8 | 324,394 | 79.86% | 79.55% ± 0.00% | 80.64% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr1e-3-bs8-pruned30 | 161,094 | 79.74% | 79.63% ± 0.00% | 80.94% |

Agent decisions:

1. run_nas chosen by slm: Baseline model is too small (50.25% accuracy), need to search for a better architecture on worker data before tuning hyperparameters or optimizing topology.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best architecture (seed-baseline) on every worker's own data to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Improve validation accuracy from 0.50255 by finding a better peer-to-peer topology for non-IID CIFAR10 data, with 5 experiments and 860 minutes remaining.
1. train_final chosen by slm: Use the best architecture found (hpo-adam-lr1e-3-bs8) with the selected topology (mesh) to maximize validation accuracy within remaining time.
1. prune_model chosen by rule: The target is met; compress the model for the Jetsons before finishing.
1. finish_workflow chosen by rule: The measured result meets the goal.

Pruning attempts: 30% → 79.74% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 79.74%, test 79.63%, installed.

## Notes

- Purpose: rerun of CIFAR-10 seed 17 after commit a48df44, which makes the
  supervisor continue from the best checkpoint by rule when the best validation
  accuracy is within 5 points of the target and at least 30 minutes remain.
- The rule was not needed in this run: the final training already reached
  79.86% validation, above the 70% target, so the existing rules pruned the
  model and finished.
- Compared with the earlier seed 17 run (69.69% validation, finished by the SLM
  just below the target), the SLM this time skipped the learned topology and
  trained the tuned model on the mesh topology chosen by the sweep. The data
  shards and baseline initialisation follow the seed; worker-side training is
  not bit-for-bit deterministic, so the baseline differs slightly (50.26% vs
  49.66% validation).
- The model installed on every worker at the end is the pruned model
  (161,094 parameters, 79.63% test).
