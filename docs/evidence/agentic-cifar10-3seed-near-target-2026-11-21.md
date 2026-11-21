# Agentic end-to-end supervisor runs: CIFAR10, 3 seeds

Generated 2026-11-21 06:05 UTC by `scripts/paper_campaign.py`.

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
| Baseline test accuracy | 46.41% ± 2.53 |
| Best model test accuracy | 76.41% ± 4.93 |
| Pruned model test accuracy | 75.17% ± 5.48 |
| Model installed on the workers at the end (test) | 75.17% ± 5.48 |
| Target met (seeds) | 3 / 3 |

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 48.70% | Final training | mesh | 324,394 | 79.26% ± 0.00% | 79.01% (161,094) | 4 / 2 / 0 | 1.85 h |
| 42 | completed | 47.65% | Continued training | mesh | 95,434 | 69.48% ± 0.00% | 67.42% (47,649) | 5 / 3 / 0 | 1.41 h |
| 2026 | completed | 42.88% | Final training | mesh | 197,290 | 80.50% ± 0.00% | 79.07% (97,700) | 4 / 2 / 0 | 1.49 h |

### Seed 17 (run `c3215d45-bd20-4a82-805a-e9f96dc1f0d1`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 49.17% | 48.70% ± 6.34% | 70.95% |
| NAS | completed |  | seed-baseline | 324,394 |  |  |  |
| HPO | completed |  | hpo-adam-lr1e-3-bs8 | 324,394 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Final training | completed | mesh | hpo-adam-lr1e-3-bs8 | 324,394 | 80.12% | 79.26% ± 0.00% | 80.56% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr1e-3-bs8-pruned30 | 161,094 | 78.98% | 79.01% ± 0.00% | 80.55% |

Agent decisions:

1. run_nas chosen by slm: Search for a better CNN architecture on worker data since baseline accuracy (0.49) is below target (0.70) and NAS can improve model performance.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best architecture (seed-baseline) on every worker's own data to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Improve validation accuracy by finding an optimal peer-to-peer topology for non-IID CIFAR-10 data after NAS and HPO have identified better architectures.
1. train_final chosen by slm: Best architecture from HPO with mesh topology found, need to train further to reach 0.7 accuracy target
1. prune_model chosen by rule: The target is met; compress the model for the Jetsons before finishing.
1. finish_workflow chosen by rule: The measured result meets the goal.

Pruning attempts: 30% → 78.98% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 78.98%, test 79.01%, installed.

### Seed 42 (run `bfdc424e-b8bd-4bd8-9e31-12d866e131be`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 47.39% | 47.65% ± 7.98% | 65.11% |
| NAS | completed |  | nas-d3-w32x64x128-k3-conv-h0-drop35 | 95,434 |  |  |  |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 95,434 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Final training | completed | mesh | hpo-adam-lr3e-4-bs32 | 95,434 | 66.28% | 65.79% ± 0.00% | 67.12% |
| APG learned topology | completed | learned | hpo-adam-lr3e-4-bs32 | 95,434 | 56.45% | 56.11% ± 2.04% | 72.61% |
| Continued training | completed | mesh | hpo-adam-lr3e-4-bs32 | 95,434 | 70.36% | 69.48% ± 0.00% | 69.97% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr3e-4-bs32-pruned30 | 47,649 | 68.92% | 67.42% ± 0.00% | 68.95% |

Agent decisions:

1. run_nas chosen by slm: Search for a better CNN architecture since the baseline validation accuracy (0.47) is significantly below the target (0.70), and NAS can find an optimal model for the specific worker data distributions.
1. run_hpo chosen by slm: Tune learning rate and batch size of the NAS-found architecture on each worker's data to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Improve validation accuracy by finding a better peer-to-peer topology for non-IID CIFAR-10 data after NAS and HPO have identified good architectures.
1. train_final chosen by slm: Train final model with best found configuration (hpo-adam-lr3e-4-bs32 on mesh topology) to maximize accuracy within remaining time.
1. learn_topology chosen by slm: The current mesh topology achieved 66.28% accuracy, which is below the 70% target. Learning a new topology from measured peer-model losses and waiting times on real workers may improve convergence and final accuracy, especially given the remaining budget of 3 experiment
1. continue_training chosen by rule: Best validation accuracy 0.6628 is 3.72 points below the target 0.7000; continue from the best checkpoint.
1. prune_model chosen by rule: The target is met; compress the model for the Jetsons before finishing.
1. finish_workflow chosen by rule: The measured result meets the goal.

Pruning attempts: 30% → 68.92% (kept).

Continued training from the best checkpoint: 70.36% (improved).

Model left on every worker: Pruning + fine-tune (worker1), global val 68.92%, test 67.42%, installed.

### Seed 2026 (run `60f28b6d-3945-4a22-9289-2e0458fa4b0b`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 43.08% | 42.88% ± 12.94% | 59.64% |
| NAS | completed |  | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 |  |  |  |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 197,290 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Final training | completed | mesh | hpo-adam-lr1e-3-bs32 | 197,290 | 79.98% | 80.50% ± 0.00% | 80.00% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr1e-3-bs32-pruned30 | 97,700 | 78.46% | 79.07% ± 0.00% | 77.49% |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved only 43% validation accuracy, which is far from the 70% target. Running NAS on each worker's data to find a better architecture is the most promising next step before considering topology optimization or hyperparameter tuning.
1. run_hpo chosen by slm: Tune learning rate and batch size of the best NAS architecture on every worker's data to improve training stability and convergence before final training.
1. optimize_topology chosen by slm: Best topology found so far is directed_ring with 0.43 accuracy; need to improve toward 0.7 target using remaining budget.
1. train_final chosen by slm: Train final model with best found topology (mesh) and architecture to maximize validation accuracy within remaining budget.
1. prune_model chosen by rule: The target is met; compress the model for the Jetsons before finishing.
1. finish_workflow chosen by rule: The measured result meets the goal.

Pruning attempts: 30% → 78.46% (kept).

Model left on every worker: Pruning + fine-tune (worker1), global val 78.46%, test 79.07%, installed.

## Notes

- Purpose: CIFAR-10 over 3 seeds with commit a48df44, which continues training
  from the best checkpoint by rule when the best validation accuracy is within
  5 points of the target and at least 30 minutes remain.
- The rule was used once, in seed 42: after the learned topology (56.45%) the best
  model stood at 66.28% validation, 3.72 points short. The rule continued from the
  best checkpoint on mesh and reached 70.36%, meeting the target. Seeds 17 and
  2026 met the target in their final training and did not need it.
- Open issue: in seed 42 the pruned model (68.92% validation, 67.42% test) was
  accepted because it is within the 2-point pruning tolerance, so the model left
  on the workers is below the 70% target although the unpruned model met it. The
  pruning acceptance does not yet check the target.
- These runs are a new campaign; the architecture search and topology choices
  differ from the 2026-11-20 campaign, so the two campaigns are reported
  separately rather than pooled.
