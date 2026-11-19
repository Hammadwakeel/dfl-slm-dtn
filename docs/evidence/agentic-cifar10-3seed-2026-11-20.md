# Agentic end-to-end supervisor runs: CIFAR10, 3 seeds

Generated 2026-11-19 20:37 UTC by `scripts/paper_campaign.py`.

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
| Baseline test accuracy | 47.09% ± 2.28 |
| Best model test accuracy | 67.35% ± 2.35 |
| Pruned model test accuracy | 67.97% ± 2.32 |
| Target met (seeds) | 1 / 3 |

Notes:

- Every run finished without an error. Seed 2026 met the 70% target (best global
  test 70.67%); seeds 17 and 42 stopped at 65.7% when the policy had no step left
  to offer (`no_worthwhile_action`): the final retrain of the chosen configuration
  was done and pruning only compresses a model within 2 points of the best.
- Seed 2026 exercised the pruning retry on the real Jetsons for the first time: the
  30% prune lost 2.6 points and was rejected (the unpruned model was restored), the
  15% retry kept 70.26% validation accuracy at 70,462 parameters.
- The topology sweep picked mesh on every seed. Before commit 95c8027 the absolute
  0.9 weakest-worker gate (calibrated for MNIST) passed no CIFAR-10 candidate and
  bandwidth savings chose chain (weakest worker 38.9% vs mesh 54.9%); the selector
  now keeps quality first in that case. A run on the old selector was stopped and
  re-run from scratch, so all three seeds use the same code.
- Known limits for future work: (1) when the target is not met, the policy offers
  no longer continuation training; (2) workers end a run holding the newest model,
  which is not always the best one (seed 17: learned topology 58.8% after the 66.8%
  final training).

## Per seed

| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| 17 | completed | 49.46% | Final training | mesh | 29,802 | 65.74% ± 0.00% | n/a | 5 / 1 / 0 | 0.91 h |
| 42 | completed | 47.78% | Pruning + fine-tune | mesh | 47,649 | 65.64% ± 0.00% | 65.64% (47,649) | 7 / 1 / 0 | 1.31 h |
| 2026 | completed | 44.02% | Final training | learned | 197,290 | 70.67% ± 3.07% | 70.29% (70,462) | 5 / 2 / 0 | 1.57 h |

### Seed 17 (run `3fce48bd-897e-4b45-88e5-e2d07e9cb343`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 49.85% | 49.46% ± 7.50% | 70.38% |
| NAS | completed |  | nas-d3-w32x32x64-k3-conv-h0-drop0 | 29,802 |  |  |  |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 29,802 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Final training | completed | mesh | hpo-adam-lr1e-3-bs32 | 29,802 | 66.80% | 65.74% ± 0.00% | 67.89% |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs32 | 29,802 | 58.76% | 59.11% ± 3.57% | 75.28% |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved only 49.8% validation accuracy, which is far from the 70% target. Running NAS on each worker's data to find a better architecture is the most promising next step before considering hyperparameter tuning or topology changes.
1. run_hpo chosen by slm: Tune learning rate and batch size of the NAS architecture on every worker's data to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Best architecture found (hpo-adam-lr1e-3-bs32) with directed_ring topology achieved 0.498 accuracy; need to improve toward 0.7 target using remaining budget.
1. train_final chosen by slm: Train final model with best found topology (mesh) and architecture to maximize validation accuracy within remaining budget.
1. learn_topology chosen by slm: The current mesh topology achieved 0.668 accuracy, which is below the 0.7 target. Learning a new topology from measured peer-model losses and waiting times on real workers may improve convergence and final accuracy, especially since we have 3 experiments left and 849 mi
1. finish_workflow chosen by rule: No further step is available.

### Seed 42 (run `8b89df52-e787-447b-8d3f-b17a045b2a7f`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 47.74% | 47.78% ± 10.86% | 65.82% |
| NAS | completed |  | nas-d3-w32x64x128-k3-conv-h0-drop35 | 95,434 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| Final training | completed | mesh | nas-d3-w32x64x128-k3-conv-h0-drop35 | 95,434 | 64.82% | 64.76% ± 0.00% | 65.74% |
| HPO | completed |  | hpo-adam-lr3e-4-bs32 | 95,434 |  |  |  |
| APG learned topology | completed | learned | hpo-adam-lr3e-4-bs32 | 95,434 | 57.89% | 57.36% ± 1.86% | 74.37% |
| Final training | completed | mesh | hpo-adam-lr3e-4-bs32 | 95,434 | 64.10% | 63.81% ± 0.00% | 64.82% |
| Pruning + fine-tune | completed | mesh | hpo-adam-lr3e-4-bs32-pruned30 | 47,649 | 66.80% | 65.64% ± 0.00% | 67.10% |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved only 47.7% validation accuracy, which is far from the 70% target. Running NAS to find a better architecture on each worker's data is the most promising next step to improve performance before considering topology optimization or hyperparameter tunin
1. optimize_topology chosen by slm: NAS found a better architecture, now optimize topology to improve validation accuracy before final training.
1. train_final chosen by slm: Train with the best architecture (NAS) and topology (mesh) found so far to maximize validation accuracy within remaining budget.
1. run_hpo chosen by slm: Tune learning rate and batch size on worker data to improve validation accuracy from 0.6482 toward target 0.7, as this is the most promising remaining action before budget expires.
1. learn_topology chosen by slm: The current mesh topology may not be optimal for this dataset; learning a new topology from measured peer-model losses could improve convergence and final accuracy within the remaining budget.
1. train_final chosen by slm: Best architecture and topology found so far are mesh with nas-d3-w32x64x128-k3-conv-h0-drop35, need to train final model to reach 0.7 accuracy target
1. prune_model chosen by slm: The model has achieved 64.82% validation accuracy, which is close to the 70% target. Pruning can make the model smaller and faster on Jetson devices while potentially improving inference speed without significantly sacrificing accuracy. With 827 minutes remaining, there's 
1. finish_workflow chosen by rule: No further step is available.

Pruning attempts: 30% → 66.80% (kept).

### Seed 2026 (run `0a643bc2-28ca-4992-992d-4e8c8fa9282d`)

| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |
|---|---|---|---|---|---|---|---|
| Baseline | completed | directed_ring | baseline | 324,394 | 44.01% | 44.02% ± 11.09% | 59.39% |
| NAS | completed |  | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197,290 |  |  |  |
| HPO | completed |  | hpo-adam-lr1e-3-bs32 | 197,290 |  |  |  |
| Topology sweep | completed | mesh |  |  |  |  |  |
| APG learned topology | completed | learned | hpo-adam-lr1e-3-bs32 | 197,290 | 61.25% | 60.99% ± 2.93% | 72.01% |
| Final training | completed | learned | hpo-adam-lr1e-3-bs32 | 197,290 | 70.93% | 70.67% ± 3.07% | 78.25% |
| Pruning + fine-tune | completed | learned | hpo-adam-lr1e-3-bs32-pruned30 | 97,700 | 68.32% | 68.00% ± 2.75% | 80.57% |
| Pruning + fine-tune | completed | learned | hpo-adam-lr1e-3-bs32-pruned15 | 70,462 | 70.26% | 70.29% ± 3.44% | 79.56% |

Agent decisions:

1. run_nas chosen by slm: The baseline model achieved only 0.44 validation accuracy, which is far from the 0.7 target. Running NAS on worker data to find a better architecture is the most promising next step before considering topology optimization or hyperparameter tuning.
1. run_hpo chosen by slm: Tune learning rate and batch size of the NAS architecture on each worker's data to improve validation accuracy before final training.
1. optimize_topology chosen by slm: Best architecture found, need to find better topology for non-IID data to reach 70% accuracy target within remaining budget.
1. learn_topology chosen by slm: The recent experiments used a mesh topology, but learning a new topology from measured peer-model losses and waiting times on real workers may yield better performance, especially given the remaining budget of 4 experiments and 837 minutes.
1. train_final chosen by slm: Final training with the best learned topology and architecture to push validation accuracy toward 0.7 target
1. prune_model chosen by rule: The target is met; compress the model for the Jetsons before finishing.
1. finish_workflow chosen by rule: The measured result meets the goal.

Pruning attempts: 30% → 68.32% (rejected), 15% → 70.26% (kept).
