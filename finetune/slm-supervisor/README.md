# Supervisor-SLM fine-tuning dataset

Built by `scripts/slm_finetune_dataset.py --episodes 40 --epsilon 0.3 --seed 2026`
from 38 fair-campaign runs (`docs/evidence/fair-*`). Each line is one decision of a simulated
supervisor run: the exact system prompt, state JSON and tools the supervisor sends to the SLM,
and one teacher tool call with a reason. Split by dataset: CIFAR-10 is the test set and never
informs the teacher; validation is seed 42 of MNIST, Fashion-MNIST and FEMNIST.

## Examples

| Split | Examples | Fair mode | Budget mode | Labels |
|---|---|---|---|---|
| train | 7032 | 4541 | 2491 | optimize_topology 1883, run_nas 1424, train_final 742, optimize_aggregation 724, optimize_sync 643, run_hpo 506, learn_topology 427, prune_model 346, finish_workflow 337 |
| val | 3161 | 2071 | 1090 | optimize_topology 814, run_nas 611, train_final 344, optimize_aggregation 327, optimize_sync 293, run_hpo 261, learn_topology 199, prune_model 164, finish_workflow 148 |
| test | 3698 | 2365 | 1333 | optimize_topology 962, run_nas 749, optimize_aggregation 428, train_final 402, optimize_sync 324, run_hpo 290, learn_topology 222, finish_workflow 195, prune_model 126 |

## Teacher priors (mean measured validation effect, points, training datasets only)

| Split | learn_topology | optimize_aggregation | optimize_sync | optimize_topology | run_hpo | run_nas |
|---|---|---|---|---|---|---|
| dirichlet_a01 | -12.64 | +0.68 | -0.82 | +24.52 | -1.16 | +6.00 |
| dirichlet_a05 | -1.80 | -0.01 | -0.30 | +6.12 | -0.32 | +3.35 |
| dirichlet_a09 | -0.73 | +0.18 | +0.01 | +3.55 | -0.28 | +2.28 |

A step is worth an experiment in budget mode when its prior is at least +1 point; in fair
mode every search step runs and the teacher orders them by prior.
