# Topology selection with a quality-gated reward (reward v2)

Measured on the 8-worker Jetson Orin Nano cluster, 2026-11-18.
Sweep `optimize-1794992581-58384f`. Raw data: [topology-reward-v2-sweep-2026-11-18.json](topology-reward-v2-sweep-2026-11-18.json).
Reproduce the tables with [topology_reward_v2_analysis.py](topology_reward_v2_analysis.py).

> **Correction (2026-11-19).** This sweep did not start from a random initialization. Workers train the installed initial model in preference to the configured architecture. At the time, that installed model was the 30%-pruned model from supervisor run `82668b5e` (4,921 parameters, weights `23377fc6…`), so every run fine-tuned the same pruned model. Per-worker accuracy before round 1 was 2–81%. The comparison between topologies is still like-for-like, because every run started from the same weights. Treat these results as the *warm-start* setting. The from-scratch sweeps on MNIST (α = 0.1, 0.3, 0.5) and CIFAR-10 are in [topology-reward-v2-paper-results-2026-11-19.md](topology-reward-v2-paper-results-2026-11-19.md).

## Summary

- The original reward (v1) cannot tell topologies apart. All five score between 0.961 and 0.969.
- v1 ranks **directed ring first**, even though it has the lowest accuracy on a shared balanced validation split (76.6%). Its rank correlation with shared-split accuracy is negative (Spearman ρ = −0.5).
- Reward v2 ranks topologies in nearly the same order as shared-split accuracy (ρ = 0.9). With its quality gate it selects **mesh**, the only topology whose weakest worker exceeds 90% accuracy (97.3 ± 0.4%) on all three seeds.
- Accuracy that workers report on their own non-IID validation slice overstates quality by up to 13.6 points for sparse topologies. All quality in v2 is therefore measured on a shared, class-balanced split.

## Setup

| Item | Value |
|---|---|
| Workers | 8 × Jetson Orin Nano 8 GB, real peer-to-peer exchange over LAN |
| Data | MNIST, Dirichlet α = 0.3 non-IID shards (one per worker) |
| Model | **Warm start** (see correction): the installed 30%-pruned model from an earlier supervisor run, 4,921 parameters, same weights for every run |
| Training | 10 rounds × 5 local epochs, semi-synchronous rounds, weighted FedAvg |
| Topologies | directed ring, bidirectional ring, mesh, chain, star |
| Seeds | 17, 42, 2026 (15 runs in total) |
| Shared validation split | 2,000 images (200 per class), class-balanced. Drawn from the preprocessing validation split, which is disjoint from every training shard. The same bytes are installed on every worker, and every worker scores its model on it after each round. |
| Test split | Not used for any decision |

## Reward definitions

**v1 (historical).**
`R = 0.5·gain + 0.2·convergence + 0.15·communication + 0.15·wait`, where:
- `gain = min((A_final − A_initial)/0.25, 1)`
- `convergence` is the first round where accuracy reached 50%
- the cost terms are normalized by fixed caps (50 MB and 30 s per worker per round)

Accuracy was each worker's local, non-IID validation accuracy.

**v2.**
`R = 0.5·Q + 0.2·S + 0.15·B + 0.15·W`, where:
- `q_r = ½·mean_i F1_{i,r} + ½·min_i F1_{i,r}` is the macro-F1 of round r on the shared split, half the mean over workers and half the weakest worker
- `Q = q_R` is the quality after the last round
- `S = mean_r q_r` is the learning speed, the area under the quality curve
- `B = 1 − bytes_sent / (model_bytes · N · (N−1) · R)`, so a full mesh scores 0
- `W = 1 − Σ peer_wait / Σ round_time`

**Selection rule.**
1. Keep only the candidates that pass the quality gate: the weakest worker must reach ≥ 90% shared-split accuracy on every seed.
2. Among those, take the highest mean R.
3. If several are within one standard deviation of the best, prefer the one with the highest B (the cheapest).
4. If no candidate passes the gate, fall back to the highest R and report that the gate was missed.

## Results (mean ± std over 3 seeds)

| Topology | v1 reward | v2 reward | Gate | Local acc. | Shared acc. | Weakest worker | Q | S | B | W | MB sent / run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| directed ring | **0.969 ± 0.002** | 0.715 ± 0.013 | 0/3 | 88.0% | 76.6 ± 0.8% | 65.3 ± 4.9% | 0.681 | 0.616 | 0.832 | 0.843 | 1.86 |
| bidirectional ring | 0.967 ± 0.001 | 0.785 ± 0.012 | 0/3 | 95.1% | 87.8 ± 1.1% | 80.9 ± 2.5% | 0.828 | 0.734 | 0.663 | 0.834 | 3.71 |
| **mesh** | 0.964 ± 0.005 | **0.786 ± 0.002** | **3/3** | 96.9% | **97.3 ± 0.4%** | **97.3 ± 0.4%** | 0.973 | 0.874 | 0.000 | 0.835 | 12.99 |
| chain | 0.961 ± 0.013 | 0.762 ± 0.004 | 0/3 | 94.3% | 85.1 ± 1.0% | 78.1 ± 1.1% | 0.789 | 0.676 | 0.705 | 0.839 | 3.25 |
| star | 0.964 ± 0.002 | 0.682 ± 0.010 | 0/3 | 91.0% | 77.4 ± 1.6% | 65.3 ± 2.8% | 0.659 | 0.614 | 0.706 | 0.825 | 3.25 |

| Ranking by | Order | Spearman ρ vs. shared accuracy |
|---|---|---|
| v1 reward | directed ring > bidirectional ring > star > mesh > chain | −0.5 |
| v2 reward | mesh > bidirectional ring > chain > directed ring > star | 0.9 |
| Shared-split accuracy (reference) | mesh > bidirectional ring > chain > star > directed ring | 1.0 |

The selector chose **mesh** (method `quality_gated_reward_v2`, gate met).

### Learning curves (shared-split mean accuracy per round, mean of 3 seeds)

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| directed ring | 0.620 | 0.669 | 0.694 | 0.724 | 0.746 | 0.726 | 0.733 | 0.757 | 0.777 | 0.766 |
| bidirectional ring | 0.588 | 0.731 | 0.763 | 0.823 | 0.827 | 0.855 | 0.868 | 0.876 | 0.854 | 0.878 |
| mesh | 0.539 | 0.684 | 0.828 | 0.934 | 0.964 | 0.951 | 0.969 | 0.972 | 0.976 | 0.973 |
| chain | 0.561 | 0.703 | 0.714 | 0.796 | 0.785 | 0.796 | 0.812 | 0.842 | 0.817 | 0.851 |
| star | 0.635 | 0.681 | 0.718 | 0.725 | 0.738 | 0.746 | 0.752 | 0.768 | 0.775 | 0.774 |

Mesh passes 90% by round 4. No other topology reaches 90% mean accuracy within 10 rounds.

## Findings

1. **Why v1 fails.**
   - Every topology improves by more than 25 points from its starting point, so the accuracy-gain term saturates at 1.0 and half of the reward weight carries no information.
   - Every topology reaches 50% in round 1–2, so the convergence term does not discriminate either.
   - What remains is the cost terms, and those favour the sparsest topology. That is why the topology with the worst model quality is ranked first.
2. **Local validation overstates quality.**
   - Workers validate on their own non-IID slice, which can lack whole classes. Local accuracy exceeds shared-split accuracy by 11.4 points for directed ring and 13.6 points for star. For mesh the two agree (96.9% vs. 97.3%), because every worker converges to the same consensus model.
   - Local macro-F1 also counted classes absent from the slice as F1 = 0, which lowered the score of workers whose accuracy was actually high.
3. **The weakest worker separates the topologies.**
   - Sparse topologies leave some workers far behind. The weakest worker reaches 65% in a directed ring or star, against 97% in a mesh.
   - Averaging accuracy over workers hides this. v2 gives the weakest worker half of the quality weight and also gates on it.
4. **Cost.**
   - Mesh sends about 7× the bytes of a directed ring (13.0 MB vs. 1.9 MB per run). This equals N − 1 = 7 times the peers per round.
   - With a 128 KB model on a LAN this is small in absolute terms, and mesh rounds were not slower (263 s vs. 280–304 s wall-clock per run).

## Sensitivity and limitations (report these)

- **The reward margin is thin.** Without the gate, mesh (0.7864 ± 0.0016) and bidirectional ring (0.7854 ± 0.0115) are within one standard deviation of each other.
  - With quality, speed and wait kept at the default 0.5 : 0.2 : 0.15 ratio, the reward alone prefers bidirectional ring once the bandwidth weight reaches 0.16. The default is 0.15.
  - Over all 969 weight vectors on a 0.05 grid with quality weight ≥ 0.2, the reward alone picks mesh in 44.8% of cases, bidirectional ring in 36.8% and directed ring in 18.3%.
- **The gate is what makes the selection robust.** Because only mesh passes the 90% weakest-worker gate, the gated selection picks mesh for every one of those weight vectors. The gate threshold (90%) is a design choice and should be stated as one.
- **Narrow scope.** MNIST, one non-IID level (α = 0.3), one fixed small model and 8 workers on one LAN. The bandwidth trade-off would weigh more with larger models or constrained links. CIFAR-10 and other α values are needed before generalising.
- **No test-set results for the sweep itself.** Test-set accuracy was not computed per topology. Workers keep only their newest model, so per-topology final models were not retained. Report shared-validation numbers as validation results. The selected topology was evaluated on the test split in a separate run (see below).
- **Three seeds.** Standard deviations are population std over 3 seeds. Confidence intervals would need more seeds.

## Corroborating end-to-end run (test split)

After the sweep, the agentic supervisor ran the full pipeline on the same cluster and dataset (supervisor run `395f597b`, outcome `objective_achieved`). This was a single run, not part of the 3-seed sweep:

| Step | Result |
|---|---|
| Topology sweep (reward v2, 4 rounds) | Selected **mesh**, independently of the 10-round sweep above. It took about 25 min, against about 55 min for the longer setting. |
| Final training on mesh | **99.12%** test accuracy |
| 30% structured pruning (accepted) | 78,890 → 38,305 parameters, **98.92%** test accuracy |

A shorter sweep reached the same selection, and the selected topology generalises to the held-out test split. One run gives no variance estimate, so report these test numbers as a single run.

## Suggested paper wording

> Reward-based topology selection must measure model quality on a shared, class-balanced validation split and account for the weakest worker. A reward computed from local non-IID validation accuracy, with a saturating accuracy-gain term, ranked a directed ring first even though it had the lowest shared-split accuracy of the five topologies (76.6%; Spearman ρ = −0.5 with true quality). The proposed quality-gated reward ranked topologies in near-agreement with shared-split accuracy (ρ = 0.9). It selected a mesh, which brought every one of the 8 workers to 97.3 ± 0.4% accuracy, compared with a weakest worker of 65.3% under the directed ring and star topologies.
