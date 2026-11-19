# Quality-gated topology selection on MNIST and CIFAR-10 (reward v2, from scratch)

All runs were measured on the 8-worker Jetson Orin Nano cluster between 2026-11-18 and 2026-11-19.
- Raw per-run numbers: [topology-reward-v2-paper-results-2026-11-19.json](topology-reward-v2-paper-results-2026-11-19.json)
- Reproduce every table with [topology_reward_v2_paper_analysis.py](topology_reward_v2_paper_analysis.py)

This report supersedes the *from-scratch* claims in [topology-reward-v2-2026-11-18.md](topology-reward-v2-2026-11-18.md). That sweep fine-tuned an installed pruned model (see its correction note), and it is kept below as the warm-start setting.

## Summary

- **Five sweeps, 75 hardware runs.** Four start from scratch: MNIST at α = 0.1, 0.3 and 0.5, and CIFAR-10 at α = 0.3. One is a warm start. Each sweep covers 5 topologies × 3 seeds.
- **Reward v1 is anti-correlated with quality.** In every setting, its rank correlation with shared-split accuracy is negative (Spearman ρ between −0.6 and −0.1). It always picks a sparse topology and never the best one.
- **Reward v2 with its quality gate picks the right topology for the regime.**
  - Under strong heterogeneity (MNIST α = 0.1 and 0.3; CIFAR-10) it selects **mesh**. Mesh brings every worker to the same accuracy: 97.5%, 98.4% and 65.5% respectively. The best sparse alternative's weakest worker reaches only 69.5%, 88.0% and 45.3%.
  - Under mild heterogeneity (MNIST α = 0.5) every topology clears the 90% weakest-worker gate. The selector then picks the highest-reward and cheapest topology, the **directed ring**: 97.4% mean and 95.6% weakest, against 98.7% for mesh, with 1/7 of mesh's traffic.
- **Local validation is misleading.** On CIFAR-10 it rates the directed ring equal to mesh (64.7% vs. 64.3%), while on the shared split they are 47.8% vs. 65.5%.

## Setup

| Item | Value |
|---|---|
| Workers | 8 × Jetson Orin Nano 8 GB. Real peer-to-peer model exchange over LAN; semi-synchronous rounds; weighted FedAvg. |
| Datasets | MNIST (28×28×1) and CIFAR-10 (32×32×3), 10 classes each, one Dirichlet non-IID shard per worker. Official test sets untouched. |
| Non-IID levels | MNIST: α = 0.1, 0.3, 0.5. CIFAR-10: α = 0.3. |
| Model | One fixed CNN for every candidate and both datasets: 3 × (conv 3×3 → BN → max-pool) with 32/64/64 channels, GAP, dense 64 (GELU), dropout 0.2, linear output. 61,194 parameters on MNIST, 61,770 on CIFAR-10. Adam, lr 10⁻³, batch 32. |
| Initialization | A fresh, untrained model is installed on every worker before each sweep, and every run starts from it. Accuracy before round 1 was 9–10%, i.e. chance. Seeds change data order and peer timing, not the initial weights. Each run's round-1 weight hash was checked against the installed model. |
| Training | MNIST: 10 rounds × 2 local epochs. CIFAR-10: 30 rounds × 2 local epochs. The warm-start sweep (2026-11-18) used 10 rounds × 5 epochs, starting from the installed 30%-pruned model (4,921 parameters). |
| Topologies | directed ring, bidirectional ring, mesh (all-to-all), chain, star |
| Seeds | 17, 42, 2026. That gives 15 runs per sweep and 60 runs across the four from-scratch sweeps. |
| Quality measurement | A shared, class-balanced validation split: 2,000 images, 200 per class. It is drawn from the preprocessing validation split, which is disjoint from every training shard. The same bytes are installed on every worker, and every worker scores its own model on it after every round. MNIST uses the same split (sha256 `0bac76ac…`) at every α. |
| Test split | Not used for any decision. |

## Reward and selection rule

**Reward v1 (historical baseline).**
`R = 0.5·gain + 0.2·convergence + 0.15·communication + 0.15·wait`
- `gain = min((A_final − A_initial)/0.25, 1)`, where A is each worker's *local* non-IID validation accuracy
- `convergence` is the first round with A ≥ 50%
- the cost terms are normalised by fixed caps (50 MB and 30 s per worker per round)

**Reward v2.**
`R = 0.5·Q + 0.2·S + 0.15·B + 0.15·W`
- `q_r = ½·mean_i F1_{i,r} + ½·min_i F1_{i,r}` is the macro-F1 on the shared split in round r: half the worker mean, half the weakest worker
- `Q = q_R` is the final quality
- `S = mean_r q_r` is the learning speed, the area under the quality curve
- `B = 1 − bytes_sent / (model_bytes · N · (N−1) · R)`, which is 0 for a full mesh
- `W = 1 − Σ peer_wait / Σ round_time`

**Selection (quality-gated).**
1. Keep the candidates whose weakest worker reaches the quality gate (≥ 90% shared-split accuracy) on every seed.
2. Among them, take the highest mean R. When several are within one standard deviation of the best, take the one with the highest B.
3. If no candidate passes the gate, fall back to the highest mean R and report that the gate was missed.

The gate states a requirement on the weakest worker. R then trades quality against cost among the topologies that meet it.

## Results (mean ± std over 3 seeds)

Column meanings:
- **Local acc.** is what workers report on their own non-IID validation slice.
- **Shared acc.** and **Weakest worker** are measured on the shared split after the last round.
- **Gate** counts the seeds on which the weakest worker reached 90%.
- **Bold** marks the selected topology and the highest v1 and v2 rewards.

### Across settings

| Setting | v1 picks | v2 reward alone picks | Gated v2 selects (gate met?) | Spearman ρ v1 / v2 | Mesh shared acc. / weakest | Best non-mesh shared acc. / weakest | Mesh bytes vs. ring |
|---|---|---|---|---|---|---|---|
| MNIST α=0.1 | star | mesh | mesh (yes) | -0.10 / 1.00 | 97.5% / 97.5% | star: 81.0% / 69.5% | 7.0× |
| MNIST α=0.3 | chain | bidir. ring | mesh (yes) | -0.60 / 0.70 | 98.4% / 98.4% | bidir. ring: 93.1% / 88.0% | 7.0× |
| MNIST α=0.5 | chain | directed ring | directed ring (yes) | -0.40 / -0.20 | 98.7% / 98.7% | bidir. ring: 97.9% / 97.0% | 7.0× |
| CIFAR-10 α=0.3 | directed ring | mesh | mesh (no) | -0.50 / 1.00 | 65.5% / 65.5% | bidir. ring: 49.4% / 45.3% | 7.0× |
| MNIST α=0.3 warm start | directed ring | mesh | mesh (yes) | -0.50 / 0.90 | 97.3% / 97.3% | bidir. ring: 87.8% / 80.9% | 7.0× |

"Best non-mesh" is the non-mesh topology with the highest shared-split accuracy.

### Per setting

#### MNIST α=0.1 (mnist, α = 0.1, R = 10, E = 2; group `optimize-1795026562-8a9b6a`)

| Topology | v1 reward | v2 reward | Gate (≥ 90%) | Local acc. | Shared acc. | Weakest worker | Q | S | B | W | MB sent / run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| directed ring | 0.958 ± 0.003 | 0.610 ± 0.023 | 0/3 | 83.0% | 68.7 ± 1.4% | 57.4 ± 4.1% | 0.557 | 0.424 | 0.865 | 0.776 | 18.5 |
| bidirectional ring | 0.958 ± 0.001 | 0.662 ± 0.003 | 0/3 | 91.7% | 75.9 ± 1.4% | 65.9 ± 1.1% | 0.650 | 0.538 | 0.731 | 0.797 | 36.9 |
| **mesh** | 0.945 ± 0.000 | **0.786 ± 0.002** | 3/3 | 97.8% | 97.5 ± 0.3% | 97.5 ± 0.3% | 0.975 | 0.842 | 0.057 | 0.812 | 129.2 |
| chain | 0.956 ± 0.001 | 0.631 ± 0.006 | 0/3 | 92.8% | 71.7 ± 1.6% | 63.8 ± 1.3% | 0.602 | 0.495 | 0.764 | 0.777 | 32.3 |
| star | **0.962 ± 0.010** | 0.687 ± 0.010 | 0/3 | 93.2% | 81.0 ± 1.3% | 69.5 ± 1.6% | 0.694 | 0.557 | 0.764 | 0.758 | 32.3 |

Selected: **mesh** (gate met: True). Spearman ρ vs. shared accuracy: v1 = -0.10, v2 = 1.00.
v1 order: star > bidirectional ring > directed ring > chain > mesh
v2 order: mesh > star > bidirectional ring > chain > directed ring
shared-accuracy order: mesh > star > bidirectional ring > chain > directed ring
Ungated weight sensitivity (969 weight vectors): mesh 64.8%, star 25.4%, directed ring 8.9%, bidirectional ring 0.9%
Gated selection by threshold: 0.4: mesh, 0.45: mesh, 0.5: mesh, 0.55: mesh, 0.6: mesh, 0.65: mesh, 0.7: mesh, 0.75: mesh, 0.8: mesh, 0.85: mesh, 0.9: mesh, 0.95: mesh (* no candidate passes; reward fallback)

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| directed ring mean | 0.222 | 0.508 | 0.573 | 0.600 | 0.622 | 0.643 | 0.655 | 0.659 | 0.668 | 0.687 |
| bidirectional ring mean | 0.205 | 0.569 | 0.650 | 0.701 | 0.713 | 0.728 | 0.742 | 0.739 | 0.773 | 0.759 |
| mesh mean | 0.117 | 0.714 | 0.909 | 0.947 | 0.959 | 0.966 | 0.973 | 0.972 | 0.975 | 0.975 |
| chain mean | 0.180 | 0.543 | 0.629 | 0.653 | 0.657 | 0.679 | 0.691 | 0.703 | 0.706 | 0.717 |
| star mean | 0.293 | 0.560 | 0.692 | 0.719 | 0.740 | 0.759 | 0.777 | 0.786 | 0.796 | 0.810 |

#### MNIST α=0.3 (mnist, α = 0.3, R = 10, E = 2; group `optimize-1795023303-b12686`)

| Topology | v1 reward | v2 reward | Gate (≥ 90%) | Local acc. | Shared acc. | Weakest worker | Q | S | B | W | MB sent / run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| directed ring | 0.962 ± 0.002 | 0.771 ± 0.014 | 0/3 | 94.9% | 88.4 ± 0.6% | 73.4 ± 6.1% | 0.797 | 0.599 | 0.865 | 0.820 | 18.5 |
| bidirectional ring | 0.959 ± 0.002 | **0.825 ± 0.011** | 0/3 | 97.7% | 93.1 ± 0.9% | 88.0 ± 1.3% | 0.894 | 0.728 | 0.731 | 0.821 | 36.9 |
| **mesh** | 0.949 ± 0.001 | 0.805 ± 0.002 | 3/3 | 98.6% | 98.4 ± 0.1% | 98.4 ± 0.1% | 0.984 | 0.871 | 0.057 | 0.867 | 129.2 |
| chain | **0.965 ± 0.001** | 0.806 ± 0.013 | 0/3 | 97.1% | 90.6 ± 0.7% | 81.2 ± 5.2% | 0.844 | 0.695 | 0.764 | 0.872 | 32.3 |
| star | 0.961 ± 0.000 | 0.700 ± 0.007 | 0/3 | 97.2% | 83.6 ± 1.9% | 68.0 ± 1.0% | 0.701 | 0.560 | 0.764 | 0.815 | 32.3 |

Selected: **mesh** (gate met: True). Spearman ρ vs. shared accuracy: v1 = -0.60, v2 = 0.70.
v1 order: chain > directed ring > star > bidirectional ring > mesh
v2 order: bidirectional ring > chain > mesh > directed ring > star
shared-accuracy order: mesh > bidirectional ring > chain > directed ring > star
Ungated weight sensitivity (969 weight vectors): mesh 40.1%, bidirectional ring 30.8%, directed ring 15.8%, chain 13.3%
Gated selection by threshold: 0.4: bidirectional ring, 0.45: bidirectional ring, 0.5: bidirectional ring, 0.55: bidirectional ring, 0.6: bidirectional ring, 0.65: bidirectional ring, 0.7: bidirectional ring, 0.75: bidirectional ring, 0.8: bidirectional ring, 0.85: bidirectional ring, 0.9: mesh, 0.95: mesh (* no candidate passes; reward fallback)

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| directed ring mean | 0.182 | 0.591 | 0.692 | 0.754 | 0.797 | 0.822 | 0.838 | 0.853 | 0.874 | 0.884 |
| bidirectional ring mean | 0.142 | 0.646 | 0.808 | 0.842 | 0.873 | 0.894 | 0.906 | 0.913 | 0.919 | 0.931 |
| mesh mean | 0.100 | 0.867 | 0.962 | 0.970 | 0.975 | 0.981 | 0.981 | 0.984 | 0.984 | 0.984 |
| chain mean | 0.140 | 0.659 | 0.776 | 0.838 | 0.860 | 0.875 | 0.891 | 0.895 | 0.898 | 0.906 |
| star mean | 0.109 | 0.588 | 0.683 | 0.731 | 0.749 | 0.773 | 0.786 | 0.803 | 0.826 | 0.836 |

#### MNIST α=0.5 (mnist, α = 0.5, R = 10, E = 2; group `optimize-1795029636-6f3416`)

| Topology | v1 reward | v2 reward | Gate (≥ 90%) | Local acc. | Shared acc. | Weakest worker | Q | S | B | W | MB sent / run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **directed ring** | 0.949 ± 0.002 | **0.884 ± 0.003** | 3/3 | 98.3% | 97.4 ± 0.2% | 95.6 ± 1.0% | 0.965 | 0.833 | 0.865 | 0.702 | 18.5 |
| bidirectional ring | 0.944 ± 0.001 | 0.871 ± 0.001 | 3/3 | 98.3% | 97.9 ± 0.1% | 97.0 ± 0.1% | 0.975 | 0.858 | 0.731 | 0.685 | 36.9 |
| mesh | 0.942 ± 0.001 | 0.800 ± 0.002 | 3/3 | 98.8% | 98.7 ± 0.1% | 98.7 ± 0.1% | 0.987 | 0.893 | 0.057 | 0.797 | 129.3 |
| chain | **0.952 ± 0.000** | 0.869 ± 0.009 | 3/3 | 98.3% | 96.9 ± 0.5% | 92.8 ± 2.8% | 0.948 | 0.846 | 0.764 | 0.740 | 32.3 |
| star | 0.932 ± 0.007 | 0.857 ± 0.006 | 3/3 | 96.8% | 97.1 ± 0.6% | 95.5 ± 1.5% | 0.963 | 0.881 | 0.764 | 0.564 | 32.3 |

Selected: **directed ring** (gate met: True). Spearman ρ vs. shared accuracy: v1 = -0.40, v2 = -0.20.
v1 order: chain > directed ring > bidirectional ring > mesh > star
v2 order: directed ring > bidirectional ring > chain > star > mesh
shared-accuracy order: mesh > bidirectional ring > directed ring > star > chain
Ungated weight sensitivity (969 weight vectors): directed ring 62.4%, mesh 24.4%, star 5.1%, chain 4.7%, bidirectional ring 3.4%
Gated selection by threshold: 0.4: directed ring, 0.45: directed ring, 0.5: directed ring, 0.55: directed ring, 0.6: directed ring, 0.65: directed ring, 0.7: directed ring, 0.75: directed ring, 0.8: directed ring, 0.85: directed ring, 0.9: directed ring, 0.95: bidirectional ring (* no candidate passes; reward fallback)

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| directed ring mean | 0.347 | 0.832 | 0.931 | 0.944 | 0.956 | 0.958 | 0.963 | 0.969 | 0.969 | 0.974 |
| bidirectional ring mean | 0.277 | 0.869 | 0.963 | 0.966 | 0.970 | 0.975 | 0.974 | 0.977 | 0.977 | 0.979 |
| mesh mean | 0.166 | 0.965 | 0.979 | 0.982 | 0.985 | 0.984 | 0.986 | 0.987 | 0.987 | 0.987 |
| chain mean | 0.276 | 0.849 | 0.953 | 0.964 | 0.964 | 0.968 | 0.968 | 0.970 | 0.970 | 0.969 |
| star mean | 0.418 | 0.927 | 0.962 | 0.958 | 0.968 | 0.969 | 0.969 | 0.972 | 0.974 | 0.971 |

#### CIFAR-10 α=0.3 (cifar10, α = 0.3, R = 30, E = 2; group `optimize-1795014072-efec7c`)

| Topology | v1 reward | v2 reward | Gate (≥ 90%) | Local acc. | Shared acc. | Weakest worker | Q | S | B | W | MB sent / run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| directed ring | **0.959 ± 0.004** | 0.491 ± 0.003 | 0/3 | 64.7% | 47.8 ± 0.3% | 42.9 ± 1.2% | 0.393 | 0.327 | 0.865 | 0.659 | 56.0 |
| bidirectional ring | 0.949 ± 0.002 | 0.496 ± 0.004 | 0/3 | 63.9% | 49.4 ± 0.4% | 45.3 ± 1.3% | 0.432 | 0.368 | 0.730 | 0.648 | 112.0 |
| **mesh** | 0.923 ± 0.004 | **0.555 ± 0.002** | 0/3 | 64.3% | 65.5 ± 0.8% | 65.5 ± 0.8% | 0.655 | 0.592 | 0.055 | 0.672 | 392.1 |
| chain | 0.957 ± 0.002 | 0.474 ± 0.002 | 0/3 | 64.3% | 47.5 ± 0.4% | 39.5 ± 0.3% | 0.383 | 0.348 | 0.764 | 0.656 | 98.0 |
| star | 0.946 ± 0.004 | 0.480 ± 0.004 | 0/3 | 58.9% | 47.6 ± 0.2% | 40.2 ± 1.0% | 0.396 | 0.341 | 0.764 | 0.665 | 98.0 |

Selected: **mesh** (gate met: False). Spearman ρ vs. shared accuracy: v1 = -0.50, v2 = 1.00.
v1 order: directed ring > chain > bidirectional ring > star > mesh
v2 order: mesh > bidirectional ring > directed ring > star > chain
shared-accuracy order: mesh > bidirectional ring > directed ring > star > chain
Ungated weight sensitivity (969 weight vectors): mesh 55.9%, directed ring 44.1%
Gated selection by threshold: 0.4: mesh, 0.45: mesh, 0.5: mesh, 0.55: mesh, 0.6: mesh, 0.65: mesh*, 0.7: mesh*, 0.75: mesh*, 0.8: mesh*, 0.85: mesh*, 0.9: mesh*, 0.95: mesh* (* no candidate passes; reward fallback)

Re-checked under `95c8027` (when no topology passes, only topologies whose weakest worker is within 0.05 of the best stay in the pool): the selection is still **mesh** at every threshold, because mesh's weakest worker (65.5%) is 20 points above the next topology (bidirectional ring, 45.3%). The method is then reported as `relative_quality_gated_reward_v2`.

| Round | 1 | 4 | 7 | 10 | 13 | 16 | 19 | 22 | 25 | 28 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| directed ring mean | 0.203 | 0.359 | 0.392 | 0.424 | 0.435 | 0.445 | 0.448 | 0.464 | 0.471 | 0.475 | 0.478 |
| bidirectional ring mean | 0.198 | 0.390 | 0.435 | 0.455 | 0.465 | 0.472 | 0.480 | 0.483 | 0.494 | 0.498 | 0.494 |
| mesh mean | 0.197 | 0.532 | 0.588 | 0.614 | 0.635 | 0.641 | 0.656 | 0.652 | 0.650 | 0.653 | 0.655 |
| chain mean | 0.211 | 0.397 | 0.439 | 0.450 | 0.461 | 0.469 | 0.462 | 0.473 | 0.471 | 0.473 | 0.475 |
| star mean | 0.183 | 0.378 | 0.423 | 0.446 | 0.461 | 0.463 | 0.466 | 0.467 | 0.475 | 0.469 | 0.476 |

#### MNIST α=0.3 warm start (mnist, α = 0.3, R = 10, E = 5; group `optimize-1794992581-58384f`)

| Topology | v1 reward | v2 reward | Gate (≥ 90%) | Local acc. | Shared acc. | Weakest worker | Q | S | B | W | MB sent / run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| directed ring | **0.969 ± 0.002** | 0.715 ± 0.013 | 0/3 | 88.0% | 76.6 ± 0.8% | 65.3 ± 4.9% | 0.681 | 0.616 | 0.832 | 0.843 | 1.9 |
| bidirectional ring | 0.967 ± 0.001 | 0.785 ± 0.011 | 0/3 | 95.1% | 87.8 ± 1.1% | 80.9 ± 2.5% | 0.828 | 0.734 | 0.663 | 0.834 | 3.7 |
| **mesh** | 0.964 ± 0.005 | **0.786 ± 0.002** | 3/3 | 96.9% | 97.3 ± 0.4% | 97.3 ± 0.4% | 0.973 | 0.874 | 0.000 | 0.835 | 13.0 |
| chain | 0.961 ± 0.013 | 0.762 ± 0.004 | 0/3 | 94.3% | 85.1 ± 1.0% | 78.1 ± 1.1% | 0.789 | 0.676 | 0.705 | 0.839 | 3.2 |
| star | 0.964 ± 0.002 | 0.682 ± 0.010 | 0/3 | 91.0% | 77.4 ± 1.6% | 65.3 ± 2.8% | 0.659 | 0.614 | 0.706 | 0.825 | 3.2 |

Selected: **mesh** (gate met: True). Spearman ρ vs. shared accuracy: v1 = -0.50, v2 = 0.90.
v1 order: directed ring > bidirectional ring > star > mesh > chain
v2 order: mesh > bidirectional ring > chain > directed ring > star
shared-accuracy order: mesh > bidirectional ring > chain > star > directed ring
Ungated weight sensitivity (969 weight vectors): mesh 44.8%, bidirectional ring 36.8%, directed ring 18.3%, chain 0.1%
Gated selection by threshold: 0.4: bidirectional ring, 0.45: bidirectional ring, 0.5: bidirectional ring, 0.55: bidirectional ring, 0.6: bidirectional ring, 0.65: bidirectional ring, 0.7: bidirectional ring, 0.75: bidirectional ring, 0.8: mesh, 0.85: mesh, 0.9: mesh, 0.95: mesh (* no candidate passes; reward fallback)

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| directed ring mean | 0.620 | 0.669 | 0.694 | 0.724 | 0.746 | 0.726 | 0.733 | 0.757 | 0.777 | 0.766 |
| bidirectional ring mean | 0.588 | 0.731 | 0.763 | 0.823 | 0.827 | 0.855 | 0.868 | 0.876 | 0.854 | 0.878 |
| mesh mean | 0.539 | 0.684 | 0.828 | 0.934 | 0.964 | 0.951 | 0.969 | 0.972 | 0.976 | 0.973 |
| chain mean | 0.561 | 0.703 | 0.714 | 0.796 | 0.785 | 0.796 | 0.812 | 0.842 | 0.817 | 0.851 |
| star mean | 0.635 | 0.681 | 0.718 | 0.725 | 0.738 | 0.746 | 0.752 | 0.768 | 0.775 | 0.774 |


## Findings

1. **Reward v1 ranks topologies roughly in reverse order of quality.**
   - In every setting v1 picks a sparse topology (star, chain or directed ring), never the most accurate one, and ranks mesh last or near last.
   - Its rank correlation with shared-split accuracy is negative everywhere (ρ between −0.6 and −0.1).
   - The reasons, visible in the per-run components:
     - Every topology improves by more than 25 points, so the gain term saturates at 1.0.
     - Every topology passes 50% accuracy in round 1–3, so the convergence term barely varies.
     - What remains is the cost terms, and those reward the sparsest graph.
2. **Reward v2 tracks quality where quality differs.**
   - ρ = 1.0 on MNIST α = 0.1 and on CIFAR-10, and ρ = 0.7 on MNIST α = 0.3.
   - On MNIST α = 0.1 and 0.3 mesh is the only topology whose weakest worker clears 90% on every seed, and the gated selector picks it. On CIFAR-10 no topology clears 90%, so the selector falls back to the highest reward, which is mesh, and reports the gate as missed.
   - On MNIST α = 0.5 all five topologies are within 1.8 points of each other (96.9–98.7%). The cost terms then dominate by design, so ρ = −0.2 there; see finding 6.
3. **Local validation cannot see the difference; the shared split can.**
   - Workers' local non-IID validation accuracy exceeds their shared-split accuracy by 5–21 points in the sparse topologies, against at most 1.2 points in mesh.
   - On CIFAR-10 local accuracy is 59–65% for *every* topology. That makes a directed ring look as good as a mesh (64.7% vs. 64.3%), while on the shared split they differ by 17.7 points (47.8% vs. 65.5%).
4. **The weakest worker separates the topologies, and the gap grows with heterogeneity.**
   - Mesh brings every worker to the same model, so its weakest worker equals its mean.
   - In sparse topologies, workers whose shards miss classes stay behind. At MNIST α = 0.1 the best non-mesh topology (star) reaches 81.0% mean but only 69.5% on its weakest worker, against 97.5% for every mesh worker. At α = 0.3 the best alternative (bidirectional ring) reaches 93.1% / 88.0%.
5. **The gate, not the weights, makes the selection robust when a cheaper topology is close.**
   - On MNIST α = 0.3 the bidirectional ring reaches 93.1% with half the bytes of a mesh. With the default weights its reward (0.825) exceeds mesh (0.805), because B rewards its lower traffic.
   - Only the gate, which its weakest worker (88.0%) misses, moves the selection to mesh. At any gate of 0.85 or below the selector returns the bidirectional ring.
   - This is the intended trade-off: the gate encodes the requirement, and R then prefers the cheapest topology that meets it.
6. **When heterogeneity is mild, the selector trades a little accuracy for a lot of bandwidth.**
   - At MNIST α = 0.5 every topology's weakest worker clears 90% on every seed (92.8–98.7%). The selector picks the directed ring:
     - 97.4 ± 0.2% mean and 95.6 ± 1.0% weakest, against 98.7% for mesh;
     - 18.5 MB per run instead of 129 MB, i.e. 1.3 points of accuracy for 7× less traffic.
   - v1 picks the chain instead, which has the lowest mean and weakest-worker accuracy of the five (96.9% / 92.8%).
   - Together with findings 2 and 5, this shows the rule adapts: full mixing is selected only when sparser graphs leave some worker below the requirement.
7. **Cost.**
   - Mesh sends N − 1 = 7 times the bytes of a directed ring in every setting: 129 MB vs. 18.5 MB per MNIST run, and 392 MB vs. 56 MB per CIFAR-10 run.
   - Wall-clock cost is smaller than the byte ratio suggests: +14% vs. the directed ring on MNIST α = 0.3 (165 s vs. 145 s), and +31% on CIFAR-10 (563 s vs. 431 s).
   - Mesh also converges in fewer rounds. It passes 90% mean accuracy on MNIST by round 3, while the directed ring never does within 10 rounds. On CIFAR-10 it passes 45% at round 3, against round 10–17 for the other topologies.
8. **Warm start gives the same picture.** The earlier sweep that fine-tuned an installed pruned model (MNIST α = 0.3, 10 rounds × 5 epochs) shows the same ordering: v1 ρ = −0.5, v2 ρ = 0.9, and mesh selected with the gate met. That makes the selection independent of whether training starts from scratch or from a compressed pre-trained model.

## Sensitivity and limitations

- **Default weights are not uniquely decisive.**
  - Over all 969 weight vectors on a 0.05 grid with quality weight ≥ 0.2, the ungated reward picks mesh in 64.8% of vectors on MNIST α = 0.1, 40.1% on α = 0.3, 24.4% on α = 0.5 and 55.9% on CIFAR-10. The rest go to sparser topologies, which win when bandwidth or wait weights are high.
  - Report the gate threshold and the weights as design choices, and include the threshold sweep. The gated selection is mesh for every threshold on MNIST α = 0.1, for thresholds ≥ 0.90 on MNIST α = 0.3, and for thresholds ≤ 0.60 on CIFAR-10 (where it is the only topology that passes). On MNIST α = 0.5 it is the directed ring for every threshold up to 0.90, and the bidirectional ring at 0.95.
- **The 90% gate is MNIST-scaled.** On CIFAR-10 no topology reaches it with this model and budget, so the selection there comes from the fallback: the highest reward in this sweep and, since `95c8027`, a gate relative to the best weakest worker, which keeps only mesh and gives the same selection. A dataset-relative gate, or one set from a centralised reference, would be more principled.
- **Model capacity limits CIFAR-10 accuracy.** A 62k-parameter CNN trained for 30 rounds × 2 epochs reaches 65.5% even with full mixing. The ranking result does not depend on reaching state-of-the-art accuracy, but absolute CIFAR-10 numbers should not be compared with centralised baselines.
- **Fixed initialization across seeds.** Seeds vary data order and timing, not the initial weights, so the standard deviations understate initialization variance.
- **Population std over 3 seeds.** Confidence intervals would need more seeds.
- **No test-set numbers per topology.** Workers keep only their newest model. The selection was made on validation data only. For test-set evidence of the selected topology, see the end-to-end supervisor run in [topology-reward-v2-2026-11-18.md](topology-reward-v2-2026-11-18.md): mesh with a NAS-selected model, 99.12% MNIST test accuracy, and 98.92% after 30% pruning.
- **One cluster and one LAN.** The bandwidth trade-off weighs more on constrained links or with larger models, where the 7× traffic of a mesh may not be affordable. That is the regime in which the gate-plus-cost rule would select a sparser topology.

## Suggested paper wording

> Reward-driven topology selection for decentralized federated learning must measure quality on a shared, class-balanced validation split and must account for the weakest worker. We compared five communication topologies on eight Jetson Orin Nano devices, with three seeds per configuration, across MNIST at three non-IID levels (Dirichlet α = 0.1, 0.3, 0.5) and CIFAR-10 (α = 0.3).
>
> A conventional reward built on local validation accuracy, a saturating accuracy-gain term and fixed cost caps ranked the topologies roughly in reverse order of their true quality (Spearman ρ between −0.6 and −0.1 against shared-split accuracy). It never selected the most accurate topology. Under strong heterogeneity, it also never selected the mesh, the only topology that brought every worker above 90%.
>
> The proposed quality-gated reward combines four terms: final and area-under-curve macro-F1 that weight the weakest worker equally with the mean, bandwidth relative to a full mesh, and idle time. A minimum weakest-worker accuracy acts as a gate. Under strong heterogeneity (α ≤ 0.3, and CIFAR-10), this reward tracked quality closely (ρ = 0.7–1.0) and selected the mesh: every worker reached 97.5–98.4% on MNIST, while the best sparse topology left its weakest worker at 69.5–88.0%. Under mild heterogeneity (α = 0.5), every topology met the 90% weakest-worker requirement, and the rule selected a directed ring that reached 97.4% (vs. 98.7% for the mesh) with one seventh of the communication.
>
> Local validation accuracy alone was misleading. On CIFAR-10 it rated a directed ring equal to a mesh (64.7% vs. 64.3%), while their shared-split accuracies were 47.8% and 65.5%.
