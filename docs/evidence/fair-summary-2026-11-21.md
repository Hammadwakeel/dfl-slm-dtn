# Fair step-by-step evaluation: Fashion-MNIST and FEMNIST

Generated 2026-11-24 22:54 UTC by `scripts/fair_summary.py`.

Eight Jetson workers, decentralized training, Dirichlet label split with α = 0.1, 0.5 and 0.9,
seeds 17, 42 and 2026. Every trained model (baseline and each step's measurement) uses the
same budget: 10 rounds × 5 local epochs from the same installed initial weights. Accuracy is
the mean over the 8 workers on the dataset's shared test split. Values are mean ± std over seeds.

## Baseline vs. final model (same training budget)

| Dataset | α | Seeds | Baseline test (%) | Final model test (%) | Gain (points) | Final params |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 3 | 42.22 ± 3.28 | 73.25 ± 10.55 | +31.03 ± 12.40 | 109,450, 129,770, 51,754 |
| Fashion-MNIST | 0.5 | 3 | 78.41 ± 1.60 | 89.67 ± 1.29 | +11.25 ± 0.32 | 109,450, 129,770, 69,770 |
| Fashion-MNIST | 0.9 | 3 | 83.61 ± 0.50 | 89.98 ± 0.90 | +6.37 ± 0.51 | 129,770, 22,682, 28,810 |
| FEMNIST | 0.1 | 3 | 33.38 ± 4.38 | 70.57 ± 4.49 | +37.19 ± 7.87 | 112,830, 133,150, 73,150 |
| FEMNIST | 0.5 | 3 | 66.73 ± 1.05 | 82.73 ± 1.23 | +16.00 ± 0.57 | 112,830, 133,150, 73,150 |
| FEMNIST | 0.9 | 3 | 71.78 ± 2.08 | 83.62 ± 1.13 | +11.84 ± 1.12 | 112,830, 133,150, 55,134 |

## Effect of each step

Change in validation and test accuracy (points) caused by each step, measured against the
configuration active just before it. "Kept" counts seeds where the change was kept; a change
more than 0.5 points below the best so far is reverted. The learned topology is the last step
and is reported without a keep/revert decision.

### Fashion-MNIST

| Step | α 0.1: Δ val | α 0.1: Δ test | α 0.1: kept | α 0.5: Δ val | α 0.5: Δ test | α 0.5: kept | α 0.9: Δ val | α 0.9: Δ test | α 0.9: kept |
|---|---|---|---|---|---|---|---|---|---|
| NAS | +8.11 ± 5.90 | +8.03 ± 5.73 | 3/3 | +4.00 ± 1.88 | +3.86 ± 2.08 | 3/3 | +2.24 ± 1.71 | +2.00 ± 1.61 | 3/3 |
| Topology sweep | +19.72 ± 8.17 | +19.57 ± 7.84 | 3/3 | +7.38 ± 2.26 | +7.02 ± 2.09 | 3/3 | +4.29 ± 1.01 | +4.34 ± 1.14 | 3/3 |
| Aggregation sweep | +1.30 ± 2.12 | +1.21 ± 2.05 | 3/3 | +0.18 ± 0.16 | +0.44 ± 0.37 | 3/3 | -0.41 ± 0.31 | -0.52 ± 0.61 | 2/3 |
| HPO | -2.63 ± 6.05 | -2.48 ± 6.47 | 1/3 | -0.86 ± 0.38 | -0.80 ± 0.74 | 1/3 | -0.38 ± 0.17 | -0.27 ± 0.43 | 1/3 |
| Synchronization sweep | -2.83 ± 5.90 | -2.79 ± 5.71 | 2/3 | -0.61 ± 1.10 | -0.59 ± 0.99 | 2/3 | +0.11 ± 0.41 | -0.08 ± 0.71 | 3/3 |
| Learned topology (APG) | -12.67 ± 9.57 | -12.43 ± 9.74 | – | -2.07 ± 0.35 | -2.05 ± 0.20 | – | -0.94 ± 0.21 | -0.82 ± 0.49 | – |

### FEMNIST

| Step | α 0.1: Δ val | α 0.1: Δ test | α 0.1: kept | α 0.5: Δ val | α 0.5: Δ test | α 0.5: kept | α 0.9: Δ val | α 0.9: Δ test | α 0.9: kept |
|---|---|---|---|---|---|---|---|---|---|
| NAS | +6.77 ± 7.26 | +6.66 ± 7.38 | 3/3 | +5.45 ± 1.79 | +5.26 ± 1.99 | 3/3 | +4.55 ± 1.52 | +4.63 ± 1.56 | 3/3 |
| Topology sweep | +28.32 ± 1.99 | +28.74 ± 2.03 | 3/3 | +10.31 ± 1.69 | +10.32 ± 1.68 | 3/3 | +6.24 ± 1.22 | +6.35 ± 1.17 | 3/3 |
| Aggregation sweep | +1.82 ± 0.80 | +1.46 ± 0.88 | 3/3 | -0.12 ± 0.50 | +0.17 ± 0.52 | 2/3 | +0.81 ± 1.17 | +0.65 ± 1.33 | 3/3 |
| HPO | -0.85 ± 0.62 | -1.03 ± 0.85 | 1/3 | +0.10 ± 0.64 | -0.36 ± 0.50 | 2/3 | -0.27 ± 0.50 | -0.31 ± 0.19 | 2/3 |
| Synchronization sweep | +0.13 ± 1.45 | -0.07 ± 1.13 | 1/3 | -0.23 ± 0.70 | +0.10 ± 0.38 | 2/3 | +0.03 ± 0.41 | +0.35 ± 0.38 | 3/3 |
| Learned topology (APG) | -15.35 ± 4.64 | -15.26 ± 4.40 | – | -3.52 ± 0.85 | -3.15 ± 0.89 | – | -1.37 ± 1.06 | -1.62 ± 0.86 | – |

## Communication per training

Every training runs 10 rounds; in each round every directed edge carries one model
(float32 weights). Directed edges per round: directed ring 8,
mesh 56; the learned (APG) graph is measured per run. The baseline uses
the baseline CNN on the directed ring; the final model is the run's best model on its topology.
The search steps (NAS, sweeps) are not counted; they cost the same for every seed.

| Dataset | α | Baseline: models / MB | Final model: models / MB | Learned (APG): edges / models / MB | Final test (%) | Learned test (%) |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 80 / 9.2 | 560 / 217.3 | 31.7 / 317 / 120.3 | 73.25 ± 10.55 | 60.51 ± 4.32 |
| Fashion-MNIST | 0.5 | 80 / 9.2 | 560 / 230.7 | 27.3 / 273 / 112.2 | 89.67 ± 1.29 | 87.68 ± 1.39 |
| Fashion-MNIST | 0.9 | 80 / 9.2 | 560 / 135.3 | 28.3 / 283 / 68.0 | 89.98 ± 0.90 | 88.75 ± 1.01 |
| FEMNIST | 0.1 | 80 / 11.4 | 560 / 238.3 | 31.0 / 310 / 132.7 | 70.57 ± 4.49 | 55.31 ± 1.45 |
| FEMNIST | 0.5 | 80 / 11.4 | 560 / 238.3 | 26.3 / 263 / 111.1 | 82.73 ± 1.23 | 79.58 ± 1.89 |
| FEMNIST | 0.9 | 80 / 11.4 | 560 / 224.8 | 28.7 / 287 / 117.8 | 83.62 ± 1.13 | 81.97 ± 1.89 |

## Learned topology (APG) diagnostics

APG runs on a 2-round mesh probe. It keeps, for each receiver, the senders whose models
have a low loss on the receiver's own validation data and arrive quickly; a small balance
term pulls the neighbourhood label mix toward the global one. The table compares the kept
and dropped senders and the label gap (total-variation distance) between the data a worker
aggregates over (itself and its senders) and the global label mix; on the mesh that gap is 0.

| Dataset | α | Senders kept per worker | Peer loss: kept / dropped | Label cosine: kept / dropped | Label gap vs global |
|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 4.0 of 7 | 2.12 / 3.00 | 0.14 / 0.11 | 0.37 |
| Fashion-MNIST | 0.5 | 3.4 of 7 | 1.40 / 1.72 | 0.45 / 0.37 | 0.18 |
| Fashion-MNIST | 0.9 | 3.5 of 7 | 1.42 / 1.70 | 0.55 / 0.55 | 0.12 |
| FEMNIST | 0.1 | 3.9 of 7 | 4.47 / 5.42 | 0.11 / 0.09 | 0.33 |
| FEMNIST | 0.5 | 3.3 of 7 | 2.93 / 3.40 | 0.40 / 0.38 | 0.17 |
| FEMNIST | 0.9 | 3.6 of 7 | 2.86 / 3.12 | 0.52 / 0.49 | 0.12 |

APG optimizes each worker's local fit and waiting time, not global accuracy. It keeps
peers whose models already suit the receiver's data, so each worker averages over only part
of the label space. The label gap and the accuracy loss of the learned topology are both
largest at α 0.1, where the label skew is strongest. The learned graph sends about half as
many models as the mesh, so it trades accuracy for communication.

## Leave-one-out ablation (order-independent)

The step effects above depend on the order the SLM chose. To remove that dependence, each
run's final configuration is retrained with the same budget, once as it is (full) and once
without each component the run kept. A component's contribution is full minus without (test
points). When a seed was ablated more than once, its contribution is the mean over those
repeats; values are mean ± std over the seeds that kept the component (n). Details per run are
in `fair-ablation-2026-11-21.md`.

| Dataset | α | Ablation passes | Full test (%) | NAS (n) | HPO (n) | Topology (n) | Aggregation (n) | Sync (n) |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 3 | 72.12 ± 10.10 | +10.79 ± 12.52 (3) | +2.82 (1) | +22.18 ± 7.75 (3) | – (0) | -3.92 (1) |
| Fashion-MNIST | 0.5 | 1 | 89.21 ± 2.23 | +1.98 ± 1.69 (3) | -0.34 (1) | +6.38 ± 0.85 (3) | -0.29 ± 0.98 (2) | +0.30 ± 0.29 (2) |
| Fashion-MNIST | 0.9 | 1 | 89.72 ± 0.98 | +1.27 ± 1.34 (2) | -0.11 (1) | +4.23 ± 1.21 (3) | +0.14 (1) | – (0) |
| FEMNIST | 0.1 | 1 | 70.18 ± 4.20 | +6.89 ± 5.40 (3) | – (0) | +29.52 ± 0.23 (3) | +2.55 ± 1.22 (2) | – (0) |
| FEMNIST | 0.5 | 1 | 82.60 ± 1.50 | +3.31 ± 0.65 (3) | – (0) | +10.96 ± 2.10 (3) | +0.20 ± 0.23 (2) | +0.07 ± 0.14 (2) |
| FEMNIST | 0.9 | 1 | 83.69 ± 1.10 | +2.71 ± 2.06 (3) | +0.43 (1) | +6.88 ± 1.17 (3) | +0.11 ± 0.06 (2) | +0.00 ± 0.13 (3) |

## Run-to-run noise

The baseline and the full configuration are retrained with the same seed in the campaign
and in every ablation pass. The spread between those repeats is run-to-run noise on the
cluster (thread scheduling, network timing and semi-synchronous deadlines are not
deterministic). The table gives the mean over seeds of the per-seed standard deviation and
the largest per-seed range (max − min), in test points.

| Dataset | α | Trainings per seed | Baseline: mean std / max range | Full: mean std / max range |
|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 4 | 2.83 / 14.29 | 2.31 / 9.38 |
| Fashion-MNIST | 0.5 | 2 | 1.25 / 3.91 | 0.43 / 1.57 |
| Fashion-MNIST | 0.9 | 2 | 1.27 / 3.28 | 0.19 / 0.36 |
| FEMNIST | 0.1 | 2 | 0.61 / 1.54 | 0.36 / 1.15 |
| FEMNIST | 0.5 | 2 | 1.36 / 3.01 | 0.32 / 0.57 |
| FEMNIST | 0.9 | 2 | 0.65 / 1.51 | 0.11 / 0.20 |

Per seed (mean ± std over the repeats; the campaign value comes first):

| Dataset | α | Seed | Baseline test (%) | Full test (%) |
|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 17 | 42.25 ± 1.08 (43.13, 43.11, 40.90, 41.87) | 81.05 ± 2.22 (82.40, 83.45, 78.82, 79.55) |
| Fashion-MNIST | 0.1 | 42 | 45.77 ± 0.99 (44.94, 46.85, 46.37, 44.93) | 61.14 ± 4.04 (61.71, 55.39, 64.77, 62.68) |
| Fashion-MNIST | 0.1 | 2026 | 48.07 ± 6.43 (38.58, 52.88, 50.39, 50.43) | 75.00 ± 0.66 (75.64, 75.39, 74.85, 74.14) |
| Fashion-MNIST | 0.5 | 17 | 79.89 ± 0.28 (79.69, 80.08) | 90.64 ± 0.02 (90.65, 90.62) |
| Fashion-MNIST | 0.5 | 42 | 74.67 ± 2.76 (76.62, 72.72) | 87.42 ± 1.11 (88.21, 86.64) |
| Fashion-MNIST | 0.5 | 2026 | 79.44 ± 0.71 (78.93, 79.94) | 90.26 ± 0.17 (90.14, 90.38) |
| Fashion-MNIST | 0.9 | 17 | 83.76 ± 0.60 (84.19, 83.34) | 90.75 ± 0.14 (90.85, 90.65) |
| Fashion-MNIST | 0.9 | 42 | 82.65 ± 0.87 (83.27, 82.03) | 88.87 ± 0.25 (89.05, 88.69) |
| Fashion-MNIST | 0.9 | 2026 | 85.02 ± 2.32 (83.38, 86.67) | 89.92 ± 0.16 (90.04, 89.81) |
| FEMNIST | 0.1 | 17 | 28.78 ± 0.55 (28.39, 29.17) | 73.62 ± 0.81 (74.20, 73.05) |
| FEMNIST | 0.1 | 42 | 34.41 ± 1.09 (35.18, 33.64) | 65.46 ± 0.14 (65.56, 65.36) |
| FEMNIST | 0.1 | 2026 | 36.44 ± 0.19 (36.58, 36.31) | 72.06 ± 0.13 (71.97, 72.15) |
| FEMNIST | 0.5 | 17 | 65.80 ± 2.13 (67.30, 64.29) | 83.75 ± 0.21 (83.90, 83.60) |
| FEMNIST | 0.5 | 42 | 66.20 ± 1.67 (67.37, 65.02) | 83.08 ± 0.35 (82.84, 83.33) |
| FEMNIST | 0.5 | 2026 | 65.31 ± 0.30 (65.52, 65.10) | 81.16 ± 0.40 (81.45, 80.88) |
| FEMNIST | 0.9 | 17 | 73.53 ± 0.83 (74.11, 72.95) | 84.75 ± 0.10 (84.68, 84.82) |
| FEMNIST | 0.9 | 42 | 71.05 ± 0.05 (71.08, 71.01) | 83.68 ± 0.09 (83.75, 83.62) |
| FEMNIST | 0.9 | 2026 | 70.90 ± 1.06 (70.14, 71.65) | 82.53 ± 0.14 (82.43, 82.63) |

## Topology or communication volume? (in-degree study)

The mesh's gain could come from receiving 7 models per round instead of 1. Each seed's final
configuration (everything else unchanged) is trained on circulant graphs where worker i averages
with workers i-1 .. i-k, with the same 10 × 5 budget, and on the directed ring for 70 rounds × 1
epoch, which receives as many models as the mesh (70 per worker) with more local compute.
k = 1 and k = 7 come from the ablation passes. The spectral gap (1 − |λ2| of the uniform averaging
matrix) measures how much one round mixes the workers' models. The learned (APG) graph of the
same seed, which keeps about 4 senders per worker, is listed for comparison with k = 4. Test %,
mean ± std over seeds; details in `topology-degree-2026-11-21.md`.

| Dataset | α | Graph | Models received per worker | Spectral gap | Test (%) | Seeds |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | in-degree 1 (ring) | 10 | 0.076 | 49.94 ± 3.26 | 3 |
| Fashion-MNIST | 0.1 | in-degree 2 | 20 | 0.195 | 58.52 ± 8.87 | 3 |
| Fashion-MNIST | 0.1 | in-degree 4 | 40 | 0.517 | 68.68 ± 7.13 | 3 |
| Fashion-MNIST | 0.1 | in-degree 7 (mesh) | 70 | 1.000 | 72.12 ± 10.10 | 3 |
| Fashion-MNIST | 0.1 | ring 70x1 | 70 | 0.076 (7× rounds) | 56.96 ± 9.29 | 3 |
| Fashion-MNIST | 0.1 | learned (APG), 4.0 senders | 40 | varies | 60.51 ± 4.32 | 3 |

## Are the keep-or-revert decisions above noise?

σ is the pooled within-seed standard deviation of global validation accuracy over repeated
trainings of the same configuration and seed. A decision compares two measurements, so a
difference beyond 2·σ·√2 (about 95 %) is *clear*; smaller differences are *within noise*. The
default rule keeps changes down to 0.5 points below the best, well inside that band, so a single
kept HPO, aggregation or synchronization decision is not evidence of an improvement; the paired
test below shows where repeats do settle it. The supervisor now takes the tolerance from the goal
(`accept_tolerance`), so a run can require a
gain larger than the noise. Every decision is listed in `decision-robustness-2026-11-21.md`.

| Dataset | α | σ (points) | 95 % band | Decisions | Clear gain | Clear loss | Within noise |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 3.58 | ±10.13 | 15 | 5 | 0 | 10 |
| Fashion-MNIST | 0.5 | 1.59 | ±4.51 | 15 | 5 | 0 | 10 |
| Fashion-MNIST | 0.9 | 1.72 | ±4.88 | 15 | 1 | 0 | 14 |
| FEMNIST | 0.1 | 0.78 | ±2.21 | 15 | 6 | 0 | 9 |
| FEMNIST | 0.5 | 1.52 | ±4.29 | 15 | 5 | 0 | 10 |
| FEMNIST | 0.9 | 0.75 | ±2.13 | 15 | 6 | 0 | 9 |

| Step | Decisions | Clear gain | Clear loss | Within noise |
|---|---|---|---|---|
| NAS | 18 | 11 | 0 | 7 |
| HPO | 18 | 0 | 0 | 18 |
| topology | 18 | 16 | 0 | 2 |
| aggregation | 18 | 1 | 0 | 17 |
| sync | 18 | 0 | 0 | 18 |

Components that only one seed kept, tested over the repeated ablation passes (full minus without,
test points, 95 % t interval):

| Dataset | α | Seed | Component | Passes | Mean | 95 % interval |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 17 | sync | 3 | -3.92 | [-8.99, +1.16] |
| Fashion-MNIST | 0.1 | 2026 | HPO | 3 | +2.82 | [+1.82, +3.82] |

## Limitations

* **Target accuracy.** Every run's goal was 99 % so that the supervisor would never stop
  early and every step would be measured. No run reached it (best final test 90.85 %);
  the target is a stopping rule, not a claim. Results are reported as gains over the baseline at
  an equal training budget.
* **Run-to-run noise.** Training on the real cluster is not deterministic. At Fashion-MNIST α 0.1 the
  same configuration and seed vary by up to about 9 points, and the spread between seeds is larger
  still, so that setting is reported over repeats and its per-step effects are wide.
* **Small effects.** HPO, aggregation and synchronization change accuracy by about ±1 point on
  average, inside the noise band; only NAS and the topology change it clearly.
* **Communication.** The selected mesh sends 7 times as many models per round as the directed ring.
  The in-degree study separates graph connectivity from the number of models received.
* **Learned topology (APG).** APG halves the mesh's communication but loses accuracy, most at α 0.1;
  it optimizes each worker's local fit and waiting time, not global accuracy.
* **Scope.** One cluster of 8 Jetson workers on Wi-Fi, two 28 × 28 grayscale datasets and three
  label-skew levels. CIFAR-10 was run with the non-fair (search-budget) supervisor, not in this
  equal-budget mode, so it is not part of these tables.

## Per-seed accuracy after each step (validation %)

### Fashion-MNIST, α 0.1

| Seed | Order of steps (validation % → kept?) | Final test |
|---|---|---|
| 17 | Baseline 43.29 → NAS 53.71 → Topology sweep 81.55 → Aggregation sweep 81.57 → HPO 77.03 (reverted) → Synchronization sweep 82.83 → Learned topology (APG) 59.56 | 82.40 |
| 42 | Baseline 45.32 → NAS 46.72 → Aggregation sweep 50.46 → Synchronization sweep 40.86 (reverted) → HPO 42.96 (reverted) → Topology sweep 61.95 → Learned topology (APG) 57.29 | 61.71 |
| 2026 | Baseline 38.61 → NAS 51.11 → Topology sweep 70.93 → Aggregation sweep 71.07 → HPO 75.22 → Synchronization sweep 75.07 → Learned topology (APG) 64.98 | 75.64 |

### Fashion-MNIST, α 0.5

| Seed | Order of steps (validation % → kept?) | Final test |
|---|---|---|
| 17 | Baseline 80.67 → NAS 86.21 → Topology sweep 91.73 → Aggregation sweep 91.93 → HPO 91.52 → Synchronization sweep 91.48 → Learned topology (APG) 89.30 | 90.65 |
| 42 | Baseline 76.72 → NAS 78.62 → Topology sweep 88.52 → Aggregation sweep 88.53 → HPO 87.52 (reverted) → Synchronization sweep 86.65 (reverted) → Learned topology (APG) 86.19 | 88.21 |
| 2026 | Baseline 78.99 → NAS 83.54 → Topology sweep 90.28 → Aggregation sweep 90.62 → HPO 89.48 (reverted) → Synchronization sweep 90.70 → Learned topology (APG) 89.03 | 90.14 |

### Fashion-MNIST, α 0.9

| Seed | Order of steps (validation % → kept?) | Final test |
|---|---|---|
| 17 | Baseline 84.90 → NAS 88.34 → Topology sweep 91.80 → Aggregation sweep 91.53 → HPO 91.23 (reverted) → Synchronization sweep 91.98 → Learned topology (APG) 90.82 | 90.85 |
| 42 | Baseline 83.34 → NAS 83.62 → Topology sweep 89.03 → Aggregation sweep 88.27 (reverted) → HPO 88.45 (reverted) → Synchronization sweep 88.68 → Learned topology (APG) 87.80 | 89.05 |
| 2026 | Baseline 83.47 → NAS 86.46 → Topology sweep 90.45 → Aggregation sweep 90.25 → HPO 89.98 → Synchronization sweep 90.20 → Learned topology (APG) 89.44 | 90.04 |

### FEMNIST, α 0.1

| Seed | Order of steps (validation % → kept?) | Final test |
|---|---|---|
| 17 | Baseline 28.65 → NAS 43.66 → Topology sweep 70.56 → Aggregation sweep 73.30 → HPO 72.95 → Synchronization sweep 74.73 → Learned topology (APG) 56.86 | 74.20 |
| 42 | Baseline 35.20 → NAS 36.55 → Topology sweep 64.01 → Aggregation sweep 65.33 → HPO 63.78 (reverted) → Synchronization sweep 64.43 (reverted) → Learned topology (APG) 55.33 | 65.56 |
| 2026 | Baseline 36.47 → NAS 40.42 → Topology sweep 71.01 → Aggregation sweep 72.41 → HPO 71.76 (reverted) → Synchronization sweep 71.91 (reverted) → Learned topology (APG) 54.22 | 71.97 |

### FEMNIST, α 0.5

| Seed | Order of steps (validation % → kept?) | Final test |
|---|---|---|
| 17 | Baseline 67.54 → NAS 74.70 → Topology sweep 83.06 → Aggregation sweep 83.06 → HPO 83.28 → Synchronization sweep 83.83 → Learned topology (APG) 81.16 | 83.90 |
| 42 | Baseline 67.44 → NAS 71.03 → Topology sweep 82.41 → Aggregation sweep 82.71 → HPO 83.40 → Synchronization sweep 82.61 (reverted) → Learned topology (APG) 79.90 | 82.84 |
| 2026 | Baseline 65.78 → NAS 71.38 → Aggregation sweep 70.71 (reverted) → Synchronization sweep 70.91 → HPO 70.32 (reverted) → Topology sweep 82.10 → Learned topology (APG) 77.72 | 81.45 |

### FEMNIST, α 0.9

| Seed | Order of steps (validation % → kept?) | Final test |
|---|---|---|
| 17 | Baseline 74.34 → NAS 77.15 → Aggregation sweep 79.20 → Synchronization sweep 79.38 → Topology sweep 84.28 → HPO 84.43 → Learned topology (APG) 83.47 | 84.68 |
| 42 | Baseline 71.28 → NAS 76.91 → Topology sweep 83.46 → Aggregation sweep 84.13 → HPO 83.31 (reverted) → Synchronization sweep 83.70 → Learned topology (APG) 83.12 | 83.75 |
| 2026 | Baseline 70.36 → NAS 75.58 → Aggregation sweep 75.29 → Synchronization sweep 75.63 → HPO 75.50 → Topology sweep 82.78 → Learned topology (APG) 80.21 | 82.43 |

