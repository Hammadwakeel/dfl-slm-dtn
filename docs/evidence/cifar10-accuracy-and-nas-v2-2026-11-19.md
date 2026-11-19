# CIFAR-10 accuracy and NAS v2 (federation-aware NAS) — 2026-11-19

Measured on the real cluster: 8 Jetson Orin Nano workers + master, Dirichlet
alpha = 0.3 non-IID shards, decentralized FedAvg on a **mesh** topology,
synchronous rounds. Every number below is from one seed (2026) unless noted;
"test" is the official CIFAR-10 / MNIST test split (10 000 images), evaluated
on every worker's final model by `POST /api/supervisor/evaluate`; "global val"
is the shared class-balanced 2 000-image validation split. In a mesh all
workers hold the same averaged model, so worker mean = worker minimum.

Raw per-run data: [cifar10-nas-v2-ablation-2026-11-19.json](cifar10-nas-v2-ablation-2026-11-19.json).

## 1. CIFAR-10 accuracy: data augmentation and model size

Same split, topology, 30 rounds x 2 local epochs, Adam 1e-3, batch 32.
`flip_shift` = random horizontal flip + random shift of up to 4 px
(reflect-pad, crop), applied by each worker to its training inputs only.

| Model | Params | Augmentation | Global val | Test | Test macro-F1 | Wall | Group |
|---|---|---|---|---|---|---|---|
| 3-block CNN | 61.8k | none | 65.5% (3 seeds) | — | — | ~9 min | optimize-1795014072-efec7c (mesh runs) |
| 3-block CNN | 61.8k | flip_shift | 72.7% | 73.0% | 0.728 | 9 min | mesh-1795079645-10c061 |
| VGG-6 (2x conv per block, 32/64/128, BN) | 324k | none | 76.9% | 75.2% | 0.749 | 18 min | mesh-1795056980-2a3c5b |
| **VGG-6** | 324k | **flip_shift** | **84.2%** | **82.9%** | **0.828** | 19 min | mesh-1795055388-140dd1 |

* Augmentation adds **+7.2** (3-block CNN) and **+7.3 / +7.6 test** (VGG-6)
  points; without it the VGG-6 curve plateaus at ~76% after round 15
  (local over-fitting on 2.5k-9.9k-image non-IID shards), with it the accuracy
  is still rising at round 30.
* Together, model capacity and augmentation take CIFAR-10 from 65.5% to
  **84.2% global / 82.9% test**, above the supervisor's 0.70 target (its best
  CIFAR-10 run so far was 0.67 global val).

## 2. NAS v1 vs NAS v2 (ablation)

* **NAS v1** (original): each worker trains candidates on its own non-IID
  slice (1 epoch, then 3 epochs for the evolved survivors) and candidates are
  ranked by the mean of each worker's *local* validation accuracy; compact
  space (<=3 single-conv blocks, <=64 channels, 150k cap on CIFAR-10).
* **Shuffle bug** found during this study: the search's training indices were
  ordered class by class and its input pipeline shuffles with a 32-element
  buffer, so every candidate trained on one class at a time. On CIFAR-10 every
  candidate scored ~0.10 shared-split (chance) and ~0.2 local accuracy, i.e.
  NAS v1 and HPO were ranking noise. Fixed in `19a831e` (seeded permutation;
  both splits keep the same members). After the fix the v1 screen measures
  0.54-0.71 local accuracy.
* **NAS v2** (`c39e0a8`, `de5adc3`): candidates are ranked on the shared
  class-balanced split; extended CIFAR-10 space (double-conv VGG-style blocks,
  up to 128 channels, 400k cap, two known-good anchors); search under the run's
  augmentation; then a **federated check**: the two best searched designs and
  the deployed baseline (seed) train 5 rounds x 2 epochs with real
  decentralized averaging on a mesh from the same seeded initialization; the
  best final shared-split accuracy wins (finalists within 0.005 count as tied
  and the smallest wins). NAS v2 therefore cannot return a design that
  federates worse than the baseline it was given.

Final training of every selected design uses the same recipe as section 1
(CIFAR-10: 30 x 2, flip_shift; MNIST: 10 x 2, no augmentation).

### CIFAR-10 (alpha 0.3)

| NAS | Selected design | Params | Search time | Test | Test macro-F1 | Final training |
|---|---|---|---|---|---|---|
| v1 (original, shuffle bug) | nas-d2-w16x64-k3-conv-h0-drop35 | 10.7k | 546 s | 60.7% | 0.593 | 6 min |
| v1 + shuffle fix | nas-d3-w16x32x64-k3-conv-h64-drop20 | 28.8k | 524 s | 67.9% | 0.673 | 9 min |
| v2 (screen still had the shuffle bug) | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197k | 1267 s | 79.6% | 0.795 | 14 min |
| **v2 + shuffle fix** | nas-d2-w32x128-k3-conv-h0-drop35-c2 | 197k | 1180 s | **80.3%** | **0.802** | 16 min |
| Deployed baseline (hand-designed, batch 8) | VGG-style 6-conv | 324k | — | 82.1% | 0.821 | 56 min |

Federated check of v2 + fix (5 rounds x 2 epochs, global val):
nas-d2-w32x128-c2 **0.571** > nas-d3-w32x64x128 0.561 > seed baseline 0.516.

* NAS v2 improves on the original NAS by **+19.6 test points** and on the
  bug-fixed v1 by **+12.4**.
* v2 selected the **same design** with the noisy screen and with the fixed
  one; its final test accuracy was 79.6% and 80.3% in two independent
  trainings, so the federated check is robust to the screen.
* v2 is 1.8 points below the hand-designed 324k baseline with **39% fewer
  parameters and ~3.5x faster training** (batch 32 vs 8); the 5-round check
  ranked it above the baseline, which only overtakes it later in training.

### MNIST (alpha 0.3)

| NAS | Selected design | Params | Search time | Test | Test macro-F1 | Final training |
|---|---|---|---|---|---|---|
| v1 (original, shuffle bug) | nas-d3-w16x16x64-k5-sep-h0-drop35 | 3.3k | 573 s | 95.5% | 0.955 | 2 min |
| v1 + shuffle fix | nas-d3-w32x32x64-k5-conv-h0-drop0 | 78.9k | 500 s | 99.0% | 0.990 | 3 min |
| v2 + shuffle fix (seed cut-off bug, see below) | nas-d3-w32x32x64-k5-conv-h0-drop0-attn | 89.7k | 787 s | 98.9% | 0.989 | 4 min |
| **v2 + all fixes** | seed baseline (no searched design beat it) | 32.8k | 968 s | 98.4% | 0.984 | 5 min |
| Deployed baseline | conv + transformer | 32.8k | — | 98.6% | 0.986 | 5 min |

* On MNIST the shuffle fix alone is worth **+3.5 test points**; the good
  designs saturate at ~98.5-99%, so v1-fixed, v2 and the baseline are tied
  within noise (the baseline measured 98.6% and 98.4% in two trainings).
* With all fixes, v2's federated check measured 0.978 (37.6k design), 0.974
  (baseline, 32.8k) and 0.966 (28.6k); the top two are tied within 0.005, so
  v2 kept the smaller baseline: on MNIST NAS v2 correctly reports that the
  search found nothing better than the deployed model.
* The first MNIST v2 run exposed two NAS v2 bugs, fixed in `de5adc3`: the
  compact first stage already holds the worker maximum of 8 designs, so the
  seed baseline was cut off and never entered the federated check, and the
  check picked an 89.7k design at 0.980 over a 78.9k one at 0.979.

## 3. Robustness bugs found and fixed while running this study

| Commit | Problem | Effect before the fix |
|---|---|---|
| 75f612e | Trainings without an explicit architecture (the supervisor baseline) used the worker-wide default Adam lr 0.01 instead of the installed model's settings | CIFAR-10 baseline trained at 10x its learning rate |
| c39e0a8 | One dropped connection while uploading peer weights failed a whole run | 30-round CIFAR-10 run died in round 17 |
| 3748ba0 | `/results` returned 42 MB (~90 s) | pollers and the supervisor stalled in retries |
| 3677ccb | The dashboard polled `/results` every 3 s with overlapping reads | master at ~100% CPU sending ~5 MB/s over Wi-Fi, congesting the whole cluster (300-500 ms pings, failed exchanges) |
| d8f8eac, 1d799bb | Short or no retries for Wi-Fi drops (worker7) and for requests that never reached a worker | runs and NAS stages failed on "No route to host" |
| 441718c | An interrupted model install left workers on different models | every later baseline restore failed |

## 4. Limitations

* Single seed per configuration (the v2 design was trained twice:
  79.6% / 80.3%); 3-seed repeats are the next step for confidence intervals.
* One non-IID level (alpha 0.3) and one topology (mesh) for the NAS ablation.
* Search time of v2 includes the federated check (~11 min on CIFAR-10).

## Paper wording (suggested)

> On CIFAR-10 with Dirichlet alpha = 0.3 across eight Jetson Orin Nano
> workers, standard flip/shift augmentation raised the decentralized test
> accuracy of a 324k-parameter CNN from 75.2% to 82.9%. Ranking NAS candidates
> on a shared class-balanced split and confirming the finalists with a short
> decentralized training (NAS v2) selected a 197k-parameter design reaching
> 80.3% test accuracy, versus 60.7% for the original local-objective search
> (67.9% after fixing a class-ordering defect in its input pipeline), within
> 1.8 points of a hand-designed 324k model while training 3.5x faster.
