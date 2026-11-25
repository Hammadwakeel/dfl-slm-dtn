# Fair evaluation across four datasets

Generated 2026-11-25 07:32 UTC by `scripts/four_dataset_table.py`.

Eight Jetson workers, decentralized training, Dirichlet label skew α (smaller = more skewed).
The on-device SLM chooses the order of the search steps; every model it compares, the baseline
included, is trained with the same budget (10 rounds × 5 local epochs, same initial weights).
Test accuracy is the mean over the 8 workers on the dataset's official test split, mean ± std
over seeds (n = number of seeds). Gain = final model minus baseline, per seed.

## Baseline vs. final model

| Dataset | α | n | Baseline test (%) | Final test (%) | Gain (points) | Selected topology | Final params |
|---|---|---|---|---|---|---|---|
| MNIST | 0.1 | 0 | pending | pending | pending | – | – |
| MNIST | 0.5 | 0 | pending | pending | pending | – | – |
| MNIST | 0.9 | 0 | pending | pending | pending | – | – |
| Fashion-MNIST | 0.1 | 3 | 42.22 ± 3.28 | 73.25 ± 10.55 | +31.03 ± 12.40 | mesh 3/3 | 97k (mean) |
| Fashion-MNIST | 0.5 | 3 | 78.41 ± 1.60 | 89.67 ± 1.29 | +11.25 ± 0.32 | mesh 3/3 | 103k (mean) |
| Fashion-MNIST | 0.9 | 3 | 83.61 ± 0.50 | 89.98 ± 0.90 | +6.37 ± 0.51 | mesh 3/3 | 60k (mean) |
| FEMNIST | 0.1 | 3 | 33.38 ± 4.38 | 70.57 ± 4.49 | +37.19 ± 7.87 | mesh 3/3 | 106k (mean) |
| FEMNIST | 0.5 | 3 | 66.73 ± 1.05 | 82.73 ± 1.23 | +16.00 ± 0.57 | mesh 3/3 | 106k (mean) |
| FEMNIST | 0.9 | 3 | 71.78 ± 2.08 | 83.62 ± 1.13 | +11.84 ± 1.12 | mesh 3/3 | 100k (mean) |
| CIFAR-10 | 0.1 | 0 | pending | pending | pending | – | – |
| CIFAR-10 | 0.5 | 0 | pending | pending | pending | – | – |
| CIFAR-10 | 0.9 | 0 | pending | pending | pending | – | – |

## Effect of each step (test points)

Change in test accuracy caused by each step, against the configuration active just before it
(the baseline or the last kept change); kept = seeds where the change was kept. The learned
topology (APG) is measured last, without a keep/revert decision.

| Dataset | α | NAS | Topology sweep | Aggregation sweep | HPO | Synchronization sweep | Learned topology (APG) |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | +8.03 ± 5.73 (3/3 kept) | +19.57 ± 7.84 (3/3 kept) | +1.21 ± 2.05 (3/3 kept) | -2.48 ± 6.47 (1/3 kept) | -2.79 ± 5.71 (2/3 kept) | -12.43 ± 9.74 |
| Fashion-MNIST | 0.5 | +3.86 ± 2.08 (3/3 kept) | +7.02 ± 2.09 (3/3 kept) | +0.44 ± 0.37 (3/3 kept) | -0.80 ± 0.74 (1/3 kept) | -0.59 ± 0.99 (2/3 kept) | -2.05 ± 0.20 |
| Fashion-MNIST | 0.9 | +2.00 ± 1.61 (3/3 kept) | +4.34 ± 1.14 (3/3 kept) | -0.52 ± 0.61 (2/3 kept) | -0.27 ± 0.43 (1/3 kept) | -0.08 ± 0.71 (3/3 kept) | -0.82 ± 0.49 |
| FEMNIST | 0.1 | +6.66 ± 7.38 (3/3 kept) | +28.74 ± 2.03 (3/3 kept) | +1.46 ± 0.88 (3/3 kept) | -1.03 ± 0.85 (1/3 kept) | -0.07 ± 1.13 (1/3 kept) | -15.26 ± 4.40 |
| FEMNIST | 0.5 | +5.26 ± 1.99 (3/3 kept) | +10.32 ± 1.68 (3/3 kept) | +0.17 ± 0.52 (2/3 kept) | -0.36 ± 0.50 (2/3 kept) | +0.10 ± 0.38 (2/3 kept) | -3.15 ± 0.89 |
| FEMNIST | 0.9 | +4.63 ± 1.56 (3/3 kept) | +6.35 ± 1.17 (3/3 kept) | +0.65 ± 1.33 (3/3 kept) | -0.31 ± 0.19 (2/3 kept) | +0.35 ± 0.38 (3/3 kept) | -1.62 ± 0.86 |

## Per seed (test %)

| Dataset | α | Seed | Baseline | Final | Final configuration |
|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 17 | 43.13 | 82.40 | nas-d3-w16x64x64-k5-conv-h0-drop35 on mesh |
| Fashion-MNIST | 0.1 | 42 | 44.94 | 61.71 | nas-d2-w64x64-k5-conv-h64-drop35 on mesh |
| Fashion-MNIST | 0.1 | 2026 | 38.58 | 75.64 | hpo-adam-lr3e-4-bs64 on mesh |
| Fashion-MNIST | 0.5 | 17 | 79.69 | 90.65 | nas-d3-w16x64x64-k5-conv-h0-drop35 on mesh |
| Fashion-MNIST | 0.5 | 42 | 76.62 | 88.21 | nas-d2-w64x64-k5-conv-h64-drop35 on mesh |
| Fashion-MNIST | 0.5 | 2026 | 78.93 | 90.14 | nas-d3-w16x32x64-k5-conv-h64-drop0 on mesh |
| Fashion-MNIST | 0.9 | 17 | 84.19 | 90.85 | nas-d3-w16x64x64-k5-conv-h0-drop0 on mesh |
| Fashion-MNIST | 0.9 | 42 | 83.27 | 89.05 | seed-baseline on mesh |
| Fashion-MNIST | 0.9 | 2026 | 83.38 | 90.04 | nas-d3-w16x16x32-k5-conv-h64-drop20 on mesh |
| FEMNIST | 0.1 | 17 | 28.39 | 74.20 | hpo-adam-lr1e-3-bs32 on mesh |
| FEMNIST | 0.1 | 42 | 35.18 | 65.56 | nas-d2-w64x64-k5-conv-h64-drop35 on mesh |
| FEMNIST | 0.1 | 2026 | 36.58 | 71.97 | nas-d3-w16x32x64-k5-conv-h64-drop0 on mesh |
| FEMNIST | 0.5 | 17 | 67.30 | 83.90 | hpo-adam-lr1e-3-bs32 on mesh |
| FEMNIST | 0.5 | 42 | 67.37 | 82.84 | hpo-adam-lr1e-3-bs32 on mesh |
| FEMNIST | 0.5 | 2026 | 65.52 | 81.45 | nas-d3-w16x32x64-k5-conv-h64-drop0 on mesh |
| FEMNIST | 0.9 | 17 | 74.11 | 84.68 | hpo-adam-lr3e-3-bs32 on mesh |
| FEMNIST | 0.9 | 42 | 71.08 | 83.75 | nas-d2-w64x64-k5-conv-h64-drop35 on mesh |
| FEMNIST | 0.9 | 2026 | 70.14 | 82.43 | hpo-adam-lr1e-3-bs32 on mesh |
