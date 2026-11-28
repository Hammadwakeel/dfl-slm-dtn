# Fair evaluation across four datasets

Generated 2026-11-28 01:10 UTC by `scripts/four_dataset_table.py`.

Eight Jetson workers, decentralized training, Dirichlet label skew α (smaller = more skewed).
The on-device SLM chooses the order of the search steps; every model it compares, the baseline
included, is trained with the same budget (10 rounds × 5 local epochs, same initial weights).
Test accuracy is the mean over the 8 workers on the dataset's official test split, mean ± std
over seeds (n = number of seeds). Gain = final model minus baseline, per seed.

## Baseline vs. final model

| Dataset | α | n | Baseline test (%) | Final test (%) | Gain (points) | Selected topology | Final params |
|---|---|---|---|---|---|---|---|
| MNIST | 0.1 | 3 | 56.33 ± 7.27 | 96.83 ± 1.33 | +40.50 ± 6.48 | mesh 3/3 | 62k (mean) |
| MNIST | 0.5 | 3 | 96.96 ± 0.98 | 98.87 ± 0.46 | +1.91 ± 1.36 | learned 2/3, mesh 1/3 | 58k (mean) |
| MNIST | 0.9 | 3 | 98.09 ± 0.66 | 98.89 ± 0.36 | +0.79 ± 0.90 | learned 2/3, directed_ring 1/3 | 36k (mean) |
| Fashion-MNIST | 0.1 | 5 | 44.26 ± 3.69 | 71.60 ± 8.55 | +27.33 ± 10.52 | mesh 4/5, learned 1/5 | 67k (mean) |
| Fashion-MNIST | 0.5 | 3 | 78.41 ± 1.60 | 89.67 ± 1.29 | +11.25 ± 0.32 | mesh 3/3 | 103k (mean) |
| Fashion-MNIST | 0.9 | 3 | 83.56 ± 0.54 | 90.05 ± 0.92 | +6.49 ± 0.63 | mesh 3/3 | 60k (mean) |
| FEMNIST | 0.1 | 3 | 33.38 ± 4.38 | 70.57 ± 4.49 | +37.19 ± 7.87 | mesh 3/3 | 106k (mean) |
| FEMNIST | 0.5 | 3 | 66.73 ± 1.05 | 82.73 ± 1.23 | +16.00 ± 0.57 | mesh 3/3 | 106k (mean) |
| FEMNIST | 0.9 | 3 | 71.78 ± 2.08 | 83.62 ± 1.13 | +11.84 ± 1.12 | mesh 3/3 | 100k (mean) |
| CIFAR-10 | 0.1 | 3 | 35.36 ± 8.31 | 61.42 ± 4.85 | +26.06 ± 3.46 | mesh 3/3 | 194k (mean) |
| CIFAR-10 | 0.5 | 3 | 59.57 ± 1.71 | 80.11 ± 2.30 | +20.54 ± 0.63 | mesh 3/3 | 273k (mean) |
| CIFAR-10 | 0.9 | 3 | 68.90 ± 3.15 | 82.44 ± 1.20 | +13.55 ± 4.29 | mesh 3/3 | 240k (mean) |

## Effect of each step (test points)

Change in test accuracy caused by each step, against the configuration active just before it
(the baseline or the last kept change); kept = seeds where the change was kept. The learned
topology (APG) is measured last, without a keep/revert decision.

| Dataset | α | NAS | Topology sweep | Aggregation sweep | HPO | Synchronization sweep | Learned topology (APG) |
|---|---|---|---|---|---|---|---|
| MNIST | 0.1 | +4.33 ± 2.40 (3/3 kept) | +35.66 ± 3.85 (3/3 kept) | -2.46 ± 5.41 (2/3 kept) | -0.73 ± 0.62 (1/3 kept) | -0.18 ± 0.09 (2/3 kept) | -17.44 ± 9.66 |
| MNIST | 0.5 | +0.56 ± 0.86 (3/3 kept) | +0.69 ± 0.98 (3/3 kept) | -0.19 ± 0.16 (2/2 kept) | -0.14 ± 0.26 (2/2 kept) | +0.12 ± 0.15 (2/2 kept) | +1.11 ± 0.34 |
| MNIST | 0.9 | +0.27 ± 0.41 (3/3 kept) | -0.01 ± 0.05 (3/3 kept) | +0.22 ± 0.25 (3/3 kept) | -0.30 ± 0.19 (3/3 kept) | -0.17 ± 0.27 (2/3 kept) | +0.41 ± 0.30 |
| Fashion-MNIST | 0.1 | +6.25 ± 5.02 (5/5 kept) | +15.76 ± 10.32 (5/5 kept) | +1.56 ± 1.81 (5/5 kept) | -1.42 ± 4.88 (2/5 kept) | -1.82 ± 4.25 (4/5 kept) | -7.98 ± 10.85 |
| Fashion-MNIST | 0.5 | +3.86 ± 2.08 (3/3 kept) | +7.02 ± 2.09 (3/3 kept) | +0.44 ± 0.37 (3/3 kept) | -0.80 ± 0.74 (1/3 kept) | -0.59 ± 0.99 (2/3 kept) | -2.05 ± 0.20 |
| Fashion-MNIST | 0.9 | +1.94 ± 1.58 (3/3 kept) | +4.50 ± 1.12 (3/3 kept) | -0.53 ± 0.60 (2/3 kept) | -0.16 ± 0.38 (1/3 kept) | -0.03 ± 0.73 (3/3 kept) | -1.12 ± 0.93 |
| FEMNIST | 0.1 | +6.66 ± 7.38 (3/3 kept) | +28.74 ± 2.03 (3/3 kept) | +1.46 ± 0.88 (3/3 kept) | -1.03 ± 0.85 (1/3 kept) | -0.07 ± 1.13 (1/3 kept) | -15.26 ± 4.40 |
| FEMNIST | 0.5 | +5.26 ± 1.99 (3/3 kept) | +10.32 ± 1.68 (3/3 kept) | +0.17 ± 0.52 (2/3 kept) | -0.36 ± 0.50 (2/3 kept) | +0.10 ± 0.38 (2/3 kept) | -3.15 ± 0.89 |
| FEMNIST | 0.9 | +4.63 ± 1.56 (3/3 kept) | +6.35 ± 1.17 (3/3 kept) | +0.65 ± 1.33 (3/3 kept) | -0.31 ± 0.19 (2/3 kept) | +0.35 ± 0.38 (3/3 kept) | -1.62 ± 0.86 |
| CIFAR-10 | 0.1 | +2.17 ± 6.52 (2/3 kept) | +19.28 ± 1.14 (3/3 kept) | +0.61 ± 0.85 (3/3 kept) | +0.70 ± 2.53 (2/3 kept) | -0.40 ± 1.55 (2/3 kept) | -19.64 ± 2.41 |
| CIFAR-10 | 0.5 | +2.36 ± 1.81 (3/3 kept) | +17.56 ± 2.32 (3/3 kept) | +0.19 ± 0.93 (2/3 kept) | -2.28 ± 3.44 (1/3 kept) | -0.10 ± 0.77 (1/3 kept) | -6.46 ± 3.00 |
| CIFAR-10 | 0.9 | +1.48 ± 3.87 (2/3 kept) | +10.92 ± 2.07 (3/3 kept) | -0.52 ± 0.54 (1/3 kept) | -1.44 ± 2.45 (1/3 kept) | -0.39 ± 0.20 (0/3 kept) | -2.12 ± 0.55 |

## Per seed (test %)

| Dataset | α | Seed | Baseline | Final | Final configuration |
|---|---|---|---|---|---|
| MNIST | 0.1 | 17 | 49.05 | 96.54 | nas-d3-w16x64x64-k5-conv-h0-drop35 on mesh |
| MNIST | 0.1 | 42 | 56.35 | 95.67 | seed-baseline on mesh |
| MNIST | 0.1 | 2026 | 63.59 | 98.29 | nas-d3-w16x16x32-k5-conv-h64-drop20 on mesh |
| MNIST | 0.5 | 17 | 95.84 | 99.20 | nas-d3-w16x64x64-k5-conv-h0-drop35 on mesh |
| MNIST | 0.5 | 42 | 97.69 | 98.34 | hpo-adam-lr3e-4-bs32 on learned |
| MNIST | 0.5 | 2026 | 97.35 | 99.06 | hpo-adam-lr3e-3-bs32-pruned30 on learned |
| MNIST | 0.9 | 17 | 97.33 | 99.10 | hpo-adam-lr3e-4-bs32-pruned30 on learned |
| MNIST | 0.9 | 42 | 98.49 | 98.47 | seed-baseline on directed_ring |
| MNIST | 0.9 | 2026 | 98.46 | 99.09 | hpo-adam-lr1e-3-bs64-pruned30 on learned |
| Fashion-MNIST | 0.1 | 7 | 48.26 | 74.09 | hpo-adam-lr3e-4-bs32 on mesh |
| Fashion-MNIST | 0.1 | 17 | 43.13 | 82.40 | nas-d3-w16x64x64-k5-conv-h0-drop35 on mesh |
| Fashion-MNIST | 0.1 | 42 | 44.94 | 61.71 | nas-d2-w64x64-k5-conv-h64-drop35 on mesh |
| Fashion-MNIST | 0.1 | 99 | 46.40 | 64.14 | nas-d3-w16x32x32-k3-conv-h64-drop20-pruned30 on learned |
| Fashion-MNIST | 0.1 | 2026 | 38.58 | 75.64 | hpo-adam-lr3e-4-bs64 on mesh |
| Fashion-MNIST | 0.5 | 17 | 79.69 | 90.65 | nas-d3-w16x64x64-k5-conv-h0-drop35 on mesh |
| Fashion-MNIST | 0.5 | 42 | 76.62 | 88.21 | nas-d2-w64x64-k5-conv-h64-drop35 on mesh |
| Fashion-MNIST | 0.5 | 2026 | 78.93 | 90.14 | nas-d3-w16x32x64-k5-conv-h64-drop0 on mesh |
| Fashion-MNIST | 0.9 | 17 | 84.19 | 90.85 | nas-d3-w16x64x64-k5-conv-h0-drop0 on mesh |
| Fashion-MNIST | 0.9 | 42 | 83.27 | 89.05 | seed-baseline on mesh |
| Fashion-MNIST | 0.9 | 2026 | 83.24 | 90.25 | hpo-adam-lr3e-3-bs64 on mesh |
| FEMNIST | 0.1 | 17 | 28.39 | 74.20 | hpo-adam-lr1e-3-bs32 on mesh |
| FEMNIST | 0.1 | 42 | 35.18 | 65.56 | nas-d2-w64x64-k5-conv-h64-drop35 on mesh |
| FEMNIST | 0.1 | 2026 | 36.58 | 71.97 | nas-d3-w16x32x64-k5-conv-h64-drop0 on mesh |
| FEMNIST | 0.5 | 17 | 67.30 | 83.90 | hpo-adam-lr1e-3-bs32 on mesh |
| FEMNIST | 0.5 | 42 | 67.37 | 82.84 | hpo-adam-lr1e-3-bs32 on mesh |
| FEMNIST | 0.5 | 2026 | 65.52 | 81.45 | nas-d3-w16x32x64-k5-conv-h64-drop0 on mesh |
| FEMNIST | 0.9 | 17 | 74.11 | 84.68 | hpo-adam-lr3e-3-bs32 on mesh |
| FEMNIST | 0.9 | 42 | 71.08 | 83.75 | nas-d2-w64x64-k5-conv-h64-drop35 on mesh |
| FEMNIST | 0.9 | 2026 | 70.14 | 82.43 | hpo-adam-lr1e-3-bs32 on mesh |
| CIFAR-10 | 0.1 | 17 | 42.38 | 65.44 | hpo-adam-lr1e-3-bs8-pruned30 on mesh |
| CIFAR-10 | 0.1 | 42 | 26.19 | 56.03 | nas-d3-w32x64x128-k3-conv-h0-drop35 on mesh |
| CIFAR-10 | 0.1 | 2026 | 37.50 | 62.79 | hpo-adam-lr3e-4-bs8 on mesh |
| CIFAR-10 | 0.5 | 17 | 58.62 | 79.08 | hpo-adam-lr3e-4-bs8 on mesh |
| CIFAR-10 | 0.5 | 42 | 61.55 | 82.75 | nas-d3-w32x64x128-k3-conv-h64-drop0-c2 on mesh |
| CIFAR-10 | 0.5 | 2026 | 58.55 | 78.50 | nas-d2-w32x128-k3-conv-h0-drop35-c2 on mesh |
| CIFAR-10 | 0.9 | 17 | 71.31 | 81.98 | baseline on mesh |
| CIFAR-10 | 0.9 | 42 | 65.34 | 83.81 | hpo-adam-lr1e-3-bs32 on mesh |
| CIFAR-10 | 0.9 | 2026 | 70.05 | 81.54 | nas-d2-w32x128-k3-conv-h0-drop35-c2-pruned30 on mesh |
