# Leave-one-out ablation of the fair campaign

Generated 2026-11-24 01:23 UTC by `scripts/ablation_campaign.py`.

For every finished fair run, the final configuration (after keep-or-revert) is retrained
with the baseline's budget (10 rounds x 5 local epochs, same seed and initial weights), once
as it is and once without each component the run kept. A component's contribution is the
test accuracy of the full configuration minus that of the configuration without it, so it
does not depend on the order in which the SLM ran the steps. Values are mean ± std over the
seeds that kept the component (n). Accuracy is the mean over the 8 workers on the test split.

## Contribution of each component (test accuracy points, full minus without)

| Dataset | α | Full (%) | Nas (n) | Hpo (n) | Topology (n) | Aggregation (n) | Sync (n) |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 71.41 ± 14.45 | +10.27 ± 15.94 (3) | +3.12 (1) | +21.57 ± 13.53 (3) | – (0) | -1.82 (1) |
| Fashion-MNIST | 0.5 | 89.21 ± 2.23 | +1.98 ± 1.69 (3) | -0.34 (1) | +6.38 ± 0.85 (3) | -0.29 ± 0.98 (2) | +0.30 ± 0.29 (2) |
| Fashion-MNIST | 0.9 | 89.72 ± 0.98 | +1.27 ± 1.34 (2) | -0.11 (1) | +4.23 ± 1.21 (3) | +0.14 (1) | – (0) |
| FEMNIST | 0.1 | 70.18 ± 4.20 | +6.89 ± 5.40 (3) | – (0) | +29.52 ± 0.23 (3) | +2.55 ± 1.22 (2) | – (0) |
| FEMNIST | 0.5 | 82.60 ± 1.50 | +3.31 ± 0.65 (3) | – (0) | +10.96 ± 2.10 (3) | +0.20 ± 0.23 (2) | +0.07 ± 0.14 (2) |
| FEMNIST | 0.9 | 83.69 ± 1.10 | +2.71 ± 2.06 (3) | +0.43 (1) | +6.88 ± 1.17 (3) | +0.11 ± 0.06 (2) | +0.00 ± 0.13 (3) |

## Repeatability

The baseline and the full configuration are each trained twice: once in the campaign and
once here, with the same seed. The difference is run-to-run noise on the real cluster
(thread scheduling, network timing and semi-synchronous deadlines are not deterministic).

| Dataset | α | Seed | Baseline test: campaign / repeat | Full test: campaign / repeat |
|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 42 | 44.94 / 46.85 | 61.71 / 55.39 |
| Fashion-MNIST | 0.1 | 2026 | 38.58 / 52.88 | 75.64 / 75.39 |
| Fashion-MNIST | 0.1 | 17 | 43.13 / 43.11 | 82.40 / 83.45 |
| Fashion-MNIST | 0.5 | 17 | 79.69 / 80.08 | 90.65 / 90.62 |
| Fashion-MNIST | 0.5 | 42 | 76.62 / 72.72 | 88.21 / 86.64 |
| Fashion-MNIST | 0.5 | 2026 | 78.93 / 79.94 | 90.14 / 90.38 |
| Fashion-MNIST | 0.9 | 17 | 84.19 / 83.34 | 90.85 / 90.65 |
| Fashion-MNIST | 0.9 | 42 | 83.27 / 82.03 | 89.05 / 88.69 |
| Fashion-MNIST | 0.9 | 2026 | 83.38 / 86.67 | 90.04 / 89.81 |
| FEMNIST | 0.1 | 17 | 28.39 / 29.17 | 74.20 / 73.05 |
| FEMNIST | 0.1 | 42 | 35.18 / 33.64 | 65.56 / 65.36 |
| FEMNIST | 0.1 | 2026 | 36.58 / 36.31 | 71.97 / 72.15 |
| FEMNIST | 0.5 | 17 | 67.30 / 64.29 | 83.90 / 83.60 |
| FEMNIST | 0.5 | 42 | 67.37 / 65.02 | 82.84 / 83.33 |
| FEMNIST | 0.5 | 2026 | 65.52 / 65.10 | 81.45 / 80.88 |
| FEMNIST | 0.9 | 17 | 74.11 / 72.95 | 84.68 / 84.82 |
| FEMNIST | 0.9 | 42 | 71.08 / 71.01 | 83.75 / 83.62 |
| FEMNIST | 0.9 | 2026 | 70.14 / 71.65 | 82.43 / 82.63 |

## Per run

| Dataset | α | Seed | Final configuration | Baseline | Full | without NAS | without HPO | without topology | without aggregation | without sync |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 42 | nas-d2-w64x64-k5-conv-h64-drop35, lr 0.001, batch 32, mesh, fedavg, semi_sync | 46.85 | 55.39 | 59.87 | – | 49.36 | – | – |
| Fashion-MNIST | 0.1 | 2026 | hpo-adam-lr3e-4-bs64, lr 0.0003, batch 64, mesh, fedavg, semi_sync | 52.88 | 75.39 | 67.27 | 72.27 | 47.44 | – | – |
| Fashion-MNIST | 0.1 | 17 | nas-d3-w16x64x64-k5-conv-h0-drop35, lr 0.001, batch 32, mesh, fedavg, sync | 43.11 | 83.45 | 56.27 | – | 52.72 | – | 85.27 |
| Fashion-MNIST | 0.5 | 17 | hpo-adam-lr1e-3-bs64, lr 0.001, batch 64, mesh, fedavg, sync | 80.08 | 90.62 | 86.98 | 90.96 | 85.15 | – | 90.52 |
| Fashion-MNIST | 0.5 | 42 | nas-d2-w64x64-k5-conv-h64-drop35, lr 0.001, batch 32, mesh, uniform, semi_sync | 72.72 | 86.64 | 86.38 | – | 79.47 | 87.62 | – |
| Fashion-MNIST | 0.5 | 2026 | nas-d3-w16x32x64-k5-conv-h64-drop0, lr 0.001, batch 32, mesh, trimmed_mean, sync | 79.94 | 90.38 | 88.35 | – | 83.88 | 89.97 | 89.87 |
| Fashion-MNIST | 0.9 | 17 | nas-d3-w16x64x64-k5-conv-h0-drop0, lr 0.001, batch 32, mesh, metropolis, semi_sync | 83.34 | 90.65 | 88.43 | – | 87.45 | 90.51 | – |
| Fashion-MNIST | 0.9 | 42 | seed-baseline, lr 0.001, batch 32, mesh, fedavg, semi_sync | 82.03 | 88.69 | – | – | 83.13 | – | – |
| Fashion-MNIST | 0.9 | 2026 | hpo-adam-lr1e-3-bs64, lr 0.001, batch 64, mesh, fedavg, semi_sync | 86.67 | 89.81 | 89.48 | 89.92 | 85.89 | – | – |
| FEMNIST | 0.1 | 17 | hpo-adam-lr1e-3-bs32, lr 0.001, batch 32, mesh, uniform, semi_sync | 29.17 | 73.05 | 60.09 | – | 43.34 | 69.64 | – |
| FEMNIST | 0.1 | 42 | nas-d2-w64x64-k5-conv-h64-drop35, lr 0.001, batch 32, mesh, fedavg, semi_sync | 33.64 | 65.36 | 62.75 | – | 35.77 | – | – |
| FEMNIST | 0.1 | 2026 | nas-d3-w16x32x64-k5-conv-h64-drop0, lr 0.001, batch 32, mesh, uniform, semi_sync | 36.31 | 72.15 | 67.06 | – | 42.88 | 70.46 | – |
| FEMNIST | 0.5 | 17 | hpo-adam-lr1e-3-bs32, lr 0.001, batch 32, mesh, uniform, sync | 64.29 | 83.60 | 80.44 | – | 74.95 | 83.57 | 83.43 |
| FEMNIST | 0.5 | 42 | hpo-adam-lr1e-3-bs32, lr 0.001, batch 32, mesh, trimmed_mean, semi_sync | 65.02 | 83.33 | 80.58 | – | 70.58 | 82.97 | – |
| FEMNIST | 0.5 | 2026 | nas-d3-w16x32x64-k5-conv-h64-drop0, lr 0.001, batch 32, mesh, fedavg, sync | 65.10 | 80.88 | 76.86 | – | 69.41 | – | 80.91 |
| FEMNIST | 0.9 | 17 | hpo-adam-lr3e-3-bs32, lr 0.003, batch 32, mesh, uniform, sync | 72.95 | 84.82 | 82.77 | 84.39 | 79.29 | 84.76 | 84.90 |
| FEMNIST | 0.9 | 42 | nas-d2-w64x64-k5-conv-h64-drop35, lr 0.001, batch 32, mesh, trimmed_mean, sync | 71.01 | 83.62 | 82.56 | – | 76.09 | 83.47 | 83.69 |
| FEMNIST | 0.9 | 2026 | hpo-adam-lr1e-3-bs32, lr 0.001, batch 32, mesh, fedavg, sync | 71.65 | 82.63 | 77.62 | – | 75.05 | – | 82.48 |
