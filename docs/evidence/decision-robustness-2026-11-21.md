# Robustness of the keep-or-revert decisions

Generated 2026-11-25 15:16 UTC by `scripts/decision_robustness.py`.

The fair campaign keeps a step's change when its measurement is at least the best global
validation accuracy so far minus 0.5 points. σ is the pooled within-seed standard
deviation of global validation accuracy over repeated trainings of the same configuration with
the same seed (campaign baseline and every ablation pass's baseline and full configuration). A
decision compares two measurements, so its noise band is 2·σ·√2 (about 95 %). A decision is
*clear* when the measured difference lies outside that band and *within noise* otherwise.

## Noise and decisions per setting

| Dataset | α | σ (points) | dof | 95 % band (points) | Decisions | Kept | Reverted | Clear gain | Clear loss | Within noise |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 2.34 | 12 | ±6.63 | 25 | 21 | 4 | 6 | 2 | 17 |
| Fashion-MNIST | 0.5 | 1.59 | 3 | ±4.51 | 15 | 12 | 3 | 5 | 0 | 10 |
| Fashion-MNIST | 0.9 | 0.74 | 2 | ±2.09 | 10 | 7 | 3 | 3 | 0 | 7 |
| FEMNIST | 0.1 | 0.78 | 3 | ±2.21 | 15 | 11 | 4 | 6 | 0 | 9 |
| FEMNIST | 0.5 | 1.52 | 3 | ±4.29 | 15 | 12 | 3 | 5 | 0 | 10 |
| FEMNIST | 0.9 | 0.75 | 3 | ±2.13 | 15 | 14 | 1 | 6 | 0 | 9 |
| **All** | | | | | 95 | 77 | 18 | 31 | 2 | 62 |

## Decisions per step

| Step | Decisions | Clear gain | Clear loss | Within noise |
|---|---|---|---|---|
| NAS | 19 | 12 | 0 | 7 |
| HPO | 19 | 0 | 1 | 18 |
| topology | 19 | 18 | 0 | 1 |
| aggregation | 19 | 1 | 0 | 18 |
| sync | 19 | 0 | 1 | 18 |

## Paired test of the components a single seed kept

Each ablation pass trains the full configuration and the configuration without the component
back to back with the same seed and initial weights. The table gives the per-pass differences
(full minus without, test points), their mean and a 95 % t interval when there are two or more
passes. An interval that contains 0 means the component's effect is not distinguishable from noise.

| Dataset | α | Seed | Component | Per-pass differences | Mean | 95 % interval |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 17 | sync | -4.03, -5.90, -1.82 | -3.92 | [-8.99, +1.16] |
| Fashion-MNIST | 0.1 | 2026 | HPO | +2.36, +2.97, +3.12 | +2.82 | [+1.82, +3.82] |
| Fashion-MNIST | 0.5 | 17 | HPO | -0.34 | -0.34 | n = 1 |
| Fashion-MNIST | 0.5 | 17 | sync | +0.10 | +0.10 | n = 1 |
| Fashion-MNIST | 0.5 | 42 | aggregation | -0.98 | -0.98 | n = 1 |
| Fashion-MNIST | 0.5 | 2026 | aggregation | +0.41 | +0.41 | n = 1 |
| Fashion-MNIST | 0.5 | 2026 | sync | +0.51 | +0.51 | n = 1 |
| Fashion-MNIST | 0.9 | 17 | aggregation | +0.14 | +0.14 | n = 1 |
| Fashion-MNIST | 0.9 | 2026 | HPO | -0.11 | -0.11 | n = 1 |
| FEMNIST | 0.1 | 17 | aggregation | +3.41 | +3.41 | n = 1 |
| FEMNIST | 0.1 | 2026 | aggregation | +1.69 | +1.69 | n = 1 |
| FEMNIST | 0.5 | 17 | aggregation | +0.03 | +0.03 | n = 1 |
| FEMNIST | 0.5 | 17 | sync | +0.17 | +0.17 | n = 1 |
| FEMNIST | 0.5 | 42 | aggregation | +0.36 | +0.36 | n = 1 |
| FEMNIST | 0.5 | 2026 | sync | -0.03 | -0.03 | n = 1 |
| FEMNIST | 0.9 | 17 | HPO | +0.43 | +0.43 | n = 1 |
| FEMNIST | 0.9 | 17 | aggregation | +0.06 | +0.06 | n = 1 |
| FEMNIST | 0.9 | 17 | sync | -0.08 | -0.08 | n = 1 |
| FEMNIST | 0.9 | 42 | aggregation | +0.15 | +0.15 | n = 1 |
| FEMNIST | 0.9 | 42 | sync | -0.07 | -0.07 | n = 1 |
| FEMNIST | 0.9 | 2026 | sync | +0.15 | +0.15 | n = 1 |

## Every decision

| Dataset | α | Seed | Step | Measured (%) | Best before (%) | Difference (points) | Kept | Verdict |
|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 7 | NAS | 49.84 | 48.73 | +1.11 | yes | within noise |
| Fashion-MNIST | 0.1 | 7 | topology | 70.03 | 49.84 | +20.19 | yes | clear gain |
| Fashion-MNIST | 0.1 | 7 | aggregation | 73.75 | 70.03 | +3.72 | yes | within noise |
| Fashion-MNIST | 0.1 | 7 | HPO | 74.40 | 73.75 | +0.65 | yes | within noise |
| Fashion-MNIST | 0.1 | 7 | sync | 74.83 | 74.40 | +0.43 | yes | within noise |
| Fashion-MNIST | 0.1 | 17 | NAS | 53.71 | 43.29 | +10.42 | yes | clear gain |
| Fashion-MNIST | 0.1 | 17 | topology | 81.55 | 53.71 | +27.84 | yes | clear gain |
| Fashion-MNIST | 0.1 | 17 | aggregation | 81.57 | 81.55 | +0.02 | yes | within noise |
| Fashion-MNIST | 0.1 | 17 | HPO | 77.03 | 81.57 | -4.53 | no | within noise |
| Fashion-MNIST | 0.1 | 17 | sync | 82.83 | 81.57 | +1.27 | yes | within noise |
| Fashion-MNIST | 0.1 | 42 | NAS | 46.72 | 45.32 | +1.40 | yes | within noise |
| Fashion-MNIST | 0.1 | 42 | aggregation | 50.46 | 46.72 | +3.74 | yes | within noise |
| Fashion-MNIST | 0.1 | 42 | sync | 40.86 | 50.46 | -9.59 | no | clear loss |
| Fashion-MNIST | 0.1 | 42 | HPO | 42.96 | 50.46 | -7.49 | no | clear loss |
| Fashion-MNIST | 0.1 | 42 | topology | 61.95 | 50.46 | +11.49 | yes | clear gain |
| Fashion-MNIST | 0.1 | 99 | NAS | 52.64 | 46.49 | +6.15 | yes | within noise |
| Fashion-MNIST | 0.1 | 99 | aggregation | 53.06 | 52.64 | +0.41 | yes | within noise |
| Fashion-MNIST | 0.1 | 99 | sync | 52.61 | 53.06 | -0.45 | yes | within noise |
| Fashion-MNIST | 0.1 | 99 | HPO | 51.48 | 53.06 | -1.58 | no | within noise |
| Fashion-MNIST | 0.1 | 99 | topology | 52.83 | 53.06 | -0.22 | yes | within noise |
| Fashion-MNIST | 0.1 | 2026 | NAS | 51.11 | 38.61 | +12.50 | yes | clear gain |
| Fashion-MNIST | 0.1 | 2026 | topology | 70.93 | 51.11 | +19.82 | yes | clear gain |
| Fashion-MNIST | 0.1 | 2026 | aggregation | 71.07 | 70.93 | +0.13 | yes | within noise |
| Fashion-MNIST | 0.1 | 2026 | HPO | 75.22 | 71.07 | +4.15 | yes | within noise |
| Fashion-MNIST | 0.1 | 2026 | sync | 75.07 | 75.22 | -0.15 | yes | within noise |
| Fashion-MNIST | 0.5 | 17 | NAS | 86.21 | 80.67 | +5.54 | yes | clear gain |
| Fashion-MNIST | 0.5 | 17 | topology | 91.73 | 86.21 | +5.52 | yes | clear gain |
| Fashion-MNIST | 0.5 | 17 | aggregation | 91.93 | 91.73 | +0.20 | yes | within noise |
| Fashion-MNIST | 0.5 | 17 | HPO | 91.52 | 91.93 | -0.42 | yes | within noise |
| Fashion-MNIST | 0.5 | 17 | sync | 91.48 | 91.93 | -0.45 | yes | within noise |
| Fashion-MNIST | 0.5 | 42 | NAS | 78.62 | 76.72 | +1.91 | yes | within noise |
| Fashion-MNIST | 0.5 | 42 | topology | 88.52 | 78.62 | +9.89 | yes | clear gain |
| Fashion-MNIST | 0.5 | 42 | aggregation | 88.53 | 88.52 | +0.02 | yes | within noise |
| Fashion-MNIST | 0.5 | 42 | HPO | 87.52 | 88.53 | -1.02 | no | within noise |
| Fashion-MNIST | 0.5 | 42 | sync | 86.65 | 88.53 | -1.88 | no | within noise |
| Fashion-MNIST | 0.5 | 2026 | NAS | 83.54 | 78.99 | +4.55 | yes | clear gain |
| Fashion-MNIST | 0.5 | 2026 | topology | 90.28 | 83.54 | +6.74 | yes | clear gain |
| Fashion-MNIST | 0.5 | 2026 | aggregation | 90.62 | 90.28 | +0.33 | yes | within noise |
| Fashion-MNIST | 0.5 | 2026 | HPO | 89.48 | 90.62 | -1.13 | no | within noise |
| Fashion-MNIST | 0.5 | 2026 | sync | 90.70 | 90.62 | +0.08 | yes | within noise |
| Fashion-MNIST | 0.9 | 17 | NAS | 88.34 | 84.90 | +3.43 | yes | clear gain |
| Fashion-MNIST | 0.9 | 17 | topology | 91.80 | 88.34 | +3.46 | yes | clear gain |
| Fashion-MNIST | 0.9 | 17 | aggregation | 91.53 | 91.80 | -0.27 | yes | within noise |
| Fashion-MNIST | 0.9 | 17 | HPO | 91.23 | 91.80 | -0.57 | no | within noise |
| Fashion-MNIST | 0.9 | 17 | sync | 91.98 | 91.80 | +0.18 | yes | within noise |
| Fashion-MNIST | 0.9 | 42 | NAS | 83.62 | 83.34 | +0.28 | yes | within noise |
| Fashion-MNIST | 0.9 | 42 | topology | 89.03 | 83.62 | +5.41 | yes | clear gain |
| Fashion-MNIST | 0.9 | 42 | aggregation | 88.27 | 89.03 | -0.77 | no | within noise |
| Fashion-MNIST | 0.9 | 42 | HPO | 88.45 | 89.03 | -0.58 | no | within noise |
| Fashion-MNIST | 0.9 | 42 | sync | 88.68 | 89.03 | -0.35 | yes | within noise |
| FEMNIST | 0.1 | 17 | NAS | 43.66 | 28.65 | +15.02 | yes | clear gain |
| FEMNIST | 0.1 | 17 | topology | 70.56 | 43.66 | +26.90 | yes | clear gain |
| FEMNIST | 0.1 | 17 | aggregation | 73.30 | 70.56 | +2.73 | yes | clear gain |
| FEMNIST | 0.1 | 17 | HPO | 72.95 | 73.30 | -0.35 | yes | within noise |
| FEMNIST | 0.1 | 17 | sync | 74.73 | 73.30 | +1.43 | yes | within noise |
| FEMNIST | 0.1 | 42 | NAS | 36.55 | 35.20 | +1.35 | yes | within noise |
| FEMNIST | 0.1 | 42 | topology | 64.01 | 36.55 | +27.46 | yes | clear gain |
| FEMNIST | 0.1 | 42 | aggregation | 65.33 | 64.01 | +1.32 | yes | within noise |
| FEMNIST | 0.1 | 42 | HPO | 63.78 | 65.33 | -1.55 | no | within noise |
| FEMNIST | 0.1 | 42 | sync | 64.43 | 65.33 | -0.90 | no | within noise |
| FEMNIST | 0.1 | 2026 | NAS | 40.42 | 36.47 | +3.95 | yes | clear gain |
| FEMNIST | 0.1 | 2026 | topology | 71.01 | 40.42 | +30.60 | yes | clear gain |
| FEMNIST | 0.1 | 2026 | aggregation | 72.41 | 71.01 | +1.40 | yes | within noise |
| FEMNIST | 0.1 | 2026 | HPO | 71.76 | 72.41 | -0.65 | no | within noise |
| FEMNIST | 0.1 | 2026 | sync | 71.91 | 72.41 | -0.50 | no | within noise |
| FEMNIST | 0.5 | 17 | NAS | 74.70 | 67.54 | +7.16 | yes | clear gain |
| FEMNIST | 0.5 | 17 | topology | 83.06 | 74.70 | +8.36 | yes | clear gain |
| FEMNIST | 0.5 | 17 | aggregation | 83.06 | 83.06 | +0.00 | yes | within noise |
| FEMNIST | 0.5 | 17 | HPO | 83.28 | 83.06 | +0.22 | yes | within noise |
| FEMNIST | 0.5 | 17 | sync | 83.83 | 83.28 | +0.55 | yes | within noise |
| FEMNIST | 0.5 | 42 | NAS | 71.03 | 67.44 | +3.59 | yes | within noise |
| FEMNIST | 0.5 | 42 | topology | 82.41 | 71.03 | +11.39 | yes | clear gain |
| FEMNIST | 0.5 | 42 | aggregation | 82.71 | 82.41 | +0.30 | yes | within noise |
| FEMNIST | 0.5 | 42 | HPO | 83.40 | 82.71 | +0.68 | yes | within noise |
| FEMNIST | 0.5 | 42 | sync | 82.61 | 83.40 | -0.78 | no | within noise |
| FEMNIST | 0.5 | 2026 | NAS | 71.38 | 65.78 | +5.60 | yes | clear gain |
| FEMNIST | 0.5 | 2026 | aggregation | 70.71 | 71.38 | -0.67 | no | within noise |
| FEMNIST | 0.5 | 2026 | sync | 70.91 | 71.38 | -0.47 | yes | within noise |
| FEMNIST | 0.5 | 2026 | HPO | 70.32 | 71.38 | -1.05 | no | within noise |
| FEMNIST | 0.5 | 2026 | topology | 82.10 | 71.38 | +10.72 | yes | clear gain |
| FEMNIST | 0.9 | 17 | NAS | 77.15 | 74.34 | +2.81 | yes | clear gain |
| FEMNIST | 0.9 | 17 | aggregation | 79.20 | 77.15 | +2.04 | yes | within noise |
| FEMNIST | 0.9 | 17 | sync | 79.38 | 79.20 | +0.19 | yes | within noise |
| FEMNIST | 0.9 | 17 | topology | 84.28 | 79.38 | +4.90 | yes | clear gain |
| FEMNIST | 0.9 | 17 | HPO | 84.43 | 84.28 | +0.15 | yes | within noise |
| FEMNIST | 0.9 | 42 | NAS | 76.91 | 71.28 | +5.63 | yes | clear gain |
| FEMNIST | 0.9 | 42 | topology | 83.46 | 76.91 | +6.55 | yes | clear gain |
| FEMNIST | 0.9 | 42 | aggregation | 84.13 | 83.46 | +0.67 | yes | within noise |
| FEMNIST | 0.9 | 42 | HPO | 83.31 | 84.13 | -0.82 | no | within noise |
| FEMNIST | 0.9 | 42 | sync | 83.70 | 84.13 | -0.43 | yes | within noise |
| FEMNIST | 0.9 | 2026 | NAS | 75.58 | 70.36 | +5.22 | yes | clear gain |
| FEMNIST | 0.9 | 2026 | aggregation | 75.29 | 75.58 | -0.29 | yes | within noise |
| FEMNIST | 0.9 | 2026 | sync | 75.63 | 75.58 | +0.05 | yes | within noise |
| FEMNIST | 0.9 | 2026 | HPO | 75.50 | 75.63 | -0.13 | yes | within noise |
| FEMNIST | 0.9 | 2026 | topology | 82.78 | 75.63 | +7.15 | yes | clear gain |
