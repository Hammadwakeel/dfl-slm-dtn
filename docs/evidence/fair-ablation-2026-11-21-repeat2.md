# Leave-one-out ablation of the fair campaign

Generated 2026-11-24 14:09 UTC by `scripts/ablation_campaign.py`.

For every finished fair run, the final configuration (after keep-or-revert) is retrained
with the baseline's budget (10 rounds x 5 local epochs, same seed and initial weights), once
as it is and once without each component the run kept. A component's contribution is the
test accuracy of the full configuration minus that of the configuration without it, so it
does not depend on the order in which the SLM ran the steps. Values are mean ± std over the
seeds that kept the component (n). Accuracy is the mean over the 8 workers on the test split.

## Contribution of each component (test accuracy points, full minus without)

| Dataset | α | Full (%) | Nas (n) | Hpo (n) | Topology (n) | Aggregation (n) | Sync (n) |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 72.12 ± 8.61 | +11.86 ± 12.23 (3) | +2.97 (1) | +22.43 ± 5.03 (3) | – (0) | -5.90 (1) |

## Repeatability

The baseline and the full configuration are each trained twice: once in the campaign and
once here, with the same seed. The difference is run-to-run noise on the real cluster
(thread scheduling, network timing and semi-synchronous deadlines are not deterministic).

| Dataset | α | Seed | Baseline test: campaign / repeat | Full test: campaign / repeat |
|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 42 | 44.94 / 44.93 | 61.71 / 62.68 |
| Fashion-MNIST | 0.1 | 2026 | 38.58 / 50.43 | 75.64 / 74.14 |
| Fashion-MNIST | 0.1 | 17 | 43.13 / 41.87 | 82.40 / 79.55 |

## Per run

| Dataset | α | Seed | Final configuration | Baseline | Full | without NAS | without HPO | without topology | without aggregation | without sync |
|---|---|---|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 42 | nas-d2-w64x64-k5-conv-h64-drop35, lr 0.001, batch 32, mesh, fedavg, semi_sync | 44.93 | 62.68 | 62.80 | – | 46.05 | – | – |
| Fashion-MNIST | 0.1 | 2026 | hpo-adam-lr3e-4-bs64, lr 0.0003, batch 64, mesh, fedavg, semi_sync | 50.43 | 74.14 | 62.77 | 71.17 | 48.50 | – | – |
| Fashion-MNIST | 0.1 | 17 | nas-d3-w16x64x64-k5-conv-h0-drop35, lr 0.001, batch 32, mesh, fedavg, sync | 41.87 | 79.55 | 55.22 | – | 54.54 | – | 85.45 |
