# Topology or communication volume? In-degree study

Generated 2026-11-25 06:52 UTC by `scripts/topology_degree_study.py`.

Every seed's final fair-campaign configuration (architecture, hyperparameters, aggregation and
synchronization unchanged) is trained on directed graphs of increasing in-degree with the same
budget (10 rounds x 5 local epochs, same seed and initial weights). In-degree k means
worker i averages its model with the models of workers i-1 .. i-k each round (a circulant graph);
k = 1 is the baseline's directed ring and k = 7 the mesh. `ring 70x1` is the ring with 70 rounds of
1 local epoch: every worker receives as many models as in the mesh (70), with more local compute
(70 epochs instead of 50). Models received counts model transfers into one worker over the run;
bytes scale with it times the model size. The spectral gap (1 - |λ2| of the uniform averaging
matrix) measures how fast one round of averaging mixes information across the 8 workers.
In-degree 1 and 7 come from the leave-one-out ablation passes (`fair-ablation-2026-11-21*.json`), so
they are averaged over every repeat. Accuracy is the mean test accuracy over the 8 workers (%).

## Mean over seeds

| Dataset | α | Graph | Models received per worker | Spectral gap | Test accuracy (%) | Seeds |
|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | in-degree 1 (ring) | 10 | 0.076 | 49.94 ± 3.26 | 3 |
| Fashion-MNIST | 0.1 | in-degree 2 | 20 | 0.195 | 58.52 ± 8.87 | 3 |
| Fashion-MNIST | 0.1 | in-degree 4 | 40 | 0.517 | 68.68 ± 7.13 | 3 |
| Fashion-MNIST | 0.1 | in-degree 7 (mesh) | 70 | 1.000 | 72.12 ± 10.10 | 3 |
| Fashion-MNIST | 0.1 | ring 70x1 | 70 | 0.076 per round, 7x the rounds | 56.96 ± 9.29 | 3 |
| FEMNIST | 0.1 | in-degree 1 (ring) | 10 | 0.076 | 40.67 ± 4.25 | 3 |
| FEMNIST | 0.1 | in-degree 2 | 20 | 0.195 | 52.14 ± 5.07 | 3 |
| FEMNIST | 0.1 | in-degree 4 | 40 | 0.517 | 61.70 ± 4.93 | 3 |
| FEMNIST | 0.1 | in-degree 7 (mesh) | 70 | 1.000 | 70.18 ± 4.20 | 3 |
| FEMNIST | 0.1 | ring 70x1 | 70 | 0.076 per round, 7x the rounds | 55.98 ± 2.05 | 3 |

## Per seed

| Dataset | α | Seed | in-degree 1 (ring) | in-degree 2 | in-degree 4 | in-degree 7 (mesh) | ring 70x1 |
|---|---|---|---|---|---|---|---|
| Fashion-MNIST | 0.1 | 42 | 47.71 ± 1.66 | 48.44 | 60.54 | 60.95 ± 4.92 | 61.06 |
| Fashion-MNIST | 0.1 | 2026 | 48.43 ± 0.96 | 65.11 | 71.65 | 74.79 ± 0.63 | 46.33 |
| Fashion-MNIST | 0.1 | 17 | 53.68 ± 0.92 | 62.00 | 73.84 | 80.61 ± 2.49 | 63.51 |
| FEMNIST | 0.1 | 17 | 43.34 | 56.09 | 64.16 | 73.05 | 55.59 |
| FEMNIST | 0.1 | 42 | 35.77 | 46.42 | 56.02 | 65.36 | 54.15 |
| FEMNIST | 0.1 | 2026 | 42.88 | 53.90 | 64.92 | 72.15 | 58.20 |
