# Results Report

Updated 2026-10-30. This report contains measured project results only. CIFAR-10 and MNIST are separate datasets and their scores are not directly comparable.

## Summary

| Experiment | Result | Evaluation split | Provenance |
|---|---|---|---|
| CIFAR-10 DFL, Phase 4 | 55.05% mean accuracy | Official 10,000-image test split | [Phase 4 report](phase4-worker-training.md), three seeds |
| CIFAR-10 centralized fixed baseline, Phase 4 | 57.06% mean accuracy | Same CIFAR-10 test split | [Phase 4 report](phase4-worker-training.md), three seeds |
| CIFAR-10 NAS DFL, Phase 6 | 61.63% mean accuracy; macro-F1 0.6136 | Official CIFAR-10 test split after NAS selection | [Phase 6 report](phase6-neural-architecture-search.md), `b1_c16_k5`, three seeds |
| CIFAR-10 centralized NAS baseline, M8 | 63.78% +/- 0.41 pp accuracy; macro-F1 0.6357 +/- 0.0038 | Same 10,000-image CIFAR-10 test split | [M8 JSON](../data/m8_centralized_nas_baseline.json), `b1_c16_k5`, three seeds |
| MNIST first hardware run, before worker fix | 32.29% accuracy; macro-F1 25.45% | Hardware validation metric, not test | Recorded pre-fix run; workers did not load the installed initial-model bundle |
| MNIST corrected hardware run | 77.71% accuracy; macro-F1 77.09% | Validation metric, 160 images per worker; not test | Accepted hardware group `bidirectional_ring-1793345933-ed39e6`, 5 rounds x 20 epochs |
| MNIST federated PC evaluation | 79.11% +/- 0.69 pp accuracy; macro-F1 0.7900 +/- 0.0082 | 300-image test split, evaluated after training | [MNIST JSON](../data/mnist_final_test_eval.json), three seeds |
| MNIST centralized PC evaluation | 83.56% +/- 2.22 pp accuracy; macro-F1 0.8358 +/- 0.0218 | Same 300-image test split | [MNIST JSON](../data/mnist_final_test_eval.json), three seeds |

The M8 centralized baseline used the Phase 6 `b1_c16_k5` architecture and matched the DFL NAS example count per seed. The older 56.93% centralized result used the fixed `b1_c8_k3` architecture and is not an architecture-matched baseline.

For MNIST, validation was used for tuning. The 300-image test split was kept for the final PC-only metrics and was not used for training or tuning. The 77.71% hardware result is validation, not test accuracy.

## MNIST Hardware Evidence

The corrected 5x20 run reported 123,248 communication bytes and 41.04 seconds duration (the earlier 360-second figure was an estimate). Per-worker final validation accuracies were worker1 80.625%, worker2 76.875%, and worker3 75.625%.

The persisted FULL_ACCESS audit events for that run were:

| Event ID | Action | Mode | Result |
|---|---|---|---|
| `execution-4` | `set_topology` to `bidirectional_ring` | FULL_ACCESS | executed |
| `execution-5` | `start_training`, 5x20 `mnist_demo` | FULL_ACCESS | executed |
| `execution-6` | SLM follow-up `start_training`, 3x2 `mnist_demo` | FULL_ACCESS | executed |

The SLM response was valid JSON and passed schema validation, but suggested less training (3x2). Its source was `slm`, status `executed`, fallback was off, and the follow-up validation accuracy was 37.71% with macro-F1 30.53%. Valid JSON does not establish correct intent.

Measured tegrastats peaks (RAM / used swap, MB):

| Run | Master | Worker1 | Worker2 | Worker3 |
|---|---:|---:|---:|---:|
| Corrected 5x20 | 1753 / 685 | 5226 / 26 | 5266 / 120 | 5197 / 81 |
| SLM 3x2 follow-up | 3512 / 685 | 5230 / 26 | 5269 / 120 | 5203 / 81 |

## Milestones

| Milestone | Status | Evidence / remaining work |
|---|---|---|
| M1: autonomy controls and audit | PASS | ASK_HUMAN default, bounded FULL_ACCESS, STOP, loops, execution audit |
| M2: user datasets | PASS | Named upload and metadata flow; no default CIFAR upload |
| M3: preprocessing and sharding | PASS | Stratified/Dirichlet processing and shard verification |
| M4: initial model and NAS | PASS, preliminary | Dataset-derived model and NAS; small-proxy scores are not a final ranking |
| M5: feedback loops | PASS | ASK_HUMAN/FULL_ACCESS loop controls and real feedback flow |
| M6: frontend | PASS | React UI, dataset flow, lint/tests/build |
| M7: failure recovery | PARTIAL | Worker crash and backend restart passed; worker reboot training acceptance timed out; master reboot and security re-check were not run |
| M8: centralized NAS baseline | PASS | PC-only `b1_c16_k5`, three seeds, same CIFAR-10 train/test split and 10x5 schedule |
| M9: demo and reporting | PASS | Read-only demo, README and results report; full suites pass |

## Limitations

- Schema-valid SLM output can still express poor intent; the MNIST feedback suggested 3x2 training after the 5x20 hardware run.
- The earlier 32.29% MNIST validation run was affected by a worker path that ignored the installed initial-model bundle; it is not the corrected result.
- A prior backend restart check was a false positive because an old orphan Python process still owned port 8000. Listener provenance must include process start time and parent chain, not just health.
- M7 remains partial: the post-worker-reboot first results request timed out; master reboot and security re-check remain pending.
- Results use three seeds. The MNIST test split has only 300 images, so its estimate is noisy; validation data was used for tuning.
- Three workers provide limited topology diversity; Ring and Mesh can be operationally similar at this scale. No GNN is implemented.
- The earlier Phase 6 centralized reference uses a different, fixed architecture; M8 is the architecture-matched centralized comparison.
- No result in this report claims SLM fine-tuning, NAS-selected architecture deployment through the generic training action, or CO2-data evaluation.
- The working tree remains uncommitted; no commit was made for M9.