# Phase 6: neural architecture search

## Design

DFL weight averaging requires identical tensor layouts at every worker. Each worker therefore evaluates the same fixed candidate list independently using only its own stratified CIFAR-10 training shard. It reserves a deterministic, class-stratified 20% validation split from that shard, trains each candidate for one epoch, and reports candidate ID, validation accuracy, parameter count, elapsed time, peak resident memory, and validation sample count. No held-out CIFAR-10 test data is sent to a worker or used by the search.

The master averages each candidate's validation accuracy across the three workers and deterministically selects the highest mean (ties prefer fewer parameters, then candidate ID). It then configures the same selected architecture on all three workers before any comparison run starts. Fixed and NAS DFL runs use the same topology, local shard, rounds, epochs, and seeds. Evaluation against the held-out test set only happens after model selection, during the training comparison.

This is a small discrete CNN search, not a differentiable or learned supernet. Each candidate is trained independently from initialization for one local epoch; that inexpensive proxy may not predict the best architecture after full DFL training. The local validation metric is an equal-weight mean of worker scores, not a centralized pooled validation score.

## Candidate space and bounds

The eight candidates vary block count, channels, and kernel size. Each block is a convolution followed by 2×2 max pooling; the first convolution uses stride two. A 10-class dense output follows flattening.

| Candidate | Convolution blocks | Channels per block | Kernel | Parameters |
|---|---:|---:|---:|---:|
| `b1_c8_k3` (fixed baseline) | 1 | 8 | 3×3 | 5,354 |
| `b1_c8_k5` | 1 | 8 | 5×5 | 5,738 |
| `b1_c16_k3` | 1 | 16 | 3×3 | 10,698 |
| `b1_c16_k5` | 1 | 16 | 5×5 | 11,466 |
| `b2_c8_k3` | 2 | 8 | 3×3 | 2,098 |
| `b2_c8_k5` | 2 | 8 | 5×5 | 3,506 |
| `b2_c16_k3` | 2 | 16 | 3×3 | 5,338 |
| `b2_c16_k5` | 2 | 16 | 5×5 | 10,202 |

The search space has a hard 12,000-parameter ceiling. Each worker search has a 300-second wall-clock budget and a 2,048 MiB process-RSS ceiling. Search executes in a separate worker-local Python process so its memory measurement and ceiling do not inherit memory retained by the long-lived training service. It checks elapsed time and RSS at each training batch and after validation inference, aborting if either bound is exceeded. RSS is process memory, not a separate GPU-memory allocation limit; Jetson uses unified system memory. Peak RSS is measured during candidate evaluation.

## SLM and dashboard integration

The schema-validated `enable_nas` action starts a real worker search followed by fixed-model DFL, selected-model DFL, and a centralized fixed-model reference. It accepts bounded rounds, epochs, seeds, and search seconds. The new Architecture Search page starts either a small one-round pilot or the full three-seed comparison and shows every candidate's measured validation accuracy, parameter count, time, and worker RSS, the selected common architecture, and the Fixed/NAS/centralized results.

The existing worker-to-worker peer weight exchange is unchanged except that the configured CNN determines the exchanged arrays. Since every peer receives the same validated architecture in the configuration request, their weight shapes match. Per-round serialized communication bytes are measured rather than assumed.

## Pilot and full comparison

The pilot group `nas-1792853182-9be865` used Phase 5's selected `bidirectional_ring`, the stratified split, one round, one local epoch, and seed 17. It completed in **65.1 seconds total**. The three worker searches took **28.0 seconds parallel wall time**, and every worker reported TensorFlow GPU `/physical_device:GPU:0`. Each worker searched 13,330–13,340 training examples and validated on 3,330 examples. Peak RSS across all search candidates and workers was **1,913.4 MiB**, below the 2,048 MiB limit.

| Candidate | Mean worker validation accuracy | Parameters | Mean candidate time | Maximum worker RSS |
|---|---:|---:|---:|---:|
| `b1_c8_k3` | 30.57% | 5,354 | 3.10 s | 1,697.9 MiB |
| `b1_c8_k5` | 32.59% | 5,738 | 3.06 s | 1,755.7 MiB |
| `b1_c16_k3` | 34.05% | 10,698 | 3.00 s | 1,836.6 MiB |
| `b1_c16_k5` | **35.97%** | 11,466 | 3.15 s | 1,873.9 MiB |
| `b2_c8_k3` | 23.68% | 2,098 | 3.61 s | 1,913.4 MiB |
| `b2_c8_k5` | 28.86% | 3,506 | 4.13 s | 1,905.1 MiB |
| `b2_c16_k3` | 31.17% | 5,338 | 3.71 s | 1,673.3 MiB |
| `b2_c16_k5` | 35.02% | 10,202 | 3.82 s | 1,726.5 MiB |

The selector chose `b1_c16_k5`. One-seed, one-round held-out metrics are preliminary and are not evidence that NAS will outperform after full training:

| Pilot variant | Mean held-out accuracy | Macro-F1 | Runtime incl. search | Peak worker RSS | Communication bytes |
|---|---:|---:|---:|---:|---:|
| Fixed DFL (`b1_c8_k3`) | 32.02% | 0.3051 | 3.2 s | 2,022.5 MiB | 124,146 |
| NAS DFL (`b1_c16_k5`) | 38.91% | 0.3865 | 31.3 s | 2,022.5 MiB | 258,756 |
| Centralized fixed reference | 40.84% | 0.3934 | 11.1 s | 1,650.8 MiB | 0 |

NAS increased measured communication bytes as expected for a larger model. The worker DFL RSS peak was close to, but below, the per-search 2,048 MiB ceiling; the ceiling applies to search, while the DFL result is reported separately.

Based on the pilot, the estimated full comparison time was about 45 minutes. The measured full group completed faster, in 20 minutes.

### Full 10-round comparison

The final group `nas-1792853466-4b3051` completed all nine runs: fixed DFL, NAS-selected DFL, and centralized fixed reference, each with seeds 17, 42, and 2026; every run used 10 rounds × 5 local epochs. Topology was `bidirectional_ring` and the worker data split was stratified. The complete group took **1,202.3 seconds (20 minutes 2 seconds)**, including architecture search. The worker searches completed in **34.6 seconds parallel wall time** on the three GPUs; maximum search RSS was **1,687.3 MiB**.

The full search selected the same model as the pilot, `b1_c16_k5` (11,466 parameters). Full-search candidate metrics:

| Candidate | Mean worker validation accuracy | Parameters | Mean candidate time | Maximum worker RSS |
|---|---:|---:|---:|---:|
| `b1_c8_k3` | 30.56% | 5,354 | 5.13 s | 1,243.7 MiB |
| `b1_c8_k5` | 32.59% | 5,738 | 3.73 s | 1,333.6 MiB |
| `b1_c16_k3` | 34.06% | 10,698 | 3.64 s | 1,438.3 MiB |
| `b1_c16_k5` | **35.90%** | 11,466 | 3.82 s | 1,468.5 MiB |
| `b2_c8_k3` | 23.68% | 2,098 | 4.71 s | 1,466.6 MiB |
| `b2_c8_k5` | 29.01% | 3,506 | 4.35 s | 1,532.1 MiB |
| `b2_c16_k3` | 31.36% | 5,338 | 4.42 s | 1,588.9 MiB |
| `b2_c16_k5` | 34.87% | 10,202 | 4.42 s | 1,687.3 MiB |

Aggregate held-out metrics are means over the three completed seeds. Runtime including search amortizes the one parallel search across the three NAS runs; the separate training-only time and full NAS group total are also shown. RAM is the maximum process RSS observed across those runs. Communication is mean bytes sent per DFL run (not including the centralized baseline).

| Variant | Mean held-out accuracy | Mean macro-F1 | Mean training runtime | Mean runtime incl. search | Peak process RSS | Mean communication bytes |
|---|---:|---:|---:|---:|---:|---:|
| Fixed DFL (`b1_c8_k3`) | 55.07% | 0.5459 | 80.24 s | 80.24 s | 2,285.8 MiB | 1,248,496 |
| NAS DFL (`b1_c16_k5`) | **61.63%** | **0.6136** | 86.36 s | 97.91 s | 2,508.9 MiB | 2,609,813 |
| Centralized fixed reference | 56.93% | 0.5653 | 214.74 s | 214.74 s | 3,383.5 MiB | 0 |

NAS beat the fixed DFL model on mean held-out accuracy by **6.56 percentage points** and macro-F1 by **0.0677**. NAS training took about 6.1 seconds more per seed, and amortized search added another 11.5 seconds per seed; average communication bytes were about **2.09×** the fixed model. The centralized reference was also measured; NAS DFL exceeded its mean accuracy by 4.70 percentage points, but that is not a like-for-like communication setting.

Per-seed held-out accuracy and macro-F1 confirm that the aggregate difference was present for all three seeds:

| Seed | Fixed accuracy | Fixed macro-F1 | NAS accuracy | NAS macro-F1 | Central fixed accuracy | Central fixed macro-F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 55.00% | 0.5426 | 61.04% | 0.6065 | 56.68% | 0.5634 |
| 42 | 55.33% | 0.5491 | 61.74% | 0.6158 | 57.70% | 0.5743 |
| 2026 | 54.88% | 0.5460 | 62.11% | 0.6185 | 56.42% | 0.5581 |

The initial full-matrix launch immediately after the pilot did not begin any training runs: the long-lived TensorFlow worker process retained roughly 2.0 GiB RSS after the pilot and its next in-process search returned HTTP 400 under the 2,048 MiB cap. The worker training services were restarted to release that allocator memory and the full matrix was retried successfully. Search was subsequently isolated in a worker-local process so later searches start with a fresh process memory budget.

After deploying the isolated-search change, an authenticated search-only verification was run once on each worker. Each evaluated all eight candidates on its GPU; no DFL runs were repeated:

| Worker | TensorFlow GPU | Candidates | Search elapsed | Peak RSS | Best local candidate |
|---|---|---:|---:|---:|---|
| worker1 | `/physical_device:GPU:0` | 8 | 34.187 s | 1,690.5 MiB | `b1_c16_k5` |
| worker2 | `/physical_device:GPU:0` | 8 | 34.519 s | 1,556.9 MiB | `b1_c16_k5` |
| worker3 | `/physical_device:GPU:0` | 8 | 34.675 s | 1,553.2 MiB | `b1_c16_k5` |

## Validation

- Python worker/master tests: **46 passed** with `python -m pytest --import-mode=importlib -q tests master\tests`.
- DTN backend tests: **18 passed**.
- Frontend Vitest: **11 passed**; TypeScript check and Vite production build succeeded.
- Deployment scripts parsed successfully; all three worker training services were redeployed and passed live authenticated GPU searches; their status services remained active. Both master services remained active.
- The full 10-round / 5-epoch / 3-seed comparison completed on the Jetsons; all nine runs finished successfully.
