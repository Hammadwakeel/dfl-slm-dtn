# Phase 5: topology optimization

## Deployment and data

All three Jetsons had internet access and installed `tcpdump 4.99.1` from Ubuntu Jammy. The package was installed on `dfl-worker1..3`; authenticated worker-training services remain active on TCP/5004 and status services remain active on TCP/5001.

The worker shards use CIFAR-10 training data, deterministic seed `2026`, and Dirichlet partitioning. Each worker stores only its own shard for each split; no complete training set was copied to a worker.

| Split | Worker 1 rows | Worker 2 rows | Worker 3 rows |
|---|---:|---:|---:|
| Dirichlet α=0.3 | 25,266 | 10,554 | 14,180 |
| Dirichlet α=0.5 | 23,266 | 11,368 | 15,366 |

The label distributions are intentionally non-IID. For example, under α=0.3, worker1 has no airplane, dog, or ship samples, while worker2 has only 1 ship image and 9 dog images. Under α=0.5, worker1 has only 4 airplane and 1 dog images; worker2 has 31 ship and 101 dog images. The generated shard metadata records exact per-class counts.

Re-create the deployed split files from the verified CIFAR-10 archive with:

```powershell
python -m scripts.prepare_cifar10 --seed 2026 --split dirichlet --alpha 0.3 --shard-tag dirichlet_a03
python -m scripts.prepare_cifar10 --seed 2026 --split dirichlet --alpha 0.5 --shard-tag dirichlet_a05
```

## Candidate graphs and semantics

Every run uses sample-count-weighted averaging of a worker's locally trained model with weights received from its configured neighbors. Workers exchange model weights directly on TCP/5004; the master coordinates but does not forward weight traffic.

| Candidate | Peer graph / direction |
|---|---|
| `directed_ring` | worker1 → worker2 → worker3 → worker1; each worker receives from its predecessor |
| `bidirectional_ring` | All three unordered pairs, with both directions |
| `mesh` | Complete graph, both directions |
| `chain` | worker1 ↔ worker2 ↔ worker3 |
| `star` | worker1 is hub: worker1 ↔ worker2 and worker1 ↔ worker3 |

With exactly three workers, `bidirectional_ring` and `mesh` have the same complete-graph adjacency and the same averaging inputs. Their results should therefore be expected to match closely (and can be identical for the same seed and split); this is a structural limitation, not a reason to report them as distinct evidence. `directed_ring` differs in its one-way exchange per edge. No APG expansion or GNN is implemented or claimed.

## Packet-capture evidence

Captures were made on all three workers with `tcpdump -i any -nn -U -w ... 'tcp port 5004'` during one-round, one-local-epoch runs using α=0.5 and seeds 17, 42, and 2026. Counts below are TCP packets grouped by source/destination worker IP from the sender's own capture; they include TCP handshake, payload, ACK, and other TCP packets on port 5004, not just model payloads.

| Short run | Pair | Worker A → B packets | Worker B → A packets | Pair total |
|---|---|---:|---:|---:|
| Directed ring | worker1 ↔ worker2 | 60 | 33 | 93 |
| Directed ring | worker2 ↔ worker3 | 64 | 31 | 95 |
| Directed ring | worker3 ↔ worker1 | 60 | 35 | 95 |
| Mesh | worker1 ↔ worker2 | 101 | 93 | 194 |
| Mesh | worker1 ↔ worker3 | 94 | 87 | 181 |
| Mesh | worker2 ↔ worker3 | 89 | 93 | 182 |

The directed-ring peer-weight direction is worker1→worker2, worker2→worker3, and worker3→worker1; reverse-direction packets are transport responses/ACKs, not additional weight-send edges. The short Ring run completed in 42.2 seconds and Mesh in 19.0 seconds. These captures directly confirm worker-to-worker TCP/5004 traffic and provide packet counts per peer pair.

## Reward and selector

The optimizer is a transparent reward-based selector over measured device runs, not a learned GNN. Reward weights and normalization values are loaded from [`topology_opt/reward_config.json`](../topology_opt/reward_config.json) and copied with the master service. The active values are recorded with each optimization group.

For a run, each component is normalized to `[0,1]`:

- **Accuracy gain**: `clip((final mean validation accuracy - initial mean validation accuracy) / 0.25, 0, 1)`. Workers measure the initial accuracy before local training in round one, so one-round smoke tests can report a real gain. Older one-round runs without that baseline show accuracy gain as unavailable instead of implying a measured zero.
- **Convergence speed**: first round whose mean validation accuracy reaches the configured target (currently 0.50); score is `(total rounds - convergence round + 1) / total rounds`, or zero if the target was not reached.
- **Communication efficiency**: average of the capped byte-efficiency score (50 MB per worker-round scale) and exchange-time-efficiency score (30 seconds per worker-round scale).
- **Peer-wait efficiency**: one minus total measured worker peer-wait seconds divided by the configured 30-second-per-worker-round scale. Runtime is still recorded and shown for comparison, but is no longer part of the selection reward.

The communication and peer-wait scales are normalization caps, not pass/fail limits. A component reaches zero when its measured cost meets or exceeds its cap. New sweeps use the current scales; already saved sweeps retain the reward and configuration recorded when they ran.

The weighted reward is `0.50 × accuracy_gain + 0.20 × convergence_speed + 0.15 × communication_efficiency + 0.15 × peer_wait_efficiency`. Candidate scores average the per-run rewards. The optimizer selector is compared with two deliberately simple baselines: a reproducible random candidate (Python seed 2026) and the best-observed candidate by mean final validation accuracy. The selected topology and dataset split become the master's active configuration. All runs and reward components are persisted in the master results store.

The authenticated SLM action `optimize_topology` accepts bounded rounds/epochs and one to three seeds. The current Topology page sweep uses α=0.3 only, for five topologies and three seeds (15 runs); the older full experiment matrix below records the earlier two-alpha configuration. `set_topology` supports all five candidates. The Topology page displays the active graph, measured reward components, optimizer choice, random baseline, and best-observed baseline.

## One-round pilot and run-time estimate

Before starting the full matrix, the optimizer ran one round × five local epochs with seed 17 for all five topologies and both alpha values (10 real candidate runs). The client-observed elapsed time was **154.1 seconds** (master group timestamps: 148.7 seconds). Scaling this measured sweep by 30 for 10 rounds × three seeds gives a rough wall-time estimate of **about 77 minutes**, below the requested three-hour threshold. The full matrix was started using that estimate.

| α | Topology | Pilot final held-out accuracy | Pilot reward |
|---:|---|---:|---:|
| 0.3 | directed_ring | 24.70% | 0.200903 |
| 0.3 | bidirectional_ring | 17.41% | 0.169861 |
| 0.3 | mesh | 17.40% | 0.172356 |
| 0.3 | chain | 21.97% | 0.201810 |
| 0.3 | star | 21.98% | 0.173693 |
| 0.5 | directed_ring | 29.15% | 0.210566 |
| 0.5 | bidirectional_ring | 23.33% | 0.189623 |
| 0.5 | mesh | 23.32% | 0.190154 |
| 0.5 | chain | 27.60% | 0.206829 |
| 0.5 | star | 26.14% | 0.190671 |

Because a one-round run has no first-to-final improvement interval and none reached the 50% convergence target, both accuracy-gain and convergence components were zero in the pilot. The pilot's reward winner and best-observed accuracy winner were both directed_ring at α=0.5. These preliminary one-seed results are not the full topology comparison.

## Full topology matrix

The full experiment matrix completed successfully in master group `optimize-1792848734-b42ae2`: five topologies × two Dirichlet alphas × seeds 17, 42, and 2026, 10 rounds, and 5 local epochs per round. All 30 runs completed on the three Jetsons. The poller measured **3,516 seconds (58 minutes 36 seconds)**; the master's group timestamps span 3,494 seconds (about 58 minutes 14 seconds), well below the three-hour limit. Every candidate is the mean of three real runs.

### Held-out accuracy and reward

Accuracy is the mean final held-out accuracy across three seeds. Reward and component columns are the measured means using the weights above. `Conv.` is zero for all candidates because none reached the configured 50% accuracy target.

| α | Topology | Final accuracy | Reward | Accuracy gain | Conv. | Communication | Runtime |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.3 | directed_ring | 40.772% | 0.447883 | 0.528889 | 0.000000 | 0.616602 | 0.606325 |
| 0.3 | bidirectional_ring | 41.200% | 0.512404 | 0.667867 | 0.000000 | 0.583726 | 0.606078 |
| 0.3 | mesh | 41.167% | 0.510548 | 0.665733 | 0.000000 | 0.579984 | 0.604555 |
| 0.3 | chain | 40.998% | 0.489710 | 0.616133 | 0.000000 | 0.610817 | 0.600139 |
| 0.3 | star | 41.000% | 0.461332 | 0.561022 | 0.000000 | 0.595919 | 0.609552 |
| 0.5 | directed_ring | 46.303% | 0.495109 | 0.583467 | 0.000000 | 0.714841 | 0.641000 |
| 0.5 | bidirectional_ring | 47.283% | 0.526124 | 0.657200 | 0.000000 | 0.678556 | 0.638271 |
| 0.5 | mesh | 47.267% | 0.517391 | 0.655867 | 0.000000 | 0.642265 | 0.620789 |
| 0.5 | chain | 46.410% | 0.509350 | 0.617333 | 0.000000 | 0.699496 | 0.638392 |
| 0.5 | star | 46.801% | 0.503717 | 0.613556 | 0.000000 | 0.676009 | 0.636917 |

For additional context, the mean measured communication and runtime per 10-round run were:

| α | Topology | Bytes sent | Exchange seconds | Run runtime seconds |
|---:|---|---:|---:|---:|
| 0.3 | directed_ring | 623,529 | 111.902 | 118.103 |
| 0.3 | bidirectional_ring | 1,244,251 | 118.661 | 118.177 |
| 0.3 | mesh | 1,244,317 | 119.783 | 118.633 |
| 0.3 | chain | 831,463 | 112.598 | 119.958 |
| 0.3 | star | 829,674 | 117.076 | 117.134 |
| 0.5 | directed_ring | 624,017 | 82.428 | 107.700 |
| 0.5 | bidirectional_ring | 1,247,174 | 90.197 | 108.519 |
| 0.5 | mesh | 1,247,166 | 101.085 | 113.763 |
| 0.5 | chain | 832,105 | 85.991 | 108.482 |
| 0.5 | star | 831,555 | 93.040 | 108.925 |

`Run runtime seconds` is the sum across rounds of the slowest worker-reported round time. Communication values are per-run totals averaged across seeds.

### Selector and baseline comparison

The measured-reward selector chose **bidirectional Ring, α=0.5**, with mean reward **0.526124** and mean final held-out accuracy **47.283%**. The seeded-random baseline selected bidirectional Ring, α=0.3 (reward **0.512404**), so the selected candidate's reward was higher by **0.013720**. The best-observed mean-accuracy baseline selected the same candidate as the reward selector. The random baseline is one deterministic draw, not a repeated statistical comparison; the best-observed baseline uses the same completed candidate set, not a separate validation set.

The topologies differed only modestly in held-out accuracy: the best-to-worst spread was **0.428 percentage points** at α=0.3 and **0.980 percentage points** at α=0.5. Bidirectional Ring and Mesh were nearly identical as expected for their equivalent three-worker graph: their mean accuracies differed by 0.033 percentage points at α=0.3 and 0.017 at α=0.5. These three-seed results show observed differences, not evidence of statistical significance. The reward selector is a transparent measured-data baseline; no GNN or APG result is claimed.

### Topology comparison within each data split

This comparison holds the Dirichlet split fixed and ranks topologies using only runs at the matching alpha; it does not mix the alpha=0.3 and alpha=0.5 results. Accuracy differences are relative to the top accuracy within that alpha.

| α | Topology | Mean held-out accuracy | Difference from best topology |
|---:|---|---:|---:|
| 0.3 | bidirectional_ring | 41.200% | 0.000 pp |
| 0.3 | mesh | 41.167% | -0.033 pp |
| 0.3 | star | 41.000% | -0.200 pp |
| 0.3 | chain | 40.998% | -0.202 pp |
| 0.3 | directed_ring | 40.772% | -0.428 pp |
| 0.5 | bidirectional_ring | 47.283% | 0.000 pp |
| 0.5 | mesh | 47.267% | -0.017 pp |
| 0.5 | star | 46.801% | -0.482 pp |
| 0.5 | chain | 46.410% | -0.873 pp |
| 0.5 | directed_ring | 46.303% | -0.980 pp |

These are descriptive means over three seeds, not statistical-significance claims. **No GNN was used**; the optimizer is the measured-reward selector described above.

## Validation

- Python worker/master tests: **37 passed** using `python -m pytest --import-mode=importlib -q tests master\tests`.
- DTN backend tests: **18 passed** using `python -m pytest -q` from `dtn\backend`.
- Frontend tests: **9 passed**; TypeScript check and Vite production build succeeded.
- All 30 real-device runs completed with 10 rounds and 5 local epochs each. The packet-capture evidence above confirms direct peer TCP/5004 traffic.
- Tagged α=0.3 and α=0.5 shard metadata is persisted separately in each generated split directory to avoid one split overwriting the other's metadata.
