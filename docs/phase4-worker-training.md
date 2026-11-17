# Phase 4: Worker training and decentralized CIFAR-10

## Dataset inspection and local preparation

The official CIFAR-10 Python archive was downloaded on the PC to the ignored path `data/raw/cifar-10-python.tar.gz`. Its MD5 matched the published archive checksum `c58f30108f718f92721af3b95e74349a`. The archive is not committed.

| Split | Shape (NHWC) | Rows | Per-class counts | Pixel range |
|---|---:|---:|---|---:|
| Training | `(50000, 32, 32, 3)` | 50,000 | 5,000 for each of the 10 classes | 0–255 |
| Held-out test | `(10000, 32, 32, 3)` | 10,000 | 1,000 for each of the 10 classes | 0–255 |

Class order: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. The test set is reserved for evaluation and is never used by worker training. A 5-example-per-class visualization is saved at [cifar10_sample_grid.png](./cifar10_sample_grid.png).

The training set was partitioned using seed `2026`, stratified independently per class, and verified disjoint:

| Worker | Images | Per-class counts | Shard size |
|---|---:|---:|---:|
| worker1 | 16,670 | 1,667 | 47,015,579 bytes |
| worker2 | 16,670 | 1,667 | 47,073,574 bytes |
| worker3 | 16,660 | 1,666 | 46,972,798 bytes |

Each shard contains all ten classes. The shard archives, parsed training data, and held-out test data are under ignored `data/`. Re-create the default split with:

```powershell
Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn'
python -m scripts.prepare_cifar10 --seed 2026 --split stratified
```

An optional non-IID Dirichlet split is available, but is off by default:

```powershell
python -m scripts.prepare_cifar10 --seed 2026 --split dirichlet --alpha 0.5
```

The deployment script sends only each worker's corresponding training shard to that worker. The held-out test data and centralized-baseline training data are used by the master evaluator and are not sent to workers.

## Hardware preflight

Before deploying the new worker service, real two-epoch TensorFlow GPU smoke training passed on all three workers using system TensorFlow `2.15.0`. A forced matrix multiplication and the training model's first variable both reported `GPU:0`.

| Worker | Smoke training time | Peak process RSS | Host available-memory change |
|---|---:|---:|---:|
| dfl-worker1 | 4,341.0 ms | 825.2 MB | -202.3 MB |
| dfl-worker2 | 4,558.8 ms | 823.8 MB | -224.4 MB |
| dfl-worker3 | 4,479.1 ms | 824.2 MB | -210.3 MB |

This smoke test used synthetic in-memory data only; it did not copy CIFAR-10 to any Jetson or run distributed training.

## Services and exchange protocol

The three workers run `dfl-slm-worker-training.service` on TCP/5004. Their status agents remain active on TCP/5001. The master SLM/coordinator runs on TCP/5003 alongside its status agent on TCP/5002. Worker credentials and peer tokens are stored in root-owned systemd environment files on each device; the APIs reject unauthenticated requests with HTTP 401. The worker firewall check found UFW inactive on all three devices, and each worker reached both peers on TCP/5004.

Each worker trains a small CIFAR-10 CNN (5,354 parameters; 21,416 bytes of float32 weights) on its own shard. Ring uses a directed cycle and averages a worker's weights with its predecessor's; Mesh exchanges with both peers and uses sample-count-weighted averaging. Workers transfer compressed weights directly to peer addresses on TCP/5004. The master separately retrieves each worker's post-aggregation weights for evaluation against the held-out test split; training images and labels stay on the workers. The dashboard polls persisted, real per-round accuracy, cross-entropy loss, macro-F1, training/round runtime, process peak RSS, exchange time, and peer byte/message counters.

After each distributed run, once the coordinator has persisted worker metrics and retrieved the final weights, it asks every worker to discard that run's model, weights, cached results, and error state. Workers wait until training is idle, clear the Keras session, collect Python garbage, and request a Linux heap trim. Cleanup acknowledgements or per-worker failures are recorded on the run as `worker_cleanup`; the coordinator's persisted metrics remain available independently.

The Dashboard also offers an all-worker memory cleanup action. It first verifies that every configured worker is reachable and idle; if any worker is training, scheduled, or unreachable, it starts no cleanup. Each worker clears only in-memory experiment/model state and drops Linux filesystem caches, preserving its dataset shards and coordinator-side experiment history. Page-cache access is provided by the separate `dfl-slm-memory-cleanup.service`, which listens only on loopback, requires a dedicated token, and exposes only the fixed cache-drop operation. Deployment verifies the helper's authenticated health endpoint before restarting the training API. If cleanup reports HTTP 401 or connection refused for a worker, redeploy just the helper and its token without reinstalling training dependencies: `.\deploy\deploy_worker_memory_cleanup.ps1 -WorkerIds worker4,worker5`. Installation allocates an SSH terminal so `sudo` can prompt for the worker account's password without storing it; a failed installation removes its staged credential files when possible. The UI displays per-worker before/after available memory and process RSS, including partial failures. Dropping filesystem caches can temporarily slow file access and does not terminate worker processes.

Both complete distributed matrices were run through the deployed master coordinator using seeds `17`, `42`, and `2026`, 10 rounds, and 5 local epochs per round. Each score below is the mean final held-out test metric across the three worker models for that seed.

| Topology | Seed | Final test accuracy | Final test loss | Final macro-F1 |
|---|---:|---:|---:|---:|
| Ring | 17 | 54.91% | 1.2896 | 54.16% |
| Ring | 42 | 55.32% | 1.2829 | 54.90% |
| Ring | 2026 | 54.93% | 1.2994 | 54.67% |
| Mesh | 17 | 54.98% | 1.2863 | 54.24% |
| Mesh | 42 | 55.31% | 1.2799 | 54.89% |
| Mesh | 2026 | 54.87% | 1.2970 | 54.59% |

Each topology completed all 30 seed-rounds (90 worker-rounds). Average worker round duration was 8.18 seconds for Ring and 8.05 seconds for Mesh. The maximum process RSS observed during the runs was 2,099.1 MB for Ring and 2,387.7 MB for Mesh. The measured model weight payload was 21,416 bytes before compression.

The workers' JSON event logs independently confirm direct peer transfers. Across the full Ring experiment, each worker logged 30 `peer_weights_sent` and 30 `peer_weights_received` events, with the expected cycle neighbors. Across Mesh, each worker logged 60 sends and 60 receives, covering both peers. Application counters sum to 90 Ring messages and 180 Mesh messages sent across the cluster. `tcpdump`, `dumpcap`, and `tshark` were not installed, so no OS packet-capture count was available.

## Failure injection

During a live 10-round/5-epoch Ring run, `dfl-worker3`'s training service was stopped. The coordinator marked the group failed when worker3's `/v1/results` connection was refused, canceled the remaining workers, and preserved the error in the experiment results. The service was restarted and its authenticated status endpoint returned successfully. No worker was left stopped.

The coordinator retries transient network failures (including timeouts and unreachable-worker errors) up to twice for read-only `/v1/results` requests before failing the group. Persistent failures and non-network errors still fail explicitly; state-changing worker requests are not retried.

## Centralized baseline

The matching master-side baseline used all 50,000 CIFAR-10 training images on `dfl-master`; only the 10,000 held-out test examples were used for evaluation. The full 10-round/5-local-epoch/3-seed baseline completed successfully (group `baseline-1792842068-89e711`).

| Seed | Final test accuracy | Final test loss | Final macro-F1 | Final round time | Peak process RSS |
|---:|---:|---:|---:|---:|---:|
| 17 | 56.72% | 1.2406 | 56.39% | 21.789 s | 3,145.7 MB |
| 42 | 57.70% | 1.2177 | 57.43% | 22.002 s | 3,445.7 MB |
| 2026 | 56.76% | 1.2465 | 56.15% | 24.609 s | 3,445.7 MB |
| **Mean** | **57.06%** | **1.2349** | **56.66%** | **22.800 s** | **3,445.7 MB max** |

End-to-end elapsed time for the three-seed baseline was approximately 614 seconds. Compared with the mean across distributed seeds, the centralized model was 2.01 percentage points higher in accuracy and 2.08 points higher in macro-F1 for both topologies; its mean test loss was lower by 0.0557 versus Ring and 0.0528 versus Mesh. The centralized run used about 3.45 GB peak process RSS on the master. Distributed worker peak RSS was 2.10 GB for Ring and 2.39 GB for Mesh. These are measurements for this model, data split, hardware, and configuration, not a general guarantee that centralized training will outperform DFL.
