# DFL-APG Specification (T2)

Source: thesis file `SLM-DFL-20261025_1_1_3_1_2_1.pptx`, Chapter 5 (slides 91-112: formulation 5.1, APG algorithm 5.2, config, results).
Everything in sections 1-4 is taken from the thesis. Section 5 lists what the thesis does NOT define, so the implementer must choose and document it (marked IMPLEMENTATION CHOICE). Section 6 adapts the experiment to our hardware.
Do not copy the thesis results (FEMNIST 0.922, CIFAR-10 0.811) as our results. Our numbers must come from our runs only.

## 1. Idea

A coordinator re-optimizes the communication graph between rounds. After round t-1 each node reports validation losses of neighbor models, waiting times, and its local class distribution. The coordinator solves a convex multi-objective problem with Accelerated Projected Gradient (Nesterov) and thresholds the result into a directed adjacency matrix used in round t. Initial topology in the thesis is Mesh.

## 2. Notation (n nodes, k classes)

- A(t-1) in {0,1}^(n x n): current directed adjacency. a_ij = 1 means a link from node i to node j.
- W(t-1) in [0,1]^(n x n): current weight matrix.
- Theta in [0,1]^(n x k): class proportions of each node's local data (row i = proportions of node i).
- L_ij: validation loss of neighbor j's model evaluated on node i's local validation set.
- T_ij: time node i waited to receive node j's model update.
- L~, T~: min-max normalized L and T (over the whole matrix).

## 3. Optimization problem (thesis eq. 5.3-5.6)

- F(W) = sum_ij (alpha * L~_ij + beta * T~_ij) * W_ij
- H(W) = (1/n) * || W Theta - (1/n) 1 1^T Theta ||_F^2 (pushes each node's neighborhood class mix toward the global mix)
- R(W) = || W - W_prev ||_F^2 (avoids sudden topology change)
- minimize g(W) = gamma*F(W) + mu*H(W) + lambda*R(W)
- constraints: W_ij in [0,1]; W_ii = 0; sum_j W_ij = 1 for every i; gamma + mu + lambda = 1
- Gradient: grad g(W) = gamma*(alpha*L~ + beta*T~) + (2*mu/n) * (W Theta - (1/n) 1 1^T Theta) Theta^T + 2*lambda*(W - W_prev)
- Binary topology: a_ij = 1 if W_ij >= epsilon, else 0.

## 4. APG algorithm (thesis 5.2, run once per round)

Initialize W_0 = W(t-1) (and W_-1 = W_0). k_0 = 1 is the standard choice (IMPLEMENTATION CHOICE, not stated).
For m = 0..M-1:

1. k_(m+1) = (1 + sqrt(1 + 4*k_m^2)) / 2
2. Y_m = W_m + ((k_m - 1) / (k_m + 1)) * (W_m - W_(m-1)) (formula as written in the thesis)
3. Z_m = Y_m - eta * grad g(Y_m)
4. W_(m+1) = Pi_S(Z_m): projection onto the feasible set S (each row projected onto the probability simplex over the n-1 off-diagonal entries; diagonal fixed to 0)
5. Stop if |g(W_(m+1)) - g(W_m)| < psi, or m reaches M.

Output W_hat(t), then A_hat(t) by thresholding with epsilon. Cost per round: O(M * n^2 * (k + log n)). Convergence rate of the objective: O(1/m^2).

Hyperparameters given by the thesis:

| Parameter | Value |
|---|---|
| alpha, beta | 0.6, 0.4 (grid search over [0.2, 0.8]) |
| gamma, mu, lambda | 0.6, 0.2, 0.2 |
| psi (stop threshold) | 0.01 |
| epsilon (edge threshold) | 0.15 |

Thesis training setup (for reference): 8 nodes, cross-entropy, Adam, lr 0.01, 3 local epochs, 100 communication rounds, initial topology Mesh, non-IID Dirichlet alpha = 0.1, baselines Mesh / Ring / PSO-GNN.

## 5. NOT defined by the thesis (IMPLEMENTATION CHOICES: pick, document in the code and in the report, test)

1. eta (step size) and M (max iterations). Suggested default: eta = 1/Lip with Lip = (2*mu/n) * sigma_max(Theta)^2 + 2*lambda (gradient Lipschitz bound of g); M = 100. Tune only on VALIDATION, if at all.
2. W(0) for the first round when starting from Mesh: uniform off-diagonal weights 1/(n-1).
3. L_ij and T_ij for pairs that are not currently linked (not measured in the current round). Suggested: keep the last measured value and log its age in rounds. Start from Mesh so every pair is measured in round 1.
4. Degenerate min-max normalization (max == min): use 0 for the whole matrix.
5. Empty row after thresholding. With n = 8, a uniform row has weight 1/7 = 0.143 < epsilon = 0.15, so a node can end with no outgoing neighbor. Suggested safeguard: keep the single largest weight in that row. Log how often it triggers.
6. Aggregation after the topology is chosen: keep the existing weighted averaging (by sample count) over the node's neighbors in A_hat plus itself. Say clearly if W is also used as aggregation weights (thesis does not state this); default: do not use W for aggregation.
7. How Theta is obtained: each node computes class proportions of its local TRAIN shard and reports them. Never use TEST data.

## 6. Adaptation to our system (hybrid: 5 simulated + 3 real Jetson)

- n = 8. node_0..node_4 simulated on the PC, node_5..node_7 = worker1..3. Same model and exposure for ring, mesh, APG.
- Rounds: thesis uses 100 rounds x 3 local epochs. Our earlier 5 rounds x 20 epochs gives APG only 4 topology updates. Use 30 rounds x 3 local epochs (about the same 90-100 local epochs of exposure as before). Report this as a deviation from the thesis.
- Data: the thesis uses non-IID Dirichlet (alpha = 0.1). With IID stratified shards Theta is almost identical for all nodes and the H term has nothing to fix. Run two settings and report them separately: (a) IID stratified, (b) Dirichlet alpha = 0.1 on the TRAIN split only (sharding seed 2026). Training seeds 17 / 42 / 2026. TEST is never sharded.
- Baselines: Mesh, Ring, APG (starting from Mesh). PSO-GNN is not implemented (no GNN in this project); say so.
- L_ij: node i evaluates the received model of node j on its own local validation rows (VALIDATION only).
- T_ij: real nodes: measured at the bridge/worker (time from the end of node i's local training until node j's update has arrived). Per-edge timing is not exposed by the current workers; adding it is a Phase C change that needs approval. Simulated nodes: T_ij = max(0, train_time_j - train_time_i) + bytes / min(bw_i, bw_j), where train_time_j = base_time / compute_j, from the SYNTHETIC profile config. Label every simulated T as SIMULATED.
- Synthetic profiles for node_0..node_4 (proposed, assumed, not measured; edit in the config file if you want different values): node_0 compute 1.0x, bandwidth 100%; node_1 0.75x, 75%; node_2 0.5x, 50%; node_3 0.35x, 25%; node_4 0.25x, 10% (relative to the measured real-worker round time and the measured LAN throughput).

## 7. Required tests

- Projection: every row sums to 1, entries in [0,1], diagonal 0, idempotent.
- Gradient check against finite differences on a small random case.
- Objective g does not increase over the APG iterations on a convex test case (allow Nesterov non-monotonicity only with a documented tolerance).
- Determinism for a fixed seed.
- Constant inputs and W_prev at the optimum: topology does not change (supports the UI verdict check "topology follows its inputs").
- Changing L, T or Theta changes the chosen edges.
- Thresholding and empty-row safeguard.

## 8. What to report

Per round: A_hat(t) (edge list), edges changed vs previous round, number of edges, g value, iterations used, safeguard hits, per-node VALIDATION accuracy, bytes (SIMULATED or measured). Final: TEST accuracy, macro-F1, macro precision/recall, confusion matrix from saved predictions, mean +/- sample SD (ddof = 1) over 3 seeds, separately for IID and Dirichlet, and separately for all-simulated vs hybrid. State clearly whether APG shows a benefit over Mesh/Ring beyond seed noise. If not, say "no demonstrated benefit".
