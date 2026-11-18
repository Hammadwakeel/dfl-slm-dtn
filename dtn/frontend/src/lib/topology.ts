import type { TrainingTopology } from "../api/client";

/** Human-readable peer links of a topology over the given workers. */
export function topologyLinks(
  topology: TrainingTopology | "centralized" | "candidate_sweep",
  workers: string[],
  learnedSenders?: Record<string, string[]>,
): string[] {
  const ordered = [...workers].sort();
  if (topology === "centralized") return ["dfl-master · centralized"];
  if (topology === "learned") {
    return Object.entries(learnedSenders ?? {})
      .sort(([left], [right]) => left.localeCompare(right))
      .flatMap(([receiver, senders]) => senders.map((sender) => `${sender} → ${receiver}`));
  }
  const pairs = new Set<string>();
  const undirected = (left: string, right: string) => pairs.add(left < right ? `${left} ↔ ${right}` : `${right} ↔ ${left}`);
  ordered.forEach((node, index) => {
    const next = ordered[(index + 1) % ordered.length];
    if (topology === "mesh") ordered.slice(index + 1).forEach((peer) => undirected(node, peer));
    else if (topology === "bidirectional_ring") undirected(node, next);
    else if (topology === "chain" && index + 1 < ordered.length) undirected(node, next);
    else if (topology === "star" && index > 0) undirected(ordered[0], node);
    else if (topology === "directed_ring" || topology === "ring" || topology === "candidate_sweep") pairs.add(`${node} → ${next}`);
  });
  return [...pairs];
}
