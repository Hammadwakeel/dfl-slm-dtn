import type { NASArchitecture, NASArchitectureLayer } from "../api/client";

export function layersFromNASArchitecture(architecture: NASArchitecture): NASArchitectureLayer[] {
  if ("layers" in architecture && architecture.layers.length > 0) return architecture.layers;

  const layers: NASArchitectureLayer[] = [];
  for (let index = 0; index < architecture.blocks; index += 1) {
    layers.push({
      type: "conv2d",
      filters: architecture.channels,
      kernel_size: architecture.kernel_size,
      activation: "relu",
    });
    layers.push({ type: "max_pooling2d", pool_size: 2 });
  }
  layers.push({ type: "global_average_pooling2d" });
  layers.push({ type: "dense", units: 10, activation: "linear" });
  return layers;
}
