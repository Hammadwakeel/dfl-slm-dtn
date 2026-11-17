import type { NASArchitectureLayer, NASLayerType } from "../api/client";

interface Props {
  layers: NASArchitectureLayer[];
  title: string;
  inputLabel?: string;
  inputShape?: string;
  layerLabels?: string[];
  note?: string;
}

const LAYER_STYLES: Record<NASLayerType, { label: string; color: string }> = {
  conv2d: { label: "Conv2D", color: "#00f5d4" },
  separable_conv2d: { label: "Separable Conv2D", color: "#00f5d4" },
  max_pooling2d: { label: "MaxPooling2D", color: "#8338ec" },
  average_pooling2d: { label: "AveragePooling2D", color: "#a855f7" },
  batch_normalization: { label: "BatchNorm", color: "#06b6d4" },
  dropout: { label: "Dropout", color: "#03045e" },
  flatten: { label: "Flatten", color: "#ffbe0b" },
  global_average_pooling2d: { label: "Global AvgPool2D", color: "#ffbe0b" },
  dense: { label: "Dense", color: "#fb5607" },
  patch_embedding: { label: "Patch Embedding", color: "#00b4d8" },
  transformer_encoder: { label: "Transformer", color: "#4361ee" },
  global_average_pooling1d: { label: "Global AvgPool1D", color: "#ffbe0b" },
};

const INPUT_COLOR = "#00f5d4";
const COLUMN_STEP = 82;
const FRONT_WIDTH = 40;
const DEPTH = 12;

function layerDetails(layer: NASArchitectureLayer): string {
  switch (layer.type) {
    case "conv2d":
    case "separable_conv2d":
      return `${layer.filters ?? 32} filters · ${layer.kernel_size ?? 3}×${layer.kernel_size ?? 3}`;
    case "max_pooling2d":
    case "average_pooling2d":
      return `${layer.pool_size ?? 2}×${layer.pool_size ?? 2}`;
    case "dropout":
      return `${Math.round((layer.rate ?? 0.25) * 100)}%`;
    case "dense":
      return `${layer.units ?? 10} units`;
    case "patch_embedding":
      return `${layer.embed_dim ?? 64} dimensions`;
    case "transformer_encoder":
      return `${layer.num_heads ?? 4} heads · FFN ${layer.ff_dim ?? 128}`;
    default:
      return "";
  }
}

function layerHeight(layer: NASArchitectureLayer): number {
  switch (layer.type) {
    case "max_pooling2d":
    case "average_pooling2d":
      return 68;
    case "flatten":
    case "global_average_pooling1d":
    case "global_average_pooling2d":
      return 48;
    case "dense":
      return Math.max(34, Math.min(86, 28 + Math.sqrt(layer.units ?? 10) * 5));
    case "transformer_encoder":
      return 92;
    default:
      return 112;
  }
}

export default function LayeredArchitectureView({
  layers,
  title,
  inputLabel = "MNIST input",
  inputShape = "28 × 28 × 1",
  layerLabels,
  note = "Schematic layer view; block sizes are illustrative, not tensor dimensions.",
}: Props) {
  const width = Math.max(600, 94 + (layers.length + 1) * COLUMN_STEP);
  const height = 220;
  const yCenter = 76;
  const entries = [
    { label: inputLabel, detail: inputShape, color: INPUT_COLOR, blockHeight: 128 },
    ...layers.map((layer, index) => ({
      label: layerLabels?.[index] ?? LAYER_STYLES[layer.type].label,
      detail: layerDetails(layer),
      color: LAYER_STYLES[layer.type].color,
      blockHeight: layerHeight(layer),
    })),
  ];
  const legend = [...new Set(layers.map((layer) => layer.type))];

  return (
    <figure className="layered-architecture">
      <figcaption>{title}</figcaption>
      <div className="layered-architecture__canvas">
        <svg
          aria-label={`${title}: ${entries.map((entry) => entry.label).join(" → ")}`}
          className="layered-architecture__svg"
          height={height}
          role="img"
          viewBox={`0 0 ${width} ${height}`}
          width={width}
        >
          {entries.map((entry, index) => {
            const x = 26 + index * COLUMN_STEP;
            const blockTop = yCenter + DEPTH - entry.blockHeight / 2;
            const blockBottom = blockTop + entry.blockHeight;
            const shade = `${entry.color}99`;
            const dark = `${entry.color}bb`;
            const nameLines = entry.label
              .replace(/([a-z])([A-Z])/g, "$1 $2")
              .split(/[\s-]+/)
              .slice(0, 3);
            return (
              <g key={`${entry.label}-${index}`}>
                {index > 0 && (
                  <path
                    className="layered-architecture__connector"
                    d={`M ${x - COLUMN_STEP + FRONT_WIDTH + DEPTH} ${yCenter + DEPTH} H ${x - 5}`}
                  />
                )}
                <polygon
                  points={`${x},${blockTop + DEPTH} ${x + DEPTH},${blockTop} ${x + FRONT_WIDTH + DEPTH},${blockTop} ${x + FRONT_WIDTH},${blockTop + DEPTH}`}
                  fill={shade}
                  stroke={entry.color}
                />
                <polygon
                  points={`${x + FRONT_WIDTH},${blockTop + DEPTH} ${x + FRONT_WIDTH + DEPTH},${blockTop} ${x + FRONT_WIDTH + DEPTH},${blockBottom} ${x + FRONT_WIDTH},${blockBottom + DEPTH}`}
                  fill={dark}
                  stroke={entry.color}
                />
                <rect
                  fill={entry.color}
                  height={entry.blockHeight}
                  stroke={entry.color}
                  width={FRONT_WIDTH}
                  x={x}
                  y={blockTop + DEPTH}
                />
                <text className="layered-architecture__index" textAnchor="middle" x={x + FRONT_WIDTH / 2} y={yCenter + DEPTH + 4}>
                  {index === 0 ? "IN" : index}
                </text>
                <text className="layered-architecture__label" textAnchor="middle" x={x + (FRONT_WIDTH + DEPTH) / 2} y="174">
                  {nameLines.map((line, lineIndex) => (
                    <tspan dy={lineIndex === 0 ? 0 : 11} key={line} x={x + (FRONT_WIDTH + DEPTH) / 2}>
                      {line}
                    </tspan>
                  ))}
                </text>
                {entry.detail && (
                  <text className="layered-architecture__detail" textAnchor="middle" x={x + (FRONT_WIDTH + DEPTH) / 2} y="204">
                    {entry.detail}
                  </text>
                )}
                <title>{`${entry.label}${entry.detail ? `: ${entry.detail}` : ""}`}</title>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="layered-architecture__legend" aria-label="Layer color legend">
        <span><i style={{ backgroundColor: INPUT_COLOR }} />Input</span>
        {legend.map((type) => (
          <span key={type}>
            <i style={{ backgroundColor: LAYER_STYLES[type].color }} />
            {LAYER_STYLES[type].label}
          </span>
        ))}
      </div>
      <p className="layered-architecture__note">{note}</p>
    </figure>
  );
}
