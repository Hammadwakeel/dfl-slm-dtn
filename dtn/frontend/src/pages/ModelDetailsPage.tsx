import type {
  ExperimentGroup,
  NASCandidate,
  NASArchitectureLayer,
} from "../api/client";

interface Props {
  candidate: NASCandidate;
  group: ExperimentGroup;
  onBack: () => void;
  onUseInModelStudio: () => void;
}

function layerDescription(layer: NASArchitectureLayer): string {
  switch (layer.type) {
    case "conv2d":
    case "separable_conv2d":
      return `${layer.type === "conv2d" ? "Conv2D" : "SeparableConv2D"} · ${layer.filters ?? "—"} filters · ${layer.kernel_size ?? "—"}×${layer.kernel_size ?? "—"} · ${layer.activation ?? "linear"}`;
    case "batch_normalization":
      return "Batch normalization";
    case "max_pooling2d":
    case "average_pooling2d":
      return `${layer.type === "max_pooling2d" ? "Max pooling" : "Average pooling"} · ${layer.pool_size ?? 2}×${layer.pool_size ?? 2}`;
    case "flatten":
      return "Flatten feature map";
    case "global_average_pooling2d":
      return "Global average pooling · 2D";
    case "global_average_pooling1d":
      return "Global average pooling · token sequence";
    case "patch_embedding":
      return `Patch embedding · ${layer.patch_size ?? "—"}×${layer.patch_size ?? "—"} patches · ${layer.embed_dim ?? "—"} dimensions`;
    case "transformer_encoder":
      return `Transformer encoder · ${layer.num_heads ?? "—"}-head self-attention · feed-forward ${layer.ff_dim ?? "—"} · ${layer.activation ?? "gelu"}`;
    case "dense":
      return `Dense · ${layer.units ?? "—"} units · ${layer.activation ?? "linear"}`;
    case "dropout":
      return `Dropout · ${((layer.rate ?? 0) * 100).toFixed(0)}%`;
  }
}

function formatNumber(value: number | undefined, digits = 4): string {
  return value === undefined ? "Not recorded" : value.toFixed(digits);
}

export default function ModelDetailsPage({
  candidate,
  group,
  onBack,
  onUseInModelStudio,
}: Props) {
  const architecture = candidate.architecture;
  const layers = "layers" in architecture ? architecture.layers : null;
  const imageShape = group.dataset_image_shape ?? [28, 28, 1];
  const classes = group.dataset_classes ?? Array.from(
    { length: candidate.validation_confusion_matrix?.length ?? 10 },
    (_, index) => String(index),
  );
  const confusion = candidate.validation_confusion_matrix;

  return (
    <section className="content model-details-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">MNIST · NAS CANDIDATE · {group.group_id}</span>
          <h1>Model details</h1>
          <p>{architecture.id} · Validation metrics · held-out test excluded.</p>
        </div>
        <div className="model-details-actions">
          <button className="quiet-button" type="button" onClick={onBack}>Back to search</button>
          <button className="primary-button" type="button" onClick={onUseInModelStudio}>Open in Model Studio</button>
        </div>
      </div>

      <article className="panel-card">
        <h2>Layer-by-layer architecture</h2>
        <div className="model-architecture-flow" role="group" aria-label="Model architecture diagram">
          <div className="model-architecture-node model-architecture-input">
            <span>INPUT</span>
            <strong>{imageShape.join(" × ")}</strong>
          </div>
          {layers ? (
            <ol className="model-architecture-layers">
              {layers.map((layer, index) => (
                <li key={`${layer.type}-${index}`}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <strong>{layerDescription(layer)}</strong>
                </li>
              ))}
            </ol>
          ) : (
            <p className="model-details-empty">
              Layer specifications unavailable.
            </p>
          )}
          <div className="model-architecture-node model-architecture-output">
            <span>OUTPUT</span>
            <strong>{classes.length} linear logits</strong>
            <small>{classes.join(" · ")}</small>
          </div>
        </div>
        <div className="model-architecture-summary">
          <div><span>Architecture</span><strong>{architecture.id}</strong></div>
          <div><span>Parameters</span><strong>{candidate.parameter_count.toLocaleString()}</strong></div>
          <div><span>Convolution blocks</span><strong>{architecture.blocks}</strong></div>
          <div><span>Attention</span><strong>{"attention" in architecture && architecture.attention ? "Multi-head self-attention" : layers?.some((layer) => layer.type === "transformer_encoder") ? "Multi-head self-attention" : "Not used"}</strong></div>
        </div>
      </article>

      <article className="panel-card">
        <h2>Evaluation results</h2>
        <div className="model-architecture-summary">
          <div><span>Mean validation accuracy</span><strong>{(candidate.mean_validation_accuracy * 100).toFixed(2)}%</strong></div>
          <div><span>Macro precision</span><strong>{formatNumber(candidate.mean_validation_macro_precision)}</strong></div>
          <div><span>Macro recall</span><strong>{formatNumber(candidate.mean_validation_macro_recall)}</strong></div>
          <div><span>Macro F1</span><strong>{formatNumber(candidate.mean_validation_macro_f1)}</strong></div>
          <div><span>Generation / fidelity</span><strong>{candidate.generation ?? "Not recorded"} / {candidate.fidelity_epochs ?? "Not recorded"} epochs</strong></div>
          <div><span>Mean candidate time</span><strong>{candidate.mean_elapsed_seconds === undefined ? "Not recorded" : `${candidate.mean_elapsed_seconds.toFixed(2)} s`}</strong></div>
          <div><span>Peak worker RSS</span><strong>{candidate.max_peak_rss_mb === undefined ? "Not recorded" : `${candidate.max_peak_rss_mb.toFixed(1)} MiB`}</strong></div>
          <div><span>Worker validation accuracy</span><strong>{candidate.worker_accuracies.map((value) => `${(value * 100).toFixed(1)}%`).join(" / ") || "Not recorded"}</strong></div>
        </div>
        {confusion ? (
          <div className="table-wrap model-confusion-wrap">
            <table className="model-confusion-matrix">
              <caption>Summed worker-validation confusion matrix · rows are true classes, columns are predicted</caption>
              <thead>
                <tr>
                  <th scope="col">True / Predicted</th>
                  {classes.map((label) => <th scope="col" key={label}>{label}</th>)}
                </tr>
              </thead>
              <tbody>
                {confusion.map((row, rowIndex) => (
                  <tr key={classes[rowIndex] ?? rowIndex}>
                    <th scope="row">{classes[rowIndex] ?? rowIndex}</th>
                    {row.map((count, columnIndex) => (
                      <td key={`${rowIndex}-${columnIndex}`}>{count.toLocaleString()}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="model-details-empty">Confusion matrix unavailable.</p>
        )}
      </article>

      <article className="panel-card">
        <h2>Framework model summary</h2>
        {candidate.model_summary ? (
          <pre className="keras-model-summary">{candidate.model_summary}</pre>
        ) : (
          <p className="model-details-empty">Framework summary unavailable.</p>
        )}
      </article>

      {group.comparison && (
        <article className="panel-card">
          <h2>Completed training comparison</h2>
          <div className="model-architecture-summary">
            <div><span>NAS held-out accuracy</span><strong>{(group.comparison.nas.mean_accuracy * 100).toFixed(2)}%</strong></div>
            <div><span>NAS held-out macro-F1</span><strong>{group.comparison.nas.mean_macro_f1.toFixed(4)}</strong></div>
            <div><span>Runtime including search</span><strong>{group.comparison.nas.mean_runtime_including_search_seconds.toFixed(1)} s</strong></div>
            <div><span>Communication</span><strong>{Math.round(group.comparison.nas.mean_communication_bytes).toLocaleString()} bytes</strong></div>
          </div>
        </article>
      )}
    </section>
  );
}
