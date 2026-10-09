import { BarChart3 } from "lucide-react";
import Panel from "./Panel.jsx";
import EmptyState from "./EmptyState.jsx";
import { MODEL_LABELS, formatValue, humanize, isScalar } from "../utils/format.js";

/** Shows exactly the scalar fields the backend reports in /metrics -> models (nothing added). */
export default function ModelMetaGrid({ meta }) {
  const entries = Object.entries(meta || {});
  if (entries.length === 0) {
    return (
      <EmptyState
        icon={BarChart3}
        title="No model metrics available"
        description="Train the models to create model_metadata.json; the backend then reports their versions and scores here."
      />
    );
  }
  return (
    <div className="grid-2">
      {entries.map(([key, info]) => (
        <Panel key={key} title={MODEL_LABELS[key] ?? humanize(key)} subtitle="Reported by /metrics">
          <dl className="detail-list compact">
            {Object.entries(info || {})
              .filter(([, v]) => isScalar(v))
              .map(([k, v]) => (
                <div key={k}>
                  <dt>{humanize(k)}</dt>
                  <dd className="mono">{formatValue(v)}</dd>
                </div>
              ))}
          </dl>
        </Panel>
      ))}
    </div>
  );
}
