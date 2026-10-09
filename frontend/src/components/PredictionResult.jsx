import { Sparkles } from "lucide-react";
import LoadingSpinner from "./LoadingSpinner.jsx";
import ErrorMessage from "./ErrorMessage.jsx";
import EmptyState from "./EmptyState.jsx";

/** Renders the idle / loading / success / error states of a prediction. `children` is a function of the response. */
export default function PredictionResult({ state, onRetry, idleTitle, idleText, icon = Sparkles, children }) {
  if (state.status === "loading") {
    return (
      <div className="card result center">
        <LoadingSpinner label="Running model…" />
      </div>
    );
  }
  if (state.status === "error") return <ErrorMessage error={state.error} onRetry={onRetry} />;
  if (state.status === "success") return <div className="card result fade-in">{children(state.data)}</div>;
  return (
    <div className="card result center">
      <EmptyState icon={icon} title={idleTitle} description={idleText} />
    </div>
  );
}

export function DetailList({ items }) {
  return (
    <dl className="detail-list">
      {items.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}
