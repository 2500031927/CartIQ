import { CheckCircle2, XCircle, HelpCircle } from "lucide-react";

export default function ModelStatusCard({ label, icon: Icon, info }) {
  const known = Boolean(info);
  const loaded = known && info.loaded;
  const StatusIcon = !known ? HelpCircle : loaded ? CheckCircle2 : XCircle;
  return (
    <div className={`card model-card ${!known ? "unknown" : loaded ? "ok" : "bad"}`}>
      <div className="model-card-top">
        <div className="stat-icon">
          <Icon size={20} />
        </div>
        <span className={`badge ${!known ? "badge-muted" : loaded ? "badge-green" : "badge-red"}`}>
          <StatusIcon size={14} /> {!known ? "Unknown" : loaded ? "Loaded" : "Not loaded"}
        </span>
      </div>
      <h4>{label}</h4>
      <dl className="detail-list compact">
        <div>
          <dt>Version</dt>
          <dd className="mono">{known ? info.version ?? "Data unavailable" : "Data unavailable"}</dd>
        </div>
        {known && info.error && (
          <div>
            <dt>Error</dt>
            <dd className="text-red">{info.error}</dd>
          </div>
        )}
      </dl>
    </div>
  );
}
