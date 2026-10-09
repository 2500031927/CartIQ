export default function StatCard({ icon: Icon, label, value, hint, accent = "blue" }) {
  const unavailable = value === null || value === undefined;
  return (
    <div className={`card stat accent-${accent}`}>
      <div className="stat-icon">
        <Icon size={20} />
      </div>
      <div className="stat-body">
        <span className="stat-label">{label}</span>
        <strong className={`stat-value ${unavailable ? "muted" : ""}`}>{unavailable ? "Data unavailable" : value}</strong>
        {hint && <span className="stat-hint">{hint}</span>}
      </div>
    </div>
  );
}
