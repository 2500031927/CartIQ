export default function LoadingSpinner({ label = "Loading…", size = 40 }) {
  return (
    <div className="spinner-wrap" role="status" aria-live="polite">
      <span className="spinner" style={{ width: size, height: size }} />
      {label && <span className="spinner-label">{label}</span>}
    </div>
  );
}
