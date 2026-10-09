import { AlertTriangle, WifiOff } from "lucide-react";

export default function ErrorMessage({ error, title, onRetry }) {
  const message = typeof error === "string" ? error : error?.message || "Something went wrong.";
  const details = Array.isArray(error?.details) ? error.details : [];
  const network = Boolean(error?.network);
  return (
    <div className="card error-box" role="alert">
      <div className="error-icon">{network ? <WifiOff size={22} /> : <AlertTriangle size={22} />}</div>
      <div className="error-body">
        <h4>{title || (network ? "API unreachable" : "Request failed")}</h4>
        <p>{message}</p>
        {details.length > 0 && (
          <ul>
            {details.map((d, i) => (
              <li key={i}>{d}</li>
            ))}
          </ul>
        )}
        {onRetry && (
          <button type="button" className="btn btn-ghost" onClick={onRetry}>
            Try again
          </button>
        )}
      </div>
    </div>
  );
}
