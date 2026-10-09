import { AlertTriangle, CheckCircle2, X, XCircle } from "lucide-react";

const ICONS = { success: CheckCircle2, warning: AlertTriangle, error: XCircle };

export default function Toast({ toast, onClose }) {
  if (!toast) return null;
  const Icon = ICONS[toast.type] ?? CheckCircle2;
  return (
    <div className={`toast toast-${toast.type}`} role="status">
      <Icon size={20} />
      <span>{toast.text}</span>
      <button className="icon-btn" onClick={onClose} aria-label="Dismiss">
        <X size={16} />
      </button>
    </div>
  );
}
