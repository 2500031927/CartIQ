import { Menu } from "lucide-react";

export default function Header({ title, health, onMenu }) {
  const { status, data, error } = health;
  const label = status === "online" ? "API Online" : status === "offline" ? "API Offline" : "Checking API…";
  const degraded = status === "online" && data && data.status !== "ok";
  return (
    <header className="header">
      <div className="header-left">
        <button className="icon-btn menu-btn" onClick={onMenu} aria-label="Open menu">
          <Menu size={22} />
        </button>
        <h1>{title}</h1>
      </div>
      <div className="header-right">
        {degraded && <span className="badge badge-amber">Some models not loaded</span>}
        <span className={`status-pill ${status}`} title={status === "offline" ? error?.message : undefined}>
          <span className="dot" /> {label}
        </span>
      </div>
    </header>
  );
}
