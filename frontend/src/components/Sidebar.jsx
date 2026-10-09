import { NavLink } from "react-router-dom";
import { Activity, LayoutDashboard, Repeat, ShieldAlert, ShoppingBasket, ShoppingCart, Users, X } from "lucide-react";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/basket-value", label: "Basket Value", icon: ShoppingBasket },
  { to: "/reorder", label: "Reorder Prediction", icon: Repeat },
  { to: "/segmentation", label: "Customer Segmentation", icon: Users },
  { to: "/anomaly", label: "Anomaly Detection", icon: ShieldAlert },
  { to: "/monitoring", label: "Model Monitoring", icon: Activity },
];

export default function Sidebar({ open, onClose }) {
  return (
    <>
      <div className={`scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">
            <ShoppingCart size={22} />
          </div>
          <div className="brand-text">
            <span className="brand-name">
              CART<em>IQ</em>
            </span>
            <span className="brand-sub">AI-Powered Retail Intelligence</span>
          </div>
          <button className="icon-btn sidebar-close" onClick={onClose} aria-label="Close menu">
            <X size={20} />
          </button>
        </div>
        <nav className="nav">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              <Icon size={19} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">CartIQ · Retail ML Platform</div>
      </aside>
    </>
  );
}
