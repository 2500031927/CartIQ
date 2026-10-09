import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import Sidebar from "./Sidebar.jsx";
import Header from "./Header.jsx";

/** App shell: sidebar + sticky header + animated page area. */
export default function Layout({ title, health, children }) {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => setMenuOpen(false), [location.pathname]);

  return (
    <div className="app-shell">
      <Sidebar open={menuOpen} onClose={() => setMenuOpen(false)} />
      <div className="main">
        <Header title={title} health={health} onMenu={() => setMenuOpen(true)} />
        <main className="content">
          <div className="page" key={location.pathname}>
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
