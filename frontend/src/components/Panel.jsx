export default function Panel({ title, subtitle, action, children, className = "" }) {
  return (
    <section className={`card panel ${className}`}>
      <header className="panel-head">
        <div>
          <h3>{title}</h3>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}
