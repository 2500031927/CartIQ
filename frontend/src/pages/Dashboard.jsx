import { useCallback, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, AlertOctagon, BarChart3, DollarSign, RefreshCw, Server, ShieldAlert, ShoppingCart, Users, Repeat, Gauge } from "lucide-react";
import StatCard from "../components/StatCard.jsx";
import Panel from "../components/Panel.jsx";
import EmptyState from "../components/EmptyState.jsx";
import ErrorMessage from "../components/ErrorMessage.jsx";
import ModelMetaGrid from "../components/ModelMetaGrid.jsx";
import { getMetrics } from "../services/api.js";
import { CHART, num } from "../utils/format.js";

const axisProps = { stroke: CHART.axis, tick: { fill: CHART.axis, fontSize: 12 }, tickLine: false };

export default function Dashboard({ health }) {
  const [metrics, setMetrics] = useState({ status: "loading", data: null, error: null });

  const load = useCallback(async () => {
    setMetrics((m) => ({ ...m, status: "loading" }));
    try {
      setMetrics({ status: "ready", data: await getMetrics(), error: null });
    } catch (error) {
      setMetrics({ status: "error", data: null, error });
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, health.status]);

  const m = metrics.data;
  const endpoints = Object.entries(m?.requests?.endpoints ?? {}).map(([path, v]) => ({
    name: path,
    count: v.count ?? 0,
    errors: v.errors ?? 0,
  }));
  const totalErrors = endpoints.reduce((s, e) => s + e.errors, 0);
  const models = health.data?.models ?? {};
  const modelCount = Object.keys(models).length;
  const loadedCount = Object.values(models).filter((x) => x.loaded).length;
  const healthPie = modelCount
    ? [
        { name: "Loaded", value: loadedCount, color: CHART.green },
        { name: "Not loaded", value: modelCount - loadedCount, color: CHART.red },
      ].filter((p) => p.value > 0)
    : [];
  const drift = m?.reorder_input_drift;
  const driftRows = Object.entries(drift?.features ?? {}).map(([name, v]) => ({ name, psi: v.psi }));
  const reorderMeta = m?.models?.reorder;
  const anomalyMeta = m?.models?.anomaly;
  const NA = "Not provided by /metrics";

  return (
    <>
      <div className="page-intro with-action">
        <div>
          <h2>Retail Intelligence Overview</h2>
          <p>Live status of your CartIQ models and API traffic.</p>
        </div>
        <button className="btn btn-ghost" onClick={load} disabled={metrics.status === "loading"}>
          <RefreshCw size={16} className={metrics.status === "loading" ? "spin" : ""} /> Refresh
        </button>
      </div>

      {metrics.status === "error" && <div className="mb"><ErrorMessage error={metrics.error} onRetry={load} title="Metrics unavailable" /></div>}

      <div className="grid-kpi">
        <StatCard icon={Users} label="Total Customers" value={null} hint={NA} accent="blue" />
        <StatCard icon={ShoppingCart} label="Total Orders" value={null} hint={NA} accent="purple" />
        <StatCard icon={DollarSign} label="Average Basket Value" value={null} hint={NA} accent="green" />
        <StatCard icon={AlertOctagon} label="Anomaly Rate" value={null} hint={NA} accent="red" />
        <StatCard icon={Activity} label="Requests Served" value={num(m?.requests?.total_requests)?.toLocaleString() ?? null} hint={m ? `${totalErrors} error response(s)` : undefined} accent="blue" />
        <StatCard icon={Server} label="Models Loaded" value={modelCount ? `${loadedCount} / ${modelCount}` : null} hint={health.status === "online" ? "From /health" : undefined} accent="green" />
      </div>

      <div className="grid-2">
        <Panel title="Model Health" subtitle="Loaded vs. not loaded models (from /health)">
          {healthPie.length ? (
            <div className="chart">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={healthPie} dataKey="value" nameKey="name" innerRadius={62} outerRadius={92} paddingAngle={4} stroke="none">
                    {healthPie.map((p) => <Cell key={p.name} fill={p.color} />)}
                  </Pie>
                  <Tooltip {...CHART.tooltip} />
                  <Legend wrapperStyle={{ color: CHART.axis }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState icon={Server} title="Health data unavailable" description="Start the FastAPI server to see model status." />
          )}
        </Panel>

        <Panel title="Prediction Analytics" subtitle="Requests and errors per endpoint since the API started">
          {endpoints.length ? (
            <div className="chart">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={endpoints} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="name" {...axisProps} tickFormatter={(v) => v.replace("/predict/", "")} />
                  <YAxis allowDecimals={false} {...axisProps} />
                  <Tooltip {...CHART.tooltip} />
                  <Legend wrapperStyle={{ color: CHART.axis }} />
                  <Bar dataKey="count" name="Requests" fill={CHART.blue} radius={[6, 6, 0, 0]} />
                  <Bar dataKey="errors" name="Errors" fill={CHART.red} radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState icon={BarChart3} title="No requests recorded yet" description="Run a prediction and traffic statistics will appear here." />
          )}
        </Panel>

        <Panel title="Customer Segments" subtitle="Segment distribution">
          <EmptyState icon={Users} title="Data unavailable" description="The API does not expose segment counts. Use the Customer Segmentation page to analyze individual customers." />
        </Panel>

        <Panel title="Reorder Insights" subtitle="Input drift of recent reorder requests vs. training data (PSI)">
          {driftRows.length ? (
            <>
              <div className="chart">
                <ResponsiveContainer width="100%" height={230}>
                  <BarChart data={driftRows} layout="vertical" margin={{ top: 4, right: 12, left: 30, bottom: 0 }}>
                    <CartesianGrid stroke={CHART.grid} horizontal={false} />
                    <XAxis type="number" {...axisProps} />
                    <YAxis type="category" dataKey="name" width={110} {...axisProps} />
                    <Tooltip {...CHART.tooltip} />
                    <Bar dataKey="psi" name="PSI" fill={CHART.purple} radius={[0, 6, 6, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="note">Drift status: <strong>{drift.status}</strong> · {drift.n_rows} recent request(s)</p>
            </>
          ) : drift?.status === "insufficient_data" ? (
            <EmptyState icon={Repeat} title="Collecting reorder requests" description={`Drift needs at least ${drift.min_rows} requests; ${drift.n_rows} recorded so far.`} />
          ) : (
            <EmptyState icon={Repeat} title="Data unavailable" description={drift?.detail ?? "No drift information reported by the API."} />
          )}
          {reorderMeta && (
            <dl className="detail-list compact top-gap">
              {Object.entries(reorderMeta).filter(([k, v]) => /pr_auc|threshold|calibrated/.test(k) && v !== null).map(([k, v]) => (
                <div key={k}><dt>{k.replace(/_/g, " ")}</dt><dd className="mono">{typeof v === "number" ? v.toFixed(4) : String(v)}</dd></div>
              ))}
            </dl>
          )}
        </Panel>

        <Panel title="Anomaly Overview" subtitle="Order anomaly detection" className="span-all">
          <EmptyState icon={ShieldAlert} title="Anomaly statistics unavailable" description={`The API reports no aggregate anomaly counts.${anomalyMeta?.version ? ` Active model version: ${anomalyMeta.version}.` : ""} Use the Anomaly Detection page to check individual orders.`} />
        </Panel>
      </div>

      <h3 className="section-title"><Gauge size={18} /> Available Model Metrics</h3>
      <ModelMetaGrid meta={m?.models} />
    </>
  );
}
