import { useCallback, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, BarChart3, Loader2, RefreshCw, Repeat, ShieldAlert, ShoppingBasket, Users } from "lucide-react";
import ModelStatusCard from "../components/ModelStatusCard.jsx";
import ModelMetaGrid from "../components/ModelMetaGrid.jsx";
import Panel from "../components/Panel.jsx";
import EmptyState from "../components/EmptyState.jsx";
import ErrorMessage from "../components/ErrorMessage.jsx";
import Toast from "../components/Toast.jsx";
import { getMetrics, reloadModels } from "../services/api.js";
import { CHART, MODEL_LABELS, formatMs, formatUptime, formatValue } from "../utils/format.js";

const CARDS = [
  { key: "basket_value", icon: ShoppingBasket },
  { key: "reorder", icon: Repeat },
  { key: "segmentation", icon: Users },
  { key: "anomaly", icon: ShieldAlert },
];
const axisProps = { stroke: CHART.axis, tick: { fill: CHART.axis, fontSize: 12 }, tickLine: false };

export default function ModelMonitoring({ health, onRefreshHealth }) {
  const [metrics, setMetrics] = useState({ status: "loading", data: null, error: null });
  const [reloading, setReloading] = useState(false);
  const [toast, setToast] = useState(null);

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

  useEffect(() => {
    if (!toast) return undefined;
    const id = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(id);
  }, [toast]);

  async function handleReload() {
    setReloading(true);
    try {
      const res = await reloadModels();
      const failed = Object.keys(res.errors ?? {});
      const loaded = Array.isArray(res.loaded) ? res.loaded : [];
      setToast(
        failed.length
          ? { type: "warning", text: `Reloaded ${loaded.length} model(s). Could not load: ${failed.join(", ")}.` }
          : { type: "success", text: `Models reloaded successfully${loaded.length ? `: ${loaded.join(", ")}` : ""}.` }
      );
      await Promise.all([onRefreshHealth(), load()]);
    } catch (error) {
      setToast({ type: "error", text: error.message });
    } finally {
      setReloading(false);
    }
  }

  const m = metrics.data;
  const endpointRows = Object.entries(m?.requests?.endpoints ?? {}).map(([path, v]) => ({ path, ...v }));
  const drift = m?.reorder_input_drift;
  const driftRows = Object.entries(drift?.features ?? {});
  const models = health.data?.models ?? {};

  return (
    <>
      <Toast toast={toast} onClose={() => setToast(null)} />
      <div className="page-intro with-action">
        <div>
          <h2>Model Monitoring</h2>
          <p>Model availability, API traffic and data drift reported by the backend.</p>
        </div>
        <div className="btn-row">
          <button className="btn btn-ghost" onClick={() => { onRefreshHealth(); load(); }}>
            <RefreshCw size={16} /> Refresh
          </button>
          <button className="btn btn-primary" onClick={handleReload} disabled={reloading || health.status === "offline"}>
            {reloading ? <Loader2 size={18} className="spin" /> : <RefreshCw size={18} />}
            {reloading ? "Reloading…" : "Reload Models"}
          </button>
        </div>
      </div>

      {health.status === "offline" && <div className="mb"><ErrorMessage error={health.error} /></div>}

      <div className="grid-models">
        {CARDS.map(({ key, icon }) => (
          <ModelStatusCard key={key} label={MODEL_LABELS[key]} icon={icon} info={models[key]} />
        ))}
      </div>

      {metrics.status === "error" && <div className="mb"><ErrorMessage error={metrics.error} onRetry={load} title="Metrics unavailable" /></div>}

      <div className="grid-2">
        <Panel title="Request Latency" subtitle="Mean and p95 latency per endpoint (ms)">
          {endpointRows.length ? (
            <div className="chart">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={endpointRows} margin={{ top: 8, right: 8, left: -12, bottom: 0 }}>
                  <CartesianGrid stroke={CHART.grid} vertical={false} />
                  <XAxis dataKey="path" {...axisProps} tickFormatter={(v) => v.replace("/predict/", "")} />
                  <YAxis {...axisProps} />
                  <Tooltip {...CHART.tooltip} formatter={(v) => `${Number(v).toFixed(1)} ms`} />
                  <Legend wrapperStyle={{ color: CHART.axis }} />
                  <Bar dataKey="mean_ms" name="Mean" fill={CHART.blue} radius={[6, 6, 0, 0]} />
                  <Bar dataKey="p95_ms" name="p95" fill={CHART.purple} radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState icon={BarChart3} title="No requests recorded yet" description="Latency appears once the API has served requests." />
          )}
        </Panel>

        <Panel title="API Traffic" subtitle="Counters since the API process started">
          <dl className="detail-list compact">
            <div><dt>Total Requests</dt><dd className="mono">{formatValue(m?.requests?.total_requests)}</dd></div>
            <div><dt>Uptime</dt><dd className="mono">{formatUptime(m?.requests?.uptime_seconds)}</dd></div>
          </dl>
          {endpointRows.length > 0 && (
            <div className="table-wrap top-gap">
              <table>
                <thead><tr><th>Endpoint</th><th>Requests</th><th>Errors</th><th>Mean</th><th>p95</th><th>Max</th></tr></thead>
                <tbody>
                  {endpointRows.map((r) => (
                    <tr key={r.path}>
                      <td className="mono">{r.path}</td><td>{formatValue(r.count)}</td><td>{formatValue(r.errors)}</td>
                      <td>{formatMs(r.mean_ms)}</td><td>{formatMs(r.p95_ms)}</td><td>{formatMs(r.max_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>
      </div>

      <Panel title="Data Drift — Reorder Inputs" subtitle="Population Stability Index of recent requests vs. training data" className="mb">
        {driftRows.length ? (
          <>
            <p className="note">Overall status: <strong>{drift.status}</strong> · max PSI {formatValue(drift.max_psi)} · {drift.n_rows} request(s)</p>
            <div className="table-wrap">
              <table>
                <thead><tr><th>Feature</th><th>PSI</th><th>Status</th></tr></thead>
                <tbody>
                  {driftRows.map(([name, v]) => (
                    <tr key={name}>
                      <td className="mono">{name}</td><td>{formatValue(v.psi)}</td>
                      <td><span className={`badge ${v.status === "stable" ? "badge-green" : v.status === "moderate" ? "badge-amber" : "badge-red"}`}>{v.status}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : drift?.status === "insufficient_data" ? (
          <EmptyState icon={Activity} title="Not enough data yet" description={`Drift needs at least ${drift.min_rows} reorder requests; ${drift.n_rows} recorded so far.`} />
        ) : (
          <EmptyState icon={Activity} title="Data unavailable" description={drift?.detail ?? "The API reported no drift information."} />
        )}
      </Panel>

      <h3 className="section-title"><BarChart3 size={18} /> Model Metadata</h3>
      <ModelMetaGrid meta={m?.models} />
    </>
  );
}
