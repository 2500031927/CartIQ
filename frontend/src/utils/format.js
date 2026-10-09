export const CHART = {
  grid: "rgba(255,255,255,0.08)",
  axis: "#8c96b2",
  blue: "#4f8cff",
  purple: "#8b5cf6",
  green: "#22c55e",
  red: "#ef4444",
  amber: "#f59e0b",
  tooltip: {
    contentStyle: {
      background: "#10162b",
      border: "1px solid rgba(255,255,255,0.12)",
      borderRadius: 12,
      color: "#e8ecf6",
    },
    itemStyle: { color: "#e8ecf6" },
    labelStyle: { color: "#8c96b2" },
    cursor: { fill: "rgba(255,255,255,0.04)" },
  },
};

export const MODEL_LABELS = {
  basket_value: "Basket Value",
  reorder: "Reorder",
  segmentation: "Customer Segmentation",
  anomaly: "Anomaly Detection",
};

export const isScalar = (v) => v === null || ["string", "number", "boolean"].includes(typeof v);

export const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);

export function humanize(key) {
  return String(key).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function formatValue(v) {
  if (v === null || v === undefined) return "Data unavailable";
  if (typeof v === "number") {
    return Number.isInteger(v) ? v.toLocaleString() : v.toLocaleString(undefined, { maximumFractionDigits: 4 });
  }
  if (typeof v === "boolean") return v ? "Yes" : "No";
  return String(v);
}

export function formatUptime(seconds) {
  if (typeof seconds !== "number") return "Data unavailable";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

export function formatMs(v) {
  return typeof v === "number" ? `${v.toFixed(1)} ms` : "Data unavailable";
}
