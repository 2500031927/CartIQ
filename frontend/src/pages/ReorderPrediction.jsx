import { Repeat } from "lucide-react";
import PredictionForm from "../components/PredictionForm.jsx";
import PredictionResult, { DetailList } from "../components/PredictionResult.jsx";
import usePrediction from "../hooks/usePrediction.js";
import { predictReorder } from "../services/api.js";

const FIELDS = [
  { name: "customer_id", label: "Customer ID", type: "text", placeholder: "e.g. C-1001", hint: "Stored only as a salted hash" },
  { name: "recency_days", label: "Recency Days", type: "number", required: true, min: 0, hint: "Days since last order" },
  { name: "frequency", label: "Frequency", type: "int", required: true, min: 1, hint: "Number of orders" },
  { name: "monetary", label: "Monetary", type: "number", required: true, min: 0, hint: "Total spend" },
  { name: "avg_order_value", label: "Average Order Value", type: "number", required: true, min: 0 },
  { name: "max_order_value", label: "Maximum Order Value", type: "number", required: true, min: 0 },
  { name: "total_items", label: "Total Items", type: "number", required: true, min: 0 },
  { name: "n_unique_products", label: "Unique Products", type: "int", required: true, min: 1 },
  { name: "tenure_days", label: "Tenure Days", type: "number", required: true, min: 0 },
  { name: "avg_gap_days", label: "Average Gap Days", type: "number", min: 0, hint: "Leave blank with only one order" },
  { name: "orders_last_30d", label: "Orders Last 30 Days", type: "int", min: 0, hint: "Defaults to 0 if blank" },
  { name: "orders_last_90d", label: "Orders Last 90 Days", type: "int", min: 0, hint: "Defaults to 0 if blank" },
  { name: "spend_last_90d", label: "Spend Last 90 Days", type: "number", min: 0, hint: "Defaults to 0 if blank" },
  { name: "country", label: "Country", type: "text", defaultValue: "United Kingdom" },
];

const SAMPLE = {
  recency_days: "20", frequency: "4", monetary: "400", avg_order_value: "100", max_order_value: "180",
  total_items: "120", n_unique_products: "25", tenure_days: "150", avg_gap_days: "35",
  orders_last_30d: "1", orders_last_90d: "2", spend_last_90d: "200",
};

function ProbabilityRing({ value, threshold }) {
  const pct = Math.max(0, Math.min(1, value));
  const r = 70;
  const c = 2 * Math.PI * r;
  const t = Math.max(0, Math.min(1, threshold));
  const ang = (t * 360 - 90) * (Math.PI / 180);
  const tick = { x1: 90 + (r - 11) * Math.cos(ang), y1: 90 + (r - 11) * Math.sin(ang), x2: 90 + (r + 11) * Math.cos(ang), y2: 90 + (r + 11) * Math.sin(ang) };
  return (
    <div className="ring">
      <svg viewBox="0 0 180 180" role="img" aria-label={`Reorder probability ${(pct * 100).toFixed(1)} percent`}>
        <defs>
          <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#4f8cff" />
            <stop offset="100%" stopColor="#8b5cf6" />
          </linearGradient>
        </defs>
        <circle cx="90" cy="90" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="14" />
        <circle cx="90" cy="90" r={r} fill="none" stroke="url(#ringGrad)" strokeWidth="14" strokeLinecap="round" strokeDasharray={`${c * pct} ${c}`} transform="rotate(-90 90 90)" />
        <line {...tick} stroke="#f59e0b" strokeWidth="3" strokeLinecap="round" />
      </svg>
      <div className="ring-center">
        <strong>{(pct * 100).toFixed(1)}%</strong>
        <span>reorder probability</span>
      </div>
    </div>
  );
}

export default function ReorderPrediction() {
  const { state, run, retry } = usePrediction(predictReorder);
  return (
    <>
      <div className="page-intro">
        <h2>Reorder Prediction</h2>
        <p>Estimate the probability that a customer will reorder within 30 days.</p>
      </div>
      <div className="split">
        <PredictionForm fields={FIELDS} sample={SAMPLE} submitLabel="Predict Reorder" loading={state.status === "loading"} onSubmit={run} />
        <PredictionResult state={state} onRetry={retry} icon={Repeat} idleTitle="No prediction yet" idleText="Enter the customer's purchase history to estimate reorder likelihood.">
          {(d) => (
            <>
              <ProbabilityRing value={d.reorder_probability} threshold={d.decision_threshold} />
              <div className={`verdict ${d.will_reorder_30d ? "verdict-good" : "verdict-warn"}`}>
                {d.will_reorder_30d ? "Likely to reorder within 30 days" : "Unlikely to reorder within 30 days"}
              </div>
              <p className="note">The amber tick on the ring marks the decision threshold.</p>
              <DetailList
                items={[
                  ["Reorder Probability", `${(d.reorder_probability * 100).toFixed(2)}%`],
                  ["Will Reorder Within 30 Days", d.will_reorder_30d ? "Yes" : "No"],
                  ["Decision Threshold", `${(d.decision_threshold * 100).toFixed(2)}%`],
                  ["Calibration Status", d.calibrated ? "Calibrated" : "Not calibrated"],
                  ["Model Name", d.model_name],
                  ["Model Version", d.model_version],
                ]}
              />
            </>
          )}
        </PredictionResult>
      </div>
    </>
  );
}
