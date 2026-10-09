import { ShieldAlert, ShieldCheck } from "lucide-react";
import PredictionForm from "../components/PredictionForm.jsx";
import PredictionResult, { DetailList } from "../components/PredictionResult.jsx";
import usePrediction from "../hooks/usePrediction.js";
import { predictAnomaly } from "../services/api.js";

const FIELDS = [
  { name: "order_value", label: "Order Value", type: "number", required: true, gt: 0 },
  { name: "n_lines", label: "Number of Lines", type: "int", required: true, min: 1 },
  { name: "total_qty", label: "Total Quantity", type: "number", required: true, gt: 0 },
  { name: "avg_unit_price", label: "Average Unit Price", type: "number", required: true, gt: 0 },
];

const SAMPLE = { order_value: "250", n_lines: "6", total_qty: "40", avg_unit_price: "4.5" };

export default function AnomalyDetection() {
  const { state, run, retry } = usePrediction(predictAnomaly);
  return (
    <>
      <div className="page-intro">
        <h2>Anomaly Detection</h2>
        <p>Detect unusual order behavior using the trained anomaly detection model.</p>
      </div>
      <div className="split">
        <PredictionForm fields={FIELDS} sample={SAMPLE} submitLabel="Check Order" loading={state.status === "loading"} onSubmit={run} />
        <PredictionResult state={state} onRetry={retry} icon={ShieldAlert} idleTitle="No order checked yet" idleText="Enter an order's summary to see whether it looks unusual.">
          {(d) => (
            <>
              <div className={`anomaly-state ${d.is_anomaly ? "is-anomaly" : "is-normal"}`}>
                {d.is_anomaly ? <ShieldAlert size={46} /> : <ShieldCheck size={46} />}
                <strong>{d.is_anomaly ? "ANOMALY DETECTED" : "NORMAL ORDER"}</strong>
              </div>
              <p className="note">Unusual does not mean fraudulent — review flagged orders manually.</p>
              <DetailList
                items={[
                  ["Anomaly Score", Number(d.anomaly_score).toFixed(6)],
                  ["Anomaly Status", d.is_anomaly ? "Anomaly detected" : "Normal order"],
                  ["Model Version", d.model_version],
                ]}
              />
              <p className="note">Higher scores indicate more unusual orders.</p>
            </>
          )}
        </PredictionResult>
      </div>
    </>
  );
}
