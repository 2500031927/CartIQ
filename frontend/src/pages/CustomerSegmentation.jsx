import { Lightbulb, Users } from "lucide-react";
import PredictionForm from "../components/PredictionForm.jsx";
import PredictionResult, { DetailList } from "../components/PredictionResult.jsx";
import usePrediction from "../hooks/usePrediction.js";
import { predictCustomerSegment } from "../services/api.js";

const FIELDS = [
  { name: "customer_id", label: "Customer ID", type: "text", placeholder: "e.g. C-1001", hint: "Stored only as a salted hash" },
  { name: "recency_days", label: "Recency Days", type: "number", required: true, min: 0, hint: "Days since last order" },
  { name: "frequency", label: "Frequency", type: "int", required: true, min: 1, hint: "Number of orders" },
  { name: "monetary", label: "Monetary", type: "number", required: true, gt: 0, hint: "Total spend (greater than 0)" },
];

const SAMPLE = { recency_days: "10", frequency: "5", monetary: "500" };

export default function CustomerSegmentation() {
  const { state, run, retry } = usePrediction(predictCustomerSegment);
  return (
    <>
      <div className="page-intro">
        <h2>Customer Segmentation</h2>
        <p>Identify customer behavior segments using RFM characteristics.</p>
      </div>
      <div className="split">
        <PredictionForm fields={FIELDS} sample={SAMPLE} submitLabel="Analyze Customer" loading={state.status === "loading"} onSubmit={run} />
        <PredictionResult state={state} onRetry={retry} icon={Users} idleTitle="No analysis yet" idleText="Provide recency, frequency and monetary values to find the customer's segment.">
          {(d) => (
            <>
              <span className="result-label">Customer Segment</span>
              <div className="segment-name">{d.segment_name}</div>
              <span className="badge badge-blue">Cluster {d.cluster_id}</span>
              <div className="callout">
                <Lightbulb size={20} />
                <div>
                  <strong>Recommended action</strong>
                  <p>{d.recommended_action}</p>
                </div>
              </div>
              <DetailList
                items={[
                  ["Cluster ID", d.cluster_id],
                  ["Segment Name", d.segment_name],
                  ["Distance to Centroid", Number(d.distance_to_centroid).toFixed(4)],
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
