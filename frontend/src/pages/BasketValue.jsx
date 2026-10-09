import { ShoppingBasket } from "lucide-react";
import PredictionForm from "../components/PredictionForm.jsx";
import PredictionResult, { DetailList } from "../components/PredictionResult.jsx";
import usePrediction from "../hooks/usePrediction.js";
import { predictBasketValue } from "../services/api.js";

const FIELDS = [
  { name: "customer_id", label: "Customer ID", type: "text", placeholder: "e.g. C-1001", hint: "Stored only as a salted hash" },
  { name: "prior_orders", label: "Prior Orders", type: "int", required: true, min: 1 },
  { name: "prior_total_spend", label: "Prior Total Spend", type: "number", required: true, min: 0 },
  { name: "prior_mean_value", label: "Prior Mean Value", type: "number", required: true, min: 0 },
  { name: "prior_median_value", label: "Prior Median Value", type: "number", required: true, min: 0 },
  { name: "prior_max_value", label: "Prior Max Value", type: "number", required: true, min: 0 },
  { name: "prior_std_value", label: "Prior Standard Deviation", type: "number", min: 0, hint: "Defaults to 0 if blank" },
  { name: "last_order_value", label: "Last Order Value", type: "number", required: true, min: 0 },
  { name: "last_order_lines", label: "Last Order Lines", type: "int", required: true, min: 1 },
  { name: "last_order_items", label: "Last Order Items", type: "int", required: true, min: 1 },
  { name: "tenure_days", label: "Tenure Days", type: "number", required: true, min: 0, hint: "First order to latest order" },
  { name: "avg_gap_days", label: "Average Gap Days", type: "number", min: 0, hint: "Leave blank with only one prior order" },
  { name: "country", label: "Country", type: "text", defaultValue: "United Kingdom" },
];

const SAMPLE = {
  prior_orders: "3", prior_total_spend: "300", prior_mean_value: "100", prior_median_value: "90",
  prior_max_value: "150", prior_std_value: "20", last_order_value: "95", last_order_lines: "8",
  last_order_items: "30", tenure_days: "120", avg_gap_days: "40",
};

export default function BasketValue() {
  const { state, run, retry } = usePrediction(predictBasketValue);
  return (
    <>
      <div className="page-intro">
        <h2>Basket Value Prediction</h2>
        <p>Predict the expected value of a customer's next basket.</p>
      </div>
      <div className="split">
        <PredictionForm fields={FIELDS} sample={SAMPLE} submitLabel="Predict Basket Value" loading={state.status === "loading"} onSubmit={run} />
        <PredictionResult state={state} onRetry={retry} icon={ShoppingBasket} idleTitle="No prediction yet" idleText="Fill in the customer's order history and run the model.">
          {(d) => (
            <>
              <span className="result-label">Predicted Basket Value</span>
              <div className="hero-value">{Number(d.predicted_basket_value).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
              <p className="note">{d.currency_note}</p>
              <DetailList items={[["Model Name", d.model_name], ["Model Version", d.model_version]]} />
            </>
          )}
        </PredictionResult>
      </div>
    </>
  );
}
