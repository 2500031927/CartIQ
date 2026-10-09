import { useMemo, useState } from "react";
import { Loader2, Wand2 } from "lucide-react";
import FormInput from "./FormInput.jsx";

/**
 * Generic validated form. Field spec:
 * { name, label, type: "text" | "number" | "int", required, min, gt, hint, defaultValue }
 * Blank optional fields are omitted from the payload (the backend then uses its own defaults).
 */
export default function PredictionForm({ fields, sample, submitLabel, loading, onSubmit }) {
  const initial = useMemo(() => Object.fromEntries(fields.map((f) => [f.name, f.defaultValue ?? ""])), [fields]);
  const [values, setValues] = useState(initial);
  const [errors, setErrors] = useState({});

  const change = (name, value) => {
    setValues((p) => ({ ...p, [name]: value }));
    setErrors((p) => ({ ...p, [name]: undefined }));
  };

  function validate() {
    const errs = {};
    for (const f of fields) {
      const raw = String(values[f.name] ?? "").trim();
      if (raw === "") {
        if (f.required) errs[f.name] = "Required";
        continue;
      }
      if (f.type === "text") continue;
      const n = Number(raw);
      if (!Number.isFinite(n)) errs[f.name] = "Enter a valid number";
      else if (f.type === "int" && !Number.isInteger(n)) errs[f.name] = "Enter a whole number";
      else if (f.gt !== undefined && n <= f.gt) errs[f.name] = `Must be greater than ${f.gt}`;
      else if (f.min !== undefined && n < f.min) errs[f.name] = `Must be at least ${f.min}`;
    }
    return errs;
  }

  function buildPayload() {
    const payload = {};
    for (const f of fields) {
      const raw = String(values[f.name] ?? "").trim();
      if (raw === "") continue;
      payload[f.name] = f.type === "text" ? raw : Number(raw);
    }
    return payload;
  }

  function handleSubmit(e) {
    e.preventDefault();
    const errs = validate();
    setErrors(errs);
    if (Object.keys(errs).length === 0) onSubmit(buildPayload());
  }

  return (
    <form className="card form-card" onSubmit={handleSubmit} noValidate>
      <div className="form-grid">
        {fields.map((f) => (
          <FormInput key={f.name} field={f} value={values[f.name]} error={errors[f.name]} disabled={loading} onChange={change} />
        ))}
      </div>
      <div className="form-actions">
        {sample && (
          <button type="button" className="btn btn-ghost" disabled={loading} onClick={() => { setValues({ ...initial, ...sample }); setErrors({}); }}>
            <Wand2 size={16} /> Fill sample values
          </button>
        )}
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? <Loader2 size={18} className="spin" /> : null}
          {loading ? "Working…" : submitLabel}
        </button>
      </div>
    </form>
  );
}
