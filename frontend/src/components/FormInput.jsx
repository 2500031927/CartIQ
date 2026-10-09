/** One labelled text input with hint / error text. `field` follows the spec used by PredictionForm. */
export default function FormInput({ field, value, error, disabled, onChange }) {
  const isText = field.type === "text";
  return (
    <label className={`field ${error ? "has-error" : ""}`}>
      <span className="field-label">
        {field.label}
        {!field.required && <small>optional</small>}
      </span>
      <input
        type="text"
        inputMode={isText ? "text" : "decimal"}
        value={value}
        onChange={(e) => onChange(field.name, e.target.value)}
        placeholder={field.placeholder ?? (isText ? "" : "0")}
        autoComplete="off"
        disabled={disabled}
      />
      {error ? <span className="field-error">{error}</span> : field.hint ? <span className="field-hint">{field.hint}</span> : null}
    </label>
  );
}
