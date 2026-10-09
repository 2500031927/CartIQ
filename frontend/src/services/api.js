const API_BASE_URL = "http://127.0.0.1:8000";

const CONNECTION_MESSAGE =
  "Unable to connect to CartIQ API. Make sure the FastAPI server is running on port 8000.";

export class ApiError extends Error {
  constructor(message, { status = null, details = [], network = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
    this.network = network;
  }
}

function describeFailure(data, status) {
  // Our backend: {"error":"validation_error","details":[{field,message}]}
  if (data && data.error === "validation_error" && Array.isArray(data.details)) {
    return {
      message: "Please correct the highlighted input values.",
      details: data.details.map((d) => (d.field ? `${d.field}: ${d.message}` : d.message)),
    };
  }
  if (data && typeof data.detail === "string") return { message: data.detail, details: [] };
  // FastAPI default validation shape: {"detail":[{loc,msg}]}
  if (data && Array.isArray(data.detail)) {
    return {
      message: "Please correct the highlighted input values.",
      details: data.detail.map((d) => `${(d.loc || []).filter((p) => p !== "body").join(".")}: ${d.msg}`),
    };
  }
  if (data && typeof data.error === "string") {
    return { message: data.detail ? `${data.error}: ${data.detail}` : data.error, details: [] };
  }
  return { message: `Request failed (HTTP ${status}).`, details: [] };
}

async function request(path, { method = "GET", body } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(CONNECTION_MESSAGE, { network: true });
  }

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      if (response.ok) throw new ApiError("The server returned an invalid (non-JSON) response.", { status: response.status });
    }
  }

  if (!response.ok) {
    const { message, details } = describeFailure(data, response.status);
    throw new ApiError(message, { status: response.status, details });
  }
  if (data === null || typeof data !== "object") {
    throw new ApiError("The server returned an empty or invalid response.", { status: response.status });
  }
  return data;
}

export const getHealth = () => request("/health");
export const getMetrics = () => request("/metrics");
export const predictBasketValue = (data) => request("/predict/basket-value", { method: "POST", body: data });
export const predictReorder = (data) => request("/predict/reorder", { method: "POST", body: data });
export const predictCustomerSegment = (data) => request("/predict/customer-segment", { method: "POST", body: data });
export const predictAnomaly = (data) => request("/predict/anomaly", { method: "POST", body: data });
export const reloadModels = () => request("/admin/reload-models", { method: "POST" });
