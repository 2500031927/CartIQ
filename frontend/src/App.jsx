import { useCallback, useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import BasketValue from "./pages/BasketValue.jsx";
import ReorderPrediction from "./pages/ReorderPrediction.jsx";
import CustomerSegmentation from "./pages/CustomerSegmentation.jsx";
import AnomalyDetection from "./pages/AnomalyDetection.jsx";
import ModelMonitoring from "./pages/ModelMonitoring.jsx";
import { getHealth } from "./services/api.js";

const TITLES = {
  "/": "Dashboard",
  "/basket-value": "Basket Value",
  "/reorder": "Reorder Prediction",
  "/segmentation": "Customer Segmentation",
  "/anomaly": "Anomaly Detection",
  "/monitoring": "Model Monitoring",
};

export default function App() {
  const location = useLocation();
  const [health, setHealth] = useState({ status: "checking", data: null, error: null });

  const refreshHealth = useCallback(async () => {
    try {
      const data = await getHealth();
      setHealth({ status: "online", data, error: null });
    } catch (error) {
      setHealth({ status: "offline", data: null, error });
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const id = setInterval(refreshHealth, 15000);
    return () => clearInterval(id);
  }, [refreshHealth]);

  return (
    <Layout title={TITLES[location.pathname] ?? "CartIQ"} health={health}>
      <Routes>
        <Route path="/" element={<Dashboard health={health} />} />
        <Route path="/basket-value" element={<BasketValue />} />
        <Route path="/reorder" element={<ReorderPrediction />} />
        <Route path="/segmentation" element={<CustomerSegmentation />} />
        <Route path="/anomaly" element={<AnomalyDetection />} />
        <Route path="/monitoring" element={<ModelMonitoring health={health} onRefreshHealth={refreshHealth} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
