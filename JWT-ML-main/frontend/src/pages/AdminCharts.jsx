import { useEffect, useState } from "react";
import api from "../api/axios";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid
} from "recharts";

const AdminCharts = () => {
  const [alertData, setAlertData] = useState([]);

  useEffect(() => {
    api.get("/admin/stats/alerts").then((res) => {
      const formatted = res.data.alerts.map(a => ({
        type: a._id,
        count: a.count
      }));
      setAlertData(formatted);
    });
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h2>Alert Statistics</h2>

      <BarChart width={600} height={300} data={alertData}>
        <CartesianGrid />
        <XAxis dataKey="type" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="count" />
      </BarChart>
    </div>
  );
};

export default AdminCharts;
