import { useEffect, useState } from "react";
import api from "../api/axios";

const AdminLogs = () => {
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    api.get("/admin/logs")
      .then((res) => {
        setLogs(res.data.logs);
      })
      .catch((err) => {
        alert("Failed to load logs");
      });
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h2>Request Logs (Dataset)</h2>

      <table border="1" cellPadding="8" cellSpacing="0">
        <thead>
          <tr>
            <th>User</th>
            <th>Endpoint</th>
            <th>Method</th>
            <th>IP Address</th>
            <th>Status</th>
            <th>Time</th>
          </tr>
        </thead>

        <tbody>
          {logs.map((log) => (
            <tr key={log._id}>
              <td>{log.userId || "Anonymous"}</td>
              <td>{log.endpoint}</td>
              <td>{log.method}</td>
              <td>{log.ipAddress}</td>
              <td>{log.statusCode}</td>
              <td>{new Date(log.createdAt).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AdminLogs;
