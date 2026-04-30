import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import api from "../api/axios";

const AdminDashboard = () => {
  const [alerts, setAlerts] = useState([]);

  const loadAlerts = () => {
    api.get("/admin/alerts").then((res) => {
      setAlerts(res.data.alerts);
    });
  };
  <Link to="/admin/risk">User Risk Scores</Link>

  useEffect(() => {
    loadAlerts();
  }, []);

  const revokeToken = async (tokenHash) => {
    try {
      await api.post("/admin/revoke", {
        tokenHash,
        reason: "Revoked by admin from dashboard",
      });

      alert("Token revoked successfully");
      loadAlerts(); // refresh list
    } catch (err) {
      alert("Failed to revoke token");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Admin Alerts</h2>
        <div style={{ marginBottom: "15px" }}>
          <Link to="/admin/charts" style={{ marginRight: "15px" }}>
            View Charts
          </Link>

          <Link to="/admin/logs" style={{ marginRight: "15px" }}>
            View Logs
          </Link>

          <Link to="/admin/risk">
            User Risk Scores
          </Link>
        </div>

        <hr />
      {alerts.length === 0 && <p>No alerts found</p>}

      {alerts.map((a) => (
        <div
          key={a._id}
          style={{
            border: "1px solid #ccc",
            padding: "10px",
            marginBottom: "10px",
          }}
        >
          <b>{a.type}</b> ({a.severity})
          <p>{a.reason}</p>

          <small>Token: {a.tokenHash.slice(0, 20)}...</small>
          <br />

          {a.actionTaken === "REVOKED" ? (
            <span style={{ color: "red" }}>Token already revoked</span>
          ) : (
            <button onClick={() => revokeToken(a.tokenHash)}>
              Revoke Token
            </button>
          )}
        </div>
      ))}
    </div>
  );
};

export default AdminDashboard;
