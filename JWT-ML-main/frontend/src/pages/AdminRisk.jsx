import { useEffect, useState } from "react";
import api from "../api/axios";

const AdminRisk = () => {
  const [scores, setScores] = useState([]);

  useEffect(() => {
    api.get("/admin/risk-scores").then((res) => {
      setScores(res.data.scores);
    });
  }, []);

  return (
    <div>
      <h2>User Risk Scores</h2>

      {scores.map((s) => (
        <div key={s._id}>
          <b>{s.userId?.email}</b>
          <p>Score: {s.score}</p>
          <p>Level: {s.level}</p>
          <hr />
        </div>
      ))}
    </div>
  );
};

export default AdminRisk;