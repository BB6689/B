import { useEffect, useState } from "react";
import api from "../api/axios";

const UserDashboard = () => {
  const [profile, setProfile] = useState(null);

  useEffect(() => {
    api.get("/user/profile").then((res) => {
      setProfile(res.data.user);
    });
  }, []);

  return (
    <div>
      <h2>User Dashboard</h2>

      {profile && (
        <>
          <p>Name: {profile.name}</p>
          <p>Email: {profile.email}</p>
        </>
      )}
    </div>
  );
};

export default UserDashboard;