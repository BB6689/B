import { BrowserRouter, Routes, Route } from "react-router-dom";
import Login from "./pages/Login";
import Register from "./pages/Register";
import UserDashboard from "./pages/UserDashboard";
import AdminDashboard from "./pages/AdminDashboard";
import ProtectedRoute from "./auth/ProtectedRoute";
import { AuthProvider } from "./auth/AuthContext";
import AdminLogs from "./pages/AdminLogs";
import AdminCharts from "./pages/AdminCharts";
import AdminRisk from "./pages/AdminRisk";





function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <UserDashboard />
              </ProtectedRoute>
            }
          />

          <Route
            path="/admin"
            element={
              <ProtectedRoute role="admin">  
                <AdminDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/logs"
            element={
              <ProtectedRoute>
                <AdminLogs />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/charts"
            element={
              <ProtectedRoute>
                <AdminCharts />
              </ProtectedRoute>
            }
          />

          <Route
            path="/admin/risk"
            element={
              <ProtectedRoute>
                <AdminRisk />
              </ProtectedRoute>
            }
          />

        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
