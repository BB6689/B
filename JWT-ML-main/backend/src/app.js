import express from "express";
import cors    from "cors";

import authRoutes      from "./routes/authRoutes.js";
import dashboardRoutes from "./routes/dashboardRoutes.js";
import analyticsRoutes from "./routes/analyticsRoutes.js";
import userRoutes      from "./routes/userRoutes.js";
import adminRoutes     from "./routes/adminRoutes.js";

import { securityMonitor } from "./middleware/securityMonitor.js";
import { verifyJWT }       from "./middleware/verifyJWT.js";
import { isAdmin }         from "./middleware/isAdmin.js";
import { loginRateLimiter } from "./middleware/rateLimiter.js";

/* ── Startup guard ────────────────────────────────────────────────── */
if (!process.env.JWT_SECRET) {
  console.error("❌ FATAL: JWT_SECRET environment variable is not set. Refusing to start.");
  process.exit(1);
}
if (process.env.JWT_SECRET.length < 32) {
  console.error("❌ FATAL: JWT_SECRET is too short (minimum 32 characters). Refusing to start.");
  process.exit(1);
}

const app = express();

// Trust the first proxy (needed for correct IP extraction behind nginx/load balancers)
app.set("trust proxy", 1);

app.use(cors());
app.use(express.json());

/* ── Public routes ────────────────────────────────────────────────── */
app.use("/api/auth", loginRateLimiter, authRoutes);

/* ── Admin-protected routes ───────────────────────────────────────── */
// FIXED: dashboard and analytics were unprotected — now require admin JWT
app.use("/api/dashboard", verifyJWT, isAdmin, dashboardRoutes);
app.use("/api/analytics", verifyJWT, isAdmin, analyticsRoutes);

/* ── User-protected routes ────────────────────────────────────────── */
app.use("/api/user",  verifyJWT, securityMonitor, userRoutes);
app.use("/api/admin", verifyJWT, securityMonitor, isAdmin, adminRoutes);

/* ── Health check ─────────────────────────────────────────────────── */
app.get("/", (req, res) => {
  res.json({ message: "JWT Abuse Detection Backend Running ✅" });
});

app.get("/api/health", (req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

export default app;
