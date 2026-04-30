import dotenv from "dotenv";
dotenv.config();

import app from "./app.js";
import { connectDB } from "./config/db.js";
import { initForest } from "./utils/isolationForest.js";

const PORT = process.env.PORT || 5000;

connectDB();

// 🔧 FIX #1: Restore ML model from disk on startup
initForest().catch(err => console.error("Forest init error:", err));

app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});

import cron from "node-cron";
import { decayRiskScores } from "./utils/riskEngine.js";

cron.schedule("0 * * * *", async () => {

  console.log("Running risk score decay...");

  await decayRiskScores();

});
