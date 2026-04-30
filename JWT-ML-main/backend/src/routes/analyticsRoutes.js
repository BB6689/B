// backend/src/routes/analyticsRoutes.js
import express from "express";
import { verifyJWT } from "../middleware/verifyJWT.js";
import { isAdmin } from "../middleware/isAdmin.js";
import {
  getApiCallsByToken,
  getSuccessFailureRate,
  getRequestTimeline,
  getAlertTypeBreakdown,
  getRiskScoreByUser,
  getRecentAlertLogs,
  getDetectionLatency,
} from "../controllers/analyticsController.js";

const router = express.Router();

// All analytics routes require admin auth
router.use(verifyJWT, isAdmin);

router.get("/api-calls",       getApiCallsByToken);    // Card 1: Total API Calls By Token
router.get("/success-rate",    getSuccessFailureRate);  // Card 2: Success/Failure Rate
router.get("/timeline",        getRequestTimeline);     // Card 3: Request Timeline
router.get("/alert-types",     getAlertTypeBreakdown);  // Card 4: Alert Type Breakdown
router.get("/risk-scores",     getRiskScoreByUser);     // Card 5: Risk Score By User
router.get("/alert-logs",      getRecentAlertLogs);     // Card 6: Recent Alerts Log
router.get("/detection-latency", getDetectionLatency);   // Card 7: Detection Latency Stats

export default router;
