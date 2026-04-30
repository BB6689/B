import express from "express";
import { verifyJWT } from "../middleware/verifyJWT.js";
import { securityMonitor } from "../middleware/securityMonitor.js";
import { isAdmin } from "../middleware/isAdmin.js";

import {
  getLogs,
  getAlerts,
  revokeToken,
  getRiskScores
} from "../controllers/adminController.js";



const router = express.Router();

router.get("/logs", verifyJWT, securityMonitor, isAdmin, getLogs);
router.get("/alerts", verifyJWT, securityMonitor, isAdmin, getAlerts);
router.post("/revoke", verifyJWT, securityMonitor, isAdmin, revokeToken);
router.get("/risk-scores", verifyJWT, securityMonitor, isAdmin, getRiskScores);

export default router;
