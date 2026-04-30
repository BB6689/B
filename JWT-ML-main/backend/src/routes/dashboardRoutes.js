import express from "express";
import RequestLog from "../models/RequestLog.js";
import Alert from "../models/Alert.js";
import RiskScore from "../models/RiskScore.js";

const router = express.Router();

router.get("/", async (req, res) => {
  try {
    const [requestGroups, responseTimes, statusBuckets, alertTypes, riskScores, alerts] = await Promise.all([
      RequestLog.aggregate([
        {
          $group: {
            _id: { tokenHash: "$tokenHash", endpoint: "$endpoint", method: "$method" },
            count: { $sum: 1 }
          }
        },
        { $sort: { count: -1 } },
        { $limit: 20 }
      ]),
      RequestLog.aggregate([
        {
          $group: {
            _id: "$endpoint",
            avgMs: { $avg: "$responseTimeMs" },
            minMs: { $min: "$responseTimeMs" },
            maxMs: { $max: "$responseTimeMs" }
          }
        },
        { $sort: { avgMs: -1 } },
        { $limit: 20 }
      ]),
      RequestLog.aggregate([
        {
          $group: {
            _id: {
              $cond: [
                {
                  $and: [
                    { $gte: ["$statusCode", 200] },
                    { $lt: ["$statusCode", 300] }
                  ]
                },
                "Success",
                "Failure"
              ]
            },
            count: { $sum: 1 }
          }
        }
      ]),
      Alert.aggregate([
        { $group: { _id: "$type", count: { $sum: 1 } } },
        { $sort: { count: -1 } }
      ]),
      RiskScore.find().populate("userId", "email").sort({ score: -1 }).limit(20),
      Alert.find().sort({ createdAt: -1 }).limit(25)
    ]);

    res.json({
      requestGroups,
      responseTimes,
      statusBuckets,
      alertTypes,
      riskScores,
      alerts
    });
  } catch (err) {
    console.error("Dashboard route error:", err);
    res.status(500).json({ message: "Unable to load dashboard data", error: err.message });
  }
});

export default router;
