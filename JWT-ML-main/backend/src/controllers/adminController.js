import RequestLog from "../models/RequestLog.js";
import Alert from "../models/Alert.js";
import RevokedToken from "../models/RevokedToken.js";
import RiskScore from "../models/RiskScore.js";

export const getRiskScores = async (req, res) => {
  const scores = await RiskScore.find().populate("userId", "email");

  res.json({ scores });
};

export const getLogs = async (req, res) => {
  const logs = await RequestLog.find().sort({ createdAt: -1 }).limit(200);
  res.json({ logs });
};

export const getAlerts = async (req, res) => {
  const alerts = await Alert.find().sort({ createdAt: -1 }).limit(200);
  res.json({ alerts });
};

export const revokeToken = async (req, res) => {
  const { tokenHash, reason } = req.body;

  if (!tokenHash) {
    return res.status(400).json({ message: "tokenHash is required" });
  }

  await RevokedToken.updateOne(
    { tokenHash },
    { $set: { tokenHash, reason: reason || "Revoked by admin" } },
    { upsert: true }
  );

  res.json({ message: "Token revoked ✅" });
};
