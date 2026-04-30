import RiskScore    from "../models/RiskScore.js";
import RevokedToken from "../models/RevokedToken.js";

/**
 * Risk Points Table
 *
 * Keys MUST match the `type` string emitted in Alert.create exactly.
 * Any mismatch silently falls through to the default (10 pts).
 *
 * Scoring philosophy:
 *   CRITICAL attacks (forgery, impossible travel, revoked reuse) → 60–80
 *   HIGH attacks (replay, privilege) → 50–60
 *   Behavioural signals → 15–30
 *   Low-confidence signals → 5–15
 */

const RISK_POINTS = {
  TOKEN_REPLAY:        60,
  RATE_ANOMALY:        20,
  PRIVILEGE_ABUSE:     50,
  DEVICE_ANOMALY:      15,
  IP_ANOMALY:          10,
  CONCURRENT_SESSION:  30,
  API_SCAN:            20,
  REVOKED_TOKEN_USE:   70,
  BEHAVIOR_ANOMALY:    25,   // covers signature + ML-detected anomalies
  IMPOSSIBLE_TRAVEL:   60,
  INVALID_TOKEN_FLOOD: 40,
  TOKEN_LIFETIME_ABUSE:15,
  LOGIN_TIME_ANOMALY:   5,
  FORGED_TOKEN:        80,   // Rule 25 — highest possible score
};

const calculateLevel = (score) => {
  if (score >= 80) return "CRITICAL";
  if (score >= 50) return "HIGH";
  if (score >= 25) return "MEDIUM";
  return "LOW";
};

export const updateRiskScore = async (userId, tokenHash, alertType) => {

  if (!userId) return;

  const points = RISK_POINTS[alertType] || 10;

  let risk = await RiskScore.findOne({ userId });

  if (!risk) {
    risk = await RiskScore.create({ userId, score: points });
  } else {
    risk.score       = Math.min(risk.score + points, 100);
    risk.lastUpdated = new Date();
  }

  risk.level = calculateLevel(risk.score);
  await risk.save();

  // Auto-revoke at HIGH or CRITICAL
  if ((risk.level === "HIGH" || risk.level === "CRITICAL") && tokenHash) {
    await RevokedToken.updateOne(
      { tokenHash },
      {
        $setOnInsert: {
          tokenHash,
          reason: `Auto-revoked — risk level ${risk.level} (score: ${risk.score})`,
        },
      },
      { upsert: true }
    );
    console.warn(`⚠️  User ${userId} reached ${risk.level} (${risk.score}pts) — token revoked`);
  }

};

/**
 * Risk Score Decay
 *
 * Only decays users idle for >30 min to avoid eroding active incident scores.
 * Decay rate scales inversely with severity:
 *   LOW    → -8/run  (recover fast — likely FP)
 *   MEDIUM → -5/run
 *   HIGH   → -3/run  (slow — operator should review first)
 *   CRITICAL → -2/run (very slow — requires explicit resolution)
 */
export const decayRiskScores = async () => {

  const thirtyMinsAgo = new Date(Date.now() - 1800000);

  const risks = await RiskScore.find({
    score:       { $gt: 0 },
    lastUpdated: { $lt: thirtyMinsAgo },
  });

  for (const risk of risks) {
    const decayAmount =
      risk.level === "CRITICAL" ? 2
      : risk.level === "HIGH"   ? 3
      : risk.level === "MEDIUM" ? 5
      : 8;

    risk.score       = Math.max(risk.score - decayAmount, 0);
    risk.level       = calculateLevel(risk.score);
    risk.lastUpdated = new Date();
    await risk.save();
  }

  if (risks.length > 0) {
    console.log(`🔄 Decayed risk scores for ${risks.length} idle users`);
  }

};
