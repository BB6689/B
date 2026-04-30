import { runSecurityChecks }  from "./detectionEngine.js";
import { updateUserBaseline, adaptUserBehavior } from "./userBehavior.js";
import RiskScore from "../models/RiskScore.js";

/**
 * processSecurityEvent
 *
 * Orchestrates the full security pipeline for every request:
 *  1. Run detection rules + ML scoring (tiered, parallel where safe)
 *  2. Risk-gated behaviour adaptation (never learn attacker patterns)
 *  3. Fire-and-forget baseline update from logs
 *
 * The ML Isolation Forest is invoked inside runSecurityChecks (Tier 4),
 * so it is already integrated — nothing extra needed here.
 */
export const processSecurityEvent = async (context) => {

  // ── 1. Detection (rules + ML) ──────────────────────────────────────
  await runSecurityChecks(context);

  // ── 2. Risk-gated behaviour adaptation ────────────────────────────
  // Only learn new IPs/devices/endpoints into the trusted baseline when
  // the user's risk is LOW or MEDIUM. Prevents attacker IPs from being
  // absorbed into the allowlist after a single successful session.
  if (context.userId) {
    const riskDoc  = await RiskScore.findOne({ userId: context.userId });
    const riskLevel = riskDoc?.level || "LOW";

    if (riskLevel === "LOW" || riskLevel === "MEDIUM") {
      await adaptUserBehavior(context);
    } else {
      console.log(`⛔ Skipping baseline adaptation — user risk: ${riskLevel}`);
    }
  }

  // ── 3. Baseline refresh (fire-and-forget) ─────────────────────────
  if (context.userId) {
    updateUserBaseline(context.userId).catch(err =>
      console.error("Baseline update error:", err.message)
    );
  }

};
