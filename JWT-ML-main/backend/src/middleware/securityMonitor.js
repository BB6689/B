import RequestLog from "../models/RequestLog.js";
import { normalizeIp } from "../utils/ipUtils.js";
import { processSecurityEvent } from "../utils/securityPipeline.js";

/**
 * securityMonitor middleware
 *
 * Calls next() immediately — detection NEVER adds latency to the API response.
 * Detection runs inside setImmediate (next event loop tick) after the response
 * is handed to the client.
 *
 * Timing measurement:
 *   - requestArrivalTime: captured at the moment the middleware runs (before next())
 *   - detectionLatencyMs: time from requestArrivalTime to the moment all detection
 *     checks complete. Stored on the RequestLog document for analytics.
 *   - The paper reports 9.7 ms average latency. This measurement lets us verify
 *     and display that figure in the admin dashboard.
 */
export const securityMonitor = async (req, res, next) => {

  // Capture arrival time BEFORE calling next() — this is the reference point
  const requestArrivalTime = Date.now();

  next();  // respond to client first, always

  setImmediate(async () => {
    let logDoc = null;

    try {
      const ipAddress = normalizeIp(
        req.headers["x-forwarded-for"]?.split(",")[0] ||
        req.ip ||
        req.socket?.remoteAddress ||
        "unknown"
      );

      const context = {
        userId:            req.user?.id  || null,
        tokenHash:         req.tokenHash || null,
        ipAddress,
        userAgent:         req.headers["user-agent"] || "unknown",
        endpoint:          req.path,
        method:            req.method,
        role:              req.user?.role || null,
        tokenValid:        req.tokenValid ?? true,
        timestamp:         new Date(requestArrivalTime),
        requestArrivalTime,   // passed into context so createAlert can record latency
      };

      // Write the log first — get the doc back so we can update latencyMs after
      logDoc = await RequestLog.create({
        userId:     context.userId,
        tokenHash:  context.tokenHash,
        ipAddress:  context.ipAddress,
        userAgent:  context.userAgent,
        endpoint:   context.endpoint,
        method:     context.method,
        tokenValid: context.tokenValid,
      });

      // Run all detection rules + ML
      await processSecurityEvent(context);

      // Final latency: request arrival → all detection checks complete
      const detectionLatencyMs = Date.now() - requestArrivalTime;

      // Patch the log document with the measured latency
      await RequestLog.updateOne(
        { _id: logDoc._id },
        { $set: { detectionLatencyMs } }
      );

      if (detectionLatencyMs > 500) {
        console.warn(`⚠️  Slow detection: ${detectionLatencyMs}ms [${context.method} ${context.endpoint}]`);
      } else {
        console.log(`⏱  Detection: ${detectionLatencyMs}ms [${context.method} ${context.endpoint}]`);
      }

    } catch (err) {
      console.error("❌ Security monitor error:", err.message);
    }
  });

};
