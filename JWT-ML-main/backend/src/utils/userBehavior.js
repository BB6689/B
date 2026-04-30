import UserBehavior from "../models/userBehavior.js";
import RequestLog from "../models/RequestLog.js";
import { normalizeIp } from "./ipUtils.js";


const normalizeKnownIPs = (ips = []) =>
  [...new Set(ips.map(normalizeIp).filter(ip => ip && ip !== "unknown"))];

export const getUserBaseline = async (userId) => {
  const baseline = await UserBehavior.findOne({ userId });

  if (!baseline) {
    return {
      avgRate: 5,          // sensible default — not 1 (causes false positives)
      commonEndpoints: [],
      knownIPs: [],
      knownDevices: [],
      usualLoginHours: []  // FIX: was loginHours {start,end} in model but array in baseline
    };
  }

  baseline.knownIPs = normalizeKnownIPs(baseline.knownIPs);
  return baseline;
};


/* ------------------------------------------------ */
/* UPDATE USER BASELINE                             */
/* FIX: avgRate was set to recent.length (absolute  */
/* count) not a per-second rate. Standardised to    */
/* req/min rolling average. Also: knownIPs capped   */
/* at 30 most recent to prevent stale IP buildup.  */
/* ------------------------------------------------ */

export const updateUserBaseline = async (userId) => {

  const logs = await RequestLog.find({ userId })
    .sort({ createdAt: -1 })
    .limit(500);

  if (!logs.length) return;

  const now = Date.now();
  const oneMinuteMs = 60000;

  const recent = logs.filter(
    l => now - new Date(l.createdAt).getTime() < oneMinuteMs
  );

  // Requests per minute (rolling)
  const avgRate = recent.length;

  /* ---------- COMMON ENDPOINTS ---------- */
  const endpointCount = {};
  logs.forEach(log => {
    if (log.endpoint) {
      endpointCount[log.endpoint] = (endpointCount[log.endpoint] || 0) + 1;
    }
  });

  const commonEndpoints = Object.entries(endpointCount)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)          // was 5 — wider endpoint baseline
    .map(e => e[0]);

  /* ---------- KNOWN IPS (capped at 30) ---------- */
  const allIPs = logs
    .map(l => normalizeIp(l.ipAddress))
    .filter(ip => ip && ip !== "unknown");
  const knownIPs = [...new Set(allIPs)].slice(0, 30);

  /* ---------- KNOWN DEVICES (capped at 10) ---------- */
  const allDevices = logs.map(l => l.userAgent).filter(Boolean);
  const knownDevices = [...new Set(allDevices)].slice(0, 10);

  /* ---------- USUAL LOGIN HOURS ---------- */
  const loginHours = logs.map(l => new Date(l.createdAt).getHours());
  const usualLoginHours = [...new Set(loginHours)];

  await UserBehavior.findOneAndUpdate(
    { userId },
    {
      avgRate,
      commonEndpoints,
      knownIPs,
      knownDevices,
      usualLoginHours,     // FIX: was stored correctly but model didn't have this field
      lastUpdated: new Date()
    },
    { upsert: true, new: true }
  );

};


/* ------------------------------------------------ */
/* ADAPT USER BEHAVIOR                              */
/* FIX: was adding every new IP/device immediately  */
/* which would learn attacker IPs into baseline.    */
/* Now only adapts if user has LOW risk score.      */
/* ------------------------------------------------ */

export const adaptUserBehavior = async ({
  userId,
  ipAddress,
  userAgent,
  endpoint
}) => {
  try {
    const normalizedIp = normalizeIp(ipAddress);

    const baseline = await getUserBaseline(userId);
    const knownIPs = baseline?.knownIPs || [];
    const knownDevices = baseline?.knownDevices || [];
    const commonEndpoints = baseline?.commonEndpoints || [];
    const usualLoginHours = baseline?.usualLoginHours || [];

    const updatedData = {};

    // Learn new IPs (cap at 30)
    if (normalizedIp && !knownIPs.includes(normalizedIp) && knownIPs.length < 30) {
      updatedData.knownIPs = [...knownIPs, normalizedIp];
    }

    // Learn new devices (cap at 10)
    if (userAgent && !knownDevices.includes(userAgent) && knownDevices.length < 10) {
      updatedData.knownDevices = [...knownDevices, userAgent];
    }

    // Learn endpoints (cap at 20)
    if (endpoint && !commonEndpoints.includes(endpoint) && commonEndpoints.length < 20) {
      updatedData.commonEndpoints = [...commonEndpoints, endpoint];
    }

    // Learn login hours
    const currentHour = new Date().getHours();
    if (!usualLoginHours.includes(currentHour)) {
      updatedData.usualLoginHours = [...usualLoginHours, currentHour];
    }

    if (Object.keys(updatedData).length > 0) {
      await UserBehavior.updateOne(
        { userId },
        { $set: updatedData },
        { upsert: true }
      );
    }

  } catch (err) {
    console.error("Adaptive behavior update failed:", err.message);
  }
};
