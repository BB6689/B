/**
 * Rate Limiter — Login Endpoint Protection
 *
 * Pure in-memory implementation. No external dependencies.
 * Tracks failed attempts per IP over a sliding window.
 *
 * Why in-memory vs redis/express-rate-limit?
 * - No extra dependency needed for the project
 * - Survives for single-server deployments (sufficient here)
 * - For multi-server deployments, swap the store to a shared
 *   Redis client without changing the interface.
 *
 * Policy: 10 requests per 15-minute window per IP.
 * Exceeding the limit returns 429 with Retry-After header.
 */

import { normalizeIp } from "../utils/ipUtils.js";

const WINDOW_MS    = 15 * 60 * 1000;  // 15 minutes
const MAX_REQUESTS = 10;

// Map: ip → { count, windowStart }
const store = new Map();

// Cleanup stale entries every 30 minutes to prevent memory leak
setInterval(() => {
  const now = Date.now();
  for (const [ip, entry] of store.entries()) {
    if (now - entry.windowStart > WINDOW_MS) {
      store.delete(ip);
    }
  }
}, 30 * 60 * 1000);

export const loginRateLimiter = (req, res, next) => {
  const ip  = normalizeIp(
    req.headers["x-forwarded-for"]?.split(",")[0] ||
    req.ip ||
    "unknown"
  );

  const now = Date.now();

  let entry = store.get(ip);

  if (!entry || now - entry.windowStart > WINDOW_MS) {
    // New window
    entry = { count: 1, windowStart: now };
    store.set(ip, entry);
    return next();
  }

  entry.count++;

  if (entry.count > MAX_REQUESTS) {
    const retryAfterSec = Math.ceil((WINDOW_MS - (now - entry.windowStart)) / 1000);
    res.set("Retry-After", String(retryAfterSec));
    return res.status(429).json({
      message: `Too many login attempts. Retry after ${retryAfterSec}s.`,
    });
  }

  next();
};
