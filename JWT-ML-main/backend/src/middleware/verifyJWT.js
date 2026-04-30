import jwt    from "jsonwebtoken";
import crypto from "crypto";
import RevokedToken from "../models/RevokedToken.js";
import RequestLog   from "../models/RequestLog.js";
import { normalizeIp } from "../utils/ipUtils.js";

/**
 * verifyJWT middleware
 *
 * Fixes & hardening applied:
 * 1. Algorithm guard: reject tokens with alg=none or any non-HS256 algorithm.
 *    This blocks algorithm-confusion attacks where an attacker strips the
 *    signature and sets alg to "none" — the token passes naive verify() calls.
 * 2. tokenValid=false is logged to RequestLog for INVALID_TOKEN_FLOOD detection.
 * 3. ip is normalised via ipUtils before storage.
 */
export const verifyJWT = async (req, res, next) => {
  try {
    const auth = req.headers.authorization;

    if (!auth || !auth.startsWith("Bearer ")) {
      return res.status(401).json({ message: "No token provided" });
    }

    const token     = auth.split(" ")[1];
    const tokenHash = crypto.createHash("sha256").update(token).digest("hex");

    // ── Algorithm guard (before signature verify) ──────────────────────
    // Decode header without verifying to check the alg claim.
    // If alg is "none" or something other than what we expect, reject immediately.
    const headerB64 = token.split(".")[0];
    let headerAlg   = null;
    try {
      const headerJson = JSON.parse(Buffer.from(headerB64, "base64url").toString());
      headerAlg = headerJson.alg?.toLowerCase();
    } catch {
      // malformed header — will fail jwt.verify below
    }

    const ALLOWED_ALGORITHMS = ["hs256", "hs384", "hs512"];
    if (headerAlg && !ALLOWED_ALGORITHMS.includes(headerAlg)) {
      await logRejected(req, tokenHash, "Algorithm not permitted");
      return res.status(401).json({ message: "Invalid token algorithm" });
    }

    // ── Revocation check ───────────────────────────────────────────────
    const revoked = await RevokedToken.findOne({ tokenHash });
    if (revoked) {
      await logRejected(req, tokenHash, "Revoked token reuse attempt");
      return res.status(401).json({ message: "Token revoked. Please login again." });
    }

    // ── Signature verification ─────────────────────────────────────────
    const decoded = jwt.verify(token, process.env.JWT_SECRET, {
      algorithms: ["HS256"],   // explicit algorithm whitelist
    });

    req.user       = { id: decoded.userId, role: decoded.role };
    req.token      = token;
    req.tokenHash  = tokenHash;
    req.tokenValid = true;

    next();

  } catch (err) {

    const rawToken  = req.headers.authorization?.split(" ")[1] || "";
    const tokenHash = rawToken
      ? crypto.createHash("sha256").update(rawToken).digest("hex")
      : null;

    await logRejected(req, tokenHash, err.message);

    return res.status(401).json({ message: "Invalid or expired token" });
  }
};

async function logRejected(req, tokenHash, reason) {
  await RequestLog.create({
    ipAddress:  normalizeIp(req.ip || req.socket?.remoteAddress),
    userAgent:  req.headers["user-agent"],
    endpoint:   req.path,
    method:     req.method,
    tokenHash:  tokenHash || null,
    tokenValid: false,
    statusCode: 401,
  }).catch(() => {});
}
