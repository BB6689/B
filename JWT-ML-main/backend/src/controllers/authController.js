import crypto   from "crypto";
import bcrypt    from "bcrypt";
import User      from "../models/User.js";
import IssuedToken from "../models/IssuedToken.js";
import { generateToken } from "../utils/token.js";

/**
 * POST /api/auth/register
 *
 * Security hardening applied:
 * - Role is never accepted from the request body.
 *   All new registrations are "user". Admin accounts are
 *   provisioned separately (e.g. seeder script or DB operation).
 * - Input sanitisation on name/email.
 */
export const register = async (req, res) => {
  try {
    const { name, email, password } = req.body;

    if (!name || !email || !password) {
      return res.status(400).json({ message: "All fields are required" });
    }

    if (password.length < 8) {
      return res.status(400).json({ message: "Password must be at least 8 characters" });
    }

    const existing = await User.findOne({ email: email.toLowerCase().trim() });
    if (existing) {
      return res.status(409).json({ message: "Email already registered" });
    }

    const passwordHash = await bcrypt.hash(password, 12); // increased from 10 to 12

    const user = await User.create({
      name:  name.trim(),
      email: email.toLowerCase().trim(),
      passwordHash,
      role:  "user",   // FIXED: role is never taken from request body
    });

    return res.status(201).json({
      message: "User registered successfully ✅",
      user: { id: user._id, name: user.name, email: user.email, role: user.role },
    });
  } catch (err) {
    return res.status(500).json({ message: "Server error", error: err.message });
  }
};

/**
 * POST /api/auth/login
 *
 * Issues JWT and records it in the IssuedToken registry (Rule 25).
 * The registry enables detectForgery to verify any arriving token
 * was actually minted here.
 *
 * Timing-safe credential check via bcrypt.compare (already constant-time).
 * Generic error message for both "user not found" and "wrong password"
 * to prevent user enumeration.
 */
export const login = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ message: "Email and password required" });
    }

    const user = await User.findOne({ email: email.toLowerCase().trim() });

    // Use a dummy compare to prevent timing attacks on non-existent users
    const dummyHash = "$2b$12$invalidhashfortimingprotection000000000000000000000000";
    const ok = user
      ? await bcrypt.compare(password, user.passwordHash)
      : await bcrypt.compare(password, dummyHash).then(() => false);

    if (!user || !ok) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    const token      = generateToken({ userId: user._id, role: user.role });
    const tokenHash  = crypto.createHash("sha256").update(token).digest("hex");

    // Register token in issuance registry for Rule 25 forgery detection
    await IssuedToken.create({ tokenHash, userId: user._id }).catch(() => {});
    // .catch avoids crashing login if the registry write fails (non-critical path)

    return res.status(200).json({
      message: "Login successful ✅",
      token,
      user: { id: user._id, name: user.name, email: user.email, role: user.role },
    });
  } catch (err) {
    return res.status(500).json({ message: "Server error", error: err.message });
  }
};
