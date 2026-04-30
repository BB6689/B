import mongoose from "mongoose";

/* FIX: added tokenValid field (was missing from schema
   even though securityMonitor set it in context).
   Added indexes for fast detection queries. */

const requestLogSchema = new mongoose.Schema(
  {
    userId:        { type: mongoose.Schema.Types.ObjectId, ref: "User" },
    tokenHash:     { type: String },
    ipAddress:     { type: String },
    userAgent:     { type: String },
    endpoint:      { type: String },
    method:        { type: String },
    statusCode:    { type: Number },
    responseTimeMs:{ type: Number },
    tokenValid:        { type: Boolean, default: true },  // FIX: was missing from schema
    detectionLatencyMs:{ type: Number, default: null }    // ms from log-write to detection complete
  },
  { timestamps: true }
);

// Indexes for every detection query pattern
requestLogSchema.index({ tokenHash: 1, createdAt: -1 });
requestLogSchema.index({ userId: 1,    createdAt: -1 });
requestLogSchema.index({ ipAddress: 1, tokenValid: 1, createdAt: -1 });

// TTL: auto-delete logs older than 30 days
requestLogSchema.index({ createdAt: 1 }, { expireAfterSeconds: 2592000 });

export default mongoose.model("RequestLog", requestLogSchema);
