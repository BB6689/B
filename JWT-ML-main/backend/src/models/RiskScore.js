import mongoose from "mongoose";

const riskScoreSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", unique: true },
    score: { type: Number, default: 0 },
    level: {
      type: String,
      enum: ["LOW", "MEDIUM", "HIGH","CRITICAL"],
      default: "LOW",
    },
    lastUpdated: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

export default mongoose.model("RiskScore", riskScoreSchema);