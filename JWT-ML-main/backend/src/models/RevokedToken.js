import mongoose from "mongoose";

const revokedTokenSchema = new mongoose.Schema(
  {
    tokenHash: { type: String, required: true, unique: true },
    reason: { type: String, default: "Revoked by system/admin" },
  },
  { timestamps: true }
);

export default mongoose.model("RevokedToken", revokedTokenSchema);
