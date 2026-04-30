import mongoose from "mongoose";

/* FIX: original model had loginHours: {start, end}
   but updateUserBaseline stored usualLoginHours as an array.
   Unified to usualLoginHours array across model + code. */

const userBehaviorSchema = new mongoose.Schema({

  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", unique: true },

  avgRate: { type: Number, default: 5 },     // req/min baseline (default 5, not 1)

  usualLoginHours: { type: [Number], default: [] },   // array of hours [0-23]

  commonEndpoints: { type: [String], default: [] },

  knownIPs: { type: [String], default: [] },

  knownDevices: { type: [String], default: [] },

  lastUpdated: { type: Date, default: Date.now }

});

export default mongoose.model("UserBehavior", userBehaviorSchema);
