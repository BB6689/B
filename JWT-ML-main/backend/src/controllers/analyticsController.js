// backend/src/controllers/analyticsController.js
// Replaces the old stub analyticsController.js
// Each function feeds exactly one card in the HTML dashboard

import RequestLog from "../models/RequestLog.js";
import Alert      from "../models/Alert.js";
import RiskScore  from "../models/RiskScore.js";
import User       from "../models/User.js";

/* ─── Helper: parse date range from query params ─────────────────────────
   Frontend sends ?range=Last+7+Days  OR  ?from=2024-01-01&to=2024-01-07
   Returns a MongoDB $gte date filter                                      */
function getDateFilter(query) {
  const { range, from, to } = query;

  if (from && to) {
    return { $gte: new Date(from), $lte: new Date(to) };
  }

  const now = new Date();
  const days = {
    "Last 24 Hours": 1,
    "Last 7 Days":   7,
    "Last 14 Days":  14,
    "Last 30 Days":  30,
    "Last 90 Days":  90,
  };
  const d = days[range] ?? 7;
  const cutoff = new Date(now.getTime() - d * 24 * 60 * 60 * 1000);
  return { $gte: cutoff };
}


/* ══════════════════════════════════════════════════════════════════════
   CARD 1 — Total API Calls By Token
   Returns: [{ token: "abc123…", count: 42 }]
   Chart: Bar or Line
══════════════════════════════════════════════════════════════════════ */
export const getApiCallsByToken = async (req, res) => {
  try {
    const dateFilter = getDateFilter(req.query);

    const data = await RequestLog.aggregate([
      { $match: { createdAt: dateFilter, tokenHash: { $ne: null } } },
      {
        $group: {
          _id: "$tokenHash",
          count: { $sum: 1 },
        },
      },
      { $sort: { count: -1 } },
      { $limit: 20 },
      {
        $project: {
          _id: 0,
          token: { $substr: ["$_id", 0, 16] }, // show first 16 chars only
          fullToken: "$_id",
          count: 1,
        },
      },
    ]);

    res.json({
      labels:   data.map(d => d.token + "…"),
      values:   data.map(d => d.count),
      raw:      data,         // for table view
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


/* ══════════════════════════════════════════════════════════════════════
   CARD 2 — Success / Failure Rate
   Returns: { success: 340, failure: 60, rate: 85.0 }
   Chart: Doughnut / Pie
══════════════════════════════════════════════════════════════════════ */
export const getSuccessFailureRate = async (req, res) => {
  try {
    const dateFilter = getDateFilter(req.query);

    const data = await RequestLog.aggregate([
      { $match: { createdAt: dateFilter } },
      {
        $group: {
          _id: {
            $cond: [{ $lt: ["$statusCode", 400] }, "success", "failure"],
          },
          count: { $sum: 1 },
        },
      },
    ]);

    const success = data.find(d => d._id === "success")?.count ?? 0;
    const failure = data.find(d => d._id === "failure")?.count ?? 0;
    const total   = success + failure;

    res.json({
      labels: ["Success", "Failure"],
      values: [success, failure],
      rate:   total > 0 ? ((success / total) * 100).toFixed(1) : 0,
      total,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


/* ══════════════════════════════════════════════════════════════════════
   CARD 3 — Request Timeline (requests per hour/day)
   Returns: [{ label: "2024-01-01", count: 120 }]
   Chart: Line or Bar
══════════════════════════════════════════════════════════════════════ */
export const getRequestTimeline = async (req, res) => {
  try {
    const dateFilter = getDateFilter(req.query);
    const per = req.query.per ?? "Day"; // "Hour" | "Day" | "Week"

    // Build the date grouping expression
    const groupBy =
      per === "Hour"
        ? {
            year:  { $year: "$createdAt" },
            month: { $month: "$createdAt" },
            day:   { $dayOfMonth: "$createdAt" },
            hour:  { $hour: "$createdAt" },
          }
        : {
            year:  { $year: "$createdAt" },
            month: { $month: "$createdAt" },
            day:   { $dayOfMonth: "$createdAt" },
          };

    const data = await RequestLog.aggregate([
      { $match: { createdAt: dateFilter } },
      { $group: { _id: groupBy, count: { $sum: 1 } } },
      { $sort: { "_id.year": 1, "_id.month": 1, "_id.day": 1, "_id.hour": 1 } },
    ]);

    const labels = data.map(d => {
      const { year, month, day, hour } = d._id;
      const base = `${year}-${String(month).padStart(2,"0")}-${String(day).padStart(2,"0")}`;
      return per === "Hour" ? `${base} ${String(hour).padStart(2,"0")}:00` : base;
    });

    res.json({
      labels,
      values: data.map(d => d.count),
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


/* ══════════════════════════════════════════════════════════════════════
   CARD 4 — Alert Type Breakdown
   Returns: [{ type: "TOKEN_REPLAY", count: 14, severity: "HIGH" }]
   Chart: Bar or Pie
══════════════════════════════════════════════════════════════════════ */
export const getAlertTypeBreakdown = async (req, res) => {
  try {
    const dateFilter = getDateFilter(req.query);

    // Optional severity filter: ?severity=HIGH,CRITICAL
    const severityFilter = req.query.severity
      ? { severity: { $in: req.query.severity.split(",") } }
      : {};

    const data = await Alert.aggregate([
      { $match: { createdAt: dateFilter, ...severityFilter } },
      {
        $group: {
          _id:      "$type",
          count:    { $sum: 1 },
          severity: { $first: "$severity" },
        },
      },
      { $sort: { count: -1 } },
    ]);

    res.json({
      labels:   data.map(d => d._id),
      values:   data.map(d => d.count),
      severity: data.map(d => d.severity),
      raw:      data.map(d => ({ type: d._id, count: d.count, severity: d.severity })),
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


/* ══════════════════════════════════════════════════════════════════════
   CARD 5 — Risk Score By User
   Returns: [{ email: "user@x.com", score: 75, level: "HIGH" }]
   Chart: Bar / Line / Table
══════════════════════════════════════════════════════════════════════ */
export const getRiskScoreByUser = async (req, res) => {
  try {
    const data = await RiskScore.find()
      .populate("userId", "email name")
      .sort({ score: -1 })
      .limit(30);

    const formatted = data.map(r => ({
      userId:      r.userId?._id,
      email:       r.userId?.email ?? "Unknown",
      name:        r.userId?.name  ?? "Unknown",
      score:       r.score,
      level:       r.level,
      lastUpdated: r.lastUpdated,
    }));

    res.json({
      labels: formatted.map(d => d.email),
      values: formatted.map(d => d.score),
      levels: formatted.map(d => d.level),
      raw:    formatted,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


/* ══════════════════════════════════════════════════════════════════════
   CARD 6 — Recent Alerts Log (table)
   Returns: [{ type, severity, reason, actionTaken, isResolved, user, time }]
══════════════════════════════════════════════════════════════════════ */
export const getRecentAlertLogs = async (req, res) => {
  try {
    const dateFilter = getDateFilter(req.query);

    // Optional filters from query
    const match = { createdAt: dateFilter };
    if (req.query.severity) match.severity    = { $in: req.query.severity.split(",") };
    if (req.query.type)     match.type        = { $in: req.query.type.split(",") };
    if (req.query.resolved !== undefined)
                            match.isResolved  = req.query.resolved === "true";
    if (req.query.action)   match.actionTaken = req.query.action;

    const alerts = await Alert.find(match)
      .populate("userId", "email")
      .sort({ createdAt: -1 })
      .limit(100);

    const rows = alerts.map(a => ({
      type:        a.type,
      severity:    a.severity,
      reason:      a.reason,
      actionTaken: a.actionTaken,
      isResolved:  a.isResolved,
      user:        a.userId?.email ?? "Unknown",
      token:       a.tokenHash ? a.tokenHash.slice(0, 16) + "…" : "—",
      time:        a.createdAt,
    }));

    res.json({ rows, total: rows.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};


/* ══════════════════════════════════════════════════════════════════════
   CARD 7 — Detection Latency Stats
   Returns per-endpoint and overall latency percentiles from RequestLog.
   This directly corresponds to the paper's performance metric:
   "average detection latency of 9.7 milliseconds per request".

   Percentiles computed: p50 (median), p90, p99, avg, min, max.
   Also returns a time-series of avg latency per hour for trending.
══════════════════════════════════════════════════════════════════════ */
export const getDetectionLatency = async (req, res) => {
  try {
    const dateFilter = getDateFilter(req.query);

    const [overallStats, perEndpoint, hourlyTrend, alertLatency] = await Promise.all([

      // Overall latency distribution across all logged requests
      RequestLog.aggregate([
        {
          $match: {
            createdAt: dateFilter,
            detectionLatencyMs: { $ne: null, $gt: 0 },
          },
        },
        {
          $group: {
            _id: null,
            avg:   { $avg: "$detectionLatencyMs" },
            min:   { $min: "$detectionLatencyMs" },
            max:   { $max: "$detectionLatencyMs" },
            total: { $sum: 1 },
            // Collect all values so we can compute percentiles
            values: { $push: "$detectionLatencyMs" },
          },
        },
        {
          $project: {
            _id: 0,
            avg:   { $round: ["$avg", 2] },
            min:   1,
            max:   1,
            total: 1,
            values: 1,
          },
        },
      ]),

      // Per-endpoint avg latency (top 15 slowest)
      RequestLog.aggregate([
        {
          $match: {
            createdAt: dateFilter,
            detectionLatencyMs: { $ne: null, $gt: 0 },
          },
        },
        {
          $group: {
            _id:   "$endpoint",
            avg:   { $avg: "$detectionLatencyMs" },
            count: { $sum: 1 },
            max:   { $max: "$detectionLatencyMs" },
          },
        },
        { $sort: { avg: -1 } },
        { $limit: 15 },
        {
          $project: {
            _id: 0,
            endpoint: "$_id",
            avg:   { $round: ["$avg", 2] },
            max:   1,
            count: 1,
          },
        },
      ]),

      // Hourly trend — avg detection latency per hour
      RequestLog.aggregate([
        {
          $match: {
            createdAt: dateFilter,
            detectionLatencyMs: { $ne: null, $gt: 0 },
          },
        },
        {
          $group: {
            _id: {
              year:  { $year:        "$createdAt" },
              month: { $month:       "$createdAt" },
              day:   { $dayOfMonth:  "$createdAt" },
              hour:  { $hour:        "$createdAt" },
            },
            avg:   { $avg: "$detectionLatencyMs" },
            count: { $sum: 1 },
          },
        },
        { $sort: { "_id.year": 1, "_id.month": 1, "_id.day": 1, "_id.hour": 1 } },
        {
          $project: {
            _id: 0,
            label: {
              $concat: [
                { $toString: "$_id.year" }, "-",
                { $toString: "$_id.month" }, "-",
                { $toString: "$_id.day" }, " ",
                { $toString: "$_id.hour" }, ":00",
              ],
            },
            avg:   { $round: ["$avg", 2] },
            count: 1,
          },
        },
      ]),

      // Alert-specific latency — avg time from request arrival to alert creation
      // (stored as detectionLatencyMs on Alert documents)
      Alert.aggregate([
        {
          $match: {
            createdAt: dateFilter,
            detectionLatencyMs: { $ne: null },
          },
        },
        {
          $group: {
            _id:   "$type",
            avg:   { $avg: "$detectionLatencyMs" },
            count: { $sum: 1 },
            max:   { $max: "$detectionLatencyMs" },
            min:   { $min: "$detectionLatencyMs" },
          },
        },
        { $sort: { avg: -1 } },
        {
          $project: {
            _id: 0,
            alertType: "$_id",
            avg:   { $round: ["$avg", 2] },
            max:   1,
            min:   1,
            count: 1,
          },
        },
      ]),
    ]);

    // Compute percentiles from the collected values array (done in JS, not Mongo)
    let p50 = null, p90 = null, p99 = null;
    if (overallStats[0]?.values?.length) {
      const sorted = [...overallStats[0].values].sort((a, b) => a - b);
      const pct = (p) => sorted[Math.floor(sorted.length * p / 100)] ?? null;
      p50 = pct(50);
      p90 = pct(90);
      p99 = pct(99);
      delete overallStats[0].values; // don't send raw array to client
    }

    res.json({
      overall: overallStats[0]
        ? { ...overallStats[0], p50, p90, p99 }
        : { avg: null, min: null, max: null, p50: null, p90: null, p99: null, total: 0 },
      perEndpoint,
      hourlyTrend: {
        labels: hourlyTrend.map(d => d.label),
        values: hourlyTrend.map(d => d.avg),
        counts: hourlyTrend.map(d => d.count),
      },
      alertLatency,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
};
