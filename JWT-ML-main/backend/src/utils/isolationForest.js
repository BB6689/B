/**
 * Isolation Forest — Unsupervised Anomaly Detection
 *
 * Why Isolation Forest for this project?
 * ─────────────────────────────────────
 * • No labelled dataset required. It learns purely from live traffic.
 * • Anomalies are isolated faster (shorter paths in trees) than normal
 *   points. This maps perfectly to JWT abuse: an attacker's request
 *   profile is rare and sits far from the dense cluster of legitimate
 *   traffic.
 * • O(n log n) training, O(log n) scoring — fast enough for inline use.
 * • Zero external dependencies — pure JS implementation.
 *
 * How it works here:
 * ─────────────────
 * Every request is converted to a numeric feature vector:
 *   [ requestsPerMin, uniqueEndpointsPerMin, hourOfDay,
 *     ipChangeFlag, uaChangeFlag, tokenAgeMinutes ]
 *
 * The forest is built lazily from the first N=256 samples, then
 * re-trains every 500 requests. Anomaly scores (0–1) feed the
 * risk engine as a continuous signal — they never replace the
 * signature rules, they *augment* them.
 *
 * An anomaly score > 0.72 raises a BEHAVIOR_ANOMALY alert.
 * This threshold is conservative to match the paper's goal of
 * minimising false negatives over false positives.
 */

import RequestLog from "../models/RequestLog.js";
import { normalizeIp } from "./ipUtils.js";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FOREST_PERSIST_PATH = path.join(__dirname, "..", "..", "data", "forest-state.json");

/* ─────────────────────────────────────────────────────────────
   ISOLATION TREE
───────────────────────────────────────────────────────────── */

class IsolationTree {
  constructor(data, currentDepth = 0, maxDepth = 10) {
    this.size = data.length;

    if (data.length <= 1 || currentDepth >= maxDepth) {
      this.isLeaf = true;
      return;
    }

    this.isLeaf = false;

    // Pick a random feature dimension
    const numFeatures = data[0].length;
    this.splitDim = Math.floor(Math.random() * numFeatures);

    const values = data.map(d => d[this.splitDim]);
    const min = Math.min(...values);
    const max = Math.max(...values);

    if (min === max) {
      this.isLeaf = true;
      return;
    }

    // Pick a random split point between min and max
    this.splitVal = min + Math.random() * (max - min);

    const left  = data.filter(d => d[this.splitDim] < this.splitVal);
    const right = data.filter(d => d[this.splitDim] >= this.splitVal);

    this.left  = new IsolationTree(left,  currentDepth + 1, maxDepth);
    this.right = new IsolationTree(right, currentDepth + 1, maxDepth);
  }

  pathLength(point, currentLength = 0) {
    if (this.isLeaf) {
      return currentLength + averagePathLength(this.size);
    }
    if (point[this.splitDim] < this.splitVal) {
      return this.left.pathLength(point, currentLength + 1);
    }
    return this.right.pathLength(point, currentLength + 1);
  }
}

/* Expected path length for a BST with n nodes (used for normalisation) */
function averagePathLength(n) {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n);
}

/* ─────────────────────────────────────────────────────────────
   ISOLATION FOREST
───────────────────────────────────────────────────────────── */

class IsolationForest {
  constructor({ numTrees = 50, sampleSize = 256 } = {}) {
    this.numTrees   = numTrees;
    this.sampleSize = sampleSize;
    this.trees      = [];
    this.trained    = false;
  }

  /**
   * Train on a 2-D array of numeric feature vectors.
   * Called internally — not exposed to callers.
   */
  train(data) {
    if (data.length < 8) return; // not enough data to build meaningful trees

    this.trees = [];
    const n = Math.min(data.length, this.sampleSize);

    for (let i = 0; i < this.numTrees; i++) {
      // Random subsample without replacement
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      const sample   = shuffled.slice(0, n);
      this.trees.push(new IsolationTree(sample));
    }

    this.sampleSize_actual = n;
    this.trained = true;
    console.log(`🌲 Isolation Forest trained on ${n} samples with ${this.numTrees} trees`);
  }

  /**
   * Score a single feature vector.
   * Returns anomaly score in [0, 1].
   * Scores close to 1.0 = highly anomalous.
   * Scores close to 0.5 = indeterminate.
   * Scores close to 0.0 = very normal.
   */
  score(point) {
    if (!this.trained || this.trees.length === 0) return 0.5; // unknown

    const avgPathLen =
      this.trees.reduce((sum, tree) => sum + tree.pathLength(point), 0) /
      this.trees.length;

    const c = averagePathLength(this.sampleSize_actual);
    if (c === 0) return 0.5;

    return Math.pow(2, -(avgPathLen / c));
  }
}

/* ─────────────────────────────────────────────────────────────
   SINGLETON + AUTO-TRAIN LIFECYCLE
───────────────────────────────────────────────────────────── */

const forest = new IsolationForest({ numTrees: 50, sampleSize: 256 });

let requestsSinceLastTrain = 0;
const RETRAIN_INTERVAL     = 500; // retrain every 500 requests
const MIN_SAMPLES_TO_TRAIN = 50;  // don't train until we have enough data

/**
 * Build the feature vector from live MongoDB logs for a given user.
 *
 * Features:
 *  [0] requestsPerMin       — volume signal
 *  [1] uniqueEndpointsPerMin — scanning signal
 *  [2] hourOfDay             — temporal signal (0–23)
 *  [3] ipChangeFlag          — 1 if IP differs from previous log
 *  [4] uaChangeFlag          — 1 if User-Agent differs from previous log
 *  [5] tokenAgeMinutes       — how old is this token (0–720 capped)
 */
async function buildFeatureVector(userId, tokenHash, ipAddress, userAgent) {
  const oneMinAgo  = new Date(Date.now() - 60000);
  const fiveMinAgo = new Date(Date.now() - 300000);

  const [recentLogs, prevLog, firstTokenLog] = await Promise.all([
    RequestLog.find({ userId, createdAt: { $gte: oneMinAgo } })
      .select("endpoint")
      .lean(),
    RequestLog.findOne({ userId })
      .sort({ createdAt: -1 })
      .select("ipAddress userAgent")
      .lean(),
    RequestLog.findOne({ tokenHash })
      .sort({ createdAt: 1 })
      .select("createdAt")
      .lean(),
  ]);

  const requestsPerMin        = recentLogs.length;
  const uniqueEndpointsPerMin = new Set(recentLogs.map(l => l.endpoint)).size;
  const hourOfDay             = new Date().getHours();

  const normalizedIp = normalizeIp(ipAddress);
  const ipChangeFlag = prevLog && normalizeIp(prevLog.ipAddress) !== normalizedIp ? 1 : 0;
  const uaChangeFlag = prevLog && prevLog.userAgent !== userAgent ? 1 : 0;

  const tokenAgeMinutes = firstTokenLog
    ? Math.min((Date.now() - new Date(firstTokenLog.createdAt).getTime()) / 60000, 720)
    : 0;

  return [
    requestsPerMin,
    uniqueEndpointsPerMin,
    hourOfDay,
    ipChangeFlag,
    uaChangeFlag,
    tokenAgeMinutes,
  ];
}

/**
 * Fetch training data from RequestLog — O(n log n) not O(n²).
 * Pre-sorts by userId + createdAt (desc), then uses sliding window.
 * Eliminates nested loops that scanned userHistory for each log.
 */
async function fetchTrainingData() {
  const logs = await RequestLog.find({ userId: { $ne: null } })
    .sort({ userId: 1, createdAt: -1 })
    .limit(2000)
    .select("userId tokenHash ipAddress userAgent endpoint createdAt")
    .lean();

  if (logs.length < MIN_SAMPLES_TO_TRAIN) return null;

  const vectors = [];
  let currentUserId = null;
  let userWindow = [];

  for (const log of logs) {
    const uid = String(log.userId);
    const logTime = new Date(log.createdAt).getTime();

    if (uid !== currentUserId) {
      currentUserId = uid;
      userWindow = [];
    }

    userWindow.push(log);
    userWindow = userWindow.filter(
      l => logTime - new Date(l.createdAt).getTime() <= 60000
    );

    const rateWindow = userWindow.filter(
      l => l !== log && logTime - new Date(l.createdAt).getTime() < 60000
    );

    const reqPerMin     = rateWindow.length + 1;
    const uniqueEp      = new Set([log.endpoint, ...rateWindow.map(l => l.endpoint)]).size;
    const hour          = new Date(log.createdAt).getHours();

    const prevLog = userWindow.length > 1 ? userWindow[1] : null;
    const ipChange = prevLog && normalizeIp(prevLog.ipAddress) !== normalizeIp(log.ipAddress) ? 1 : 0;
    const uaChange = prevLog && prevLog.userAgent !== log.userAgent ? 1 : 0;

    const firstTokenLog = userWindow
      .filter(l => l.tokenHash === log.tokenHash)
      .sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt))[0];

    const tokenAgeMin = firstTokenLog
      ? Math.min((logTime - new Date(firstTokenLog.createdAt).getTime()) / 60000, 720)
      : 0;

    vectors.push([reqPerMin, uniqueEp, hour, ipChange, uaChange, tokenAgeMin]);
  }

  return vectors;
}

/**
 * Save forest state to disk for persistence across restarts.
 */
async function persistForest() {
  try {
    const state = {
      numTrees: forest.numTrees,
      sampleSize: forest.sampleSize,
      sampleSize_actual: forest.sampleSize_actual,
      trained: forest.trained,
      trees: forest.trees.map(t => serializeTree(t)),
      timestamp: Date.now(),
    };
    const dir = path.dirname(FOREST_PERSIST_PATH);
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(FOREST_PERSIST_PATH, JSON.stringify(state));
  } catch (err) {
    console.warn("⚠️  Forest persistence failed:", err.message);
  }
}

/**
 * Load forest state from disk after restart.
 */
async function restoreForest() {
  try {
    const data = await fs.readFile(FOREST_PERSIST_PATH, "utf-8");
    const state = JSON.parse(data);
    const ageMs = Date.now() - state.timestamp;
    const ageHours = ageMs / (1000 * 60 * 60);

    if (ageHours < 24 && state.trees && state.trees.length > 0) {
      forest.numTrees = state.numTrees;
      forest.sampleSize = state.sampleSize;
      forest.sampleSize_actual = state.sampleSize_actual;
      forest.trained = state.trained;
      forest.trees = state.trees.map(t => deserializeTree(t));
      console.log(`🌲 Isolation Forest restored (age: ${ageHours.toFixed(1)}h, trees: ${forest.trees.length})`);
      return true;
    } else {
      console.log(`⚠️  Forest state too old (${ageHours.toFixed(1)}h) — will retrain`);
      return false;
    }
  } catch (err) {
    console.log(`ℹ️  No persisted forest found — will train on first request`);
    return false;
  }
}

function serializeTree(tree) {
  return {
    isLeaf: tree.isLeaf,
    size: tree.size,
    splitDim: tree.splitDim,
    splitVal: tree.splitVal,
    left: tree.left ? serializeTree(tree.left) : null,
    right: tree.right ? serializeTree(tree.right) : null,
  };
}

function deserializeTree(obj) {
  const tree = Object.create(IsolationTree.prototype);
  tree.isLeaf = obj.isLeaf;
  tree.size = obj.size;
  tree.splitDim = obj.splitDim;
  tree.splitVal = obj.splitVal;
  tree.left = obj.left ? deserializeTree(obj.left) : null;
  tree.right = obj.right ? deserializeTree(obj.right) : null;
  return tree;
}

/**
 * (Re-)train the forest if conditions are met.
 * Fire-and-forget — never blocks the calling path.
 */
async function maybeRetrain() {
  requestsSinceLastTrain++;

  const shouldTrain =
    (!forest.trained && requestsSinceLastTrain >= MIN_SAMPLES_TO_TRAIN) ||
    (requestsSinceLastTrain >= RETRAIN_INTERVAL);

  if (!shouldTrain) return;

  try {
    const data = await fetchTrainingData();
    if (data && data.length >= MIN_SAMPLES_TO_TRAIN) {
      forest.train(data);
      await persistForest();  // 🔧 FIX #1: Save state after training
      requestsSinceLastTrain = 0;
    }
  } catch (err) {
    console.error("Isolation Forest retrain error:", err.message);
  }
}

/* ─────────────────────────────────────────────────────────────
   PUBLIC API
───────────────────────────────────────────────────────────── */

/**
 * Score a request for anomalousness.
 * Returns { score: 0–1, vector: [...] }
 *
 * score ≥ 0.72 → anomalous (caller decides what to do with this)
 * score ≥ 0.85 → strongly anomalous
 */
export async function scoreRequest({ userId, tokenHash, ipAddress, userAgent }) {
  // Kick off background retrain (non-blocking)
  maybeRetrain().catch(() => {});

  if (!userId) return { score: 0.5, vector: null };

  try {
    const vector = await buildFeatureVector(userId, tokenHash, ipAddress, userAgent);
    const score  = forest.score(vector);
    return { score, vector };
  } catch (err) {
    console.error("IF scoring error:", err.message);
    return { score: 0.5, vector: null };
  }
}

/**
 * Initialize forest (restore from disk if available).
 * Call this on app startup.
 */
export async function initForest() {
  const restored = await restoreForest();
  if (!restored) {
    console.log("🌲 Forest will train on first request batch");
  }
}

/**
 * Expose training status for the health/admin endpoint.
 */
export function forestStatus() {
  return {
    trained:   forest.trained,
    numTrees:  forest.numTrees,
    sampleSize: forest.sampleSize_actual ?? 0,
    requestsSinceLastTrain,
  };
}
