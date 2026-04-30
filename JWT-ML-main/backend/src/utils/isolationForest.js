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
 * Every request is converted to a 7-feature numeric vector:
 *   [0] requestsPerMin        — volume signal
 *   [1] uniqueEndpointsPerMin  — API scanning signal
 *   [2] hourSin               — sin(2π·hour/24), circular time encoding
 *   [3] hourCos               — cos(2π·hour/24), circular time encoding
 *   [4] ipUnknownFlag          — 1 if IP not in user's baseline knownIPs
 *   [5] uaChangeFlag           — 1 if User-Agent differs from previous log
 *   [6] tokenAgeMinutes        — how old is this token (0–720 capped)
 *
 * Fixes vs prior version:
 *   #3 Fisher-Yates shuffle  — replaces biased sort() subsample
 *   #4 Circular hour encoding — sin/cos eliminates 23→0 discontinuity
 *   #5 IP unknown flag        — checks user baseline, not just prev-log diff
 *   #8 Min-max normalization  — computed from training batch, applied to
 *      every scoring call so high-variance features don't dominate splits
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
import { getUserBaseline } from "./userBehavior.js";
import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const TWO_PI = 2 * Math.PI;

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
    this.numTrees    = numTrees;
    this.sampleSize  = sampleSize;
    this.trees       = [];
    this.trained     = false;
    this.featureMins = null;  // per-feature training mins — used for live normalization
    this.featureMaxs = null;  // per-feature training maxs
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
      // Fisher-Yates uniform shuffle — unbiased, O(n). Replaces biased sort() (#3)
      const sample = fisherYates(data).slice(0, n);
      this.trees.push(new IsolationTree(sample));
    }

    this.sampleSize_actual = n;
    this.trained = true;
    console.log(`�� Isolation Forest trained on ${n} samples with ${this.numTrees} trees`);
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
   HELPERS
───────────────────────────────────────────────────────────── */

/**
 * Fisher-Yates uniform shuffle — O(n), unbiased (#3).
 * Replaces the biased Array.sort(() => Math.random() - 0.5) pattern.
 */
function fisherYates(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/**
 * Compute per-feature min-max normalization params from a batch of training
 * vectors, and return both the normalized vectors and the params (#8).
 *
 * Without this, requestsPerMin (0–1000+) would dominate split selection
 * over binary flags like ipUnknownFlag (0 or 1), reducing their influence.
 */
function computeNormParams(vectors) {
  const numFeatures = vectors[0].length;
  const mins = new Array(numFeatures).fill(Infinity);
  const maxs = new Array(numFeatures).fill(-Infinity);

  for (const v of vectors) {
    for (let i = 0; i < numFeatures; i++) {
      if (v[i] < mins[i]) mins[i] = v[i];
      if (v[i] > maxs[i]) maxs[i] = v[i];
    }
  }

  const normalized = vectors.map(v =>
    v.map((val, i) => {
      const range = maxs[i] - mins[i];
      return range === 0 ? 0 : (val - mins[i]) / range;
    })
  );
  return { normalized, mins, maxs };
}

/**
 * Apply pre-computed normalization to a single vector during live scoring.
 * Clamps to [0, 1] to handle values outside the training range.
 */
function applyNorm(vector, mins, maxs) {
  return vector.map((val, i) => {
    const range = maxs[i] - mins[i];
    return range === 0 ? 0 : Math.min(1, Math.max(0, (val - mins[i]) / range));
  });
}

/* ─────────────────────────────────────────────────────────────
   SINGLETON + AUTO-TRAIN LIFECYCLE
───────────────────────────────────────────────────────────── */

const forest = new IsolationForest({ numTrees: 50, sampleSize: 256 });

let requestsSinceLastTrain = 0;
const RETRAIN_INTERVAL     = 500; // retrain every 500 requests
const MIN_SAMPLES_TO_TRAIN = 50;  // don't train until we have enough data

/**
 * Build the raw (pre-normalization) 7-feature vector for live scoring.
 *
 * [0] requestsPerMin        — volume signal
 * [1] uniqueEndpointsPerMin  — scanning signal
 * [2] hourSin               — sin(2π·hour/24), circular encoding (#4)
 * [3] hourCos               — cos(2π·hour/24), circular encoding (#4)
 * [4] ipUnknownFlag          — 1 if IP not in baseline knownIPs (#5)
 * [5] uaChangeFlag           — 1 if UA differs from most recent log
 * [6] tokenAgeMinutes        — token age capped at 720 min (12 h)
 */
async function buildFeatureVector(userId, tokenHash, ipAddress, userAgent) {
  const oneMinAgo = new Date(Date.now() - 60000);

  const [recentLogs, prevLog, firstTokenLog, baseline] = await Promise.all([
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
    getUserBaseline(userId),
  ]);

  const requestsPerMin        = recentLogs.length;
  const uniqueEndpointsPerMin = new Set(recentLogs.map(l => l.endpoint)).size;

  // Circular hour encoding — eliminates the 23→0 discontinuity (#4)
  const hour    = new Date().getHours();
  const hourSin = Math.sin(TWO_PI * hour / 24);
  const hourCos = Math.cos(TWO_PI * hour / 24);

  // IP unknown flag — checks established baseline, not just last-log diff (#5)
  const normalizedIp  = normalizeIp(ipAddress);
  const knownIPs      = baseline?.knownIPs ?? [];
  const ipUnknownFlag = knownIPs.length > 0 && !knownIPs.includes(normalizedIp) ? 1 : 0;

  const uaChangeFlag = prevLog && prevLog.userAgent !== userAgent ? 1 : 0;

  const tokenAgeMinutes = firstTokenLog
    ? Math.min((Date.now() - new Date(firstTokenLog.createdAt).getTime()) / 60000, 720)
    : 0;

  return [
    requestsPerMin,
    uniqueEndpointsPerMin,
    hourSin,
    hourCos,
    ipUnknownFlag,
    uaChangeFlag,
    tokenAgeMinutes,
  ];
}

/**
 * Fetch training data from RequestLog — O(n log n), sliding window.
 *
 * Returns { vectors, mins, maxs } — vectors are already min-max normalized (#8).
 * The mins/maxs are stored on the forest singleton and applied during live scoring.
 */
async function fetchTrainingData() {
  const logs = await RequestLog.find({ userId: { $ne: null } })
    .sort({ userId: 1, createdAt: -1 })
    .limit(2000)
    .select("userId tokenHash ipAddress userAgent endpoint createdAt")
    .lean();

  if (logs.length < MIN_SAMPLES_TO_TRAIN) return null;

  const rawVectors  = [];
  let currentUserId = null;
  let userWindow    = [];
  const userSeenIPs = {};  // uid → Set of IPs seen in newer logs for this user (#5)

  for (const log of logs) {
    const uid     = String(log.userId);
    const logTime = new Date(log.createdAt).getTime();

    if (uid !== currentUserId) {
      currentUserId = uid;
      userWindow    = [];
    }

    if (!userSeenIPs[uid]) userSeenIPs[uid] = new Set();

    userWindow.push(log);
    userWindow = userWindow.filter(
      l => logTime - new Date(l.createdAt).getTime() <= 60000
    );

    const rateWindow = userWindow.filter(
      l => l !== log && logTime - new Date(l.createdAt).getTime() < 60000
    );

    const reqPerMin = rateWindow.length + 1;
    const uniqueEp  = new Set([log.endpoint, ...rateWindow.map(l => l.endpoint)]).size;

    // Circular hour encoding (#4)
    const hour    = new Date(log.createdAt).getHours();
    const hourSin = Math.sin(TWO_PI * hour / 24);
    const hourCos = Math.cos(TWO_PI * hour / 24);

    // IP unknown flag: not seen in any newer log for this user (#5)
    const normIp        = normalizeIp(log.ipAddress);
    const seenIPs       = userSeenIPs[uid];
    const ipUnknownFlag = seenIPs.size > 0 && !seenIPs.has(normIp) ? 1 : 0;
    seenIPs.add(normIp);

    const prevLog  = userWindow.length > 1 ? userWindow[1] : null;
    const uaChange = prevLog && prevLog.userAgent !== log.userAgent ? 1 : 0;

    const firstTokenLog = userWindow
      .filter(l => l.tokenHash === log.tokenHash)
      .sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt))[0];

    const tokenAgeMin = firstTokenLog
      ? Math.min((logTime - new Date(firstTokenLog.createdAt).getTime()) / 60000, 720)
      : 0;

    rawVectors.push([reqPerMin, uniqueEp, hourSin, hourCos, ipUnknownFlag, uaChange, tokenAgeMin]);
  }

  // Compute normalization params and return normalized vectors (#8)
  return computeNormParams(rawVectors);
}

/**
 * Save forest state to disk for persistence across restarts.
 */
async function persistForest() {
  try {
    const state = {
      numTrees:          forest.numTrees,
      sampleSize:        forest.sampleSize,
      sampleSize_actual: forest.sampleSize_actual,
      trained:           forest.trained,
      featureMins:       forest.featureMins,
      featureMaxs:       forest.featureMaxs,
      trees:             forest.trees.map(t => serializeTree(t)),
      timestamp:         Date.now(),
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
      forest.numTrees          = state.numTrees;
      forest.sampleSize        = state.sampleSize;
      forest.sampleSize_actual = state.sampleSize_actual;
      forest.trained           = state.trained;
      forest.featureMins       = state.featureMins || null;
      forest.featureMaxs       = state.featureMaxs || null;
      forest.trees             = state.trees.map(t => deserializeTree(t));
      console.log(`�� Isolation Forest restored (age: ${ageHours.toFixed(1)}h, trees: ${forest.trees.length}, normalization: ${forest.featureMins ? "yes" : "no"})`);
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
    const result = await fetchTrainingData();
    if (result && result.normalized.length >= MIN_SAMPLES_TO_TRAIN) {
      forest.featureMins = result.mins;
      forest.featureMaxs = result.maxs;
      forest.train(result.normalized);   // trains on normalized vectors
      await persistForest();
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
    const rawVector = await buildFeatureVector(userId, tokenHash, ipAddress, userAgent);
    // Apply same normalization used during training (#8)
    const scoringVector = forest.featureMins && forest.featureMaxs
      ? applyNorm(rawVector, forest.featureMins, forest.featureMaxs)
      : rawVector;
    const score = forest.score(scoringVector);
    return { score, vector: rawVector };  // return raw vector for human-readable alert reasons
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
    console.log("�� Forest will train on first request batch");
  }
}

/**
 * Expose training status for the health/admin endpoint.
 */
export function forestStatus() {
  return {
    trained:             forest.trained,
    numTrees:            forest.numTrees,
    sampleSize:          forest.sampleSize_actual ?? 0,
    normalizationReady:  !!(forest.featureMins && forest.featureMaxs),
    requestsSinceLastTrain,
  };
}