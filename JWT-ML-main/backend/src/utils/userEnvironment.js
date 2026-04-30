import RequestLog from "../models/RequestLog.js";
import geoip from "geoip-lite";


export const getKnownIPs = async (userId) => {
  const logs = await RequestLog.find({ userId })
    .select("ipAddress")
    .sort({ createdAt: -1 })
    .limit(100);

  const ips = logs.map(l => l.ipAddress).filter(Boolean);
  return [...new Set(ips)];
};


/* ------------------------------------------------ */
/* GEO DISTANCE                                     */
/* FIX: original used Euclidean on lat/lon degrees  */
/* which gives wrong km values (lon degrees shrink  */
/* near poles). Replaced with Haversine formula.    */
/* ------------------------------------------------ */

export const geoDistance = (ip1, ip2) => {

  const g1 = geoip.lookup(ip1);
  const g2 = geoip.lookup(ip2);

  if (!g1 || !g2) return 0;

  const toRad = (deg) => (deg * Math.PI) / 180;

  const lat1 = g1.ll[0], lon1 = g1.ll[1];
  const lat2 = g2.ll[0], lon2 = g2.ll[1];

  const R = 6371; // Earth radius in km
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c; // actual km
};
