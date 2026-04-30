import net from "net";

export const normalizeIp = (rawIp) => {
  if (!rawIp || typeof rawIp !== "string") return "unknown";

  const ip = rawIp.split(",")[0].trim();
  if (!ip) return "unknown";

  const stripped = ip.split("%")[0];

  if (stripped === "::1" || stripped === "0:0:0:0:0:0:0:1") {
    return "127.0.0.1";
  }

  if (stripped.startsWith("::ffff:")) {
    return stripped.replace("::ffff:", "");
  }

  if (net.isIP(stripped)) {
    return stripped;
  }

  return ip;
};
