// Use /app prefix for API calls (reverse proxied through nginx to logic service)
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || getDefaultApiBaseUrl();

// Use /app prefix for WebSocket (reverse proxied through nginx to logic service)
const WS_BASE_URL =
  import.meta.env.VITE_WS_BASE_URL || getDefaultWsBaseUrl();

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export function wsUrl(path: string): string {
  return `${WS_BASE_URL}${path}`;
}

function getDefaultApiBaseUrl(): string {
  if (typeof window === "undefined") {
    // fallback for non-browser environments
    return "http://localhost:3000/app";
  }

  const protocol = window.location.protocol;
  const hostname = window.location.hostname;
  const port = window.location.port || (protocol === "https:" ? "443" : "80");

  // Use same origin with /app prefix
  return `${protocol}//${hostname}${port !== "80" && port !== "443" ? `:${port}` : ""}/app`;
}

function getDefaultWsBaseUrl(): string {
  if (typeof window === "undefined") {
    return "ws://localhost:3000/app";
  }

  const isHttps = window.location.protocol === "https:";
  const wsProtocol = isHttps ? "wss:" : "ws:";
  const hostname = window.location.hostname;
  const port = window.location.port || (isHttps ? "443" : "80");

  // Use same origin with /app prefix
  return `${wsProtocol}//${hostname}${port !== "80" && port !== "443" ? `:${port}` : ""}/app`;
}

// Simple fetch wrapper
export async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(apiUrl(path));
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `GET ${path} failed`);
  }
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `POST ${path} failed`);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return res.json() as Promise<T>;
}

export async function apiUpload<T>(path: string, file: File): Promise<T> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(apiUrl(path), {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Upload to ${path} failed`);
  }
  return res.json() as Promise<T>;
}

/**
 * Rewrites MinIO URLs to use the /images reverse proxy path.
 * This allows images stored in MinIO to be accessed through the same
 * origin as the web app, which works with Cloudflare tunnels.
 *
 * Example: http://localhost:9000/bucket/image.png
 *       -> /images/bucket/image.png
 */
export function rewriteMinioUrl(url: string | null | undefined): string | null {
  if (!url) return null;

  if (typeof window === "undefined") {
    return url;
  }

  try {
    const parsed = new URL(url);
    // Check if this looks like a MinIO URL (port 9000 or matches known MinIO hostnames)
    if (parsed.port === "9000" || parsed.hostname === "minio" || parsed.hostname === "localhost") {
      // Use the /images proxy path instead of direct MinIO access
      return `/images${parsed.pathname}${parsed.search}`;
    }
  } catch {
    // If URL parsing fails, return original
  }

  return url;
}

