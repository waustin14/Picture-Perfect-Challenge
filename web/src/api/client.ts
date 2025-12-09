const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || getDefaultApiBaseUrl();

const WS_BASE_URL =
  import.meta.env.VITE_WS_BASE_URL || getDefaultWsBaseUrl();

const MINIO_PORT = import.meta.env.VITE_MINIO_PORT || "9000";

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export function wsUrl(path: string): string {
  return `${WS_BASE_URL}${path}`;
}

function getDefaultApiBaseUrl(): string {
    if (typeof window === "undefined") {
        // fallback for non-browser environments
        return "http://localhost:8000";
    }

    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const apiPort = "8000";

    return `${protocol}//${hostname}:${apiPort}`;
}

function getDefaultWsBaseUrl(): string {
  if (typeof window === "undefined") {
    return "ws://localhost:8000";
  }

  const isHttps = window.location.protocol === "https:";
  const wsProtocol = isHttps ? "wss:" : "ws:";
  const hostname = window.location.hostname;
  const apiPort = "8000";

  return `${wsProtocol}//${hostname}:${apiPort}`;
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
 * Rewrites MinIO URLs to use the current browser hostname.
 * This allows images stored in MinIO to be accessed when the app
 * is running on a remote server (not just localhost).
 *
 * Example: http://localhost:9000/bucket/image.png
 *       -> http://192.168.1.100:9000/bucket/image.png
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
      const currentHostname = window.location.hostname;
      const protocol = window.location.protocol;
      parsed.hostname = currentHostname;
      parsed.port = MINIO_PORT;
      parsed.protocol = protocol;
      return parsed.toString();
    }
  } catch {
    // If URL parsing fails, return original
  }

  return url;
}

