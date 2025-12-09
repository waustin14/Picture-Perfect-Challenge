const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || getDefaultApiBaseUrl();

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

