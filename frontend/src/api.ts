const apiBase = import.meta.env.VITE_API_BASE_URL ?? "";

export async function api<T>(path: string, options: RequestInit = {}, adminKey?: string): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  if (adminKey) headers.set("X-Admin-API-Key", adminKey);
  let response: Response;
  try {
    response = await fetch(`${apiBase}${path}`, { ...options, headers });
  } catch (cause) {
    const reason = cause instanceof Error ? ` (${cause.message})` : "";
    throw new Error(`Cannot reach the backend API${reason}. Start it from the project folder with: python -m app.cli api`);
  }
  if (!response.ok) {
    const body = await response.text();
    let detail = body;
    try {
      const parsed = JSON.parse(body) as { detail?: string };
      detail = parsed.detail ?? body;
    } catch {
      detail = body;
    }
    if (response.headers.get("content-type")?.includes("text/html")) {
      detail = `The frontend development proxy returned HTTP ${response.status}. Start the backend in a second VS Code terminal from C:\\CIS\\motor_monitoring with: .\\.venv\\Scripts\\Activate.ps1; python -m app.cli api`;
    } else if (!detail) {
      detail = `Backend API returned HTTP ${response.status}. Check the API terminal for its error and restart it with: python -m app.cli api`;
    }
    throw new Error(detail.slice(0, 500) || `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export interface Prediction {
  id: number;
  message_id: string;
  machine_id: string;
  source: "live" | "replay";
  recording_id: string | null;
  window_index: number | null;
  predicted_condition: string;
  probabilities: Record<string, number>;
  calibrated_probabilities?: Record<string, number>;
  decision_status?: string;
  reliability?: {
    sensor_quality?: { status: string; constant_channels?: string[] };
    out_of_distribution?: { status: string; outside_feature_count?: number };
    drift?: { status: string; window_size?: number; psi_max?: number | null; threshold?: number };
    uncertainty?: { maximum_calibrated_probability?: number; acceptance_threshold?: number };
  };
  explanation?: {
    method?: string;
    explained_class?: string;
    interpretation?: string;
    top_features?: Array<{
      feature: string;
      value: number;
      attribution?: number;
      probability_delta?: number;
    }>;
  };
  human_review?: {
    review_label: string | null;
    reviewed_by: string | null;
    review_note: string | null;
    reviewed_at: string | null;
  };
  model_version: string;
  received_at: string;
  latency_ms: number;
}

export interface EventRecord {
  id: number;
  event_type: string;
  severity: string;
  message: string;
  created_at: string;
  acknowledged: boolean;
}
