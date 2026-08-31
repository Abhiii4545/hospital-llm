/**
 * Client for the RECKON FastAPI backend.
 *
 * Every call reports failure explicitly rather than returning an empty object.
 * This UI shows an adjudicator what a claim is worth; a silently empty response
 * rendered as "no deductions" would be the most expensive possible bug here.
 */

// NEXT_PUBLIC_ prefix is required: Next only inlines those into the CLIENT
// bundle. A bare `process.env.RECKON_API` is undefined in the browser, so every
// request went to "undefined/health" and failed silently.
export const API_BASE =
  process.env.NEXT_PUBLIC_RECKON_API ?? "http://127.0.0.1:8000";

export interface OcrLine {
  text: string;
  score: number;
  box: [number, number, number, number];
}

export interface OcrResult {
  width: number;
  height: number;
  lines: OcrLine[];
}

export interface LineItem {
  serial_no?: string | null;
  description?: string | null;
  service_date?: string | null;
  category?: string | null;
  quantity?: string | null;
  unit_rate?: string | null;
  amount?: string | null;
  hsn_code?: string | null;
  is_payable?: string | null;
  deduction_reason?: string | null;
  [key: string]: string | null | undefined;
}

export interface ReckonDocument {
  hospital: Record<string, string | null>;
  patient: Record<string, string | null>;
  insurance: Record<string, string | null>;
  totals: Record<string, string | null>;
  line_items: LineItem[];
}

export interface Reconciliation {
  balanced: boolean;
  complete: boolean;
  flags: string[];
}

export interface ExtractResult {
  engine: string;
  warning?: string;
  document: ReckonDocument;
  reconciliation: Reconciliation;
  latency_seconds: number;
}

export interface Deduction {
  rule_id: string;
  clause: string;
  reason: string;
  amount: string;
  line_index: number | null;
  line_description: string | null;
}

export interface AdjudicationResult {
  gross: string;
  payable: string;
  total_deducted: string;
  deductions: Deduction[];
  by_rule: Record<string, string>;
  notes: string[];
  audit_trail: string;
}

export interface Health {
  status: string;
  git_sha: string | null;
  extraction_available: boolean;
  adjudication_available: boolean;
  note: string;
}

async function unwrap<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* the body was not JSON; the status line is all we have */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export async function getHealth(): Promise<Health> {
  return unwrap<Health>(await fetch(`${API_BASE}/health`, { cache: "no-store" }));
}

export async function ocrPage(file: File): Promise<OcrResult> {
  const body = new FormData();
  body.append("file", file);
  return unwrap<OcrResult>(
    await fetch(`${API_BASE}/ocr`, { method: "POST", body })
  );
}

/** Heuristic extraction. Needs no trained model; accuracy is poor by design. */
export async function extractHeuristic(file: File): Promise<ExtractResult> {
  const body = new FormData();
  body.append("file", file);
  return unwrap<ExtractResult>(
    await fetch(`${API_BASE}/extract-heuristic`, { method: "POST", body })
  );
}

/** Trained Donut heads. 503 until a checkpoint is configured. */
export async function extractModel(files: File[]): Promise<ExtractResult> {
  const body = new FormData();
  files.forEach((f) => body.append("files", f));
  return unwrap<ExtractResult>(
    await fetch(`${API_BASE}/extract`, { method: "POST", body })
  );
}

export async function adjudicate(
  document: unknown,
  options: { sum_insured?: number; co_pay_percent?: number; deductible?: number } = {}
): Promise<AdjudicationResult> {
  return unwrap<AdjudicationResult>(
    await fetch(`${API_BASE}/adjudicate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document, ...options }),
    })
  );
}

export async function logCorrection(payload: {
  page_id: string;
  field: string;
  predicted?: string | null;
  corrected: string;
  note?: string;
}): Promise<{ logged: boolean; path: string }> {
  return unwrap(
    await fetch(`${API_BASE}/corrections`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
  );
}

/**
 * Best OCR box for a field value, for highlighting.
 *
 * PRESENTATION ONLY. The model emits no coordinates; these come from a separate
 * OCR pass and are used purely to draw a rectangle. They never feed the model
 * and never touch a metric.
 */
export function findBox(value: string | null | undefined, lines: OcrLine[]): OcrLine | null {
  if (!value) return null;
  const needle = value.toLowerCase().replace(/\s+/g, "");
  let best: OcrLine | null = null;
  let bestScore = 0;
  for (const line of lines) {
    const hay = line.text.toLowerCase().replace(/\s+/g, "");
    if (!hay) continue;
    let score = 0;
    if (hay.includes(needle)) score = needle.length / hay.length;
    else if (needle.includes(hay)) score = hay.length / needle.length;
    if (score > bestScore) {
      best = line;
      bestScore = score;
    }
  }
  return bestScore >= 0.5 ? best : null;
}
