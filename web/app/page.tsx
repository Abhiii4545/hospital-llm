"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import FieldEditor from "@/components/FieldEditor";
import LineItemTable from "@/components/LineItemTable";
import PageViewer from "@/components/PageViewer";
import {
  API_BASE,
  adjudicate,
  extractHeuristic,
  extractModel,
  findBox,
  getHealth,
  logCorrection,
  ocrPage,
  type AdjudicationResult,
  type ExtractResult,
  type Health,
  type OcrLine,
  type ReckonDocument,
} from "@/lib/api";

const EMPTY: ReckonDocument = {
  hospital: {}, patient: {}, insurance: {}, totals: {}, line_items: [],
};

export default function ReviewPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [ocr, setOcr] = useState<{ w: number; h: number; lines: OcrLine[] }>({
    w: 0, h: 0, lines: [],
  });
  const [document, setDocument] = useState<ReckonDocument>(EMPTY);
  const [extraction, setExtraction] = useState<ExtractResult | null>(null);
  const [result, setResult] = useState<AdjudicationResult | null>(null);
  const [edited, setEdited] = useState<Set<string>>(new Set());
  const [activeField, setActiveField] = useState<string | null>(null);
  const [highlight, setHighlight] = useState<OcrLine | null>(null);
  const [showAllBoxes, setShowAllBoxes] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [coPay, setCoPay] = useState(10);
  const [sumInsured, setSumInsured] = useState(500000);
  const dropRef = useRef<HTMLDivElement | null>(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setHealthError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const pageId = useMemo(
    () => (file ? file.name.replace(/\.[^.]+$/, "") : "unknown"),
    [file]
  );

  const run = useCallback(
    async (label: string, task: () => Promise<void>) => {
      setBusy(label);
      setError(null);
      try {
        await task();
      } catch (e: any) {
        setError(String(e?.message ?? e));
      } finally {
        setBusy(null);
      }
    },
    []
  );

  const onFile = (chosen: File) => {
    setFile(chosen);
    setDocument(EMPTY);
    setExtraction(null);
    setResult(null);
    setEdited(new Set());
    setOcr({ w: 0, h: 0, lines: [] });
    setHighlight(null);
  };

  const loadSample = () =>
    run("Fetching sample…", async () => {
      const index = Math.floor(Math.random() * 200);
      const response = await fetch(`${API_BASE}/sample?index=${index}`);
      if (!response.ok) throw new Error("No synthetic corpus on the server.");
      const blob = await response.blob();
      onFile(new File([blob], `sample_${index}.png`, { type: "image/png" }));
    });

  const doOcr = () =>
    file &&
    run("Running OCR…", async () => {
      const out = await ocrPage(file);
      setOcr({ w: out.width, h: out.height, lines: out.lines });
    });

  const doExtract = (useModel: boolean) =>
    file &&
    run(useModel ? "Running model…" : "Running heuristics…", async () => {
      const out = useModel ? await extractModel([file]) : await extractHeuristic(file);
      setExtraction(out);
      setDocument(out.document);
      setResult(null);
      if (!ocr.lines.length) {
        const boxes = await ocrPage(file);
        setOcr({ w: boxes.width, h: boxes.height, lines: boxes.lines });
      }
    });

  const doAdjudicate = () =>
    run("Adjudicating…", async () => {
      setResult(
        await adjudicate(document, {
          co_pay_percent: coPay,
          sum_insured: sumInsured,
        })
      );
    });

  const setField = (path: string, value: string) => {
    const [block, name] = path.split(".");
    setDocument((previous) => ({
      ...previous,
      [block]: { ...(previous as any)[block], [name]: value },
    }));
    setEdited((previous) => new Set(previous).add(path));
  };

  const setItem = (index: number, key: string, value: string) => {
    setDocument((previous) => {
      const items = previous.line_items.slice();
      items[index] = { ...items[index], [key]: value };
      return { ...previous, line_items: items };
    });
    setEdited((previous) => new Set(previous).add(`line_items[${index}].${key}`));
  };

  const focusField = (path: string, value: string | null) => {
    setActiveField(path);
    setHighlight(findBox(value, ocr.lines));
  };

  const submitCorrections = () =>
    run("Logging corrections…", async () => {
      for (const path of Array.from(edited)) {
        const [block, name] = path.split(".");
        const value =
          block.startsWith("line_items")
            ? ""
            : ((document as any)[block]?.[name] ?? "");
        await logCorrection({
          page_id: pageId,
          field: path,
          predicted: extraction
            ? ((extraction.document as any)[block]?.[name] ?? null)
            : null,
          corrected: String(value),
        });
      }
      setEdited(new Set());
    });

  return (
    <>
      <header className="top">
        <h1>RECKON v2 — claim review</h1>
        {health && (
          <>
            <span className={`badge ${health.adjudication_available ? "ok" : "off"}`}>
              adjudication {health.adjudication_available ? "ready" : "off"}
            </span>
            <span className={`badge ${health.extraction_available ? "ok" : "off"}`}>
              model {health.extraction_available ? "loaded" : "not trained"}
            </span>
            {health.git_sha && (
              <span className="badge">{health.git_sha.slice(0, 7)}</span>
            )}
          </>
        )}
        {healthError && <span className="badge off">API unreachable</span>}
        <span style={{ flex: 1 }} />
        {busy && <span className="muted">{busy}</span>}
      </header>

      <div className="wrap">
        {healthError && (
          <div className="callout bad" style={{ marginTop: 16 }}>
            Cannot reach the API at <code>{API_BASE}</code> — {healthError}.
            Start it with <code>uv run uvicorn reckon.serve.api:app</code>.
          </div>
        )}
        {error && (
          <div className="callout bad" style={{ marginTop: 16 }}>{error}</div>
        )}
        {extraction?.warning && (
          <div className="callout" style={{ marginTop: 16 }}>
            <strong>{extraction.engine}.</strong> {extraction.warning}
          </div>
        )}

        <div className="grid">
          <section className="panel">
            <h2>Page</h2>
            <div
              ref={dropRef}
              className={`drop ${dragOver ? "over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                const dropped = e.dataTransfer.files?.[0];
                if (dropped) onFile(dropped);
              }}
              onClick={() => window.document.getElementById("picker")?.click()}
              style={{ marginBottom: 12 }}
            >
              {file ? file.name : "Drop a bill page here, or click to choose"}
              <input
                id="picker"
                type="file"
                accept="image/*"
                style={{ display: "none" }}
                onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
              />
            </div>

            <div className="row" style={{ marginBottom: 12 }}>
              <button disabled={!file || !!busy} onClick={() => doExtract(false)}>
                Extract (heuristics)
              </button>
              <button
                className="ghost"
                disabled={!file || !!busy || !health?.extraction_available}
                onClick={() => doExtract(true)}
                title={
                  health?.extraction_available
                    ? "Run the trained Donut heads"
                    : "No trained checkpoint is configured"
                }
              >
                Extract (model)
              </button>
              <button className="ghost" disabled={!file || !!busy} onClick={doOcr}>
                OCR only
              </button>
              <button className="ghost" disabled={!!busy} onClick={loadSample}>
                Try a sample
              </button>
              <label className="muted" style={{ display: "flex", gap: 6 }}>
                <input
                  type="checkbox"
                  style={{ width: "auto" }}
                  checked={showAllBoxes}
                  onChange={(e) => setShowAllBoxes(e.target.checked)}
                />
                all boxes
              </label>
            </div>

            <PageViewer
              imageUrl={imageUrl}
              ocrWidth={ocr.w}
              ocrHeight={ocr.h}
              highlight={highlight}
              allLines={ocr.lines}
              showAll={showAllBoxes}
            />
            {ocr.lines.length > 0 && (
              <p className="muted" style={{ marginTop: 8 }}>
                {ocr.lines.length} OCR boxes. Highlighting is presentation only —
                the model emits no coordinates.
              </p>
            )}
          </section>

          <section>
            <div className="panel" style={{ marginBottom: 18 }}>
              <h2>Extracted fields</h2>
              <FieldEditor
                document={document}
                edited={edited}
                onChange={setField}
                onFocusField={focusField}
                activeField={activeField}
              />
              <div className="row" style={{ marginTop: 10 }}>
                <button
                  className="ghost"
                  disabled={!edited.size || !!busy}
                  onClick={submitCorrections}
                >
                  Log {edited.size} correction{edited.size === 1 ? "" : "s"}
                </button>
                <span className="muted">
                  Corrections become training data from the real distribution.
                </span>
              </div>
            </div>

            <div className="panel" style={{ marginBottom: 18 }}>
              <h2>Line items</h2>
              <LineItemTable
                items={document.line_items}
                adjudication={result}
                onEdit={setItem}
              />
            </div>

            <div className="panel">
              <h2>Adjudication</h2>
              <div className="row" style={{ marginBottom: 11 }}>
                <label className="muted" style={{ width: 110 }}>Sum insured</label>
                <input
                  type="number"
                  style={{ width: 130 }}
                  value={sumInsured}
                  onChange={(e) => setSumInsured(Number(e.target.value))}
                />
                <label className="muted" style={{ width: 70 }}>Co-pay %</label>
                <input
                  type="number"
                  style={{ width: 80 }}
                  value={coPay}
                  onChange={(e) => setCoPay(Number(e.target.value))}
                />
                <button disabled={!!busy} onClick={doAdjudicate}>
                  Adjudicate
                </button>
              </div>

              {extraction && (
                <div
                  className={`callout ${
                    extraction.reconciliation.balanced ? "good" : ""
                  }`}
                >
                  {extraction.reconciliation.balanced
                    ? "Bill reconciles: line items match the printed totals."
                    : `Reconciliation flags: ${extraction.reconciliation.flags.join(
                        "; "
                      )}`}
                </div>
              )}

              {result && (
                <>
                  <div className="kpi" style={{ marginBottom: 12 }}>
                    <div>
                      <span>Gross</span>
                      <strong>₹{result.gross}</strong>
                    </div>
                    <div>
                      <span>Deducted</span>
                      <strong style={{ color: "var(--bad)" }}>
                        −₹{result.total_deducted}
                      </strong>
                    </div>
                    <div>
                      <span>Payable</span>
                      <strong style={{ color: "var(--good)" }}>
                        ₹{result.payable}
                      </strong>
                    </div>
                  </div>
                  <h2>Audit trail</h2>
                  <pre>{result.audit_trail}</pre>
                </>
              )}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}
