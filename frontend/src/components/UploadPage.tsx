import React, { useState, useCallback, useEffect, useRef } from "react";
import { DataMode, Profile, TSDetectionResult } from "../types";
import { Badge } from "./UI";
import { detectTimeseries } from "../lib/api";

interface UploadPageProps {
  onAnalyze: (
    file: File,
    mode: DataMode,
    targetCol: string | null,
    profile: Profile,
    datetimeCol: string | null,
    tsTargetCol: string | null,
  ) => void;
  onCompare: () => void;
  onHome: () => void;
  loading: boolean;
}

const PROFILES: { id: Profile; label: string; desc: string }[] = [
  { id: "standard", label: "STANDARD", desc: "General purpose defaults" },
  { id: "finance", label: "FINANCE", desc: "Strict leakage, low anomaly tolerance" },
  { id: "healthcare", label: "HEALTHCARE", desc: "High missing tolerance, strict cardinality" },
  { id: "nlp", label: "NLP / TEXT", desc: "Relaxed cardinality thresholds" },
];

const selectStyle: React.CSSProperties = {
  width: "100%",
  background: "var(--bg-3)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-md)",
  padding: "0.6rem 0.75rem",
  color: "var(--text-0)",
  fontFamily: "var(--font-mono)",
  fontSize: "0.78rem",
  outline: "none",
  cursor: "pointer",
};

export default function UploadPage({ onAnalyze, onCompare, onHome, loading }: UploadPageProps) {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState<DataMode>("labeled");
  const [targetCol, setTargetCol] = useState("");
  const [profile, setProfile] = useState<Profile>("standard");
  const [dragging, setDragging] = useState(false);
  const [showSlowMessage, setShowSlowMessage] = useState(false);
  const slowTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Time series state
  const [tsDetection, setTsDetection] = useState<TSDetectionResult | null>(null);
  const [datetimeCol, setDatetimeCol] = useState<string>("");
  const [tsTargetCol, setTsTargetCol] = useState<string>("");
  const [tsDetecting, setTsDetecting] = useState(false);

  useEffect(() => {
    if (loading) {
      slowTimerRef.current = setTimeout(() => setShowSlowMessage(true), 20000);
    } else {
      if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
      setShowSlowMessage(false);
    }
    return () => {
      if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
    };
  }, [loading]);

  const handleFile = async (f: File) => {
    setFile(f);
    setTsDetection(null);
    setDatetimeCol("");
    setTsTargetCol("");
    setTsDetecting(true);
    try {
      const detection = await detectTimeseries(f);
      setTsDetection(detection);
      if (detection.auto_datetime) setDatetimeCol(detection.auto_datetime);
      if (detection.auto_target) setTsTargetCol(detection.auto_target);
    } catch {
      // silently ignore — ts detection is best-effort
    } finally {
      setTsDetecting(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith(".csv")) handleFile(f);
  }, []);

  const handleSubmit = () => {
    if (!file) return;
    onAnalyze(
      file,
      mode,
      mode === "labeled" && targetCol ? targetCol : null,
      profile,
      datetimeCol || null,
      tsTargetCol || null,
    );
  };

  const hasTs = tsDetection && tsDetection.datetime_candidates.length > 0;

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-start", paddingTop: "0.5rem", gap: "3rem" }}>

      {/* Header */}
      <div style={{ textAlign: "center", animation: "fadeIn 0.5s ease", width: "99%" }}>
        <div style={{ textAlign: "left", marginBottom: "1.5rem" }}>
          <button
            onClick={onHome}
            style={{ background: "transparent", border: "none", color: "var(--text-2)", fontFamily: "var(--font-mono)", fontSize: "0.7rem", letterSpacing: "0.15em", cursor: "pointer", textTransform: "uppercase" }}
          >
            ← Back
          </button>
        </div>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.3em", color: "var(--accent)", marginBottom: "0.75rem", textTransform: "uppercase" }}>
          ML Diagnostics Platform
        </div>
        <h1
          onClick={onHome}
          style={{ fontFamily: "var(--font-mono)", fontSize: "clamp(2rem, 5vw, 3.5rem)", fontWeight: 600, letterSpacing: "-0.02em", color: "var(--text-0)", lineHeight: 1, cursor: "pointer" }}
        >
          ATHENA
        </h1>
        <div style={{ marginTop: "0.75rem", color: "var(--text-2)", fontFamily: "var(--font-mono)", fontSize: "0.75rem", letterSpacing: "0.05em" }}>
          Upload a dataset. Surface what matters.
        </div>
      </div>

      {/* Main panel */}
      <div style={{ width: "100%", maxWidth: "960px", display: "flex", flexDirection: "column", gap: "1rem", animation: "fadeIn 0.5s ease 0.1s both" }}>

        {/* Drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById("file-input")?.click()}
          style={{
            border: `1px dashed ${dragging ? "var(--accent)" : file ? "var(--green-dim)" : "var(--border-bright)"}`,
            borderRadius: "var(--radius-lg)",
            minHeight: "160px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            padding: "2.5rem",
            textAlign: "center",
            cursor: "pointer",
            background: dragging ? "var(--accent-glow)" : file ? "var(--green-glow)" : "var(--bg-2)",
            transition: "all var(--transition)",
          }}
        >
          <input
            id="file-input"
            type="file"
            accept=".csv"
            style={{ display: "none" }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
          />
          {file ? (
            <div style={{ fontFamily: "var(--font-mono)" }}>
              <div style={{ color: "var(--green)", fontSize: "0.8rem", marginBottom: "0.25rem" }}>
                {tsDetecting ? "⟳ SCANNING..." : "✓ FILE LOADED"}
              </div>
              <div style={{ color: "var(--text-0)", fontSize: "0.9rem" }}>{file.name}</div>
              <div style={{ color: "var(--text-2)", fontSize: "0.7rem", marginTop: "0.25rem" }}>
                {(file.size / 1024).toFixed(1)} KB
              </div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem", opacity: 0.4 }}>⬆</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--text-1)" }}>DROP CSV HERE</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-2)", marginTop: "0.25rem" }}>or click to browse</div>
            </div>
          )}
        </div>

        {/* Time series override — only shown when datetime cols detected */}
        {hasTs && (
          <div style={{
            background: "var(--bg-2)",
            border: "1px solid var(--accent-dim)",
            borderRadius: "var(--radius-lg)",
            padding: "1.25rem",
            display: "flex",
            flexDirection: "column",
            gap: "0.75rem",
            animation: "fadeIn 0.3s ease",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.12em", color: "var(--accent)", textTransform: "uppercase" }}>
                Time Series Detected
              </div>
              <Badge variant="accent">AUTO</Badge>
            </div>
            <div style={{ display: "flex", gap: "0.75rem" }}>
              <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                <label style={{ fontFamily: "var(--font-mono)", fontSize: "0.63rem", color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                  Datetime Column
                </label>
                <select value={datetimeCol} onChange={e => setDatetimeCol(e.target.value)} style={selectStyle}>
                  <option value="">Auto-detect</option>
                  {tsDetection.datetime_candidates.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
              <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                <label style={{ fontFamily: "var(--font-mono)", fontSize: "0.63rem", color: "var(--text-2)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
                  Forecast Target
                </label>
                <select value={tsTargetCol} onChange={e => setTsTargetCol(e.target.value)} style={selectStyle}>
                  <option value="">Auto-detect</option>
                  {tsDetection.target_candidates.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              </div>
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", color: "var(--text-3)" }}>
              Time series EDA and baseline models will run alongside standard analysis.
            </div>
          </div>
        )}

        {/* Mode selector */}
        <div style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "1.25rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.12em", color: "var(--text-2)", textTransform: "uppercase" }}>
            Dataset Mode
          </div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            {(["labeled", "unlabeled"] as DataMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                style={{
                  flex: 1,
                  padding: "0.6rem",
                  border: `1px solid ${mode === m ? "var(--accent)" : "var(--border)"}`,
                  borderRadius: "var(--radius-md)",
                  background: mode === m ? "var(--accent-glow)" : "var(--bg-3)",
                  color: mode === m ? "var(--accent)" : "var(--text-2)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.72rem",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                  transition: "all var(--transition)",
                }}
              >
                {m === "labeled" ? "Supervised" : "Unsupervised"}
              </button>
            ))}
          </div>
          {mode === "labeled" && (
            <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
              <label style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.12em", color: "var(--text-2)", textTransform: "uppercase" }}>
                Target Column
              </label>
              <input
                value={targetCol}
                onChange={(e) => setTargetCol(e.target.value)}
                placeholder="e.g. Loan_Status"
                style={{ background: "var(--bg-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "0.6rem 0.75rem", color: "var(--text-0)", fontFamily: "var(--font-mono)", fontSize: "0.8rem", outline: "none", transition: "border-color var(--transition)" }}
                onFocus={(e) => (e.target.style.borderColor = "var(--accent)")}
                onBlur={(e) => (e.target.style.borderColor = "var(--border)")}
              />
            </div>
          )}
        </div>

        {/* Profile selector */}
        <div style={{ background: "var(--bg-2)", border: "1px solid var(--border)", borderRadius: "var(--radius-lg)", padding: "1.25rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.12em", color: "var(--text-2)", textTransform: "uppercase" }}>
            Analysis Profile
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem" }}>
            {PROFILES.map((p) => (
              <button
                key={p.id}
                onClick={() => setProfile(p.id)}
                style={{
                  padding: "0.6rem 0.75rem",
                  border: `1px solid ${profile === p.id ? "var(--accent)" : "var(--border)"}`,
                  borderRadius: "var(--radius-md)",
                  background: profile === p.id ? "var(--accent-glow)" : "var(--bg-3)",
                  color: "var(--text-0)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "0.7rem",
                  cursor: "pointer",
                  transition: "all var(--transition)",
                  textAlign: "left",
                }}
              >
                <div style={{ color: profile === p.id ? "var(--accent)" : "var(--text-1)", marginBottom: "0.15rem" }}>{p.label}</div>
                <div style={{ fontSize: "0.62rem", color: "var(--text-2)" }}>{p.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: "0.75rem" }}>
          <button
            onClick={handleSubmit}
            disabled={!file || loading}
            style={{
              flex: 1,
              padding: "0.85rem",
              border: "1px solid var(--accent)",
              borderRadius: "var(--radius-md)",
              background: !file || loading ? "var(--bg-3)" : "var(--accent-glow)",
              color: !file || loading ? "var(--text-2)" : "var(--accent)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.8rem",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              cursor: !file || loading ? "not-allowed" : "pointer",
              transition: "all var(--transition)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "0.5rem",
            }}
          >
            {loading ? (<><span style={{ animation: "blink 1s infinite" }}>■</span> ANALYZING</>) : "RUN ANALYSIS →"}
          </button>
          <button
            onClick={onCompare}
            style={{
              padding: "0.85rem 1.25rem",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-md)",
              background: "var(--bg-2)",
              color: "var(--text-1)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.75rem",
              letterSpacing: "0.08em",
              cursor: "pointer",
              transition: "all var(--transition)",
              whiteSpace: "nowrap",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = "var(--border-bright)"; e.currentTarget.style.color = "var(--text-0)"; }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.color = "var(--text-1)"; }}
          >
            COMPARE DATASETS
          </button>
        </div>

        {showSlowMessage && (
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.9rem", color: "var(--text-2)", textAlign: "center", letterSpacing: "0.05em", animation: "fadeIn 0.5s ease" }}>
            This might take a few minutes...
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-3)", letterSpacing: "0.08em", animation: "fadeIn 0.5s ease 0.2s both" }}>
        EDA · ML READINESS · DRIFT DETECTION · TIME SERIES
      </div>
    </div>
  );
}