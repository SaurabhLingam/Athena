import React, { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, ReferenceLine } from "recharts";
import { DriftResult } from "../types";
import { Card, SectionHeader, Alert, StatBox, ProgressBar, Badge, DataTable } from "./UI";
import { compareDatasets } from "../lib/api";

interface ComparePageProps {
  onBack: () => void;
}

export default function ComparePage({ onBack }: ComparePageProps) {
  const [trainFile, setTrainFile] = useState<File | null>(null);
  const [testFile, setTestFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DriftResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleRun = async () => {
    if (!trainFile || !testFile) return;
    setLoading(true);
    setError(null);
    try {
      const r = await compareDatasets(trainFile, testFile);
      setResult(r);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const FileDropZone = ({
    label,
    file,
    onFile,
  }: {
    label: string;
    file: File | null;
    onFile: (f: File) => void;
  }) => {
    const [drag, setDrag] = useState(false);
    return (
      <div
        onDragOver={e => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f) onFile(f); }}
        onClick={() => document.getElementById(`file-${label}`)?.click()}
        style={{
          flex: 1,
          border: `1px dashed ${drag ? "var(--accent)" : file ? "var(--green-dim)" : "var(--border-bright)"}`,
          borderRadius: "var(--radius-lg)",
          padding: "2rem",
          textAlign: "center",
          cursor: "pointer",
          background: drag ? "var(--accent-glow)" : file ? "var(--green-glow)" : "var(--bg-2)",
          transition: "all var(--transition)",
        }}
      >
        <input
          id={`file-${label}`}
          type="file"
          accept=".csv"
          style={{ display: "none" }}
          onChange={e => { const f = e.target.files?.[0]; if (f) onFile(f); }}
        />
        <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.12em", color: "var(--text-2)", marginBottom: "0.5rem" }}>
          {label}
        </div>
        {file ? (
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--green)" }}>{file.name}</div>
        ) : (
          <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--text-2)" }}>DROP CSV</div>
        )}
      </div>
    );
  };

  // Build drift bar chart data — sorted by drift score desc
  const driftChartData = result
    ? result.column_drift
        .slice()
        .sort((a, b) => b.drift_score - a.drift_score)
        .map(col => ({
          column: col.column.length > 14 ? col.column.slice(0, 12) + "…" : col.column,
          fullName: col.column,
          drift: parseFloat(col.drift_score.toFixed(1)),
          train_mean: col.train_mean != null ? parseFloat(col.train_mean.toFixed(3)) : undefined,
          test_mean: col.test_mean != null ? parseFloat(col.test_mean.toFixed(3)) : undefined,
          pct_diff: col.train_mean != null && col.test_mean != null && col.train_mean !== 0
            ? parseFloat((((col.test_mean - col.train_mean) / Math.abs(col.train_mean)) * 100).toFixed(2))
            : undefined,
        }))
    : [];

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Top bar */}
      <div style={{
        borderBottom: "1px solid var(--border)",
        padding: "0.75rem 2rem",
        display: "flex",
        alignItems: "center",
        gap: "1rem",
        background: "var(--bg-1)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <button
          onClick={onBack}
          style={{ background: "none", border: "none", color: "var(--text-2)", fontFamily: "var(--font-mono)", fontSize: "0.72rem", cursor: "pointer", letterSpacing: "0.08em", padding: 0 }}
          onMouseEnter={e => e.currentTarget.style.color = "var(--text-0)"}
          onMouseLeave={e => e.currentTarget.style.color = "var(--text-2)"}
        >
          ← BACK
        </button>
        <div style={{ width: "1px", height: "16px", background: "var(--border)" }} />
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--text-0)", fontWeight: 600 }}>ATHENA</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-2)" }}>/ TRAIN vs TEST</span>
      </div>

      <div style={{ padding: "2rem", maxWidth: "1200px", margin: "0 auto", width: "100%", display: "flex", flexDirection: "column", gap: "1.5rem" }}>

        {/* Upload zone */}
        <div style={{ display: "flex", gap: "1rem" }}>
          <FileDropZone label="TRAIN.CSV" file={trainFile} onFile={setTrainFile} />
          <div style={{ display: "flex", alignItems: "center", padding: "0 0.5rem", fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-3)" }}>vs</div>
          <FileDropZone label="TEST.CSV" file={testFile} onFile={setTestFile} />
        </div>

        <button
          onClick={handleRun}
          disabled={!trainFile || !testFile || loading}
          style={{
            padding: "0.85rem",
            border: `1px solid ${trainFile && testFile && !loading ? "var(--accent)" : "var(--border)"}`,
            borderRadius: "var(--radius-md)",
            background: trainFile && testFile && !loading ? "var(--accent-glow)" : "var(--bg-3)",
            color: trainFile && testFile && !loading ? "var(--accent)" : "var(--text-2)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.8rem",
            letterSpacing: "0.1em",
            textTransform: "uppercase" as const,
            cursor: trainFile && testFile && !loading ? "pointer" : "not-allowed",
            transition: "all var(--transition)",
          }}
        >
          {loading ? "ANALYZING DRIFT..." : "RUN DRIFT ANALYSIS →"}
        </button>

        {error && <Alert variant="error">{error}</Alert>}

        {/* Results */}
        {result && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>

            {/* Summary stats */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
              <StatBox
                label="Overall Drift Score"
                value={result.overall_drift_score}
                sub={result.overall_drift_score > 50 ? "⚠ Significant drift" : "Within tolerance"}
                accent={result.overall_drift_score > 50 ? "var(--red)" : result.overall_drift_score > 25 ? "var(--yellow)" : "var(--green)"}
              />
              <StatBox label="Drifted Columns" value={result.column_drift.filter(c => c.drift_score > 25).length} accent="var(--yellow)" />
              <StatBox label="New Categories" value={Object.keys(result.new_categories ?? {}).length} accent={Object.keys(result.new_categories ?? {}).length > 0 ? "var(--red)" : "var(--green)"} />
            </div>

            {result.summary && <Alert variant={result.overall_drift_score > 50 ? "warning" : "info"}>{result.summary}</Alert>}

            {/* Drift score bar chart */}
            <Card>
              <SectionHeader label="Drift Score by Column" />
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.63rem", color: "var(--text-2)", marginBottom: "0.75rem", letterSpacing: "0.08em" }}>
                <span style={{ color: "var(--green)" }}>■</span> Low (&lt;25) &nbsp;
                <span style={{ color: "var(--yellow)" }}>■</span> Moderate (25–50) &nbsp;
                <span style={{ color: "var(--red)" }}>■</span> High (&gt;50)
              </div>
              <ResponsiveContainer width="100%" height={Math.max(200, driftChartData.length * 28)}>
                <BarChart
                  data={driftChartData}
                  layout="vertical"
                  margin={{ top: 4, right: 48, left: 8, bottom: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                  <XAxis
                    type="number"
                    domain={[0, 100]}
                    tick={{ fontFamily: "var(--font-mono)", fontSize: 9, fill: "var(--text-2)" }}
                    tickFormatter={v => `${v}`}
                  />
                  <YAxis
                    type="category"
                    dataKey="column"
                    tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--text-1)" }}
                    width={110}
                  />
                  <Tooltip
                    contentStyle={{ background: "var(--bg-2)", border: "1px solid var(--border)", fontFamily: "var(--font-mono)", fontSize: "0.7rem" }}
                    labelStyle={{ color: "var(--text-1)" }}
                    itemStyle={{ color: "var(--text-1)" }}
                    formatter={(v: any, _: any, props: any) => [
                      `${v} drift score`,
                      props.payload.fullName,
                    ]}
                    labelFormatter={() => ""}
                  />
                  <ReferenceLine x={25} stroke="var(--yellow)" strokeDasharray="3 3" strokeOpacity={0.5} />
                  <ReferenceLine x={50} stroke="var(--red)" strokeDasharray="3 3" strokeOpacity={0.5} />
                  <Bar dataKey="drift" radius={[0, 3, 3, 0]}>
                    {driftChartData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.drift > 50 ? "var(--red)" : entry.drift > 25 ? "var(--yellow)" : "var(--green)"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

          {driftChartData.some(d => d.pct_diff !== undefined) && (
            <Card>
              <SectionHeader label="Train vs Test Mean Shift (% difference)" />
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.63rem", color: "var(--text-2)", marginBottom: "0.75rem" }}>
                Positive = test mean higher than train. Normalized per feature.
              </div>
              <ResponsiveContainer width="100%" height={Math.max(200, driftChartData.filter(d => d.pct_diff !== undefined).length * 32)}>
                <BarChart
                  data={driftChartData.filter(d => d.pct_diff !== undefined)}
                  layout="vertical"
                  margin={{ top: 4, right: 48, left: 8, bottom: 4 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fontFamily: "var(--font-mono)", fontSize: 9, fill: "var(--text-2)" }}
                    tickFormatter={v => `${v}%`}
                  />
                  <YAxis
                    type="category"
                    dataKey="column"
                    tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--text-1)" }}
                    width={110}
                  />
                  <ReferenceLine x={0} stroke="var(--border)" strokeOpacity={0.8} />
                  <Tooltip
                    contentStyle={{ background: "var(--bg-2)", border: "1px solid var(--border)", fontFamily: "var(--font-mono)", fontSize: "0.7rem" }}
                    labelStyle={{ color: "var(--text-1)" }}
                    itemStyle={{ color: "var(--text-1)" }}
                    formatter={(v: any, _: any, props: any) => [`${v}%`, props.payload.fullName]}
                    labelFormatter={() => "Mean shift"}
                  />
                  <Bar dataKey="pct_diff" radius={[0, 3, 3, 0]} name="% Shift">
                    {driftChartData.filter(d => d.pct_diff !== undefined).map((entry, i) => (
                      <Cell key={i} fill={(entry.pct_diff ?? 0) > 0 ? "var(--accent)" : "var(--red)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-2)", marginTop: "0.5rem" }}>
                <span style={{ color: "var(--accent)" }}>■</span> Test &gt; Train &nbsp;
                <span style={{ color: "var(--red)" }}>■</span> Test &lt; Train
              </div>
            </Card>
          )}

            {/* New categories */}
            {Object.keys(result.new_categories ?? {}).length > 0 && (
              <Card>
                <SectionHeader label="New Categories in Test (unseen in train)" />
                {Object.entries(result.new_categories).map(([col, cats]) => (
                  <div key={col} style={{ marginBottom: "0.75rem" }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--accent)", marginBottom: "0.4rem" }}>{col}</div>
                    <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
                      {cats.map(c => <Badge key={c} variant="red">{c}</Badge>)}
                    </div>
                  </div>
                ))}
              </Card>
            )}

            {/* Missing rate delta */}
            {Object.keys(result.missing_rate_delta ?? {}).length > 0 && (
              <Card>
                <SectionHeader label="Missing Rate Changes (Train → Test)" />
                <DataTable
                  columns={["Column", "Delta"]}
                  rows={Object.entries(result.missing_rate_delta)
                    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
                    .map(([col, delta]) => [
                      <span style={{ color: "var(--text-1)" }}>{col}</span>,
                      <span style={{ color: delta > 0.1 ? "var(--red)" : delta > 0.05 ? "var(--yellow)" : "var(--green)" }}>
                        {delta > 0 ? "+" : ""}{(delta * 100).toFixed(1)}%
                      </span>,
                    ])}
                />
              </Card>
            )}
          </div>
        )}
      </div>
    </div>
  );
}