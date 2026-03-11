import React, { useState, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine
} from "recharts";
import { AnalysisResult } from "../types";
import {
  Card, SectionHeader, Badge, StatBox, ProgressBar,
  Alert, Tabs, ScoreGauge, DataTable, Divider
} from "./UI";
import { downloadScript } from "../lib/api";
import { generateReport } from "../lib/reportgen";

interface ReportPageProps {
  result: AnalysisResult;
  filename: string;
  onBack: () => void;
}

export default function ReportPage({ result, filename, onBack }: ReportPageProps) {
  const [tab, setTab] = useState("overview");
  const [isPrinting, setIsPrinting] = useState(false);

  useEffect(() => {
    if (!isPrinting) return;
    const handler = () => setIsPrinting(false);
    window.addEventListener("afterprint", handler);
    window.print();
    return () => window.removeEventListener("afterprint", handler);
  }, [isPrinting]);

  const { eda, ml_diagnostics: ml } = result;

  const hasMissing = Object.values(eda.missingness.missing_values_percent_cols).some(v => v > 0);
  const mlScore = ml.ML_readiness_score ?? null;
  const structScore = eda.data_structural_report.data_structural_score;

  const tabs = [
    { id: "overview", label: "OVERVIEW" },
    { id: "features", label: "FEATURES" },
    { id: "quality", label: "QUALITY" },
    { id: "ml", label: "ML DIAGNOSTICS" },
    ...(eda.nlp ? [{ id: "nlp", label: "NLP ANALYSIS" }] : [])
  ];

  // Build skewness chart data from numeric summary
  const skewnessChartData = eda.numeric_summary
    .filter(n => n.skewness != null)
    .map(n => ({
      column: n.column.length > 14 ? n.column.slice(0, 12) + "…" : n.column,
      fullName: n.column,
      skewness: parseFloat(n.skewness.toFixed(3)),
      abs: Math.abs(n.skewness),
    }))
    .sort((a, b) => b.abs - a.abs);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Top bar */}
      <div
        data-print-hide
        style={{
          borderBottom: "1px solid var(--border)",
          padding: "0.75rem 2rem",
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          background: "var(--bg-1)",
          position: "sticky",
          top: 0,
          zIndex: 100,
        }}
      >
        <button
          onClick={onBack}
          style={{
            background: "none",
            border: "none",
            color: "var(--text-2)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.72rem",
            cursor: "pointer",
            letterSpacing: "0.08em",
            padding: 0,
          }}
          onMouseEnter={e => e.currentTarget.style.color = "var(--text-0)"}
          onMouseLeave={e => e.currentTarget.style.color = "var(--text-2)"}
        >
          ← BACK
        </button>
        <div style={{ width: "1px", height: "16px", background: "var(--border)" }} />
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8rem", color: "var(--text-0)", fontWeight: 600 }}>
          ATHENA
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-2)" }}>
          / {filename}
        </span>
        <div style={{ flex: 1 }} />
        <Badge variant={result.mode === "labeled" ? "accent" : "neutral"}>
          {result.mode.toUpperCase()}
        </Badge>

        <button
          onClick={() => downloadScript(eda, `${filename}_preprocessing.py`)}
          style={{
            padding: "0.35rem 0.75rem",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            background: "transparent",
            color: "var(--text-1)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.65rem",
            letterSpacing: "0.08em",
            cursor: "pointer",
            transition: "all var(--transition)",
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = "var(--green-dim)"; e.currentTarget.style.color = "var(--green)"; }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.color = "var(--text-1)"; }}
        >
          ↓ SCRIPT.PY
        </button>
        <button
          onClick={() => generateReport(result, filename)}
          style={{
            padding: "0.35rem 0.75rem",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            background: "transparent",
            color: "var(--text-1)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.65rem",
            letterSpacing: "0.08em",
            cursor: "pointer",
            transition: "all var(--transition)",
          }}
          onMouseEnter={e => { e.currentTarget.style.borderColor = "var(--accent-dim)"; e.currentTarget.style.color = "var(--accent)"; }}
          onMouseLeave={e => { e.currentTarget.style.borderColor = "var(--border)"; e.currentTarget.style.color = "var(--text-1)"; }}
        >
          ↓ REPORT
        </button>
      </div>

      <div style={{ padding: "2rem", display: "flex", flexDirection: "column", gap: "1.5rem", maxWidth: "1200px", margin: "0 auto", width: "100%" }}>

        <Tabs data-print-hide tabs={tabs} active={tab} onChange={setTab} />

        {/* ── OVERVIEW ── */}
        {(tab === "overview" || isPrinting) && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}>
              <StatBox label="Rows" value={eda.basic.rows.toLocaleString()} />
              <StatBox label="Columns" value={eda.basic.cols} />
              <StatBox label="Numeric Features" value={eda.col_views.numeric_features.length} />
              <StatBox label="Categorical" value={eda.col_views.categorical_features.length} />
              <StatBox label="Identifiers" value={eda.col_views.identifiers.length} />
              {eda.free_text && <StatBox label="Free Text" value={eda.free_text.count} accent={eda.free_text.count > 0 ? "var(--yellow)" : undefined} />}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <Card>
                <SectionHeader label="ML Readiness Score" />
                <div style={{ display: "flex", alignItems: "center", gap: "2rem" }}>
                  {mlScore !== null ? (
                    <ScoreGauge score={mlScore} />
                  ) : (
                    <div style={{ fontFamily: "var(--font-mono)", color: "var(--text-2)", fontSize: "0.8rem" }}>N/A — unlabeled</div>
                  )}
                  {ml.component_scores && (
                    <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                      {Object.entries(ml.component_scores).map(([k, v]) => (
                        <ProgressBar key={k} label={k.replace(/_/g, " ").toUpperCase()} value={v} />
                      ))}
                    </div>
                  )}
                </div>
              </Card>

              <Card>
                <SectionHeader label="Data Structure Score" />
                <div style={{ display: "flex", alignItems: "center", gap: "2rem" }}>
                  <ScoreGauge score={structScore} />
                  <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--text-2)" }}>DROP COLUMNS</div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: eda.data_structural_report.action_plan.drop_columns.length > 0 ? "var(--red)" : "var(--green)" }}>
                      {eda.data_structural_report.action_plan.drop_columns.length > 0
                        ? eda.data_structural_report.action_plan.drop_columns.join(", ")
                        : "None"}
                    </div>
                    <Divider />
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--text-2)" }}>LOG TRANSFORMS</div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.85rem", color: "var(--accent)" }}>
                      {[...eda.data_structural_report.action_plan.apply_log_transform, ...eda.data_structural_report.action_plan.apply_log1p_transform].join(", ") || "None"}
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {eda.data_structural_report.critical_alerts.length > 0 && (
              <Card>
                <SectionHeader label="Critical Alerts" accent />
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {eda.data_structural_report.critical_alerts.map((a, i) => (
                    <Alert key={i} variant="error">{a}</Alert>
                  ))}
                </div>
              </Card>
            )}

            {ml.penalties && Object.keys(ml.penalties).length > 0 && (
              <Card>
                <SectionHeader label="ML Penalties" />
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {Object.entries(ml.penalties).flatMap(([, issues]) =>
                    issues.map((issue, i) => <Alert key={i} variant="warning">{issue}</Alert>)
                  )}
                </div>
              </Card>
            )}

            {ml.recommendations && Object.keys(ml.recommendations).length > 0 && (
              <Card>
                <SectionHeader label="Recommendations" />
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                  {Object.entries(ml.recommendations).flatMap(([section, recs]) =>
                    recs.map((rec, i) => (
                      <div key={i} style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
                        <Badge variant="accent">{section}</Badge>
                        <span style={{ fontSize: "0.82rem", color: "var(--text-1)" }}>{rec}</span>
                      </div>
                    ))
                  )}
                </div>
              </Card>
            )}
          </div>
        )}

        {/* ── FEATURES ── */}
        {(tab === "features" || isPrinting) && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>

            <Card>
              <SectionHeader label="Column Classification" />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "0.75rem" }}>
                {[
                  { label: "NUMERIC FEATURES", cols: eda.col_views.numeric_features, color: "var(--accent)" },
                  { label: "CATEGORICAL", cols: eda.col_views.categorical_features, color: "var(--green)" },
                  { label: "IDENTIFIERS", cols: eda.col_views.identifiers, color: "var(--red)" },
                  { label: "ENTITY KEYS", cols: eda.col_views.entity_keys, color: "var(--yellow)" },
                ].map(({ label, cols, color }) => (
                  <div key={label} style={{ background: "var(--bg-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "0.75rem" }}>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.62rem", letterSpacing: "0.1em", color: "var(--text-2)", marginBottom: "0.5rem" }}>{label}</div>
                    {cols.length === 0
                      ? <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-3)" }}>none</span>
                      : cols.map(c => (
                          <div key={c} style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color, marginBottom: "0.2rem" }}>{c}</div>
                        ))
                    }
                  </div>
                ))}
              </div>
            </Card>

            {/* Skewness Chart */}
            {skewnessChartData.length > 0 && (
              <Card>
                <SectionHeader label="Feature Skewness" />
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.63rem", color: "var(--text-2)", marginBottom: "0.75rem", letterSpacing: "0.08em" }}>
                  <span style={{ color: "var(--green)" }}>■</span> Near-normal (&lt;1) &nbsp;
                  <span style={{ color: "var(--yellow)" }}>■</span> Moderate (1–2) &nbsp;
                  <span style={{ color: "var(--red)" }}>■</span> Highly skewed (&gt;2) &nbsp;
                  <span style={{ color: "var(--text-3)" }}>— dashed lines at ±1</span>
                </div>
                <ResponsiveContainer width="100%" height={Math.max(200, skewnessChartData.length * 28)}>
                  <BarChart
                    data={skewnessChartData}
                    layout="vertical"
                    margin={{ top: 4, right: 48, left: 8, bottom: 4 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                    <XAxis
                      type="number"
                      tick={{ fontFamily: "var(--font-mono)", fontSize: 9, fill: "var(--text-2)" }}
                      tickFormatter={v => v.toFixed(1)}
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
                      formatter={(v: any, _: any, props: any) => [`${v}`, props.payload.fullName]}
                      labelFormatter={() => "Skewness"}
                    />
                    <ReferenceLine x={1} stroke="var(--yellow)" strokeDasharray="3 3" strokeOpacity={0.6} />
                    <ReferenceLine x={-1} stroke="var(--yellow)" strokeDasharray="3 3" strokeOpacity={0.6} />
                    <ReferenceLine x={0} stroke="var(--border)" strokeOpacity={0.8} />
                    <Bar dataKey="skewness" radius={[0, 3, 3, 0]}>
                      {skewnessChartData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={
                            entry.abs > 2 ? "var(--red)" :
                            entry.abs > 1 ? "var(--yellow)" :
                            "var(--green)"
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            )}

            {eda.numeric_summary.length > 0 && (
              <Card>
                <SectionHeader label="Numeric Feature Summary" />
                <DataTable
                  columns={["Column", "Mean", "Std", "Min", "Max", "Skew", "Missing%"]}
                  rows={eda.numeric_summary.map(n => [
                    <span style={{ color: "var(--accent)" }}>{n.column}</span>,
                    n.mean?.toFixed(2) ?? "—",
                    n.std?.toFixed(2) ?? "—",
                    n.min?.toFixed(2) ?? "—",
                    n.max?.toFixed(2) ?? "—",
                    <span style={{ color: Math.abs(n.skewness) > 1 ? "var(--yellow)" : "var(--text-1)" }}>
                      {n.skewness?.toFixed(3) ?? "—"}
                    </span>,
                    <span style={{ color: n.missing_pct > 5 ? "var(--red)" : "var(--text-1)" }}>
                      {n.missing_pct?.toFixed(1)}%
                    </span>,
                  ])}
                />
              </Card>
            )}

            {eda.categorical_summary.length > 0 && (
              <Card>
                <SectionHeader label="Categorical Feature Summary" />
                <DataTable
                  columns={["Column", "Unique", "Top Category %", "Missing%", "High Card."]}
                  rows={eda.categorical_summary.map(c => [
                    <span style={{ color: "var(--green)" }}>{c.column}</span>,
                    c.n_unique,
                    `${(c.most_frequent_pct * 100).toFixed(1)}%`,
                    <span style={{ color: c.missing_pct > 5 ? "var(--red)" : "var(--text-1)" }}>
                      {c.missing_pct.toFixed(1)}%
                    </span>,
                    c.high_cardinality ? <Badge variant="yellow">HIGH</Badge> : <Badge variant="green">OK</Badge>,
                  ])}
                />
              </Card>
            )}

            {eda.free_text && eda.free_text.count > 0 && (
              <Card>
                <SectionHeader label="Free Text Columns Detected" />
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginBottom: "0.75rem" }}>
                  <Alert variant="warning">
                    {eda.free_text.count} free-text column(s) detected. These were excluded from ML training automatically.
                  </Alert>
                </div>
                <DataTable
                  columns={["Column", "Avg Length", "Unique Ratio", "Recommendation"]}
                  rows={eda.free_text.free_text_columns.map(f => [
                    <span style={{ color: "var(--yellow)" }}>{f.column}</span>,
                    f.avg_length,
                    f.unique_ratio.toFixed(3),
                    <span style={{ fontSize: "0.7rem", color: "var(--text-2)" }}>{f.recommendation}</span>,
                  ])}
                />
              </Card>
            )}

            {eda.feature_correlation && (
              <Card>
                <SectionHeader label="Feature → Target Correlation" />
                <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                  {Object.entries(eda.feature_correlation.numeric_target_correlation).map(([col, vals]) => (
                    <div key={col} style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--accent)", width: "160px", flexShrink: 0 }}>{col}</span>
                      <ProgressBar
                        value={Math.abs(vals.pearson) * 100}
                        color={Math.abs(vals.pearson) > 0.5 ? "var(--red)" : "var(--accent)"}
                        showValue={false}
                      />
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: "var(--text-1)", width: "80px", flexShrink: 0 }}>
                        r={vals.pearson.toFixed(3)}
                      </span>
                    </div>
                  ))}
                  <Divider />
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-2)", letterSpacing: "0.1em", marginBottom: "0.5rem" }}>CATEGORICAL IMPACT</div>
                  {eda.feature_correlation.categorical_target_impact.map((item) => (
                    <div key={item.column} style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--green)", width: "160px", flexShrink: 0 }}>{item.column}</span>
                      <Badge variant={item.likely_influential ? "yellow" : "neutral"}>
                        {item.likely_influential ? "INFLUENTIAL" : "WEAK"}
                      </Badge>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-2)" }}>
                        shift={item.distribution_shift?.toFixed(3) ?? "—"}
                      </span>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        )}

        {/* ── QUALITY ── */}
        {(tab === "quality" || isPrinting) && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>

            <Card>
              <SectionHeader label="Missing Values" />
              {!hasMissing ? (
                <Alert variant="success">No missing values detected across all features.</Alert>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
                  {Object.entries(eda.missingness.missing_values_percent_cols)
                    .filter(([, v]) => v > 0)
                    .sort(([, a], [, b]) => b - a)
                    .map(([col, pct]) => (
                      <div key={col} style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", color: "var(--text-1)", width: "180px", flexShrink: 0 }}>{col}</span>
                        <ProgressBar
                          value={pct}
                          color={pct > 30 ? "var(--red)" : pct > 10 ? "var(--yellow)" : "var(--orange)"}
                          showValue={false}
                        />
                        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.72rem", color: pct > 30 ? "var(--red)" : "var(--yellow)", width: "48px", flexShrink: 0 }}>
                          {pct.toFixed(1)}%
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </Card>

            <Card>
              <SectionHeader label="Outlier Analysis" />
              {(eda.outliers?.column_outliers?.length ?? 0) === 0 ? (
                <Alert variant="success">No significant outliers detected.</Alert>
              ) : (
                <DataTable
                  columns={["Column", "Skewness", "Kurtosis", "Outlier %", "Distribution", "Action"]}
                  rows={eda.outliers.column_outliers.map(o => [
                    <span style={{ color: "var(--accent)" }}>{o.column}</span>,
                    <span style={{ color: Math.abs(o.skewness) > 1 ? "var(--yellow)" : "var(--text-1)" }}>{o.skewness?.toFixed(3)}</span>,
                    o.kurtosis?.toFixed(3),
                    <span style={{ color: o.outlier_pct > 5 ? "var(--red)" : "var(--text-1)" }}>{o.outlier_pct?.toFixed(1)}%</span>,
                    <Badge variant={o.distribution_type === "Gaussian-like" ? "green" : "yellow"}>
                      {o.distribution_type === "Gaussian-like" ? "GAUSSIAN" : "SKEWED"}
                    </Badge>,
                    o.suggest_log_transform ? <Badge variant="accent">LOG1P</Badge> : <span style={{ color: "var(--text-3)" }}>—</span>,
                  ])}
                />
              )}
            </Card>

            <Card>
              <SectionHeader label="Feature Redundancy" />
              <div style={{ marginBottom: "0.75rem" }}>
                <Badge variant={
                  eda.feature_redundancy.overall_redundancy_risk === "Low" ? "green" :
                  eda.feature_redundancy.overall_redundancy_risk === "Moderate" ? "yellow" : "red"
                }>
                  {eda.feature_redundancy.overall_redundancy_risk?.toUpperCase()} RISK
                </Badge>
              </div>
              {(eda.feature_redundancy.highly_correlated_pairs?.length ?? 0) === 0 ? (
                <Alert variant="success">No strongly correlated feature pairs detected.</Alert>
              ) : (
                <DataTable
                  columns={["Feature 1", "Feature 2", "Correlation", "Action"]}
                  rows={eda.feature_redundancy.highly_correlated_pairs.map(p => [
                    <span style={{ color: "var(--accent)" }}>{p.feature_1}</span>,
                    <span style={{ color: "var(--accent)" }}>{p.feature_2}</span>,
                    <span style={{ color: "var(--red)" }}>{p.correlation?.toFixed(3)}</span>,
                    <span style={{ fontSize: "0.7rem", color: "var(--text-2)" }}>{p.recommended_action}</span>,
                  ])}
                />
              )}
            </Card>
          </div>
        )}

        {/* ── ML DIAGNOSTICS ── */}
        {(tab === "ml" || isPrinting) && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>

            {ml.status === "ok" && ml.baseline && (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
                  <StatBox
                    label={`Baseline ${ml.metric?.toUpperCase()}`}
                    value={ml.baseline.mean_score.toFixed(3)}
                    sub={`±${ml.baseline.std_score.toFixed(3)} over ${ml.baseline.n_folds} folds`}
                    accent="var(--accent)"
                  />
                  <StatBox
                    label="Overfitting"
                    value={ml.baseline.learning_dynamics.overfitting_detected ? "DETECTED" : "CLEAN"}
                    accent={ml.baseline.learning_dynamics.overfitting_detected ? "var(--yellow)" : "var(--green)"}
                  />
                  <StatBox
                    label="Train-Val Gap"
                    value={ml.baseline.learning_dynamics.mean_train_val_gap.toFixed(3)}
                    accent={ml.baseline.learning_dynamics.mean_train_val_gap > 0.1 ? "var(--yellow)" : "var(--green)"}
                  />
                  {ml.leakage && (
                    <StatBox
                      label="Leakage Score"
                      value={ml.leakage.leakage_score}
                      sub={ml.leakage.leakage_score > 30 ? "⚠ Review suspicious features" : "Clean"}
                      accent={ml.leakage.leakage_score > 30 ? "var(--red)" : "var(--green)"}
                    />
                  )}
                </div>

                <Card>
                  <SectionHeader label="Cross-Validation Fold Scores" />
                  <div style={{ display: "flex", gap: "0.75rem" }}>
                    {ml.baseline.fold_scores.map((s, i) => (
                      <div key={i} style={{
                        flex: 1,
                        background: "var(--bg-3)",
                        border: "1px solid var(--border)",
                        borderRadius: "var(--radius-md)",
                        padding: "0.75rem",
                        textAlign: "center",
                        fontFamily: "var(--font-mono)",
                      }}>
                        <div style={{ fontSize: "0.6rem", color: "var(--text-2)", marginBottom: "0.25rem" }}>FOLD {i + 1}</div>
                        <div style={{ fontSize: "1.1rem", color: "var(--accent)" }}>{s.toFixed(4)}</div>
                      </div>
                    ))}
                  </div>
                </Card>

                {ml.leakage && ml.leakage.suspicious_features.length > 0 && (
                  <Card>
                    <SectionHeader label="Suspicious Features (Leakage Risk)" />
                    <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
                      {ml.leakage.suspicious_features.map(f => (
                        <Badge key={f} variant="red">{f}</Badge>
                      ))}
                    </div>
                    <Alert variant="warning">
                      Review these features — they may be causally downstream of the target or derived from it.
                    </Alert>
                  </Card>
                )}

                {ml.stability && (
                  <Card>
                    <SectionHeader label="Stability Analysis" />
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}>
                      <StatBox label="Stability Score" value={ml.stability.mean_score.toFixed(3)} accent="var(--accent)" />
                      <StatBox
                        label="Stability"
                        value={ml.stability.is_unstable ? "UNSTABLE" : "STABLE"}
                        accent={ml.stability.is_unstable ? "var(--red)" : "var(--green)"}
                      />
                      <StatBox label="Score Std" value={ml.stability.std_score.toFixed(3)} />
                      <StatBox label="Loss Gap Ratio" value={ml.stability.loss_gap_ratio.toFixed(3)} />
                    </div>
                  </Card>
                )}

                {/* Learning Curves */}
                {ml.baseline?.fold_curves && ml.baseline.fold_curves.length > 0 && (
                  <Card>
                    <SectionHeader label="Learning Curves" />
                    <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
                      {ml.baseline.fold_curves.map((fc: any) => {
                        const data = fc.train.map((v: number, i: number) => ({
                          iter: i + 1,
                          train: v,
                          val: fc.val[i] ?? null,
                        }));
                        return (
                          <div key={fc.fold}>
                            <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-2)", letterSpacing: "0.1em", marginBottom: "0.5rem" }}>
                              FOLD {fc.fold}
                            </div>
                            <ResponsiveContainer width="100%" height={200}>
                              <LineChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 24 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                                <XAxis
                                  dataKey="iter"
                                  tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--text-2)" }}
                                  label={{ value: "Iteration", position: "insideBottom", offset: -10, style: { fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--text-2)" } }}
                                />
                                <YAxis tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--text-2)" }} width={60} />
                                <Tooltip
                                  contentStyle={{ background: "var(--bg-2)", border: "1px solid var(--border)", fontFamily: "var(--font-mono)", fontSize: "0.7rem" }}
                                  itemStyle={{ color: "var(--text-1)" }}
                                  labelStyle={{ color: "var(--text-2)" }}
                                />
                                <Legend wrapperStyle={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem" }} />
                                <Line type="monotone" dataKey="train" stroke="var(--accent)" dot={false} strokeWidth={1.5} name="Train Loss" />
                                <Line type="monotone" dataKey="val" stroke="var(--green)" dot={false} strokeWidth={1.5} name="Val Loss" strokeDasharray="4 2" />
                              </LineChart>
                            </ResponsiveContainer>
                          </div>
                        );
                      })}
                    </div>
                  </Card>
                )}
              </>
            )}

            {/* Unlabeled: clustering */}
            {ml.clustering && (
              <Card>
                <SectionHeader label="Clustering Analysis" />
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
                  <StatBox label="Best K" value={ml.clustering.best_k} accent="var(--accent)" />
                  <StatBox label="Best Silhouette" value={ml.clustering.best_silhouette?.toFixed(3)} />
                </div>
                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-2)", letterSpacing: "0.1em", marginBottom: "0.5rem" }}>
                  ELBOW CURVE — SILHOUETTE SCORE VS K
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart
                    data={Object.entries(ml.clustering.silhouette_scores ?? {}).map(([k, v]) => ({ k: Number(k), silhouette: Number(v) }))}
                    margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis
                      dataKey="k"
                      tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--text-2)" }}
                      label={{ value: "K", position: "insideBottom", offset: -2, style: { fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--text-2)" } }}
                    />
                    <YAxis tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--text-2)" }} width={60} domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{ background: "var(--bg-2)", border: "1px solid var(--border)", fontFamily: "var(--font-mono)", fontSize: "0.7rem" }}
                      labelStyle={{ color: "var(--text-2)" }}
                      itemStyle={{ color: "var(--text-1)" }}
                      formatter={(v: any) => [Number(v).toFixed(4), "Silhouette"]}
                      labelFormatter={(k) => `k=${k}`}
                    />
                    <Line type="monotone" dataKey="silhouette" stroke="var(--accent)" dot={{ fill: "var(--accent)", r: 4 }} strokeWidth={2} name="Silhouette" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            )}

            {/* Anomaly detection */}
            {ml.anomaly_detection && (
              <Card>
                <SectionHeader label="Anomaly Detection" />
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
                  <StatBox
                    label="Anomaly Rate"
                    value={`${ml.anomaly_detection.anomaly_percent?.toFixed(1)}%`}
                    accent={
                      ml.anomaly_detection.predominantly_binary
                        ? "var(--text-2)"
                        : ml.anomaly_detection.anomaly_percent > 15 ? "var(--red)" : "var(--green)"
                    }
                  />
                  <StatBox label="Method" value={ml.anomaly_detection.method} />
                  <StatBox label="Mean Score" value={ml.anomaly_detection.mean_score?.toFixed(3)} />
                </div>

                {ml.anomaly_detection.score_distribution && (
                  <>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-2)", letterSpacing: "0.1em", marginBottom: "0.5rem" }}>
                      ANOMALY SCORE DISTRIBUTION
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart
                        data={(() => {
                          const scores = ml.anomaly_detection.score_distribution;
                          const min = Math.min(...scores);
                          const max = Math.max(...scores);
                          const bins = 30;
                          const binSize = (max - min) / bins;
                          const counts = Array(bins).fill(0);
                          scores.forEach((s: number) => {
                            const idx = Math.min(Math.floor((s - min) / binSize), bins - 1);
                            counts[idx]++;
                          });
                          return counts.map((count, i) => ({
                            bin: (min + i * binSize).toFixed(3),
                            count,
                          }));
                        })()}
                        margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="bin" tick={{ fontFamily: "var(--font-mono)", fontSize: 9, fill: "var(--text-2)" }} interval={5} />
                        <YAxis tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--text-2)" }} width={40} />
                        <Tooltip
                          contentStyle={{ background: "var(--bg-2)", border: "1px solid var(--border)", fontFamily: "var(--font-mono)", fontSize: "0.7rem" }}
                          labelStyle={{ color: "var(--text-1)" }}
                          itemStyle={{ color: "var(--text-1)" }}
                          formatter={(v: any) => [v, "Count"]}
                          labelFormatter={(l) => `Score: ${l}`}
                        />
                        <Bar dataKey="count" radius={[2, 2, 0, 0]}>
                          {(() => {
                            const scores = ml.anomaly_detection.score_distribution;
                            const min = Math.min(...scores);
                            const binSize = (Math.max(...scores) - min) / 30;
                            return Array(30).fill(0).map((_, i) => (
                              <Cell key={i} fill={(min + i * binSize) < 0 ? "var(--red)" : "var(--accent)"} />
                            ));
                          })()}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", color: "var(--text-2)", marginTop: "0.5rem" }}>
                      <span style={{ color: "var(--red)" }}>■</span> Anomaly (score &lt; 0) &nbsp;
                      <span style={{ color: "var(--accent)" }}>■</span> Normal (score ≥ 0)
                    </div>
                  </>
                )}

                {ml.anomaly_detection.predominantly_binary && (
                  <div style={{ marginTop: "0.5rem", fontSize: "0.72rem", color: "var(--text-2)", fontFamily: "var(--font-mono)" }}>
                    ⚠ Unreliable on predominantly binary features
                  </div>
                )}
              </Card>
            )}

            {ml.status !== "ok" && (
              <Alert variant="error">ML Diagnostics status: {ml.status}</Alert>
            )}
          </div>
        )}

        {/* ── NLP ANALYSIS ── */}
        {tab === "nlp" && eda.nlp && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem", animation: "fadeIn 0.3s ease" }}>
            {(eda.nlp.nlp_analysis ?? []).map((col: any) => (
              <Card key={col.column}>
                <SectionHeader label={`${col.column}`} />

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
                  <StatBox label="Avg Words" value={col.text_stats.avg_word_count} />
                  <StatBox label="Avg Chars" value={col.text_stats.avg_char_length} />
                  <StatBox label="Vocab Size" value={col.text_stats.vocab_size.toLocaleString()} />
                  <StatBox label="Lexical Diversity" value={col.text_stats.lexical_diversity} />
                </div>

                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.1em", color: "var(--text-2)", marginBottom: "0.5rem" }}>TEXT QUALITY</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
                  <StatBox label="Empty Ratio" value={`${(col.text_quality.empty_ratio * 100).toFixed(1)}%`} accent={col.text_quality.empty_ratio > 0.05 ? "var(--red)" : "var(--green)"} />
                  <StatBox label="Duplicate Ratio" value={`${(col.text_quality.duplicate_ratio * 100).toFixed(1)}%`} accent={col.text_quality.duplicate_ratio > 0.1 ? "var(--yellow)" : "var(--green)"} />
                  <StatBox label="URL Ratio" value={`${(col.text_quality.url_ratio * 100).toFixed(1)}%`} />
                  <StatBox label="Number Ratio" value={`${(col.text_quality.number_ratio * 100).toFixed(1)}%`} />
                </div>

                <Alert variant={col.ml_readiness.suitable_for_embeddings ? "success" : "warning"}>
                  {col.ml_readiness.vectorization_recommendation}
                </Alert>

                <Divider />

                <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.1em", color: "var(--text-2)", marginBottom: "0.5rem", marginTop: "0.5rem" }}>TOP WORDS</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.4rem", marginBottom: "1rem" }}>
                  {col.text_stats.top_words.map((w: any) => (
                    <div key={w.word} style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.72rem",
                      padding: "0.2rem 0.5rem",
                      background: "var(--bg-3)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius-sm)",
                      color: "var(--accent)",
                    }}>
                      {w.word} <span style={{ color: "var(--text-3)" }}>{w.count}</span>
                    </div>
                  ))}
                </div>

                {col.per_class_analysis && (
                  <>
                    <Divider />
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: "0.65rem", letterSpacing: "0.1em", color: "var(--text-2)", marginBottom: "0.75rem", marginTop: "0.5rem" }}>PER-CLASS ANALYSIS</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                      {col.per_class_analysis.map((cls: any) => (
                        <div key={cls.class} style={{ background: "var(--bg-3)", border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "0.75rem" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
                            <Badge variant="accent">{cls.class}</Badge>
                            <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.7rem", color: "var(--text-2)" }}>
                              {cls.count.toLocaleString()} samples · avg {cls.avg_word_count} words · avg {cls.avg_char_length} chars
                            </span>
                          </div>
                          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem" }}>
                            {cls.top_words.map((w: any) => (
                              <div key={w.word} style={{
                                fontFamily: "var(--font-mono)",
                                fontSize: "0.68rem",
                                padding: "0.15rem 0.4rem",
                                background: "var(--bg-4)",
                                border: "1px solid var(--border)",
                                borderRadius: "var(--radius-sm)",
                                color: "var(--green)",
                              }}>
                                {w.word} <span style={{ color: "var(--text-3)" }}>{w.count}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}