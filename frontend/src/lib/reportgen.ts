import { AnalysisResult } from "../types";

function fmt(n: number | null | undefined, decimals = 2): string {
  if (n == null || isNaN(n)) return "—";
  return n.toFixed(decimals);
}

function pct(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return "—";
  return n.toFixed(1) + "%";
}

function scoreColor(score: number): string {
  if (score >= 80) return "#00ff88";
  if (score >= 60) return "#ffcc00";
  return "#ff4466";
}

function buildTable(headers: string[], rows: string[][]): string {
  return `
    <table>
      <thead>
        <tr>${headers.map(h => `<th>${h}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`).join("")}
      </tbody>
    </table>
  `;
}

function section(title: string, content: string): string {
  return `
    <div class="section">
      <div class="section-title">${title}</div>
      ${content}
    </div>
  `;
}

function badge(text: string, color = "#00d4ff"): string {
  return `<span class="badge" style="border-color:${color};color:${color}">${text}</span>`;
}

function alertBox(text: string, variant: "error" | "warning" | "success" | "info" = "info"): string {
  const colors: Record<string, string> = {
    error: "#ff4466",
    warning: "#ffcc00",
    success: "#00ff88",
    info: "#00d4ff",
  };
  const c = colors[variant];
  return `<div class="alert" style="border-left-color:${c};color:${c}">${text}</div>`;
}

// ── SVG chart helpers ─────────────────────────────────────────────────────────

function svgHorizontalBarChart(
  data: { label: string; value: number; color: string }[],
  options: { width?: number; barHeight?: number; maxValue?: number; referenceLines?: number[] } = {}
): string {
  const W = options.width ?? 700;
  const BAR_H = options.barHeight ?? 18;
  const GAP = 8;
  const LABEL_W = 130;
  const VAL_W = 50;
  const CHART_W = W - LABEL_W - VAL_W - 16;
  const MAX = options.maxValue ?? Math.max(...data.map(d => Math.abs(d.value)), 1);
  const H = data.length * (BAR_H + GAP) + 20;

  const bars = data.map((d, i) => {
    const y = i * (BAR_H + GAP) + 10;
    const barW = (Math.abs(d.value) / MAX) * CHART_W;
    const isNeg = d.value < 0;
    const zeroX = LABEL_W + (isNeg ? CHART_W / 2 : 0);
    const barX = isNeg ? LABEL_W + CHART_W / 2 - barW : LABEL_W;
    return `
      <text x="${LABEL_W - 6}" y="${y + BAR_H / 2 + 4}" text-anchor="end" fill="#8fa8c0" font-size="10" font-family="IBM Plex Mono,monospace">${d.label}</text>
      <rect x="${barX}" y="${y}" width="${Math.max(barW, 1)}" height="${BAR_H}" fill="${d.color}" rx="2"/>
      <text x="${barX + barW + 4}" y="${y + BAR_H / 2 + 4}" fill="${d.color}" font-size="9" font-family="IBM Plex Mono,monospace">${d.value}</text>
    `;
  }).join("");

  const refLines = (options.referenceLines ?? []).map(rx => {
    const x = LABEL_W + (rx / MAX) * CHART_W;
    return `<line x1="${x}" y1="5" x2="${x}" y2="${H - 5}" stroke="#ffcc00" stroke-dasharray="3,3" stroke-opacity="0.5"/>`;
  }).join("");

  return `
    <svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0.5rem 0">
      <rect width="${W}" height="${H}" fill="transparent"/>
      ${refLines}
      ${bars}
    </svg>
  `;
}

function svgLineChart(
  series: { data: number[]; color: string; name: string; dashed?: boolean }[],
  options: { width?: number; height?: number; xLabel?: string } = {}
): string {
  const W = options.width ?? 700;
  const H = options.height ?? 180;
  const PAD = { top: 10, right: 20, bottom: 30, left: 50 };
  const CW = W - PAD.left - PAD.right;
  const CH = H - PAD.top - PAD.bottom;

  const allValues = series.flatMap(s => s.data);
  const minV = Math.min(...allValues);
  const maxV = Math.max(...allValues);
  const range = maxV - minV || 1;
  const maxLen = Math.max(...series.map(s => s.data.length));

  const toX = (i: number) => PAD.left + (i / (maxLen - 1)) * CW;
  const toY = (v: number) => PAD.top + CH - ((v - minV) / range) * CH;

  const paths = series.map(s => {
    const pts = s.data.map((v, i) => `${toX(i)},${toY(v)}`).join(" L ");
    return `<polyline points="${s.data.map((v, i) => `${toX(i)},${toY(v)}`).join(" ")}" fill="none" stroke="${s.color}" stroke-width="1.5" ${s.dashed ? 'stroke-dasharray="5,3"' : ''}/>`;
  }).join("");

  // Y axis ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(t => {
    const v = minV + t * range;
    const y = PAD.top + CH - t * CH;
    return `
      <line x1="${PAD.left - 4}" x2="${PAD.left}" y1="${y}" y2="${y}" stroke="#1e2d3d"/>
      <text x="${PAD.left - 6}" y="${y + 3}" text-anchor="end" fill="#4d6478" font-size="9" font-family="IBM Plex Mono,monospace">${v.toFixed(2)}</text>
      <line x1="${PAD.left}" x2="${PAD.left + CW}" y1="${y}" y2="${y}" stroke="#1e2d3d" stroke-dasharray="2,4"/>
    `;
  }).join("");

  // X axis ticks
  const xTickCount = Math.min(8, maxLen);
  const xTicks = Array.from({ length: xTickCount }, (_, i) => {
    const idx = Math.round((i / (xTickCount - 1)) * (maxLen - 1));
    const x = toX(idx);
    return `<text x="${x}" y="${H - PAD.bottom + 14}" text-anchor="middle" fill="#4d6478" font-size="9" font-family="IBM Plex Mono,monospace">${idx + 1}</text>`;
  }).join("");

  const legend = series.map((s, i) => `
    <rect x="${PAD.left + i * 120}" y="${H - 8}" width="16" height="3" fill="${s.color}" ${s.dashed ? 'stroke-dasharray="3,2"' : ''}/>
    <text x="${PAD.left + i * 120 + 20}" y="${H - 4}" fill="#8fa8c0" font-size="9" font-family="IBM Plex Mono,monospace">${s.name}</text>
  `).join("");

  return `
    <svg width="${W}" height="${H + 16}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0.5rem 0">
      <rect width="${W}" height="${H + 16}" fill="transparent"/>
      <line x1="${PAD.left}" x2="${PAD.left}" y1="${PAD.top}" y2="${PAD.top + CH}" stroke="#1e2d3d"/>
      <line x1="${PAD.left}" x2="${PAD.left + CW}" y1="${PAD.top + CH}" y2="${PAD.top + CH}" stroke="#1e2d3d"/>
      ${yTicks}
      ${xTicks}
      ${paths}
      ${legend}
      ${options.xLabel ? `<text x="${PAD.left + CW / 2}" y="${H - 2}" text-anchor="middle" fill="#4d6478" font-size="9" font-family="IBM Plex Mono,monospace">${options.xLabel}</text>` : ""}
    </svg>
  `;
}

function svgHistogram(
  values: number[],
  options: { width?: number; height?: number; bins?: number; colorFn?: (binStart: number) => string } = {}
): string {
  const W = options.width ?? 700;
  const H = options.height ?? 160;
  const BINS = options.bins ?? 30;
  const PAD = { top: 10, right: 20, bottom: 24, left: 44 };
  const CW = W - PAD.left - PAD.right;
  const CH = H - PAD.top - PAD.bottom;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const binSize = (max - min) / BINS;
  const counts = Array(BINS).fill(0);
  values.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / binSize), BINS - 1);
    counts[idx]++;
  });
  const maxCount = Math.max(...counts);

  const barW = CW / BINS;
  const bars = counts.map((count, i) => {
    const binStart = min + i * binSize;
    const barH = (count / maxCount) * CH;
    const x = PAD.left + i * barW;
    const y = PAD.top + CH - barH;
    const color = options.colorFn ? options.colorFn(binStart) : "#00d4ff";
    return `<rect x="${x + 0.5}" y="${y}" width="${Math.max(barW - 1, 1)}" height="${barH}" fill="${color}" rx="1"/>`;
  }).join("");

  // X axis labels
  const xLabels = [0, 0.25, 0.5, 0.75, 1].map(t => {
    const v = min + t * (max - min);
    const x = PAD.left + t * CW;
    return `<text x="${x}" y="${H - 4}" text-anchor="middle" fill="#4d6478" font-size="9" font-family="IBM Plex Mono,monospace">${v.toFixed(3)}</text>`;
  }).join("");

  // Y axis
  const yLabel = `<text x="${PAD.left - 6}" y="${PAD.top + CH / 2}" text-anchor="middle" fill="#4d6478" font-size="9" font-family="IBM Plex Mono,monospace" transform="rotate(-90,${PAD.left - 24},${PAD.top + CH / 2})">Count</text>`;

  return `
    <svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0.5rem 0">
      <rect width="${W}" height="${H}" fill="transparent"/>
      <line x1="${PAD.left}" x2="${PAD.left}" y1="${PAD.top}" y2="${PAD.top + CH}" stroke="#1e2d3d"/>
      <line x1="${PAD.left}" x2="${PAD.left + CW}" y1="${PAD.top + CH}" y2="${PAD.top + CH}" stroke="#1e2d3d"/>
      ${yLabel}
      ${bars}
      ${xLabels}
    </svg>
  `;
}

export function generateReport(result: AnalysisResult, filename: string): void {
  const { eda, ml_diagnostics: ml } = result;
  const timestamp = new Date().toLocaleString();
  const mlScore = ml.ML_readiness_score ?? null;
  const structScore = eda.data_structural_report.data_structural_score;

  // ── SECTION: Header ──────────────────────────────────────────────
  const headerHtml = `
    <div class="report-header">
      <div class="report-logo">ATHENA</div>
      <div class="report-meta">
        <div class="report-filename">${filename}</div>
        <div class="report-timestamp">${timestamp}</div>
        <div class="report-mode">${badge(result.mode.toUpperCase())}</div>
      </div>
      <div class="report-scores">
        <div class="score-block">
          <div class="score-label">ML READINESS</div>
          <div class="score-value" style="color:${mlScore != null ? scoreColor(mlScore) : "#4d6478"}">${mlScore ?? "N/A"}<span class="score-denom">/100</span></div>
        </div>
        <div class="score-block">
          <div class="score-label">DATA STRUCTURE</div>
          <div class="score-value" style="color:${scoreColor(structScore)}">${structScore}<span class="score-denom">/100</span></div>
        </div>
      </div>
    </div>
  `;

  // ── SECTION: Overview ────────────────────────────────────────────
  const overviewHtml = section("OVERVIEW", `
    <div class="stat-grid">
      <div class="stat-box"><div class="stat-label">ROWS</div><div class="stat-value">${eda.basic.rows.toLocaleString()}</div></div>
      <div class="stat-box"><div class="stat-label">COLUMNS</div><div class="stat-value">${eda.basic.cols}</div></div>
      <div class="stat-box"><div class="stat-label">NUMERIC</div><div class="stat-value">${eda.col_views.numeric_features.length}</div></div>
      <div class="stat-box"><div class="stat-label">CATEGORICAL</div><div class="stat-value">${eda.col_views.categorical_features.length}</div></div>
      <div class="stat-box"><div class="stat-label">IDENTIFIERS</div><div class="stat-value">${eda.col_views.identifiers.length}</div></div>
      ${eda.free_text ? `<div class="stat-box"><div class="stat-label">FREE TEXT</div><div class="stat-value" style="color:${eda.free_text.count > 0 ? "#ffcc00" : "inherit"}">${eda.free_text.count}</div></div>` : ""}
    </div>

    ${ml.component_scores ? `
    <div class="component-scores">
      ${Object.entries(ml.component_scores).map(([k, v]) => `
        <div class="comp-score-row">
          <span class="comp-score-label">${k.replace(/_/g, " ").toUpperCase()}</span>
          <div class="comp-score-bar-wrap">
            <div class="comp-score-bar" style="width:${v}%;background:${scoreColor(v as number)}"></div>
          </div>
          <span class="comp-score-val" style="color:${scoreColor(v as number)}">${v}/100</span>
        </div>
      `).join("")}
    </div>` : ""}

    ${eda.data_structural_report.action_plan.drop_columns.length > 0
      ? alertBox(`DROP COLUMNS: ${eda.data_structural_report.action_plan.drop_columns.join(", ")}`, "error")
      : ""}
    ${[...eda.data_structural_report.action_plan.apply_log_transform, ...eda.data_structural_report.action_plan.apply_log1p_transform].length > 0
      ? alertBox(`LOG TRANSFORMS: ${[...eda.data_structural_report.action_plan.apply_log_transform, ...eda.data_structural_report.action_plan.apply_log1p_transform].join(", ")}`, "info")
      : ""}
    ${eda.data_structural_report.critical_alerts.length > 0
      ? eda.data_structural_report.critical_alerts.map(a => alertBox(a, "error")).join("")
      : ""}
    ${ml.penalties && Object.keys(ml.penalties).length > 0
      ? Object.entries(ml.penalties).flatMap(([, issues]) => (issues as string[]).map(i => alertBox(i, "warning"))).join("")
      : ""}
    ${ml.recommendations && Object.keys(ml.recommendations).length > 0
      ? Object.entries(ml.recommendations).flatMap(([sec, recs]) =>
          (recs as string[]).map(r => `<div class="rec-row">${badge(sec, "#00d4ff")} <span class="rec-text">${r}</span></div>`)
        ).join("")
      : ""}
  `);

  // ── SECTION: Column classification ───────────────────────────────
  const colGroups = [
    { label: "NUMERIC FEATURES", cols: eda.col_views.numeric_features, color: "#00d4ff" },
    { label: "CATEGORICAL", cols: eda.col_views.categorical_features, color: "#00ff88" },
    { label: "IDENTIFIERS", cols: eda.col_views.identifiers, color: "#ff4466" },
    { label: "ENTITY KEYS", cols: eda.col_views.entity_keys, color: "#ffcc00" },
  ];

  const columnsHtml = section("COLUMN CLASSIFICATION", `
    <div class="col-grid">
      ${colGroups.map(g => `
        <div class="col-group">
          <div class="col-group-label">${g.label}</div>
          ${g.cols.length === 0
            ? `<div class="col-group-none">none</div>`
            : g.cols.map(c => `<div class="col-item" style="color:${g.color}">${c}</div>`).join("")}
        </div>
      `).join("")}
    </div>
  `);

  // ── SECTION: Numeric feature summary + skewness chart ────────────
  const skewnessData = eda.numeric_summary
    .filter(n => n.skewness != null)
    .map(n => ({
      label: n.column.length > 16 ? n.column.slice(0, 14) + "…" : n.column,
      value: parseFloat(n.skewness.toFixed(3)),
      color: Math.abs(n.skewness) > 2 ? "#ff4466" : Math.abs(n.skewness) > 1 ? "#ffcc00" : "#00ff88",
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const skewnessChartHtml = skewnessData.length > 0 ? `
    <div class="section-subtitle" style="margin-top:1.25rem;margin-bottom:0.5rem">FEATURE SKEWNESS</div>
    <div style="font-size:0.62rem;color:#4d6478;margin-bottom:0.5rem;font-family:'IBM Plex Mono',monospace">
      <span style="color:#00ff88">■</span> Near-normal (&lt;1) &nbsp;
      <span style="color:#ffcc00">■</span> Moderate (1–2) &nbsp;
      <span style="color:#ff4466">■</span> Highly skewed (&gt;2) &nbsp;
      dashed lines at ±1
    </div>
    ${svgHorizontalBarChart(skewnessData, {
      maxValue: Math.max(...skewnessData.map(d => Math.abs(d.value)), 1),
      referenceLines: [1],
    })}
  ` : "";

  const numericHtml = eda.numeric_summary.length > 0 ? section("NUMERIC FEATURE SUMMARY",
    buildTable(
      ["Column", "Mean", "Std", "Min", "Max", "Skew", "Missing%"],
      eda.numeric_summary.map(n => [
        `<span style="color:#00d4ff">${n.column}</span>`,
        fmt(n.mean),
        fmt(n.std),
        fmt(n.min),
        fmt(n.max),
        `<span style="color:${Math.abs(n.skewness) > 1 ? "#ffcc00" : "inherit"}">${fmt(n.skewness, 3)}</span>`,
        `<span style="color:${n.missing_pct > 5 ? "#ff4466" : "inherit"}">${pct(n.missing_pct)}</span>`,
      ])
    ) + skewnessChartHtml
  ) : "";

  // ── SECTION: Categorical feature summary ─────────────────────────
  const categoricalHtml = eda.categorical_summary.length > 0 ? section("CATEGORICAL FEATURE SUMMARY",
    buildTable(
      ["Column", "Unique", "Top Category %", "Missing%", "High Cardinality"],
      eda.categorical_summary.map(c => [
        `<span style="color:#00ff88">${c.column}</span>`,
        String(c.n_unique),
        pct(c.most_frequent_pct * 100),
        `<span style="color:${c.missing_pct > 5 ? "#ff4466" : "inherit"}">${pct(c.missing_pct)}</span>`,
        c.high_cardinality
          ? `<span style="color:#ffcc00">HIGH</span>`
          : `<span style="color:#00ff88">OK</span>`,
      ])
    )
  ) : "";

  // ── SECTION: Missing values ───────────────────────────────────────
  const missingEntries = Object.entries(eda.missingness.missing_values_percent_cols).filter(([, v]) => v > 0);
  const missingHtml = section("MISSING VALUES",
    missingEntries.length === 0
      ? alertBox("No missing values detected.", "success")
      : buildTable(
          ["Column", "Missing %"],
          missingEntries
            .sort(([, a], [, b]) => b - a)
            .map(([col, pctVal]) => [
              col,
              `<span style="color:${pctVal > 30 ? "#ff4466" : pctVal > 10 ? "#ffcc00" : "#ff8833"}">${pct(pctVal)}</span>`,
            ])
        )
  );

  // ── SECTION: Outlier analysis ─────────────────────────────────────
  const outliersHtml = section("OUTLIER ANALYSIS",
    (eda.outliers?.column_outliers?.length ?? 0) === 0
      ? alertBox("No significant outliers detected.", "success")
      : buildTable(
          ["Column", "Skewness", "Kurtosis", "Outlier %", "Distribution", "Action"],
          eda.outliers.column_outliers.map(o => [
            `<span style="color:#00d4ff">${o.column}</span>`,
            `<span style="color:${Math.abs(o.skewness) > 1 ? "#ffcc00" : "inherit"}">${fmt(o.skewness, 3)}</span>`,
            fmt(o.kurtosis, 3),
            `<span style="color:${o.outlier_pct > 5 ? "#ff4466" : "inherit"}">${pct(o.outlier_pct)}</span>`,
            o.distribution_type === "Gaussian-like"
              ? `<span style="color:#00ff88">GAUSSIAN</span>`
              : `<span style="color:#ffcc00">SKEWED</span>`,
            o.suggest_log_transform ? `<span style="color:#00d4ff">LOG1P</span>` : "—",
          ])
        )
  );

  // ── SECTION: Feature redundancy ───────────────────────────────────
  const redundancyHtml = section("FEATURE REDUNDANCY", `
    <div style="margin-bottom:0.75rem">
      ${badge(
        (eda.feature_redundancy.overall_redundancy_risk ?? "Low").toUpperCase() + " RISK",
        eda.feature_redundancy.overall_redundancy_risk === "Low" ? "#00ff88"
          : eda.feature_redundancy.overall_redundancy_risk === "Moderate" ? "#ffcc00" : "#ff4466"
      )}
    </div>
    ${(eda.feature_redundancy.highly_correlated_pairs?.length ?? 0) === 0
      ? alertBox("No strongly correlated feature pairs detected.", "success")
      : buildTable(
          ["Feature 1", "Feature 2", "Correlation", "Action"],
          eda.feature_redundancy.highly_correlated_pairs.map(p => [
            `<span style="color:#00d4ff">${p.feature_1}</span>`,
            `<span style="color:#00d4ff">${p.feature_2}</span>`,
            `<span style="color:#ff4466">${fmt(p.correlation, 3)}</span>`,
            `<span style="color:#4d6478;font-size:0.8em">${p.recommended_action}</span>`,
          ])
        )}
  `);

  // ── SECTION: ML Diagnostics ───────────────────────────────────────
  let mlHtml = "";

  if (ml.status === "ok" && ml.baseline) {
    // Learning curves SVG
    const learningCurvesSvg = ml.baseline.fold_curves && ml.baseline.fold_curves.length > 0
      ? ml.baseline.fold_curves.map((fc: any) => `
          <div class="section-subtitle" style="margin-top:1rem">FOLD ${fc.fold} — LEARNING CURVE</div>
          ${svgLineChart(
            [
              { data: fc.train, color: "#00d4ff", name: "Train Loss" },
              { data: fc.val, color: "#00ff88", name: "Val Loss", dashed: true },
            ],
            { xLabel: "Iteration" }
          )}
        `).join("")
      : "";

    mlHtml = section("ML DIAGNOSTICS", `
      <div class="stat-grid">
        <div class="stat-box">
          <div class="stat-label">BASELINE ${(ml.metric ?? "").toUpperCase()}</div>
          <div class="stat-value" style="color:#00d4ff">${fmt(ml.baseline.mean_score, 3)}</div>
          <div class="stat-sub">±${fmt(ml.baseline.std_score, 3)} over ${ml.baseline.n_folds} folds</div>
        </div>
        <div class="stat-box">
          <div class="stat-label">OVERFITTING</div>
          <div class="stat-value" style="color:${ml.baseline.learning_dynamics.overfitting_detected ? "#ffcc00" : "#00ff88"}">
            ${ml.baseline.learning_dynamics.overfitting_detected ? "DETECTED" : "CLEAN"}
          </div>
        </div>
        <div class="stat-box">
          <div class="stat-label">TRAIN-VAL GAP</div>
          <div class="stat-value" style="color:${ml.baseline.learning_dynamics.mean_train_val_gap > 0.1 ? "#ffcc00" : "#00ff88"}">
            ${fmt(ml.baseline.learning_dynamics.mean_train_val_gap, 3)}
          </div>
        </div>
        ${ml.leakage ? `
        <div class="stat-box">
          <div class="stat-label">LEAKAGE SCORE</div>
          <div class="stat-value" style="color:${ml.leakage.leakage_score > 30 ? "#ff4466" : "#00ff88"}">${ml.leakage.leakage_score}</div>
        </div>` : ""}
      </div>

      <div style="margin-top:1rem">
        <div class="section-subtitle">CROSS-VALIDATION FOLD SCORES</div>
        <div class="fold-grid">
          ${ml.baseline.fold_scores.map((s: number, i: number) => `
            <div class="fold-box">
              <div class="fold-label">FOLD ${i + 1}</div>
              <div class="fold-score">${s.toFixed(4)}</div>
            </div>
          `).join("")}
        </div>
      </div>

      ${ml.leakage && ml.leakage.suspicious_features.length > 0 ? `
      <div style="margin-top:1rem">
        <div class="section-subtitle">SUSPICIOUS FEATURES (LEAKAGE RISK)</div>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.5rem">
          ${ml.leakage.suspicious_features.map((f: string) => badge(f, "#ff4466")).join("")}
        </div>
        ${alertBox("Review these features — they may be causally downstream of the target.", "warning")}
      </div>` : ""}

      ${ml.stability ? `
      <div style="margin-top:1rem">
        <div class="section-subtitle">STABILITY ANALYSIS</div>
        <div class="stat-grid">
          <div class="stat-box"><div class="stat-label">STABILITY SCORE</div><div class="stat-value" style="color:#00d4ff">${fmt(ml.stability.mean_score, 3)}</div></div>
          <div class="stat-box"><div class="stat-label">STABILITY</div><div class="stat-value" style="color:${ml.stability.is_unstable ? "#ff4466" : "#00ff88"}">${ml.stability.is_unstable ? "UNSTABLE" : "STABLE"}</div></div>
          <div class="stat-box"><div class="stat-label">SCORE STD</div><div class="stat-value">${fmt(ml.stability.std_score, 3)}</div></div>
          <div class="stat-box"><div class="stat-label">LOSS GAP RATIO</div><div class="stat-value">${fmt(ml.stability.loss_gap_ratio, 3)}</div></div>
        </div>
      </div>` : ""}

      ${learningCurvesSvg}
    `);
  } else if (ml.clustering) {
    // Elbow curve SVG
    const elbowData = Object.entries(ml.clustering.silhouette_scores ?? {}).map(([k, v]) => ({
      k: Number(k),
      sil: Number(v),
    }));

    const elbowSvg = elbowData.length > 0 ? (() => {
      const W = 500;
      const H = 180;
      const PAD = { top: 10, right: 20, bottom: 30, left: 50 };
      const CW = W - PAD.left - PAD.right;
      const CH = H - PAD.top - PAD.bottom;
      const minK = Math.min(...elbowData.map(d => d.k));
      const maxK = Math.max(...elbowData.map(d => d.k));
      const toX = (k: number) => PAD.left + ((k - minK) / (maxK - minK || 1)) * CW;
      const toY = (v: number) => PAD.top + CH - v * CH;
      const pts = elbowData.map(d => `${toX(d.k)},${toY(d.sil)}`).join(" ");
      const dots = elbowData.map(d => {
        const isBest = d.k === ml.clustering.best_k;
        return `<circle cx="${toX(d.k)}" cy="${toY(d.sil)}" r="${isBest ? 6 : 4}" fill="${isBest ? "#00ff88" : "#00d4ff"}"/>
                <text x="${toX(d.k)}" y="${toY(d.sil) - 8}" text-anchor="middle" fill="${isBest ? "#00ff88" : "#4d6478"}" font-size="9" font-family="IBM Plex Mono,monospace">k=${d.k}</text>`;
      }).join("");
      const yTicks = [0, 0.25, 0.5, 0.75, 1].map(t => {
        const y = PAD.top + CH - t * CH;
        return `<line x1="${PAD.left - 3}" x2="${PAD.left + CW}" y1="${y}" y2="${y}" stroke="#1e2d3d" stroke-dasharray="2,4"/>
                <text x="${PAD.left - 5}" y="${y + 3}" text-anchor="end" fill="#4d6478" font-size="9" font-family="IBM Plex Mono,monospace">${t.toFixed(2)}</text>`;
      }).join("");
      return `
        <svg width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0.5rem 0">
          <rect width="${W}" height="${H}" fill="transparent"/>
          <line x1="${PAD.left}" x2="${PAD.left}" y1="${PAD.top}" y2="${PAD.top + CH}" stroke="#1e2d3d"/>
          <line x1="${PAD.left}" x2="${PAD.left + CW}" y1="${PAD.top + CH}" y2="${PAD.top + CH}" stroke="#1e2d3d"/>
          ${yTicks}
          <polyline points="${pts}" fill="none" stroke="#00d4ff" stroke-width="2"/>
          ${dots}
          <text x="${PAD.left + CW / 2}" y="${H - 2}" text-anchor="middle" fill="#4d6478" font-size="9" font-family="IBM Plex Mono,monospace">K</text>
          <text x="${PAD.left - 30}" y="${PAD.top + CH / 2}" text-anchor="middle" fill="#4d6478" font-size="9" font-family="IBM Plex Mono,monospace" transform="rotate(-90,${PAD.left - 30},${PAD.top + CH / 2})">Silhouette</text>
        </svg>
      `;
    })() : "";

    // Anomaly histogram SVG
    const anomalyHistSvg = ml.anomaly_detection?.score_distribution
      ? `
        <div class="section-subtitle" style="margin-top:1rem">ANOMALY SCORE DISTRIBUTION</div>
        <div style="font-size:0.62rem;color:#4d6478;margin-bottom:0.4rem;font-family:'IBM Plex Mono',monospace">
          <span style="color:#ff4466">■</span> Anomaly (score &lt; 0) &nbsp;
          <span style="color:#00d4ff">■</span> Normal (score ≥ 0)
        </div>
        ${svgHistogram(ml.anomaly_detection.score_distribution, {
          colorFn: (binStart) => binStart < 0 ? "#ff4466" : "#00d4ff",
        })}
      `
      : "";

    mlHtml = section("ML DIAGNOSTICS — UNSUPERVISED", `
      <div class="stat-grid">
        <div class="stat-box"><div class="stat-label">BEST K</div><div class="stat-value" style="color:#00d4ff">${ml.clustering.best_k}</div></div>
        <div class="stat-box"><div class="stat-label">BEST SILHOUETTE</div><div class="stat-value">${fmt(ml.clustering.best_silhouette, 3)}</div></div>
        ${ml.anomaly_detection ? `
        <div class="stat-box">
          <div class="stat-label">ANOMALY RATE</div>
          <div class="stat-value" style="color:${ml.anomaly_detection.anomaly_percent > 15 ? "#ff4466" : "#00ff88"}">${pct(ml.anomaly_detection.anomaly_percent)}</div>
        </div>
        <div class="stat-box"><div class="stat-label">METHOD</div><div class="stat-value" style="font-size:0.75rem">${ml.anomaly_detection.method}</div></div>
        ` : ""}
      </div>

      <div class="section-subtitle" style="margin-top:1rem">ELBOW CURVE — SILHOUETTE SCORE VS K</div>
      ${elbowSvg}

      ${anomalyHistSvg}
    `);
  }

  // ── SECTION: NLP Analysis ─────────────────────────────────────────
  let nlpHtml = "";
  if (eda.nlp && eda.nlp.nlp_analysis?.length > 0) {
    nlpHtml = eda.nlp.nlp_analysis.map((col: any) => {
      const topWordsHtml = col.text_stats.top_words
        .map((w: any) => `<span class="word-tag">${w.word} <span style="color:#4d6478">${w.count}</span></span>`)
        .join("");

      const perClassHtml = col.per_class_analysis ? `
        <div class="section-subtitle" style="margin-top:1rem">PER-CLASS ANALYSIS</div>
        ${col.per_class_analysis.map((cls: any) => `
          <div class="col-group" style="margin-bottom:0.5rem">
            <div style="display:flex;gap:0.5rem;align-items:center;margin-bottom:0.5rem">
              ${badge(cls.class, "#00d4ff")}
              <span style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#4d6478">
                ${cls.count.toLocaleString()} samples · avg ${cls.avg_word_count} words · avg ${cls.avg_char_length} chars
              </span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:0.3rem">
              ${cls.top_words.map((w: any) => `<span class="word-tag" style="color:#00ff88">${w.word} <span style="color:#4d6478">${w.count}</span></span>`).join("")}
            </div>
          </div>
        `).join("")}
      ` : "";

      return section(`NLP — ${col.column}`, `
        <div class="stat-grid">
          <div class="stat-box"><div class="stat-label">AVG WORDS</div><div class="stat-value">${col.text_stats.avg_word_count}</div></div>
          <div class="stat-box"><div class="stat-label">AVG CHARS</div><div class="stat-value">${col.text_stats.avg_char_length}</div></div>
          <div class="stat-box"><div class="stat-label">VOCAB SIZE</div><div class="stat-value">${col.text_stats.vocab_size.toLocaleString()}</div></div>
          <div class="stat-box"><div class="stat-label">LEXICAL DIVERSITY</div><div class="stat-value">${col.text_stats.lexical_diversity}</div></div>
        </div>

        <div class="section-subtitle">TEXT QUALITY</div>
        <div class="stat-grid" style="margin-bottom:1rem">
          <div class="stat-box"><div class="stat-label">EMPTY RATIO</div><div class="stat-value" style="color:${col.text_quality.empty_ratio > 0.05 ? "#ff4466" : "#00ff88"}">${pct(col.text_quality.empty_ratio * 100)}</div></div>
          <div class="stat-box"><div class="stat-label">DUPLICATE RATIO</div><div class="stat-value" style="color:${col.text_quality.duplicate_ratio > 0.1 ? "#ffcc00" : "#00ff88"}">${pct(col.text_quality.duplicate_ratio * 100)}</div></div>
          <div class="stat-box"><div class="stat-label">URL RATIO</div><div class="stat-value">${pct(col.text_quality.url_ratio * 100)}</div></div>
          <div class="stat-box"><div class="stat-label">NUMBER RATIO</div><div class="stat-value">${pct(col.text_quality.number_ratio * 100)}</div></div>
        </div>

        ${alertBox(col.ml_readiness.vectorization_recommendation, col.ml_readiness.suitable_for_embeddings ? "success" : "warning")}

        <div class="section-subtitle" style="margin-top:1rem">TOP WORDS</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1rem">
          ${topWordsHtml}
        </div>

        ${perClassHtml}
      `);
    }).join("");
  }

  // ── CSS ───────────────────────────────────────────────────────────
  const css = `
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #080b0f;
      color: #e2eaf4;
      font-family: 'IBM Plex Sans', sans-serif;
      font-size: 13px;
      line-height: 1.6;
      padding: 2.5rem;
      max-width: 960px;
      margin: 0 auto;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }

    .report-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      border-bottom: 1px solid #1e2d3d;
      padding-bottom: 1.5rem;
      margin-bottom: 2rem;
    }

    .report-logo { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; color: #00d4ff; letter-spacing: 0.15em; }
    .report-meta { display: flex; flex-direction: column; gap: 0.25rem; }
    .report-filename { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: #e2eaf4; }
    .report-timestamp { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #4d6478; }

    .report-scores { display: flex; gap: 2rem; }
    .score-block { text-align: right; }
    .score-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: #4d6478; letter-spacing: 0.12em; margin-bottom: 0.25rem; }
    .score-value { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 600; }
    .score-denom { font-size: 0.9rem; color: #4d6478; }

    .section { margin-bottom: 2rem; page-break-inside: avoid; }
    .section-title { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.15em; color: #4d6478; border-bottom: 1px solid #1e2d3d; padding-bottom: 0.5rem; margin-bottom: 1rem; }
    .section-subtitle { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; letter-spacing: 0.12em; color: #4d6478; margin-bottom: 0.5rem; }

    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
    .stat-box { background: #0d1117; border: 1px solid #1e2d3d; border-radius: 4px; padding: 0.75rem; }
    .stat-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; color: #4d6478; letter-spacing: 0.1em; margin-bottom: 0.25rem; }
    .stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; font-weight: 500; }
    .stat-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #4d6478; margin-top: 0.2rem; }

    .component-scores { margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.5rem; }
    .comp-score-row { display: flex; align-items: center; gap: 0.75rem; }
    .comp-score-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: #4d6478; width: 120px; flex-shrink: 0; }
    .comp-score-bar-wrap { flex: 1; height: 4px; background: #1e2d3d; border-radius: 2px; overflow: hidden; }
    .comp-score-bar { height: 100%; border-radius: 2px; }
    .comp-score-val { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; width: 60px; text-align: right; flex-shrink: 0; }

    table { width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; }
    th { text-align: left; color: #4d6478; font-size: 0.6rem; letter-spacing: 0.1em; padding: 0.4rem 0.6rem; border-bottom: 1px solid #1e2d3d; }
    td { padding: 0.45rem 0.6rem; border-bottom: 1px solid #111820; color: #8fa8c0; }
    tr:last-child td { border-bottom: none; }

    .badge { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; letter-spacing: 0.1em; padding: 0.15rem 0.4rem; border: 1px solid; border-radius: 3px; }
    .alert { padding: 0.5rem 0.75rem; border-left: 2px solid; margin-bottom: 0.4rem; font-size: 0.78rem; background: rgba(0,0,0,0.2); }
    .word-tag { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; padding: 0.15rem 0.4rem; background: #0d1117; border: 1px solid #1e2d3d; border-radius: 3px; color: #00d4ff; }
    .rec-row { display: flex; gap: 0.5rem; align-items: flex-start; margin-bottom: 0.4rem; }
    .rec-text { font-size: 0.78rem; color: #8fa8c0; }

    .col-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
    .col-group { background: #0d1117; border: 1px solid #1e2d3d; border-radius: 4px; padding: 0.75rem; }
    .col-group-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; color: #4d6478; letter-spacing: 0.1em; margin-bottom: 0.5rem; }
    .col-item { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; margin-bottom: 0.2rem; }
    .col-group-none { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #2a3f55; }

    .fold-grid { display: flex; gap: 0.5rem; }
    .fold-box { flex: 1; background: #0d1117; border: 1px solid #1e2d3d; border-radius: 4px; padding: 0.6rem; text-align: center; }
    .fold-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; color: #4d6478; margin-bottom: 0.25rem; }
    .fold-score { font-family: 'IBM Plex Mono', monospace; font-size: 1rem; color: #00d4ff; }

    @media print {
      body { padding: 1.5rem; }
      .section { page-break-inside: avoid; }
    }
  `;

  // ── Assemble ──────────────────────────────────────────────────────
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>athena-report-${filename.replace(/\.[^.]+$/, "")}</title>
  <style>${css}</style>
</head>
<body>
  ${headerHtml}
  ${overviewHtml}
  ${columnsHtml}
  ${numericHtml}
  ${categoricalHtml}
  ${missingHtml}
  ${outliersHtml}
  ${redundancyHtml}
  ${mlHtml}
  ${nlpHtml}
</body>
</html>`;

  const win = window.open("", "_blank");
  if (!win) {
    alert("Popup blocked — allow popups for this site to download the report.");
    return;
  }
  win.document.write(html);
  win.document.close();
  win.onload = () => {
    win.focus();
    win.print();
    win.onafterprint = () => win.close();
  };
}