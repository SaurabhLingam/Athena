import React from "react";

/* ── Card ── */
export function Card({
  children,
  className = "",
  style,
}: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      className={className}
      style={{
        background: "var(--bg-2)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-lg)",
        padding: "1.25rem 1.5rem",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

/* ── Section Header ── */
export function SectionHeader({
  label,
  sub,
  accent = false,
}: {
  label: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div style={{ marginBottom: "1rem" }}>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.7rem",
          letterSpacing: "0.15em",
          textTransform: "uppercase",
          color: accent ? "var(--accent)" : "var(--text-2)",
          marginBottom: "0.25rem",
        }}
      >
        {label}
      </div>
      {sub && (
        <div style={{ color: "var(--text-1)", fontSize: "0.85rem" }}>{sub}</div>
      )}
    </div>
  );
}

/* ── Badge ── */
type BadgeVariant = "green" | "red" | "yellow" | "accent" | "neutral";
export function Badge({
  children,
  variant = "neutral",
}: {
  children: React.ReactNode;
  variant?: BadgeVariant;
}) {
  const colors: Record<BadgeVariant, { bg: string; color: string; border: string }> = {
    green: { bg: "var(--green-glow)", color: "var(--green)", border: "var(--green-dim)" },
    red: { bg: "var(--red-glow)", color: "var(--red)", border: "var(--red-dim)" },
    yellow: { bg: "rgba(255,204,0,0.08)", color: "var(--yellow)", border: "var(--yellow-dim)" },
    accent: { bg: "var(--accent-glow)", color: "var(--accent)", border: "var(--accent-dim)" },
    neutral: { bg: "var(--bg-3)", color: "var(--text-1)", border: "var(--border)" },
  };
  const c = colors[variant];
  return (
    <span
      style={{
        display: "inline-block",
        padding: "0.15rem 0.5rem",
        borderRadius: "var(--radius-sm)",
        background: c.bg,
        color: c.color,
        border: `1px solid ${c.border}`,
        fontFamily: "var(--font-mono)",
        fontSize: "0.7rem",
        letterSpacing: "0.05em",
      }}
    >
      {children}
    </span>
  );
}

/* ── Stat Box ── */
export function StatBox({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string | number;
  sub?: string;
  accent?: string;
}) {
  return (
    <div
      style={{
        background: "var(--bg-3)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        padding: "1rem 1.25rem",
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.65rem",
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--text-2)",
          marginBottom: "0.4rem",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "1.6rem",
          fontWeight: 600,
          color: accent ?? "var(--text-0)",
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      {sub && (
        <div
          style={{
            fontSize: "0.72rem",
            color: "var(--text-2)",
            marginTop: "0.3rem",
            fontFamily: "var(--font-mono)",
          }}
        >
          {sub}
        </div>
      )}
    </div>
  );
}

/* ── Progress Bar ── */
export function ProgressBar({
  value,
  max = 100,
  color,
  label,
  showValue = true,
}: {
  value: number;
  max?: number;
  color?: string;
  label?: string;
  showValue?: boolean;
}) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  const autoColor =
    pct >= 75 ? "var(--green)" : pct >= 45 ? "var(--yellow)" : "var(--red)";
  const barColor = color ?? autoColor;

  return (
    <div style={{ width: "100%" }}>
      {(label || showValue) && (
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: "0.35rem",
            fontFamily: "var(--font-mono)",
            fontSize: "0.72rem",
          }}
        >
          {label && <span style={{ color: "var(--text-1)" }}>{label}</span>}
          {showValue && (
            <span style={{ color: barColor }}>
              {value}
              {max === 100 ? "/100" : ""}
            </span>
          )}
        </div>
      )}
      <div
        style={{
          height: "4px",
          background: "var(--bg-4)",
          borderRadius: "2px",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pct}%`,
            background: barColor,
            borderRadius: "2px",
            transition: "width 0.6s cubic-bezier(0.4,0,0.2,1)",
            boxShadow: `0 0 8px ${barColor}66`,
          }}
        />
      </div>
    </div>
  );
}

/* ── Alert ── */
export function Alert({
  children,
  variant = "neutral",
}: {
  children: React.ReactNode;
  variant?: "error" | "warning" | "success" | "info" | "neutral";
}) {
  const map = {
    error: { border: "var(--red-dim)", bg: "var(--red-glow)", icon: "✕", color: "var(--red)" },
    warning: { border: "var(--yellow-dim)", bg: "rgba(255,204,0,0.06)", icon: "⚠", color: "var(--yellow)" },
    success: { border: "var(--green-dim)", bg: "var(--green-glow)", icon: "✓", color: "var(--green)" },
    info: { border: "var(--accent-dim)", bg: "var(--accent-glow)", icon: "i", color: "var(--accent)" },
    neutral: { border: "var(--border)", bg: "var(--bg-3)", icon: "—", color: "var(--text-1)" },
  };
  const s = map[variant];
  return (
    <div
      style={{
        display: "flex",
        gap: "0.75rem",
        padding: "0.75rem 1rem",
        background: s.bg,
        border: `1px solid ${s.border}`,
        borderRadius: "var(--radius-md)",
        fontSize: "0.82rem",
        color: "var(--text-1)",
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          color: s.color,
          flexShrink: 0,
          fontSize: "0.75rem",
          marginTop: "1px",
        }}
      >
        {s.icon}
      </span>
      <span>{children}</span>
    </div>
  );
}

/* ── Tabs ── */
export function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: string; label: string }[];
  active: string;
  onChange: (id: string) => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        gap: "0",
        background: "var(--bg-2)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        padding: "3px",
        width: "fit-content",
      }}
    >
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          style={{
            padding: "0.4rem 1rem",
            borderRadius: "var(--radius-sm)",
            border: "none",
            background: active === t.id ? "var(--bg-4)" : "transparent",
            color: active === t.id ? "var(--text-0)" : "var(--text-2)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.72rem",
            letterSpacing: "0.08em",
            cursor: "pointer",
            transition: "all var(--transition)",
            borderBottom: active === t.id ? "1px solid var(--accent)" : "1px solid transparent",
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

/* ── Divider ── */
export function Divider() {
  return (
    <div
      style={{
        height: "1px",
        background: "var(--border)",
        margin: "1.5rem 0",
      }}
    />
  );
}

/* ── Score Gauge ── */
export function ScoreGauge({ score }: { score: number }) {
  const color =
    score >= 75 ? "var(--green)" : score >= 45 ? "var(--yellow)" : "var(--red)";
  const label =
    score >= 75 ? "READY" : score >= 45 ? "CAUTION" : "NOT READY";

  const r = 52;
  const circumference = 2 * Math.PI * r;
  const arc = circumference * 0.75;
  const offset = arc - (score / 100) * arc;

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "0.5rem" }}>
      <svg width="140" height="100" viewBox="0 0 140 110">
        <circle
          cx="70" cy="80" r={r}
          fill="none"
          stroke="var(--bg-4)"
          strokeWidth="8"
          strokeDasharray={`${arc} ${circumference}`}
          strokeDashoffset="0"
          strokeLinecap="round"
          transform="rotate(135 70 80)"
        />
        <circle
          cx="70" cy="80" r={r}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeDasharray={`${arc} ${circumference}`}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(135 70 80)"
          style={{
            transition: "stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)",
            filter: `drop-shadow(0 0 6px ${color})`,
          }}
        />
        <text
          x="70" y="76"
          textAnchor="middle"
          fill={color}
          fontFamily="IBM Plex Mono, monospace"
          fontWeight="600"
          fontSize="22"
        >
          {score}
        </text>
        <text
          x="70" y="92"
          textAnchor="middle"
          fill="var(--text-2)"
          fontFamily="IBM Plex Mono, monospace"
          fontSize="9"
          letterSpacing="2"
        >
          / 100
        </text>
      </svg>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.65rem",
          letterSpacing: "0.2em",
          color,
          background: color + "18",
          border: `1px solid ${color}44`,
          padding: "0.15rem 0.6rem",
          borderRadius: "var(--radius-sm)",
        }}
      >
        {label}
      </span>
    </div>
  );
}

/* ── Table ── */
export function DataTable({
  columns,
  rows,
}: {
  columns: string[];
  rows: (string | number | React.ReactNode)[][];
}) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontFamily: "var(--font-mono)",
          fontSize: "0.75rem",
        }}
      >
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c}
                style={{
                  textAlign: "left",
                  padding: "0.5rem 0.75rem",
                  borderBottom: "1px solid var(--border-bright)",
                  color: "var(--text-2)",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  fontSize: "0.65rem",
                  fontWeight: 500,
                  whiteSpace: "nowrap",
                }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              style={{
                borderBottom: "1px solid var(--border)",
                transition: "background var(--transition)",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "var(--bg-3)")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.background = "transparent")
              }
            >
              {row.map((cell, j) => (
                <td
                  key={j}
                  style={{
                    padding: "0.5rem 0.75rem",
                    color: "var(--text-1)",
                  }}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ── Spinner ── */
export function Spinner() {
  return (
    <div
      style={{
        width: "20px",
        height: "20px",
        border: "2px solid var(--border)",
        borderTop: "2px solid var(--accent)",
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite",
      }}
    />
  );
}
