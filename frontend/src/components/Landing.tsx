import { useEffect, useRef } from "react";

const css = `
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Serif+Display:ital@0;1&display=swap');

  .lp-root *, .lp-root *::before, .lp-root *::after {
    box-sizing: border-box; margin: 0; padding: 0;
  }

  .lp-root {
    --bg:       #080c10;
    --bg-2:     #0d1318;
    --bg-3:     #111820;
    --border:   #1e2d3d;
    --accent:   #00e5c0;
    --accent-2: #00b4ff;
    --red:      #ff4466;
    --yellow:   #ffd166;
    --text-1:   #e2eaf4;
    --text-2:   #7a9ab8;
    --text-3:   #3d5a75;
    --mono:     'Space Mono', monospace;
    --serif:    'DM Serif Display', serif;

    background: var(--bg);
    color: var(--text-1);
    font-family: var(--mono);
    overflow-x: hidden;
    min-height: 100vh;
  }

  .lp-noise {
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 1000;
    opacity: 0.4;
  }

  .lp-grid-bg {
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 48px 48px;
    opacity: 0.25;
    pointer-events: none;
  }

  /* NAV */
  .lp-nav {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 3rem;
    border-bottom: 1px solid var(--border);
    background: rgba(8, 12, 16, 0.85);
    backdrop-filter: blur(12px);
  }

  .lp-nav-logo {
    font-family: var(--mono);
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
  }

  .lp-nav-logo span { color: var(--text-2); font-weight: 400; }

  .lp-nav-links {
    display: flex;
    gap: 2.5rem;
    list-style: none;
  }

  .lp-nav-links a {
    color: var(--text-2);
    text-decoration: none;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    transition: color 0.2s;
  }

  .lp-nav-links a:hover { color: var(--accent); }

  .lp-nav-cta {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.5rem 1.25rem;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
  }

  .lp-nav-cta:hover { background: var(--accent); color: var(--bg); }

  /* HERO */
  .lp-hero {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 8rem 3rem 4rem;
    text-align: center;
  }

  .lp-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    border: 1px solid var(--border);
    background: var(--bg-2);
    padding: 0.35rem 0.85rem;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-2);
    margin-bottom: 2.5rem;
    animation: lpFadeUp 0.6s ease both;
  }

  .lp-badge-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: lpPulse 2s infinite;
  }

  @keyframes lpPulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .lp-hero-title {
    font-family: var(--serif);
    font-size: clamp(3.5rem, 8vw, 7.5rem);
    line-height: 1;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
    animation: lpFadeUp 0.6s 0.1s ease both;
  }

  .lp-hero-title-accent { color: var(--accent); font-style: italic; }

  .lp-hero-subtitle {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-2);
    margin-bottom: 2rem;
    animation: lpFadeUp 0.6s 0.2s ease both;
  }

  .lp-hero-desc {
    font-size: 0.85rem;
    color: var(--text-2);
    max-width: 560px;
    line-height: 1.8;
    margin-bottom: 3rem;
    animation: lpFadeUp 0.6s 0.3s ease both;
  }

  .lp-hero-actions {
    display: flex;
    gap: 1rem;
    align-items: center;
    animation: lpFadeUp 0.6s 0.4s ease both;
  }

  .lp-btn-primary {
    background: var(--accent);
    color: var(--bg);
    border: none;
    font-family: var(--mono);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.85rem 2rem;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    display: inline-block;
  }

  .lp-btn-primary:hover { background: #00ffda; transform: translateY(-1px); }

  .lp-btn-ghost {
    background: transparent;
    color: var(--text-2);
    border: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.85rem 2rem;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    display: inline-block;
  }

  .lp-btn-ghost:hover { border-color: var(--text-2); color: var(--text-1); }

  /* PREVIEW */
  .lp-preview-wrap {
    margin-top: 5rem;
    width: 100%;
    max-width: 1100px;
    position: relative;
    animation: lpFadeUp 0.8s 0.5s ease both;
  }

  .lp-preview-glow {
    position: absolute;
    inset: -40px;
    background: radial-gradient(ellipse at 50% 60%, rgba(0,229,192,0.08) 0%, transparent 70%);
    pointer-events: none;
  }

  .lp-preview-frame {
    border: 1px solid var(--border);
    background: var(--bg-2);
    overflow: hidden;
  }

  .lp-preview-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg-3);
  }

  .lp-preview-dot { width: 10px; height: 10px; border-radius: 50%; }

  .lp-preview-bar-title {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--text-2);
    text-transform: uppercase;
    margin-left: 0.5rem;
  }

  .lp-preview-content {
    padding: 1.5rem;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
  }

  .lp-preview-card {
    background: var(--bg-3);
    border: 1px solid var(--border);
    padding: 1rem;
  }

  .lp-preview-card-label {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 0.4rem;
  }

  .lp-preview-card-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
  }

  .lp-preview-card-value.blue { color: var(--accent-2); }
  .lp-preview-card-value.yellow { color: var(--yellow); }

  .lp-preview-card-sub { font-size: 0.6rem; color: var(--text-3); margin-top: 0.25rem; }

  .lp-preview-chart {
    grid-column: span 2;
    background: var(--bg-3);
    border: 1px solid var(--border);
    padding: 1rem;
    height: 120px;
    overflow: hidden;
  }

  .lp-preview-chart-label {
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 0.75rem;
  }

  .lp-preview-bars {
    display: flex;
    align-items: flex-end;
    gap: 4px;
    height: 70px;
  }

  .lp-preview-bar-item { flex: 1; border-radius: 1px; }

  .lp-preview-tags {
    grid-column: span 4;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .lp-preview-tag {
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.6rem;
    border: 1px solid var(--border);
    color: var(--text-2);
  }

  .lp-preview-tag.green { border-color: var(--accent); color: var(--accent); }
  .lp-preview-tag.red   { border-color: var(--red);    color: var(--red);    }
  .lp-preview-tag.yellow{ border-color: var(--yellow); color: var(--yellow); }

  /* TICKER */
  .lp-ticker-wrap {
    overflow: hidden;
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 0.75rem 0;
    background: var(--bg-2);
  }

  .lp-ticker-track {
    display: flex;
    gap: 3rem;
    animation: lpTicker 30s linear infinite;
    white-space: nowrap;
  }

  .lp-ticker-item {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-3);
    flex-shrink: 0;
  }

  .lp-ticker-item span { color: var(--accent); margin-right: 0.5rem; }

  @keyframes lpTicker {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
  }

  /* SECTIONS */
  .lp-section {
    padding: 6rem 3rem;
    max-width: 1200px;
    margin: 0 auto;
  }

  .lp-section-label {
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
  }

  .lp-section-title {
    font-family: var(--serif);
    font-size: clamp(2rem, 4vw, 3.5rem);
    line-height: 1.1;
    margin-bottom: 1.5rem;
  }

  .lp-section-desc {
    font-size: 0.82rem;
    color: var(--text-2);
    line-height: 1.9;
    max-width: 560px;
  }

  /* STEPS */
  .lp-steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5px;
    margin-top: 4rem;
    background: var(--border);
  }

  .lp-step {
    background: var(--bg);
    padding: 2.5rem;
    position: relative;
    overflow: hidden;
  }

  .lp-step-num {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--text-3);
    margin-bottom: 1.5rem;
  }

  .lp-step-icon {
    font-size: 1.2rem;
    color: var(--accent);
    margin-bottom: 1rem;
    font-weight: 700;
  }

  .lp-step-title {
    font-family: var(--serif);
    font-size: 1.3rem;
    margin-bottom: 0.75rem;
  }

  .lp-step-desc { font-size: 0.75rem; color: var(--text-2); line-height: 1.8; }

  .lp-step-line {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent);
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.4s ease;
  }

  .lp-step:hover .lp-step-line { transform: scaleX(1); }

  /* METRICS */
  .lp-metrics {
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    background: var(--bg-2);
  }

  .lp-metric {
    padding: 2.5rem 3rem;
    border-right: 1px solid var(--border);
  }

  .lp-metric:last-child { border-right: none; }

  .lp-metric-value {
    font-family: var(--serif);
    font-size: 2.8rem;
    color: var(--accent);
    line-height: 1;
    margin-bottom: 0.4rem;
  }

  .lp-metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-2);
  }

  /* FEATURES */
  .lp-features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    margin-top: 4rem;
    background: var(--border);
  }

  .lp-feature-card {
    background: var(--bg);
    padding: 2rem;
    transition: background 0.2s;
  }

  .lp-feature-card:hover { background: var(--bg-2); }

  .lp-feature-icon {
    width: 36px; height: 36px;
    border: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    margin-bottom: 1rem;
    color: var(--accent);
    font-weight: 700;
  }

  .lp-feature-title {
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
  }

  .lp-feature-desc { font-size: 0.72rem; color: var(--text-2); line-height: 1.8; }

  /* TERMINAL */
  .lp-terminal {
    margin-top: 4rem;
    border: 1px solid var(--border);
    background: var(--bg-2);
    overflow: hidden;
  }

  .lp-terminal-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--border);
    background: var(--bg-3);
  }

  .lp-terminal-body { padding: 1.5rem; font-size: 0.75rem; line-height: 2; }

  .t-comment { color: var(--text-3); }
  .t-key     { color: var(--accent-2); }
  .t-val     { color: var(--accent); }
  .t-str     { color: var(--yellow); }
  .t-num     { color: #ff9d5c; }
  .t-warn    { color: var(--red); }
  .t-prompt  { color: var(--text-3); }

  /* CTA */
  .lp-cta {
    text-align: center;
    padding: 8rem 3rem;
    position: relative;
  }

  .lp-cta-glow {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse, rgba(0,229,192,0.06) 0%, transparent 70%);
    pointer-events: none;
  }

  .lp-cta .lp-section-title { max-width: 600px; margin: 0 auto 1.5rem; }
  .lp-cta .lp-section-desc  { max-width: 460px; margin: 0 auto 3rem; }

  /* FOOTER */
  .lp-footer {
    border-top: 1px solid var(--border);
    padding: 2rem 3rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .lp-footer-logo {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: var(--text-3);
    text-transform: uppercase;
  }

  .lp-footer-copy { font-size: 0.65rem; color: var(--text-3); letter-spacing: 0.1em; }

  /* ANIMATIONS */
  @keyframes lpFadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .lp-reveal {
    opacity: 0;
    transform: translateY(24px);
    transition: opacity 0.7s ease, transform 0.7s ease;
  }

  .lp-reveal.visible { opacity: 1; transform: none; }

  /* RESPONSIVE */
  @media (max-width: 768px) {
    .lp-nav { padding: 1rem 1.5rem; }
    .lp-nav-links { display: none; }
    .lp-hero { padding: 6rem 1.5rem 3rem; }
    .lp-steps, .lp-features-grid { grid-template-columns: 1fr; }
    .lp-metrics { grid-template-columns: repeat(2, 1fr); }
    .lp-preview-content { grid-template-columns: repeat(2, 1fr); }
    .lp-section { padding: 4rem 1.5rem; }
    .lp-footer { flex-direction: column; gap: 1rem; text-align: center; }
  }
`;

const TICKER_ITEMS = [
  'EDA', 'Leakage Detection', 'Baseline Probing', 'Outlier Analysis',
  'NLP Readiness', 'Feature Skewness', 'Class Imbalance', 'Drift Comparison',
  'Clustering', 'Stability Analysis', 'Preprocessing Script', 'ML Readiness Score',
  'Feature Redundancy', 'Log Transform Detection', 'Free Text Detection', 'Bivariate Analysis',
];

const SKEW_DATA   = [0.3, 0.8, 1.2, 2.1, 0.5, 1.8, 0.4, 0.9, 2.4, 0.6, 1.1, 0.7];
const SKEW_COLORS = ['#00e5c0','#00e5c0','#ffd166','#ff4466','#00e5c0','#ff4466','#00e5c0','#00e5c0','#ff4466','#00e5c0','#ffd166','#00e5c0'];
const ANOMALY_DATA = [2,4,8,14,22,31,28,20,15,9,5,3,2,1];
const ANOMALY_MAX  = Math.max(...ANOMALY_DATA);

const FEATURES = [
  { icon: '#', title: 'ML Readiness Score',     desc: 'A composite score across data health, trainability, and leakage risk. Know instantly if your data is ready.' },
  { icon: '!', title: 'Leakage Detection',       desc: 'Automatically flags identifier columns, downstream features, and suspiciously correlated variables that could inflate your metrics.' },
  { icon: '+', title: 'Baseline Probe',          desc: 'Runs a 3-fold cross-validated LightGBM baseline to give you a realistic performance floor before you invest in modeling.' },
  { icon: '*', title: 'Outlier and Skew Analysis', desc: 'Per-feature skewness, kurtosis, outlier percentages and log-transform recommendations with visual distribution charts.' },
  { icon: '~', title: 'NLP Readiness',           desc: 'Detects free-text columns automatically and runs vocabulary analysis, readability scoring, and TF-IDF vectorization recommendations.' },
  { icon: '=', title: 'Drift Comparison',        desc: 'Upload train and test splits to get per-column distribution drift scores and mean shift analysis across all numeric features.' },
  { icon: 'x', title: 'Feature Redundancy',      desc: 'Correlation matrix analysis catches highly redundant feature pairs before they hurt your model interpretability.' },
  { icon: '@', title: 'Analysis Profiles',       desc: 'Switch between Standard, Finance, Healthcare, and NLP profiles to apply domain-specific thresholds and recommendations.' },
  { icon: 'v', title: 'Preprocessing Script',    desc: 'One-click export of a production-ready Python script with imputation, encoding, transforms, and feature drops tailored to your data.' },
];

export default function Landing({ onLaunch }: { onLaunch: () => void }) {
  const revealRefs = useRef<HTMLDivElement[]>([]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            (e.target as HTMLElement).style.transitionDelay = '0s';
            e.target.classList.add('visible');
          }
        });
      },
      { threshold: 0.1 }
    );
    revealRefs.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, []);

  const addReveal = (el: HTMLDivElement | null) => {
    if (el && !revealRefs.current.includes(el)) revealRefs.current.push(el);
  };

  const doubled = [...TICKER_ITEMS, ...TICKER_ITEMS];

  return (
    <>
      <style>{css}</style>
      <div className="lp-root">
        <div className="lp-noise" />
        <div className="lp-grid-bg" />

        {/* NAV */}
        <nav className="lp-nav">
          <div className="lp-nav-logo">Athena <span>// ML Diagnostics</span></div>

          <button onClick={onLaunch} className="lp-nav-cta">Launch App &rarr;</button>
        </nav>

        {/* HERO */}
        <div className="lp-hero">
          <div className="lp-badge">
            <div className="lp-badge-dot" />
            Currently in Beta
          </div>

          <h1 className="lp-hero-title">
            Know your data<br />
            <span className="lp-hero-title-accent">before you train.</span>
          </h1>
          <div className="lp-hero-subtitle">// ML Readiness &middot; EDA &middot; Diagnostics</div>

          <p className="lp-hero-desc">
            Upload a CSV. Get a full diagnostic report covering outliers, leakage, class imbalance,
            feature redundancy, and NLP readiness in under two minutes.
            Download a preprocessing script and ship faster.
          </p>

          <div className="lp-hero-actions">
            <button onClick={onLaunch} className="lp-btn-primary">Analyze a Dataset &rarr;</button>
            <a href="#how" className="lp-btn-ghost">See how it works</a>
          </div>

          {/* DASHBOARD PREVIEW */}
          <div className="lp-preview-wrap">
            <div className="lp-preview-glow" />
            <div className="lp-preview-frame">
              <div className="lp-preview-bar">
                <div className="lp-preview-dot" style={{ background: '#ff5f57' }} />
                <div className="lp-preview-dot" style={{ background: '#febc2e' }} />
                <div className="lp-preview-dot" style={{ background: '#28c840' }} />
                <span className="lp-preview-bar-title">Athena / titanic.csv. ML Diagnostics</span>
              </div>
              <div className="lp-preview-content">
                <div className="lp-preview-card">
                  <div className="lp-preview-card-label">ML Readiness</div>
                  <div className="lp-preview-card-value">87</div>
                  <div className="lp-preview-card-sub">/ 100 score</div>
                </div>
                <div className="lp-preview-card">
                  <div className="lp-preview-card-label">Baseline Acc.</div>
                  <div className="lp-preview-card-value blue">0.821</div>
                  <div className="lp-preview-card-sub">3-fold CV</div>
                </div>
                <div className="lp-preview-card">
                  <div className="lp-preview-card-label">Overfitting</div>
                  <div className="lp-preview-card-value">CLEAN</div>
                  <div className="lp-preview-card-sub">gap 0.012</div>
                </div>
                <div className="lp-preview-card">
                  <div className="lp-preview-card-label">Leakage Score</div>
                  <div className="lp-preview-card-value yellow">2</div>
                  <div className="lp-preview-card-sub">features flagged</div>
                </div>

                <div className="lp-preview-chart">
                  <div className="lp-preview-chart-label">Feature Skewness</div>
                  <div className="lp-preview-bars">
                    {SKEW_DATA.map((v, i) => (
                      <div
                        key={i}
                        className="lp-preview-bar-item"
                        style={{ height: `${(v / 2.4) * 100}%`, background: SKEW_COLORS[i], opacity: 0.8 }}
                      />
                    ))}
                  </div>
                </div>

                <div className="lp-preview-chart">
                  <div className="lp-preview-chart-label">Anomaly Score Distribution</div>
                  <div className="lp-preview-bars">
                    {ANOMALY_DATA.map((v, i) => (
                      <div
                        key={i}
                        className="lp-preview-bar-item"
                        style={{ height: `${(v / ANOMALY_MAX) * 100}%`, background: i < 5 ? '#ff4466' : '#00b4ff', opacity: 0.7 }}
                      />
                    ))}
                  </div>
                </div>

                <div className="lp-preview-tags">
                  <span className="lp-preview-tag green">No missing values</span>
                  <span className="lp-preview-tag green">Balanced classes</span>
                  <span className="lp-preview-tag yellow">Age: log transform recommended</span>
                  <span className="lp-preview-tag red">PassengerId: identifier leak</span>
                  <span className="lp-preview-tag green">Low redundancy</span>
                  <span className="lp-preview-tag yellow">Cabin: 77% missing</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* TICKER */}
        <div className="lp-ticker-wrap">
          <div className="lp-ticker-track">
            {doubled.map((item, i) => (
              <span key={i} className="lp-ticker-item"><span>+</span>{item}</span>
            ))}
          </div>
        </div>

        {/* HOW IT WORKS */}
        <section className="lp-section" id="how">
          <div ref={addReveal} className="lp-reveal">
            <div className="lp-section-label">// Process</div>
            <h2 className="lp-section-title">Three steps.<br />Full picture.</h2>
            <p className="lp-section-desc">No setup. No API keys. No notebooks. Drop a CSV and get diagnostic intelligence built for ML practitioners.</p>
          </div>
          <div ref={addReveal} className="lp-steps lp-reveal">
            {[
              { n: '01', icon: '^', title: 'Upload your CSV',    desc: 'Drop any tabular dataset up to 200k rows. Label a target column or run unlabeled diagnostics. Athena handles the rest.' },
              { n: '02', icon: '*', title: 'Get your report',    desc: 'Full EDA, ML diagnostics, leakage detection, outlier analysis, NLP readiness, clustering, and baseline model probing in one pass.' },
              { n: '03', icon: 'v', title: 'Download and ship',  desc: 'Export a ready-to-run Python preprocessing script tailored to your dataset. Or download the full HTML report for stakeholders.' },
            ].map((s) => (
              <div key={s.n} className="lp-step">
                <div className="lp-step-line" />
                <div className="lp-step-num">{s.n}</div>
                <div className="lp-step-icon">{s.icon}</div>
                <div className="lp-step-title">{s.title}</div>
                <p className="lp-step-desc">{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* METRICS */}
        <div ref={addReveal} className="lp-metrics lp-reveal">
          {[
            { val: '22+',   label: 'Diagnostic checks'   },
            { val: '200k',  label: 'Max rows supported'  },
            { val: '<2min', label: 'Full analysis time'  },
            { val: '0',     label: 'Setup required'      },
          ].map((m) => (
            <div key={m.label} className="lp-metric">
              <div className="lp-metric-value">{m.val}</div>
              <div className="lp-metric-label">{m.label}</div>
            </div>
          ))}
        </div>

        {/* FEATURES */}
        <section className="lp-section" id="features">
          <div ref={addReveal} className="lp-reveal">
            <div className="lp-section-label">// Capabilities</div>
            <h2 className="lp-section-title">Everything you need<br /><em>before</em> you model.</h2>
          </div>
          <div ref={addReveal} className="lp-features-grid lp-reveal">
            {FEATURES.map((f) => (
              <div key={f.title} className="lp-feature-card">
                <div className="lp-feature-icon">{f.icon}</div>
                <div className="lp-feature-title">{f.title}</div>
                <p className="lp-feature-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* TERMINAL */}
        <section className="lp-section">
          <div ref={addReveal} className="lp-reveal">
            <div className="lp-section-label">// Sample Output</div>
            <h2 className="lp-section-title">What Athena<br /><em>actually tells you.</em></h2>
            <p className="lp-section-desc">Real diagnostic output from the Food Inspections dataset (200k rows, Chicago open data).</p>
          </div>
          <div ref={addReveal} className="lp-terminal lp-reveal">
            <div className="lp-terminal-bar">
              <div className="lp-preview-dot" style={{ background: '#ff5f57' }} />
              <div className="lp-preview-dot" style={{ background: '#febc2e' }} />
              <div className="lp-preview-dot" style={{ background: '#28c840' }} />
              <span style={{ fontSize: '0.65rem', letterSpacing: '0.15em', color: 'var(--text-3)', marginLeft: '0.5rem', textTransform: 'uppercase' }}>
                Athena Diagnostic Report. food_inspections.csv
              </span>
            </div>
            <div className="lp-terminal-body">
              <div><span className="t-comment">// Column Classification</span></div>
              <div><span className="t-key">identifiers</span>    &nbsp;&nbsp;&nbsp;<span className="t-prompt">-&gt;</span>  <span className="t-str">["Inspection ID", "Location"]</span></div>
              <div><span className="t-key">free_text</span>      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="t-prompt">-&gt;</span>  <span className="t-str">["Violations"]</span>  <span className="t-comment">// avg 1164 chars, NLP recommended</span></div>
              <div><span className="t-key">categorical</span>    &nbsp;&nbsp;&nbsp;<span className="t-prompt">-&gt;</span>  <span className="t-str">["Facility Type", "Risk", "City", "Zip", "Inspection Type"]</span></div>
              <div><span className="t-key">numeric</span>        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="t-prompt">-&gt;</span>  <span className="t-str">["Latitude", "Longitude"]</span></div>
              <br />
              <div><span className="t-comment">// ML Diagnostics</span></div>
              <div><span className="t-key">baseline_accuracy</span>  <span className="t-prompt">-&gt;</span>  <span className="t-num">0.143</span>  <span className="t-comment">// random chance = 0.143, too few features</span></div>
              <div><span className="t-key">overfitting</span>        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="t-prompt">-&gt;</span>  <span className="t-val">CLEAN</span>  <span className="t-comment">// train-val gap 0.006</span></div>
              <div><span className="t-key">leakage_score</span>      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="t-prompt">-&gt;</span>  <span className="t-num">0</span></div>
              <br />
              <div><span className="t-comment">// Recommendations</span></div>
              <div><span className="t-warn">!</span>  <span className="t-key">Violations</span> contains rich free text. apply TF-IDF or sentence embeddings</div>
              <div><span className="t-warn">!</span>  Only 2 usable numeric features detected, insufficient signal without NLP</div>
              <div><span className="t-val">+</span>  No missing values in numeric features</div>
              <div><span className="t-val">+</span>  Class distribution acceptable (Pass: 48%, Fail: 18%)</div>
              <br />
              <div><span className="t-comment">// Action Plan</span></div>
              <div><span className="t-key">drop</span>          &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span className="t-prompt">-&gt;</span>  <span className="t-str">["Inspection ID", "Location"]</span></div>
              <div><span className="t-key">nlp_pipeline</span>  &nbsp;&nbsp;&nbsp;<span className="t-prompt">-&gt;</span>  <span className="t-str">["Violations"]</span>  <span className="t-comment">// script generated</span></div>
            </div>
          </div>
        </section>

        {/* CTA */}
        <div className="lp-cta">
          <div className="lp-cta-glow" />
          <div className="lp-section-label">// Get Started</div>
          <h2 ref={addReveal} className="lp-section-title lp-reveal">
            Stop guessing.<br /><em>Start analyzing.</em>
          </h2>
          <p ref={addReveal} className="lp-section-desc lp-reveal">
            Drop your CSV and get a full ML readiness report in minutes. Free, no signup required.
          </p>
          <div ref={addReveal} className="lp-reveal" style={{ marginTop: '2rem' }}>
            <button onClick={onLaunch} className="lp-btn-primary">Analyze Your Dataset &rarr;</button>
          </div>
        </div>

        {/* FOOTER */}
        <footer className="lp-footer">
          <div className="lp-footer-logo">Athena // ML Diagnostics</div>
          <div className="lp-footer-copy">Built with Python &middot; FastAPI &middot; React &middot; LightGBM</div>
        </footer>
      </div>
    </>
  );
}