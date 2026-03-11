import { useState } from "react";
import { AnalysisResult, AppView, DataMode, Profile } from "./types";
import { analyzeDataset } from "./lib/api";
import UploadPage from "./components/UploadPage";
import ReportPage from "./components/ReportPage";
import ComparePage from "./components/ComparePage";
import Landing from "./components/Landing"; 
import "./index.css";

export default function App() {
  const [view, setView] = useState<AppView>("landing");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [filename, setFilename] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async (
    file: File,
    mode: DataMode,
    targetCol: string | null,
    profile: Profile
  ) => {
    setLoading(true);
    setError(null);
    setFilename(file.name);
    try {
      const r = await analyzeDataset(file, mode, targetCol, profile);
      setResult(r);
      setView("report");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
          }
          @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
          `}</style>

      {view === "landing" && (
        <Landing onLaunch={() => setView("upload")} />
      )}
      {view === "upload" && (
        <>
          {error && (
            <div style={{
              position: "fixed",
              top: "1rem",
              left: "50%",
              transform: "translateX(-50%)",
              background: "var(--red-glow)",
              border: "1px solid var(--red-dim)",
              borderRadius: "var(--radius-md)",
              padding: "0.75rem 1.25rem",
              fontFamily: "var(--font-mono)",
              fontSize: "0.75rem",
              color: "var(--red)",
              zIndex: 999,
              maxWidth: "500px",
            }}>
              ✕ {error}
            </div>
          )}
          <UploadPage
            onAnalyze={handleAnalyze}
            onCompare={() => setView("compare")}
            onHome={() => setView("landing")}
            loading={loading}
            />
        </>
      )}

      {view === "report" && result && (
        <ReportPage
          result={result}
          filename={filename}
          onBack={() => setView("upload")}
        />
      )}

      {view === "compare" && (
        <ComparePage onBack={() => setView("upload")} />
      )}
    </>
  );
}
