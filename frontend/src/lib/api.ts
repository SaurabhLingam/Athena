import { AnalysisResult, DriftResult } from "../types";

const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export async function analyzeDataset(
  file: File,
  dataType: "labeled" | "unlabeled",
  targetCol: string | null,
  profile: string
): Promise<AnalysisResult> {
  const params = new URLSearchParams({ data_type: dataType, profile });
  if (targetCol) params.set("target_col", targetCol);

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/analyze?${params}`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? "Analysis failed");
  }

  return res.json();
}

export async function compareDatasets(
  trainFile: File,
  testFile: File
): Promise<DriftResult> {
  const form = new FormData();
  form.append("train_file", trainFile);
  form.append("test_file", testFile);

  const res = await fetch(`${BASE}/compare`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? "Comparison failed");
  }

  return res.json();
}

export async function downloadScript(eda: object, filename = "preprocessing.py") {
  const res = await fetch(`${BASE}/script`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ eda }),
  });

  if (!res.ok) throw new Error("Script generation failed");

  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}