"use client";

import { useState } from "react";
import ImageUploader from "./components/ImageUploader";
import PredictionResult from "./components/PredictionResult";

const API_URL = "http://localhost:8000";

export default function Home() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ── Call FastAPI /predict ──────────────────────────────────────────────────
  const handlePredict = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(
          data?.detail || `Server returned ${res.status} ${res.statusText}`
        );
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      if (err.name === "TypeError" && err.message === "Failed to fetch") {
        setError(
          "Cannot reach the backend. Make sure the FastAPI server is running on port 8000."
        );
      } else {
        setError(err.message || "An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError("");
  };

  return (
    <main className="flex-1 flex flex-col">
      {/* ── Background gradient blobs ─────────────────────────────────────── */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-[500px] h-[500px] rounded-full bg-purple-600/10 blur-[128px]" />
        <div className="absolute top-1/2 -right-40 w-[400px] h-[400px] rounded-full bg-blue-600/8 blur-[128px]" />
        <div className="absolute -bottom-20 left-1/3 w-[350px] h-[350px] rounded-full bg-indigo-600/8 blur-[128px]" />
      </div>

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <header className="border-b border-[var(--border-color)] backdrop-blur-md bg-[var(--background)]/60">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-purple-500/25">
            EV
          </div>
          <div>
            <h1 className="text-base font-semibold tracking-tight">
              EverLearn Vision
            </h1>
            <p className="text-xs text-[var(--muted)]">
              AI-powered image classifier
            </p>
          </div>
        </div>
      </header>

      {/* ── Content ───────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col items-center justify-start pt-12 pb-20 px-6">
        <div className="w-full max-w-xl space-y-6">
          {/* Hero */}
          <div className="text-center space-y-2 mb-4">
            <h2 className="text-2xl font-bold tracking-tight">
              Classify any image{" "}
              <span className="bg-gradient-to-r from-purple-400 to-indigo-400 bg-clip-text text-transparent">
                instantly
              </span>
            </h2>
            <p className="text-sm text-[var(--muted)] max-w-md mx-auto">
              Upload a photo and our fine-tuned ResNet model will predict its
              class with a confidence score.
            </p>
          </div>

          {/* Upload */}
          <ImageUploader onFileSelect={setFile} disabled={loading} />

          {/* Action buttons */}
          <div className="flex items-center gap-3">
            <button
              id="predict-button"
              onClick={handlePredict}
              disabled={!file || loading}
              className="flex-1 py-3 rounded-xl font-semibold text-sm text-white
                         bg-gradient-to-r from-purple-600 to-indigo-600
                         hover:from-purple-500 hover:to-indigo-500
                         disabled:opacity-40 disabled:cursor-not-allowed
                         transition-all shadow-lg shadow-purple-600/20
                         hover:shadow-purple-500/30 active:scale-[0.98]"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  {/* Spinner */}
                  <svg
                    className="animate-spin-slow w-4 h-4"
                    viewBox="0 0 24 24"
                    fill="none"
                  >
                    <circle
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="3"
                      className="opacity-20"
                    />
                    <path
                      d="M12 2a10 10 0 0 1 10 10"
                      stroke="currentColor"
                      strokeWidth="3"
                      strokeLinecap="round"
                    />
                  </svg>
                  Classifying…
                </span>
              ) : (
                "Classify Image"
              )}
            </button>

            {(result || file) && !loading && (
              <button
                id="reset-button"
                onClick={handleReset}
                className="py-3 px-5 rounded-xl text-sm font-medium
                           border border-[var(--border-color)]
                           hover:bg-[var(--surface-hover)]
                           transition-all active:scale-[0.98]"
              >
                Reset
              </button>
            )}
          </div>

          {/* Error */}
          {error && (
            <div
              className="text-sm text-red-400 bg-red-400/10 border border-red-400/20
                          rounded-xl px-5 py-3 animate-fade-in-up"
              role="alert"
            >
              <span className="font-medium">Error:</span> {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <PredictionResult
              label={result.label}
              confidence={result.confidence}
              allProbabilities={result.all_probabilities}
            />
          )}
        </div>
      </div>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer className="border-t border-[var(--border-color)] backdrop-blur-md bg-[var(--background)]/60">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between text-xs text-[var(--muted)]">
          <p>EverLearn Vision • ResNet-18 • PyTorch + FastAPI</p>
          <p>
            Backend:{" "}
            <code className="bg-[var(--surface)] px-1.5 py-0.5 rounded text-[var(--accent-light)]">
              localhost:8000
            </code>
          </p>
        </div>
      </footer>
    </main>
  );
}
