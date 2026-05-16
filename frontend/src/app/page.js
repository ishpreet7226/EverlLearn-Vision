"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import UploadBox from "./components/UploadBox";
import PredictionCard from "./components/PredictionCard";
import FeedbackPanel from "./components/FeedbackPanel";
import HistoryList from "./components/HistoryList";
import SkeletonLoader from "./components/SkeletonLoader";
import DarkModeToggle from "./components/DarkModeToggle";
import { useToast } from "./components/Toast";

const getApiUrl = () => {
  if (typeof window !== "undefined") {
    return `http://${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
};

let historyId = 0;

export default function Home() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [classList, setClassList] = useState([]);
  const [modelVersion, setModelVersion] = useState(null);
  const [history, setHistory] = useState([]);
  const { toast } = useToast();
  const resultRef = useRef(null);

  // Fetch class list + model info on mount
  useEffect(() => {
    fetch(`${getApiUrl()}/`)
      .then((res) => res.json())
      .then((data) => {
        if (data.classes) setClassList(data.classes);
        if (data.model_version) setModelVersion(data.model_version);
      })
      .catch(() => {});
  }, []);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const res = await fetch(`${getApiUrl()}/predict`, { method: "POST", body: formData });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail || `Server returned ${res.status} ${res.statusText}`);
      }

      const data = await res.json();
      setResult(data);
      toast(`Predicted: ${data.label} (${(data.confidence * 100).toFixed(1)}%)`, "success");

      // Scroll to result
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 200);
    } catch (err) {
      if (err.name === "TypeError" && err.message === "Failed to fetch") {
        setError("Cannot reach the backend. Make sure the FastAPI server is running on port 8000.");
      } else {
        setError(err.message || "An unexpected error occurred.");
      }
      toast("Prediction failed", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setFileName("");
    setPreviewUrl(null);
    setResult(null);
    setError("");
  };

  const handleFileSelect = (f) => {
    setFile(f);
    setFileName(f.name);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const handleFeedback = (type, actualLabel) => {
    // Add to history
    const entry = {
      id: ++historyId,
      imageUrl: previewUrl,
      label: type === "correct" ? result.label : actualLabel,
      confidence: result.confidence,
      timestamp: new Date().toLocaleTimeString(),
    };
    setHistory((prev) => [entry, ...prev]);
  };

  return (
    <main className="flex-1 flex flex-col">
      {/* Background gradient blobs */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-[500px] h-[500px] rounded-full bg-purple-600/10 blur-[128px]" />
        <div className="absolute top-1/2 -right-40 w-[400px] h-[400px] rounded-full bg-blue-600/8 blur-[128px]" />
        <div className="absolute -bottom-20 left-1/3 w-[350px] h-[350px] rounded-full bg-indigo-600/8 blur-[128px]" />
      </div>

      {/* Header */}
      <header className="border-b border-[var(--border-color)] backdrop-blur-md bg-[var(--background)]/60 sticky top-0 z-40">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <motion.div
              className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center text-white font-bold text-sm shadow-lg shadow-purple-500/25"
              whileHover={{ rotate: 12, scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              EL
            </motion.div>
            <div>
              <h1 className="text-base font-semibold tracking-tight">EverLearn ML</h1>
              <p className="text-xs text-[var(--muted)]">Self-improving classifier</p>
            </div>
          </div>
          <DarkModeToggle />
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 flex flex-col items-center justify-start pt-12 pb-20 px-6">
        <div className="w-full max-w-xl space-y-6">

          {/* Hero */}
          <motion.div
            className="text-center space-y-3 mb-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
          >
            <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
              Classify any image{" "}
              <span className="bg-gradient-to-r from-purple-400 via-indigo-400 to-blue-400 bg-clip-text text-transparent">
                instantly
              </span>
            </h2>
            <motion.p
              className="text-sm text-[var(--muted)] max-w-md mx-auto leading-relaxed"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.5 }}
            >
              Self-Improving Image Classification System — upload a photo and our fine-tuned ResNet model will predict its class. Your feedback makes it smarter over time.
            </motion.p>
          </motion.div>

          {/* Upload */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
          >
            <UploadBox onFileSelect={handleFileSelect} disabled={loading} />
          </motion.div>

          {/* Action buttons */}
          <motion.div
            className="flex items-center gap-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            <motion.button
              id="predict-button"
              onClick={handlePredict}
              disabled={!file || loading}
              whileHover={!file || loading ? {} : { scale: 1.02 }}
              whileTap={!file || loading ? {} : { scale: 0.98 }}
              className="flex-1 py-3 rounded-xl font-semibold text-sm text-white
                         bg-gradient-to-r from-purple-600 to-indigo-600
                         hover:from-purple-500 hover:to-indigo-500
                         disabled:opacity-40 disabled:cursor-not-allowed
                         transition-all shadow-lg shadow-purple-600/20
                         hover:shadow-purple-500/30"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin-slow w-4 h-4" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-20" />
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                  Classifying…
                </span>
              ) : (
                "Classify Image"
              )}
            </motion.button>

            <AnimatePresence>
              {(result || file) && !loading && (
                <motion.button
                  id="reset-button"
                  onClick={handleReset}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="py-3 px-5 rounded-xl text-sm font-medium
                             border border-[var(--border-color)]
                             hover:bg-[var(--surface-hover)] transition-colors"
                >
                  Reset
                </motion.button>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                className="text-sm text-red-400 bg-red-400/10 border border-red-400/20 rounded-xl px-5 py-3"
                role="alert"
              >
                <span className="font-medium">Error:</span> {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Loading skeleton */}
          <AnimatePresence>
            {loading && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <SkeletonLoader />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Results */}
          <div ref={resultRef}>
            <AnimatePresence>
              {result && !loading && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <PredictionCard
                    label={result.label}
                    confidence={result.confidence}
                    allProbabilities={result.all_probabilities}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Feedback */}
          <AnimatePresence>
            {result && !loading && (
              <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
                <FeedbackPanel
                  predictedLabel={result.label}
                  confidence={result.confidence}
                  fileName={fileName}
                  classList={classList}
                  onFeedbackSent={handleFeedback}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* History */}
          {history.length > 0 && (
            <motion.div
              className="pt-6 border-t border-[var(--border-color)]"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <HistoryList history={history} />
            </motion.div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-[var(--border-color)] backdrop-blur-md bg-[var(--background)]/60">
        <div className="max-w-3xl mx-auto px-6 py-4 flex items-center justify-between text-xs text-[var(--muted)]">
          <p>EverLearn ML • ResNet-18{modelVersion ? ` v${modelVersion}` : ""} • PyTorch + FastAPI</p>
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
