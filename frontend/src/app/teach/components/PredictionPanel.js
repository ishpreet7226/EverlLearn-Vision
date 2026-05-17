"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const getApiUrl = () => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (typeof window !== "undefined") {
    return `http://${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
};

export default function PredictionPanel({ isModelReady = false, mode = "manual" }) {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [valImages, setValImages] = useState(null); // { classes: { className: [{filename, data_url}] } }
  const [loadingVal, setLoadingVal] = useState(false);
  const [activeTab, setActiveTab] = useState("upload"); // upload | validate
  const inputRef = useRef(null);

  // Load val images after model is ready (for folder mode)
  useEffect(() => {
    if (isModelReady && mode === "folder") {
      fetchValImages();
    }
  }, [isModelReady, mode]);

  const fetchValImages = async () => {
    setLoadingVal(true);
    try {
      const res = await fetch(`${getApiUrl()}/teach/val-images`);
      if (res.ok) {
        const data = await res.json();
        setValImages(data);
      }
    } catch {
      // silent fail
    } finally {
      setLoadingVal(false);
    }
  };

  const handleFileSelect = (f) => {
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError("");
  };

  const handleValImageClick = async (dataUrl, filename) => {
    // Convert data URL to file for prediction
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    const f = new File([blob], filename, { type: blob.type });
    setFile(f);
    setPreviewUrl(dataUrl);
    setResult(null);
    setError("");
    setActiveTab("upload");
    // Auto-predict
    predictFile(f);
  };

  const handleFiles = (fileList) => {
    if (!fileList || !isModelReady) return;
    const f = Array.from(fileList).find((f) => f.type.startsWith("image/"));
    if (f) handleFileSelect(f);
  };

  const predictFile = async (fileToPredict) => {
    const f = fileToPredict || file;
    if (!f || !isModelReady) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", f);
      const res = await fetch(`${getApiUrl()}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const d = await res.json().catch(() => null);
        throw new Error(d?.detail || `Error ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = () => predictFile();

  const sortedProbs = result?.all_probabilities
    ? Object.entries(result.all_probabilities).sort((a, b) => b[1] - a[1])
    : [];

  const hasValImages = valImages && Object.keys(valImages.classes || {}).length > 0;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-[var(--foreground)]">Prediction</h3>

      {!isModelReady ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-card p-8 text-center"
        >
          <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-[var(--surface)] flex items-center justify-center">
            <svg className="w-6 h-6 text-[var(--muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <p className="text-sm text-[var(--muted)]">Train a model first</p>
          <p className="text-xs text-[var(--muted)] mt-1">
            Add images to your classes and click "Train Model"
          </p>
        </motion.div>
      ) : (
        <div className="space-y-3">
          {/* Tab switcher (only show if val images available) */}
          {hasValImages && (
            <div className="flex rounded-lg bg-[var(--surface)] p-0.5 gap-0.5">
              <button
                onClick={() => setActiveTab("upload")}
                className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === "upload"
                    ? "bg-purple-500/15 text-purple-400 shadow-sm"
                    : "text-[var(--muted)] hover:text-[var(--foreground)]"
                }`}
              >
                Upload Image
              </button>
              <button
                onClick={() => setActiveTab("validate")}
                className={`flex-1 py-1.5 rounded-md text-xs font-medium transition-all ${
                  activeTab === "validate"
                    ? "bg-purple-500/15 text-purple-400 shadow-sm"
                    : "text-[var(--muted)] hover:text-[var(--foreground)]"
                }`}
              >
                Test Val Images ({valImages?.total || 0})
              </button>
            </div>
          )}

          <AnimatePresence mode="wait">
            {/* Upload tab */}
            {activeTab === "upload" && (
              <motion.div
                key="upload-tab"
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                className="space-y-3"
              >
                {/* Upload zone */}
                <motion.div
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFiles(e.dataTransfer.files); }}
                  onClick={() => inputRef.current?.click()}
                  whileHover={{ scale: 1.01 }}
                  whileTap={{ scale: 0.99 }}
                  className={`glass-card cursor-pointer transition-all overflow-hidden
                    ${dragOver ? "border-purple-500 bg-purple-500/10" : ""}`}
                >
                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={(e) => { handleFiles(e.target.files); e.target.value = ""; }}
                  />
                  {previewUrl ? (
                    <div className="relative">
                      <img src={previewUrl} alt="Preview" className="w-full aspect-video object-cover rounded-t-xl" />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent rounded-t-xl" />
                      <p className="absolute bottom-2 left-3 text-xs text-white/80">Click to change image</p>
                    </div>
                  ) : (
                    <div className="p-8 text-center">
                      <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-purple-500/10 flex items-center justify-center">
                        <svg className="w-6 h-6 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                        </svg>
                      </div>
                      <p className="text-sm font-medium text-[var(--foreground)]">Upload test image</p>
                      <p className="text-xs text-[var(--muted)] mt-1">or drag & drop here</p>
                    </div>
                  )}
                </motion.div>

                {/* Classify button */}
                {file && (
                  <motion.button
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handlePredict}
                    disabled={loading}
                    className="w-full py-3 rounded-xl font-semibold text-sm text-white
                               bg-gradient-to-r from-purple-600 to-indigo-600
                               hover:from-purple-500 hover:to-indigo-500
                               disabled:opacity-60 transition-all shadow-lg shadow-purple-600/20"
                  >
                    {loading ? (
                      <span className="flex items-center justify-center gap-2">
                        <svg className="animate-spin-slow w-4 h-4" viewBox="0 0 24 24" fill="none">
                          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-20" />
                          <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                        </svg>
                        Classifying…
                      </span>
                    ) : "Classify"}
                  </motion.button>
                )}
              </motion.div>
            )}

            {/* Validate tab — val images gallery */}
            {activeTab === "validate" && hasValImages && (
              <motion.div
                key="validate-tab"
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                className="space-y-3"
              >
                <p className="text-xs text-[var(--muted)]">
                  Click any validation image to test the model's prediction
                </p>
                {Object.entries(valImages.classes).map(([cls, images]) => (
                  <div key={cls} className="glass-card overflow-hidden">
                    <div className="px-3 py-2 border-b border-[var(--border-color)]">
                      <span className="text-xs font-semibold text-[var(--foreground)] capitalize">
                        {cls}
                      </span>
                      <span className="text-[10px] text-[var(--muted)] ml-2">
                        {images.length} image{images.length !== 1 ? "s" : ""} (ground truth)
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1.5 p-2">
                      {images.map((img, i) => (
                        <motion.button
                          key={i}
                          whileHover={{ scale: 1.08 }}
                          whileTap={{ scale: 0.95 }}
                          onClick={() => handleValImageClick(img.data_url, img.filename)}
                          className="w-14 h-14 rounded-lg overflow-hidden bg-[var(--surface)]
                                     ring-2 ring-transparent hover:ring-purple-500/50 transition-all
                                     cursor-pointer"
                          title={`Test: ${img.filename} (expected: ${cls})`}
                        >
                          <img
                            src={img.data_url}
                            alt={img.filename}
                            className="w-full h-full object-cover"
                          />
                        </motion.button>
                      ))}
                    </div>
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2"
              >
                ⚠️ {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Results */}
          <AnimatePresence>
            {result && !loading && (
              <motion.div
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="glass-card p-4 space-y-4"
              >
                {/* Top prediction */}
                <div className="flex items-center gap-3">
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", damping: 12 }}
                    className="w-12 h-12 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-600/10
                               flex items-center justify-center shrink-0"
                  >
                    <span className="text-lg">🏷️</span>
                  </motion.div>
                  <div className="flex-1 min-w-0">
                    <p className="text-lg font-bold text-[var(--foreground)] truncate capitalize">
                      {result.label}
                    </p>
                    <p className="text-xs text-[var(--muted)]">
                      Confidence: {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* All probabilities */}
                <div className="space-y-2">
                  {sortedProbs.map(([cls, prob], i) => (
                    <div key={cls}>
                      <div className="flex justify-between text-xs mb-1">
                        <span className={`font-medium capitalize ${i === 0 ? "text-emerald-400" : "text-[var(--foreground)]"}`}>
                          {cls}
                        </span>
                        <span className="text-[var(--muted)]">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="prob-bar-track">
                        <motion.div
                          className={`prob-bar-fill ${i === 0 ? "top" : ""}`}
                          initial={{ width: 0 }}
                          animate={{ width: `${prob * 100}%` }}
                          transition={{ duration: 0.6, delay: i * 0.08 }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
