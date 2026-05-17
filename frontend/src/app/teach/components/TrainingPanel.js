"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const BACKBONES = [
  { value: "resnet18", label: "ResNet-18", desc: "Fast, lightweight" },
  { value: "resnet50", label: "ResNet-50", desc: "More accurate" },
  { value: "efficientnet_b0", label: "EfficientNet-B0", desc: "Balanced" },
  { value: "mobilenet_v3_small", label: "MobileNet-V3", desc: "Tiny & fast" },
];

const getApiUrl = () => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (typeof window !== "undefined") {
    return `http://${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
};

/**
 * Training state machine:
 * idle → uploading → training → complete → error
 */
export default function TrainingPanel({
  classes,
  mode = "manual",
  folderDatasetReady = false,
  onTrainingComplete,
  onTrainingStart,
}) {
  const [state, setState] = useState("idle"); // idle | uploading | training | complete | error
  const [backbone, setBackbone] = useState("resnet18");
  const [epochs, setEpochs] = useState(10);
  const [lr, setLr] = useState(0.001);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState("");
  const [epochHistory, setEpochHistory] = useState([]); // Array of per-epoch metrics
  const pollRef = useRef(null);
  const lastEpochRef = useRef(0);

  // Validation depends on mode
  const isReady = mode === "folder"
    ? folderDatasetReady
    : classes.length >= 2 &&
      classes.every((c) => c.images.length >= 2) &&
      classes.every((c) => c.name.trim().length > 0);

  // ── Polling for training status ────────────────────────────────────────────
  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    lastEpochRef.current = 0;

    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${getApiUrl()}/teach/training-status`);
        const data = await res.json();
        setProgress(data);

        // Accumulate epoch history (only when we see a new epoch with metrics)
        if (
          data.status === "training" &&
          data.epoch > lastEpochRef.current &&
          data.train_acc != null
        ) {
          lastEpochRef.current = data.epoch;
          setEpochHistory((prev) => [
            ...prev,
            {
              epoch: data.epoch,
              train_loss: data.train_loss,
              train_acc: data.train_acc,
              val_loss: data.val_loss,
              val_acc: data.val_acc,
              epoch_time: data.epoch_time,
            },
          ]);
        }

        if (data.status === "complete") {
          // Add final epoch if not already captured
          if (data.epoch > lastEpochRef.current && data.train_acc != null) {
            setEpochHistory((prev) => [
              ...prev,
              {
                epoch: data.epoch,
                train_loss: data.train_loss,
                train_acc: data.train_acc,
                val_loss: data.val_loss,
                val_acc: data.val_acc,
                epoch_time: data.epoch_time,
              },
            ]);
          }
          setState("complete");
          clearInterval(pollRef.current);
          pollRef.current = null;
          if (onTrainingComplete) onTrainingComplete(data);
        } else if (data.status === "error") {
          setState("error");
          setError(data.error || "Training failed.");
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch {
        // Server might be busy, keep polling
      }
    }, 2000);
  }, [onTrainingComplete]);

  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // ── Start training ─────────────────────────────────────────────────────────
  const handleTrain = async () => {
    setError("");
    setProgress(null);
    setEpochHistory([]);
    lastEpochRef.current = 0;
    if (onTrainingStart) onTrainingStart();

    try {
      // In manual mode: upload dataset first
      if (mode === "manual") {
        setState("uploading");
        const formData = new FormData();
        for (const cls of classes) {
          for (const img of cls.images) {
            formData.append("files", img.file);
            formData.append("class_names", cls.name);
          }
        }

        const uploadRes = await fetch(`${getApiUrl()}/teach/upload-dataset`, {
          method: "POST",
          body: formData,
        });

        if (!uploadRes.ok) {
          const d = await uploadRes.json().catch(() => null);
          throw new Error(d?.detail || `Upload failed: ${uploadRes.status}`);
        }
      }

      // Start training (dataset already on server for folder mode)
      setState("training");
      const trainForm = new FormData();
      trainForm.append("backbone", backbone);
      trainForm.append("epochs", epochs.toString());
      trainForm.append("lr", lr.toString());
      trainForm.append("batch_size", "32");

      const trainRes = await fetch(`${getApiUrl()}/teach/train`, {
        method: "POST",
        body: trainForm,
      });

      if (!trainRes.ok) {
        const d = await trainRes.json().catch(() => null);
        throw new Error(d?.detail || `Training failed: ${trainRes.status}`);
      }

      // Poll for progress
      startPolling();
    } catch (err) {
      setState("error");
      setError(err.message || "Something went wrong.");
    }
  };

  const epochProgress = progress?.total_epochs
    ? (progress.epoch / progress.total_epochs) * 100
    : 0;

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-[var(--foreground)]">Training</h3>

      {/* Config — only when idle */}
      <AnimatePresence mode="wait">
        {(state === "idle" || state === "error") && (
          <motion.div
            key="config"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-3"
          >
            {/* Backbone selector */}
            <div>
              <label className="text-xs text-[var(--muted)] mb-1.5 block">Model Architecture</label>
              <div className="grid grid-cols-2 gap-2">
                {BACKBONES.map((b) => (
                  <motion.button
                    key={b.value}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => setBackbone(b.value)}
                    className={`p-2.5 rounded-xl text-left border transition-all ${
                      backbone === b.value
                        ? "border-purple-500/50 bg-purple-500/10 shadow-sm shadow-purple-500/10"
                        : "border-[var(--border-color)] hover:border-purple-500/30"
                    }`}
                  >
                    <p className="text-xs font-semibold text-[var(--foreground)]">{b.label}</p>
                    <p className="text-[10px] text-[var(--muted)]">{b.desc}</p>
                  </motion.button>
                ))}
              </div>
            </div>

            {/* Epochs & LR */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-[var(--muted)] mb-1.5 block">Epochs</label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={epochs}
                  onChange={(e) => setEpochs(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full py-2 px-3 rounded-lg text-sm bg-[var(--surface)] border border-[var(--border-color)]
                             text-[var(--foreground)] focus:outline-none focus:ring-2 focus:ring-purple-500/40"
                />
              </div>
              <div>
                <label className="text-xs text-[var(--muted)] mb-1.5 block">Learning Rate</label>
                <input
                  type="number"
                  step={0.0001}
                  min={0.00001}
                  max={1}
                  value={lr}
                  onChange={(e) => setLr(parseFloat(e.target.value) || 0.001)}
                  className="w-full py-2 px-3 rounded-lg text-sm bg-[var(--surface)] border border-[var(--border-color)]
                             text-[var(--foreground)] focus:outline-none focus:ring-2 focus:ring-purple-500/40"
                />
              </div>
            </div>

            {/* Validation hints */}
            {!isReady && (
              <div className="text-xs text-amber-400/80 bg-amber-500/10 border border-amber-500/20 rounded-lg px-3 py-2 space-y-0.5">
                {mode === "folder" ? (
                  <p>• Upload a dataset folder first (on the left)</p>
                ) : (
                  <>
                    {classes.length < 2 && <p>• Need at least 2 classes</p>}
                    {classes.filter((c) => c.images.length < 2).length > 0 && (
                      <p>• Each class needs at least 2 images</p>
                    )}
                    {classes.filter((c) => !c.name.trim()).length > 0 && (
                      <p>• All classes need a name</p>
                    )}
                  </>
                )}
              </div>
            )}

            {/* Error */}
            {error && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2"
              >
                ⚠️ {error}
              </motion.div>
            )}

            {/* Train button */}
            <motion.button
              whileHover={isReady ? { scale: 1.02 } : {}}
              whileTap={isReady ? { scale: 0.98 } : {}}
              onClick={handleTrain}
              disabled={!isReady}
              className="w-full py-3 rounded-xl font-semibold text-sm text-white
                         bg-gradient-to-r from-purple-600 to-indigo-600
                         hover:from-purple-500 hover:to-indigo-500
                         disabled:opacity-40 disabled:cursor-not-allowed
                         transition-all shadow-lg shadow-purple-600/20
                         hover:shadow-purple-500/30"
            >
              Train Model
            </motion.button>
          </motion.div>
        )}

        {/* Uploading state */}
        {state === "uploading" && (
          <motion.div
            key="uploading"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="glass-card p-5 text-center space-y-3"
          >
            <div className="w-12 h-12 mx-auto rounded-full bg-purple-500/15 flex items-center justify-center">
              <svg className="w-6 h-6 text-purple-400 animate-spin-slow" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-20" />
                <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
            </div>
            <p className="text-sm font-medium text-[var(--foreground)]">Uploading dataset…</p>
            <p className="text-xs text-[var(--muted)]">Sending images to the server</p>
          </motion.div>
        )}

        {/* Training state */}
        {state === "training" && (
          <motion.div
            key="training"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            {/* Progress card */}
            <div className="glass-card p-4 space-y-3">
              {/* Progress header */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                  <span className="text-xs font-medium text-purple-400">Training…</span>
                </div>
                <span className="text-xs text-[var(--muted)]">
                  Epoch {progress?.epoch || 0}/{progress?.total_epochs || epochs}
                </span>
              </div>

              {/* Progress bar */}
              <div className="w-full h-2.5 rounded-full bg-[var(--surface)] overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-purple-500 to-indigo-500"
                  initial={{ width: 0 }}
                  animate={{ width: `${epochProgress}%` }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                />
              </div>

              {/* Live metrics */}
              {progress?.train_acc != null && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="grid grid-cols-2 gap-2"
                >
                  <div className="rounded-lg bg-[var(--surface)] p-2.5">
                    <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider">Train Acc</p>
                    <p className="text-lg font-bold text-[var(--foreground)]">
                      {(progress.train_acc * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="rounded-lg bg-[var(--surface)] p-2.5">
                    <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider">Val Acc</p>
                    <p className="text-lg font-bold text-[var(--foreground)]">
                      {(progress.val_acc * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="rounded-lg bg-[var(--surface)] p-2.5">
                    <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider">Train Loss</p>
                    <p className="text-lg font-bold text-[var(--foreground)]">
                      {progress.train_loss?.toFixed(4)}
                    </p>
                  </div>
                  <div className="rounded-lg bg-[var(--surface)] p-2.5">
                    <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider">Best Val</p>
                    <p className="text-lg font-bold text-emerald-400">
                      {(progress.best_val_acc * 100).toFixed(1)}%
                    </p>
                  </div>
                </motion.div>
              )}
            </div>

            {/* Epoch History Table (live) */}
            {epochHistory.length > 0 && (
              <EpochTable epochs={epochHistory} />
            )}
          </motion.div>
        )}

        {/* Complete state */}
        {state === "complete" && (
          <motion.div
            key="complete"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-4"
          >
            {/* Success banner */}
            <div className="glass-card p-5 text-center space-y-3">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", damping: 12, delay: 0.1 }}
                className="w-14 h-14 mx-auto rounded-full bg-emerald-500/15 flex items-center justify-center"
              >
                <svg className="w-7 h-7 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </motion.div>

              <div>
                <p className="text-sm font-semibold text-emerald-400">Model Ready!</p>
                <p className="text-xs text-[var(--muted)] mt-1">
                  Upload a test image on the right to try it out
                </p>
              </div>
            </div>

            {/* Final results summary */}
            {progress && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-card p-4 space-y-3"
              >
                <h4 className="text-xs uppercase tracking-widest text-[var(--muted)]">
                  Final Results
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  <ResultStat label="Best Val Accuracy" value={`${(progress.best_val_acc * 100).toFixed(2)}%`} highlight />
                  <ResultStat label="Final Train Acc" value={`${(progress.train_acc * 100).toFixed(2)}%`} />
                  <ResultStat label="Final Val Acc" value={`${(progress.val_acc * 100).toFixed(2)}%`} />
                  <ResultStat label="Final Train Loss" value={progress.train_loss?.toFixed(4)} />
                  <ResultStat label="Final Val Loss" value={progress.val_loss?.toFixed(4)} />
                  <ResultStat label="Epochs Trained" value={`${progress.epoch}/${progress.total_epochs}`} />
                </div>
                {progress.classes && (
                  <div className="pt-2 border-t border-[var(--border-color)]">
                    <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider mb-1">Classes</p>
                    <div className="flex flex-wrap gap-1.5">
                      {progress.classes.map((cls) => (
                        <span
                          key={cls}
                          className="text-xs px-2 py-0.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20 capitalize"
                        >
                          {cls}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </motion.div>
            )}

            {/* Epoch history table */}
            {epochHistory.length > 0 && (
              <EpochTable epochs={epochHistory} />
            )}

            {/* Train again */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => {
                setState("idle");
                setProgress(null);
                setEpochHistory([]);
              }}
              className="w-full py-2.5 rounded-xl text-xs font-medium
                         border border-[var(--border-color)] text-[var(--muted)]
                         hover:text-purple-400 hover:border-purple-500/30
                         hover:bg-purple-500/5 transition-all"
            >
              Train again with new settings →
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}


// ── Epoch History Table ────────────────────────────────────────────────────────
function EpochTable({ epochs }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card overflow-hidden"
    >
      <div className="px-4 py-2.5 border-b border-[var(--border-color)]">
        <h4 className="text-xs uppercase tracking-widest text-[var(--muted)]">
          Epoch History
        </h4>
      </div>
      <div className="overflow-x-auto max-h-52 overflow-y-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-[var(--background)]">
            <tr className="text-[var(--muted)] border-b border-[var(--border-color)]">
              <th className="py-2 px-3 text-left font-medium">#</th>
              <th className="py-2 px-2 text-right font-medium">Train Loss</th>
              <th className="py-2 px-2 text-right font-medium">Train Acc</th>
              <th className="py-2 px-2 text-right font-medium">Val Loss</th>
              <th className="py-2 px-2 text-right font-medium">Val Acc</th>
              <th className="py-2 px-3 text-right font-medium">Time</th>
            </tr>
          </thead>
          <tbody>
            {epochs.map((e, i) => {
              const isLast = i === epochs.length - 1;
              const isBest = e.val_acc === Math.max(...epochs.map((ep) => ep.val_acc));
              return (
                <motion.tr
                  key={e.epoch}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03 }}
                  className={`border-b border-[var(--border-color)]/50 transition-colors
                    ${isLast ? "bg-purple-500/5" : "hover:bg-[var(--surface-hover)]"}
                    ${isBest ? "text-emerald-400" : "text-[var(--foreground)]"}`}
                >
                  <td className="py-1.5 px-3 font-medium">
                    {e.epoch}
                    {isBest && <span className="ml-1 text-emerald-400">★</span>}
                  </td>
                  <td className="py-1.5 px-2 text-right font-mono">{e.train_loss?.toFixed(4)}</td>
                  <td className="py-1.5 px-2 text-right font-mono">{(e.train_acc * 100).toFixed(1)}%</td>
                  <td className="py-1.5 px-2 text-right font-mono">{e.val_loss?.toFixed(4)}</td>
                  <td className="py-1.5 px-2 text-right font-mono">{(e.val_acc * 100).toFixed(1)}%</td>
                  <td className="py-1.5 px-3 text-right text-[var(--muted)]">
                    {e.epoch_time ? `${e.epoch_time}s` : "—"}
                  </td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}


// ── Result Stat Card ───────────────────────────────────────────────────────────
function ResultStat({ label, value, highlight = false }) {
  return (
    <div className="rounded-lg bg-[var(--surface)] p-2.5">
      <p className="text-[10px] text-[var(--muted)] uppercase tracking-wider">{label}</p>
      <p className={`text-sm font-bold ${highlight ? "text-emerald-400" : "text-[var(--foreground)]"}`}>
        {value ?? "—"}
      </p>
    </div>
  );
}
