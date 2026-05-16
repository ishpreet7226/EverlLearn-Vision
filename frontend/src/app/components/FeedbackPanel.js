"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useToast } from "./Toast";

const getApiUrl = () => {
  if (typeof window !== "undefined") {
    return `http://${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
};

export default function FeedbackPanel({
  predictedLabel,
  confidence,
  fileName,
  classList = [],
  onFeedbackSent,
}) {
  const [state, setState] = useState("idle");
  const [actualLabel, setActualLabel] = useState("");
  const [error, setError] = useState("");
  const { toast } = useToast();

  const handleCorrect = () => {
    setState("sent");
    toast("Prediction confirmed! ✅", "success");
    if (onFeedbackSent) onFeedbackSent("correct");
  };

  const handleWrong = () => setState("wrong");

  const handleSubmitCorrection = async () => {
    if (!actualLabel || actualLabel === predictedLabel) return;
    setState("sending");
    setError("");
    try {
      const res = await fetch(`${getApiUrl()}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_name: fileName,
          predicted_label: predictedLabel,
          actual_label: actualLabel,
          confidence,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail || `Error ${res.status}`);
      }
      setState("sent");
      toast("Feedback submitted — model will improve! 🧠", "success");
      if (onFeedbackSent) onFeedbackSent("wrong", actualLabel);
    } catch (err) {
      setError(err.message || "Failed to submit feedback");
      toast("Failed to submit feedback", "error");
      setState("wrong");
    }
  };

  if (state === "sent") {
    return (
      <motion.div className="glass-card p-5" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3">
          <motion.div className="w-10 h-10 rounded-xl bg-emerald-500/15 flex items-center justify-center" initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ type: "spring", damping: 15, delay: 0.1 }}>
            <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
          </motion.div>
          <div>
            <p className="text-sm font-medium text-emerald-400">Feedback recorded!</p>
            <p className="text-xs text-[var(--muted)]">{actualLabel ? `Correction: ${predictedLabel} → ${actualLabel}. This will improve the model.` : "Prediction confirmed as correct."}</p>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div className="glass-card p-5" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} layout>
      <h3 className="text-xs uppercase tracking-widest text-[var(--muted)] mb-4">Was this prediction correct?</h3>
      <div className="flex items-center gap-3 mb-4">
        <motion.button id="feedback-correct-button" onClick={handleCorrect} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }} disabled={state !== "idle"} className="flex-1 py-3 px-4 rounded-xl text-sm font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 hover:bg-emerald-500/25 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
          Correct
        </motion.button>
        <motion.button id="feedback-wrong-button" onClick={handleWrong} whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }} disabled={state !== "idle"} className={`flex-1 py-3 px-4 rounded-xl text-sm font-semibold border transition-colors flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed ${state === "wrong" || state === "sending" ? "bg-red-500/20 text-red-400 border-red-500/40" : "bg-red-500/10 text-red-400 border-red-500/20 hover:bg-red-500/20"}`}>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
          Wrong
        </motion.button>
      </div>
      <AnimatePresence>
        {(state === "wrong" || state === "sending") && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} transition={{ type: "spring", damping: 25, stiffness: 300 }} className="overflow-hidden">
            <div className="pt-1 space-y-3">
              <div>
                <label htmlFor="actual-label-select" className="text-xs text-[var(--muted)] mb-1.5 block">Select the correct label</label>
                <select id="actual-label-select" value={actualLabel} onChange={(e) => setActualLabel(e.target.value)} disabled={state === "sending"} className="w-full py-2.5 px-3 rounded-lg text-sm bg-[var(--surface)] border border-[var(--border-color)] text-[var(--foreground)] focus:outline-none focus:ring-2 focus:ring-purple-500/40 disabled:opacity-50">
                  <option value="">— Choose class —</option>
                  {classList.filter((c) => c.toLowerCase() !== predictedLabel.toLowerCase()).map((cls) => (<option key={cls} value={cls}>{cls.charAt(0).toUpperCase() + cls.slice(1)}</option>))}
                </select>
              </div>
              <motion.button id="submit-feedback-button" onClick={handleSubmitCorrection} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} disabled={!actualLabel || actualLabel === predictedLabel || state === "sending"} className="w-full py-2.5 rounded-xl text-sm font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 text-white hover:from-purple-500 hover:to-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-lg shadow-purple-600/20">
                {state === "sending" ? "Sending…" : "Submit Correction"}
              </motion.button>
              {error && <p className="text-xs text-red-400">⚠️ {error}</p>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
