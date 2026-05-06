"use client";

import { useState } from "react";

const API_URL = "http://localhost:8000";

/**
 * FeedbackForm — Correction UI shown after a prediction.
 *
 * Props:
 *   predictedLabel: string         — the model's prediction
 *   confidence: number             — 0-1 softmax probability
 *   fileName: string               — original filename of the uploaded image
 *   classList: string[]             — available class names for the dropdown
 *   onFeedbackSent: () => void     — callback after successful submission
 */
export default function FeedbackForm({
  predictedLabel,
  confidence,
  fileName,
  classList = [],
  onFeedbackSent,
}) {
  const [actualLabel, setActualLabel] = useState("");
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    if (!actualLabel || actualLabel === predictedLabel) return;

    setSending(true);
    setError("");

    try {
      const res = await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_name: fileName,
          predicted_label: predictedLabel,
          actual_label: actualLabel,
          confidence: confidence,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => null);
        throw new Error(data?.detail || `Error ${res.status}`);
      }

      setSent(true);
      if (onFeedbackSent) onFeedbackSent();
    } catch (err) {
      setError(err.message || "Failed to submit feedback");
    } finally {
      setSending(false);
    }
  };

  // Already submitted — show confirmation
  if (sent) {
    return (
      <div className="glass-card p-5 animate-fade-in-up">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/15 flex items-center justify-center">
            <svg
              className="w-5 h-5 text-emerald-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-emerald-400">
              Feedback submitted!
            </p>
            <p className="text-xs text-[var(--muted)]">
              Correction: {predictedLabel} → {actualLabel}. This will improve
              the model in the next retraining cycle.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card p-5 animate-fade-in-up">
      <h3 className="text-xs uppercase tracking-widest text-[var(--muted)] mb-3">
        Was this prediction wrong?
      </h3>

      <div className="flex items-end gap-3">
        {/* Class dropdown */}
        <div className="flex-1">
          <label
            htmlFor="actual-label-select"
            className="text-xs text-[var(--muted)] mb-1 block"
          >
            Select the correct label
          </label>
          <select
            id="actual-label-select"
            value={actualLabel}
            onChange={(e) => setActualLabel(e.target.value)}
            disabled={sending}
            className="w-full py-2.5 px-3 rounded-lg text-sm
                       bg-[var(--surface)] border border-[var(--border-color)]
                       text-[var(--foreground)]
                       focus:outline-none focus:ring-2 focus:ring-purple-500/40
                       disabled:opacity-50"
          >
            <option value="">— Choose class —</option>
            {classList
              .filter((c) => c.toLowerCase() !== predictedLabel.toLowerCase())
              .map((cls) => (
                <option key={cls} value={cls}>
                  {cls.charAt(0).toUpperCase() + cls.slice(1)}
                </option>
              ))}
          </select>
        </div>

        {/* Submit button */}
        <button
          id="submit-feedback-button"
          onClick={handleSubmit}
          disabled={!actualLabel || actualLabel === predictedLabel || sending}
          className="py-2.5 px-5 rounded-lg text-sm font-medium
                     bg-amber-500/15 text-amber-400 border border-amber-500/30
                     hover:bg-amber-500/25 
                     disabled:opacity-40 disabled:cursor-not-allowed
                     transition-all active:scale-[0.98]"
        >
          {sending ? "Sending..." : "Submit Correction"}
        </button>
      </div>

      {/* Error */}
      {error && (
        <p className="text-xs text-red-400 mt-2">⚠️ {error}</p>
      )}
    </div>
  );
}
