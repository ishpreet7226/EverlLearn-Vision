"use client";

/**
 * PredictionResult — Displays the classification output.
 *
 * Props:
 *   label: string                          — predicted class name
 *   confidence: number                     — 0-1 softmax probability
 *   allProbabilities: { class: prob, ... } — per-class probabilities
 */
export default function PredictionResult({
  label,
  confidence,
  allProbabilities,
}) {
  const pct = (confidence * 100).toFixed(1);
  const isHigh = confidence >= 0.75;

  // Sort classes by probability (descending)
  const sorted = Object.entries(allProbabilities).sort(
    ([, a], [, b]) => b - a
  );

  // SVG confidence ring calculations
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - confidence);

  return (
    <div className="space-y-6 animate-fade-in-up">
      {/* ── Top: label + confidence ring ────────────────────────────────────── */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-6">
          {/* Confidence ring */}
          <svg
            className={`confidence-ring shrink-0 ${isHigh ? "high" : ""}`}
            viewBox="0 0 120 120"
          >
            <circle className="track" cx="60" cy="60" r={radius} />
            <circle
              className="fill"
              cx="60"
              cy="60"
              r={radius}
              strokeDasharray={circumference}
              strokeDashoffset={offset}
            />
            <text
              x="60"
              y="56"
              textAnchor="middle"
              className="fill-[var(--foreground)] text-xl font-bold"
              style={{ fontSize: "20px" }}
            >
              {pct}%
            </text>
            <text
              x="60"
              y="74"
              textAnchor="middle"
              className="fill-[var(--muted)]"
              style={{ fontSize: "10px" }}
            >
              confidence
            </text>
          </svg>

          {/* Label */}
          <div className="flex-1 min-w-0">
            <p className="text-xs uppercase tracking-widest text-[var(--muted)] mb-1">
              Prediction
            </p>
            <h2
              className={`text-3xl font-bold capitalize ${
                isHigh ? "text-emerald-400" : "text-amber-400"
              }`}
            >
              {label}
            </h2>
            <p className="text-sm text-[var(--muted)] mt-1">
              {isHigh
                ? "High confidence — model is sure"
                : "Moderate confidence — consider a clearer image"}
            </p>
          </div>
        </div>
      </div>

      {/* ── Probability bars ────────────────────────────────────────────────── */}
      <div className="glass-card p-6">
        <h3 className="text-xs uppercase tracking-widest text-[var(--muted)] mb-4">
          All class probabilities
        </h3>

        <div className="space-y-3 stagger">
          {sorted.map(([cls, prob], idx) => {
            const barPct = (prob * 100).toFixed(1);
            const isTop = idx === 0;

            return (
              <div key={cls} className="animate-fade-in-up">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium capitalize flex items-center gap-2">
                    {cls}
                    {isTop && (
                      <span className="text-[10px] bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full">
                        predicted
                      </span>
                    )}
                  </span>
                  <span className="text-sm tabular-nums text-[var(--muted)]">
                    {barPct}%
                  </span>
                </div>
                <div className="prob-bar-track">
                  <div
                    className={`prob-bar-fill ${isTop ? "top" : ""}`}
                    style={{ width: `${barPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
