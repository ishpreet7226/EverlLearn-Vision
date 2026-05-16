"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";

/**
 * PredictionCard — Displays classification output with animated confidence gauge.
 *
 * Props:
 *   label: string                          — predicted class name
 *   confidence: number                     — 0-1 softmax probability
 *   allProbabilities: { class: prob, ... } — per-class probabilities
 */
export default function PredictionCard({ label, confidence, allProbabilities }) {
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

  // Animate the ring on mount
  const [animatedOffset, setAnimatedOffset] = useState(circumference);
  useEffect(() => {
    // Small delay so the spring animation is visible
    const t = setTimeout(() => setAnimatedOffset(offset), 100);
    return () => clearTimeout(t);
  }, [offset]);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.08 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 16 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { type: "spring", damping: 25, stiffness: 300 },
    },
  };

  return (
    <motion.div
      className="space-y-6"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* ── Top: label + confidence ring ───────────────────────────────────── */}
      <motion.div className="glass-card p-6" variants={itemVariants}>
        <div className="flex items-center gap-6">
          {/* Confidence ring */}
          <svg
            className={`confidence-ring shrink-0 ${isHigh ? "high" : ""}`}
            viewBox="0 0 120 120"
          >
            <circle className="track" cx="60" cy="60" r={radius} />
            <motion.circle
              className="fill"
              cx="60"
              cy="60"
              r={radius}
              strokeDasharray={circumference}
              initial={{ strokeDashoffset: circumference }}
              animate={{ strokeDashoffset: animatedOffset }}
              transition={{ duration: 1.2, ease: [0.22, 1, 0.36, 1] }}
            />
            <text
              x="60"
              y="56"
              textAnchor="middle"
              className="fill-[var(--foreground)] font-bold"
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
            <motion.h2
              className={`text-3xl font-bold capitalize ${
                isHigh ? "text-emerald-400" : "text-amber-400"
              }`}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ type: "spring", damping: 20, delay: 0.3 }}
            >
              {label}
            </motion.h2>
            <p className="text-sm text-[var(--muted)] mt-1">
              {isHigh
                ? "High confidence — model is sure"
                : "Moderate confidence — consider a clearer image"}
            </p>
          </div>
        </div>
      </motion.div>

      {/* ── Probability bars ──────────────────────────────────────────────── */}
      <motion.div className="glass-card p-6" variants={itemVariants}>
        <h3 className="text-xs uppercase tracking-widest text-[var(--muted)] mb-4">
          All class probabilities
        </h3>

        <div className="space-y-3">
          {sorted.map(([cls, prob], idx) => {
            const barPct = (prob * 100).toFixed(1);
            const isTop = idx === 0;

            return (
              <motion.div
                key={cls}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{
                  delay: 0.4 + idx * 0.08,
                  type: "spring",
                  damping: 25,
                }}
              >
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
                  <motion.div
                    className={`prob-bar-fill ${isTop ? "top" : ""}`}
                    initial={{ width: 0 }}
                    animate={{ width: `${barPct}%` }}
                    transition={{
                      duration: 0.8,
                      delay: 0.5 + idx * 0.08,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                  />
                </div>
              </motion.div>
            );
          })}
        </div>
      </motion.div>
    </motion.div>
  );
}
