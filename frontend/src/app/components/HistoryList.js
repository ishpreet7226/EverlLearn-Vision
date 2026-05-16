"use client";

import { motion } from "framer-motion";

/**
 * HistoryList — Displays previous predictions as animated cards.
 *
 * Props:
 *   history: Array<{ id, imageUrl, label, confidence, timestamp }>
 */
export default function HistoryList({ history = [] }) {
  if (history.length === 0) {
    return (
      <motion.div
        className="glass-card p-8 text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="w-16 h-16 rounded-2xl bg-[var(--surface-hover)] flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-[var(--muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <p className="text-sm text-[var(--muted)]">No predictions yet</p>
        <p className="text-xs text-[var(--muted)] mt-1">Upload an image to get started</p>
      </motion.div>
    );
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.08 } },
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 20, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1, transition: { type: "spring", damping: 25 } },
  };

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible">
      <h3 className="text-xs uppercase tracking-widest text-[var(--muted)] mb-4">
        Prediction History
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {history.map((item) => (
          <motion.div
            key={item.id}
            variants={cardVariants}
            whileHover={{ y: -4, scale: 1.02 }}
            className="glass-card p-4 cursor-default"
          >
            <div className="flex items-start gap-3">
              {item.imageUrl && (
                <img
                  src={item.imageUrl}
                  alt={item.label}
                  className="w-14 h-14 rounded-lg object-cover ring-1 ring-[var(--border-color)] shrink-0"
                />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold capitalize truncate">{item.label}</p>
                <div className="flex items-center gap-2 mt-1">
                  <div className="flex-1 h-1.5 rounded-full bg-[var(--surface-hover)] overflow-hidden">
                    <motion.div
                      className={`h-full rounded-full ${item.confidence >= 0.75 ? "bg-emerald-400" : "bg-amber-400"}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${(item.confidence * 100).toFixed(0)}%` }}
                      transition={{ duration: 0.6, delay: 0.2 }}
                    />
                  </div>
                  <span className="text-xs tabular-nums text-[var(--muted)]">
                    {(item.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <p className="text-[10px] text-[var(--muted)] mt-1.5">{item.timestamp}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
