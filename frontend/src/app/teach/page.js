"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import DatasetBuilder, { DEFAULT_CLASSES } from "./components/DatasetBuilder";
import TrainingPanel from "./components/TrainingPanel";
import PredictionPanel from "./components/PredictionPanel";
import DarkModeToggle from "../components/DarkModeToggle";

export default function TeachPage() {
  const [classes, setClasses] = useState(
    DEFAULT_CLASSES.map((c) => ({ ...c, images: [] }))
  );
  const [isModelReady, setIsModelReady] = useState(false);
  const [isTraining, setIsTraining] = useState(false);

  return (
    <main className="min-h-screen flex flex-col">
      {/* Background gradient blobs */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-[500px] h-[500px] rounded-full bg-purple-600/10 blur-[128px]" />
        <div className="absolute top-1/3 -right-40 w-[400px] h-[400px] rounded-full bg-blue-600/8 blur-[128px]" />
        <div className="absolute -bottom-20 left-1/3 w-[350px] h-[350px] rounded-full bg-indigo-600/8 blur-[128px]" />
      </div>

      {/* Header */}
      <header className="border-b border-[var(--border-color)] backdrop-blur-md bg-[var(--background)]/60 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/">
              <motion.div
                className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600
                           flex items-center justify-center text-white font-bold text-sm
                           shadow-lg shadow-purple-500/25 cursor-pointer"
                whileHover={{ rotate: 12, scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
              >
                EL
              </motion.div>
            </Link>
            <div>
              <h1 className="text-base font-semibold tracking-tight">
                EverLearn ML
                <span className="text-purple-400 ml-1.5 text-xs font-normal">/ Teach</span>
              </h1>
              <p className="text-xs text-[var(--muted)]">Build your own image classifier</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="text-xs text-[var(--muted)] hover:text-[var(--foreground)] transition-colors"
            >
              ← Back to Predict
            </Link>
            <DarkModeToggle />
          </div>
        </div>
      </header>

      {/* Hero */}
      <motion.div
        className="text-center py-8 px-6"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">
          Teach your{" "}
          <span className="bg-gradient-to-r from-purple-400 via-indigo-400 to-blue-400 bg-clip-text text-transparent">
            own model
          </span>
        </h2>
        <p className="text-sm text-[var(--muted)] mt-2 max-w-lg mx-auto">
          Create classes, upload images, train a model, and test predictions — all from the browser.
          Just like Teachable Machine.
        </p>
      </motion.div>

      {/* Three-panel layout */}
      <div className="flex-1 px-4 pb-8">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-[1fr_320px_1fr] gap-5">
          {/* ── Left: Dataset Builder ──────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1, duration: 0.5 }}
            className="min-w-0"
          >
            <DatasetBuilder
              classes={classes}
              setClasses={setClasses}
              disabled={isTraining}
            />
          </motion.div>

          {/* ── Center: Training Panel ─────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="lg:sticky lg:top-20 lg:self-start"
          >
            <TrainingPanel
              classes={classes}
              onTrainingStart={() => {
                setIsTraining(true);
                setIsModelReady(false);
              }}
              onTrainingComplete={() => {
                setIsTraining(false);
                setIsModelReady(true);
              }}
            />
          </motion.div>

          {/* ── Right: Prediction Panel ────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="min-w-0"
          >
            <PredictionPanel isModelReady={isModelReady} />
          </motion.div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-[var(--border-color)] backdrop-blur-md bg-[var(--background)]/60">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between text-xs text-[var(--muted)]">
          <p>EverLearn ML · Teachable Machine · PyTorch + FastAPI</p>
          <p>
            <Link href="/" className="hover:text-purple-400 transition-colors">
              Switch to Predict mode →
            </Link>
          </p>
        </div>
      </footer>
    </main>
  );
}
