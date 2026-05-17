"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import DatasetBuilder, { DEFAULT_CLASSES } from "./teach/components/DatasetBuilder";
import FolderUploader from "./teach/components/FolderUploader";
import TrainingPanel from "./teach/components/TrainingPanel";
import PredictionPanel from "./teach/components/PredictionPanel";
import DarkModeToggle from "./components/DarkModeToggle";

const MODES = [
  {
    id: "manual",
    label: "Manual",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
    desc: "Create classes & upload images one by one",
  },
  {
    id: "folder",
    label: "Upload Folder",
    icon: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
      </svg>
    ),
    desc: "Upload a dataset folder — auto-detect classes & split",
  },
];

export default function Home() {
  const [mode, setMode] = useState("manual"); // manual | folder
  const [classes, setClasses] = useState(
    DEFAULT_CLASSES.map((c) => ({ ...c, images: [] }))
  );
  const [isModelReady, setIsModelReady] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [folderDatasetReady, setFolderDatasetReady] = useState(false);
  const [folderClasses, setFolderClasses] = useState([]);

  const handleFolderDatasetReady = (data) => {
    setFolderDatasetReady(true);
    setFolderClasses(data.classes || []);
  };

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
            <motion.div
              className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600
                         flex items-center justify-center text-white font-bold text-sm
                         shadow-lg shadow-purple-500/25"
              whileHover={{ rotate: 12, scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              EL
            </motion.div>
            <div>
              <h1 className="text-base font-semibold tracking-tight">EverLearn ML</h1>
              <p className="text-xs text-[var(--muted)]">Self-Improving Image Classifier</p>
            </div>
          </div>
          <DarkModeToggle />
        </div>
      </header>

      {/* Hero */}
      <motion.div
        className="text-center py-6 px-6"
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
        </p>

        {/* Mode switcher */}
        <div className="flex justify-center mt-5">
          <div className="inline-flex rounded-xl bg-[var(--surface)] border border-[var(--border-color)] p-1 gap-1">
            {MODES.map((m) => (
              <motion.button
                key={m.id}
                onClick={() => {
                  if (!isTraining) {
                    setMode(m.id);
                    setIsModelReady(false);
                    setFolderDatasetReady(false);
                  }
                }}
                whileHover={!isTraining ? { scale: 1.02 } : {}}
                whileTap={!isTraining ? { scale: 0.98 } : {}}
                disabled={isTraining}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium
                            transition-all disabled:cursor-not-allowed
                  ${mode === m.id
                    ? "bg-purple-500/15 text-purple-400 shadow-sm shadow-purple-500/10"
                    : "text-[var(--muted)] hover:text-[var(--foreground)]"
                  }`}
              >
                {m.icon}
                <span className="hidden sm:inline">{m.label}</span>
                <span className="sm:hidden">{m.label.split(" ")[0]}</span>
              </motion.button>
            ))}
          </div>
        </div>
        <p className="text-[10px] text-[var(--muted)] mt-2">
          {MODES.find((m) => m.id === mode)?.desc}
        </p>
      </motion.div>

      {/* Three-panel layout */}
      <div className="flex-1 px-4 pb-8">
        <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-[1fr_340px_1fr] gap-5">
          {/* Left: Dataset Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1, duration: 0.5 }}
            className="min-w-0"
          >
            <AnimatePresence mode="wait">
              {mode === "manual" ? (
                <motion.div
                  key="manual"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                >
                  <DatasetBuilder
                    classes={classes}
                    setClasses={setClasses}
                    disabled={isTraining}
                  />
                </motion.div>
              ) : (
                <motion.div
                  key="folder"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                >
                  <FolderUploader
                    onDatasetReady={handleFolderDatasetReady}
                    disabled={isTraining}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>

          {/* Center: Training Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="lg:sticky lg:top-20 lg:self-start"
          >
            <TrainingPanel
              classes={mode === "manual" ? classes : []}
              mode={mode}
              folderDatasetReady={folderDatasetReady}
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

          {/* Right: Prediction Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="min-w-0"
          >
            <PredictionPanel isModelReady={isModelReady} mode={mode} />
          </motion.div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-[var(--border-color)] backdrop-blur-md bg-[var(--background)]/60">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-center text-xs text-[var(--muted)]">
          <p>EverLearn ML · PyTorch + FastAPI</p>
        </div>
      </footer>
    </main>
  );
}
