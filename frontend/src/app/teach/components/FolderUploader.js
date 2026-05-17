"use client";

import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const CLASS_COLORS = [
  "text-purple-400 bg-purple-500/10 border-purple-500/20",
  "text-blue-400 bg-blue-500/10 border-blue-500/20",
  "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
  "text-amber-400 bg-amber-500/10 border-amber-500/20",
  "text-rose-400 bg-rose-500/10 border-rose-500/20",
  "text-cyan-400 bg-cyan-500/10 border-cyan-500/20",
  "text-indigo-400 bg-indigo-500/10 border-indigo-500/20",
  "text-pink-400 bg-pink-500/10 border-pink-500/20",
];

const getApiUrl = () => {
  if (typeof window !== "undefined") {
    return `http://${window.location.hostname}:8000`;
  }
  return "http://localhost:8000";
};

export default function FolderUploader({ onDatasetReady, disabled = false }) {
  const inputRef = useRef(null);
  const [state, setState] = useState("idle"); // idle | parsing | uploading | ready | error
  const [parsedClasses, setParsedClasses] = useState(null); // { className: File[] }
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState("");
  const [folderName, setFolderName] = useState("");

  // Parse folder structure from webkitdirectory input
  const handleFolderSelect = (fileList) => {
    if (!fileList || fileList.length === 0) return;

    setState("parsing");
    setError("");

    const files = Array.from(fileList);

    // Group by parent folder (class name)
    const classMap = {};
    let rootName = "";

    for (const file of files) {
      const path = file.webkitRelativePath || file.name;
      const parts = path.split("/");

      if (parts.length < 2) continue; // skip root-level files
      if (!file.type.startsWith("image/")) continue; // skip non-images

      // Root folder name for display
      if (!rootName && parts.length >= 1) rootName = parts[0];

      // Class = parent folder of the file
      const className = parts[parts.length - 2];
      if (className === rootName && parts.length === 2) {
        // Files directly in root — skip (no class folder)
        continue;
      }
      if (className.startsWith(".") || className.startsWith("_")) continue;

      if (!classMap[className]) classMap[className] = [];
      classMap[className].push(file);
    }

    setFolderName(rootName);

    if (Object.keys(classMap).length < 2) {
      setState("error");
      setError(
        `Found ${Object.keys(classMap).length} class folder(s). Need at least 2. ` +
        `Make sure your folder has subfolders like: ${rootName}/cats/*.jpg, ${rootName}/dogs/*.jpg`
      );
      return;
    }

    setParsedClasses(classMap);
    setState("ready");
  };

  // Upload to backend
  const handleUpload = async () => {
    if (!parsedClasses) return;

    setState("uploading");
    setError("");

    try {
      const formData = new FormData();
      for (const [className, files] of Object.entries(parsedClasses)) {
        for (const file of files) {
          formData.append("files", file);
          formData.append("paths", file.webkitRelativePath || `${className}/${file.name}`);
        }
      }

      const res = await fetch(`${getApiUrl()}/teach/upload-folder`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const d = await res.json().catch(() => null);
        throw new Error(d?.detail || `Upload failed: ${res.status}`);
      }

      const data = await res.json();
      setUploadResult(data);
      if (onDatasetReady) onDatasetReady(data);
    } catch (err) {
      setState("error");
      setError(err.message || "Upload failed.");
    }
  };

  const reset = () => {
    setState("idle");
    setParsedClasses(null);
    setUploadResult(null);
    setError("");
    setFolderName("");
  };

  const totalImages = parsedClasses
    ? Object.values(parsedClasses).reduce((sum, files) => sum + files.length, 0)
    : 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-[var(--foreground)]">Dataset</h3>
          <p className="text-xs text-[var(--muted)]">
            Upload a folder with class subfolders
          </p>
        </div>
        {(state === "ready" || state === "uploading") && (
          <button
            onClick={reset}
            className="text-xs text-[var(--muted)] hover:text-red-400 transition-colors"
          >
            Reset
          </button>
        )}
      </div>

      <AnimatePresence mode="wait">
        {/* Idle — folder picker */}
        {state === "idle" && (
          <motion.div
            key="picker"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <input
              ref={inputRef}
              type="file"
              /* @ts-ignore */
              webkitdirectory="true"
              directory=""
              multiple
              className="hidden"
              onChange={(e) => {
                handleFolderSelect(e.target.files);
                e.target.value = "";
              }}
              disabled={disabled}
            />
            <motion.button
              whileHover={disabled ? {} : { scale: 1.01 }}
              whileTap={disabled ? {} : { scale: 0.99 }}
              onClick={() => !disabled && inputRef.current?.click()}
              disabled={disabled}
              className="w-full rounded-xl border-2 border-dashed border-[var(--border-color)]
                         hover:border-purple-500/50 hover:bg-purple-500/5
                         disabled:opacity-40 disabled:cursor-not-allowed
                         transition-all py-10 px-6 text-center"
            >
              <div className="flex flex-col items-center gap-3">
                <div className="w-14 h-14 rounded-2xl bg-purple-500/10 flex items-center justify-center">
                  <svg className="w-7 h-7 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-medium text-[var(--foreground)]">
                    Select Dataset Folder
                  </p>
                  <p className="text-xs text-[var(--muted)] mt-1">
                    Subfolders become class names automatically
                  </p>
                </div>
                <div className="text-[10px] text-[var(--muted)] bg-[var(--surface)] rounded-lg px-3 py-1.5 mt-1">
                  dataset/ → cats/ + dogs/ → auto train/val split
                </div>
              </div>
            </motion.button>
          </motion.div>
        )}

        {/* Parsing */}
        {state === "parsing" && (
          <motion.div
            key="parsing"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="glass-card p-6 text-center"
          >
            <svg className="w-8 h-8 mx-auto text-purple-400 animate-spin-slow" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-20" />
              <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
            </svg>
            <p className="text-sm text-[var(--muted)] mt-3">Scanning folder…</p>
          </motion.div>
        )}

        {/* Ready — show parsed classes */}
        {(state === "ready" || state === "uploading") && parsedClasses && (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-3"
          >
            {/* Folder name badge */}
            <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
              <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
              </svg>
              <span className="font-medium text-[var(--foreground)]">{folderName}</span>
              <span>· {totalImages} images · {Object.keys(parsedClasses).length} classes</span>
            </div>

            {/* Class list */}
            <div className="space-y-2">
              {Object.entries(parsedClasses).map(([name, files], i) => {
                const color = CLASS_COLORS[i % CLASS_COLORS.length];
                // Show up to 6 thumbnails
                const previews = files.slice(0, 6);
                return (
                  <motion.div
                    key={name}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    className="glass-card overflow-hidden"
                  >
                    <div className={`flex items-center gap-2 px-3 py-2 border-b border-[var(--border-color)]`}>
                      <span className={`text-xs font-semibold px-2 py-0.5 rounded-full border capitalize ${color}`}>
                        {name}
                      </span>
                      <span className="text-xs text-[var(--muted)]">
                        {files.length} image{files.length !== 1 ? "s" : ""}
                      </span>
                      <span className="text-[10px] text-[var(--muted)] ml-auto">
                        ~{Math.round(files.length * 0.8)} train · ~{Math.max(1, Math.round(files.length * 0.2))} val
                      </span>
                    </div>
                    {/* Thumbnail strip */}
                    <div className="flex gap-1 p-2 overflow-hidden">
                      {previews.map((f, j) => (
                        <div key={j} className="w-10 h-10 rounded-md overflow-hidden bg-[var(--surface)] shrink-0">
                          <img
                            src={URL.createObjectURL(f)}
                            alt={f.name}
                            className="w-full h-full object-cover"
                          />
                        </div>
                      ))}
                      {files.length > 6 && (
                        <div className="w-10 h-10 rounded-md bg-[var(--surface)] flex items-center justify-center shrink-0
                                        text-[10px] text-[var(--muted)]">
                          +{files.length - 6}
                        </div>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </div>

            {/* Upload & split button */}
            <motion.button
              whileHover={state !== "uploading" ? { scale: 1.02 } : {}}
              whileTap={state !== "uploading" ? { scale: 0.98 } : {}}
              onClick={handleUpload}
              disabled={state === "uploading"}
              className="w-full py-3 rounded-xl font-semibold text-sm text-white
                         bg-gradient-to-r from-purple-600 to-indigo-600
                         hover:from-purple-500 hover:to-indigo-500
                         disabled:opacity-60 transition-all
                         shadow-lg shadow-purple-600/20 flex items-center justify-center gap-2"
            >
              {state === "uploading" ? (
                <>
                  <svg className="animate-spin-slow w-4 h-4" viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-20" />
                    <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                  </svg>
                  Uploading & splitting…
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  Upload & Create Train/Val Split
                </>
              )}
            </motion.button>

            <p className="text-[10px] text-center text-[var(--muted)]">
              80% of images → training · 20% → validation (for testing)
            </p>
          </motion.div>
        )}

        {/* Error */}
        {state === "error" && (
          <motion.div
            key="error"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="space-y-3"
          >
            <div className="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-3">
              ⚠️ {error}
            </div>
            <button
              onClick={reset}
              className="text-xs text-purple-400 hover:text-purple-300 transition-colors"
            >
              ← Try again
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Upload result */}
      <AnimatePresence>
        {uploadResult && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card p-3 flex items-center gap-3"
          >
            <div className="w-8 h-8 rounded-lg bg-emerald-500/15 flex items-center justify-center shrink-0">
              <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-emerald-400">Dataset ready!</p>
              <p className="text-[10px] text-[var(--muted)]">
                {uploadResult.total_images} images · {uploadResult.classes.length} classes · Train/val split complete
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
