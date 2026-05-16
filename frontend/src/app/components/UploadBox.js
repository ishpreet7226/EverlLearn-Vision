"use client";

import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const ACCEPTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/bmp",
  "image/webp",
];

/**
 * UploadBox — Drag-and-drop upload zone with animated preview.
 *
 * Props:
 *   onFileSelect(file: File)  — called when a valid image is chosen
 *   disabled: boolean         — disables interaction during prediction
 */
export default function UploadBox({ onFileSelect, disabled = false }) {
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState("");
  const [fileSize, setFileSize] = useState("");
  const [dimensions, setDimensions] = useState("");
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const handleFile = useCallback(
    (file) => {
      setError("");

      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError(
          `Unsupported format: ${file.type || "unknown"}. Use JPEG, PNG, BMP, or WEBP.`
        );
        return;
      }

      if (file.size > 15 * 1024 * 1024) {
        setError("File is too large. Maximum size is 15 MB.");
        return;
      }

      setFileName(file.name);
      setFileSize((file.size / 1024).toFixed(1) + " KB");

      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        setDimensions(`${img.naturalWidth} × ${img.naturalHeight}`);
      };
      img.src = url;

      // Clean up old preview
      if (preview) URL.revokeObjectURL(preview);
      setPreview(url);
      onFileSelect(file);
    },
    [onFileSelect, preview]
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer?.files?.[0];
      if (file) handleFile(file);
    },
    [disabled, handleFile]
  );

  const onDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setDragOver(true);
  };

  const onDragLeave = () => setDragOver(false);

  const onBrowse = () => {
    if (!disabled) inputRef.current?.click();
  };

  const onInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <motion.div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={onBrowse}
        whileHover={disabled ? {} : { scale: 1.01, borderColor: "rgba(124,92,252,0.5)" }}
        whileTap={disabled ? {} : { scale: 0.99 }}
        className={`upload-zone flex flex-col items-center justify-center gap-3 p-10 text-center ${
          dragOver ? "drag-over" : ""
        } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        {/* Animated icon */}
        <motion.div
          className="w-14 h-14 rounded-2xl bg-[rgba(124,92,252,0.1)] flex items-center justify-center"
          animate={dragOver ? { y: [0, -8, 0] } : { y: 0 }}
          transition={{ repeat: dragOver ? Infinity : 0, duration: 0.6 }}
        >
          <svg
            className="w-7 h-7 text-[var(--accent-light)]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16"
            />
          </svg>
        </motion.div>

        <p className="text-sm text-[var(--muted)]">
          <span className="text-[var(--accent-light)] font-medium">
            Click to browse
          </span>{" "}
          or drag and drop an image
        </p>
        <p className="text-xs text-[var(--muted)]">
          JPEG, PNG, BMP, WEBP • Max 15 MB
        </p>

        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(",")}
          onChange={onInputChange}
          className="hidden"
          id="image-upload-input"
        />
      </motion.div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="text-sm text-red-400 bg-red-400/10 border border-red-400/20 rounded-xl px-4 py-2"
          >
            ⚠️ {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Preview */}
      <AnimatePresence>
        {preview && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="glass-card p-4"
          >
            <div className="flex items-start gap-4">
              <motion.img
                src={preview}
                alt="Upload preview"
                className="w-24 h-24 rounded-lg object-cover ring-1 ring-[var(--border-color)]"
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", damping: 20 }}
              />
              <div className="flex-1 min-w-0 space-y-1">
                <motion.p
                  className="text-sm font-medium truncate"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.1 }}
                >
                  {fileName}
                </motion.p>
                <motion.p
                  className="text-xs text-[var(--muted)]"
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.15 }}
                >
                  {fileSize}
                </motion.p>
                {dimensions && (
                  <motion.p
                    className="text-xs text-[var(--muted)]"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                  >
                    {dimensions} px
                  </motion.p>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
