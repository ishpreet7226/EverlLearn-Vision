"use client";

import { useRef, useState } from "react";
import { motion } from "framer-motion";

/**
 * Reusable multi-file image uploader with drag-and-drop.
 * Used by ClassCard (dataset building) and PredictionPanel (single image).
 */
export default function ImageUploader({
  onFilesSelect,
  multiple = true,
  disabled = false,
  compact = false,
  label = "Upload Images",
  sublabel = "or drag & drop here",
}) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (fileList) => {
    if (!fileList || disabled) return;
    const imgs = Array.from(fileList).filter((f) =>
      f.type.startsWith("image/")
    );
    if (imgs.length > 0 && onFilesSelect) {
      onFilesSelect(imgs);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  return (
    <motion.div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      whileHover={disabled ? {} : { scale: 1.01 }}
      whileTap={disabled ? {} : { scale: 0.99 }}
      className={`
        relative rounded-xl border-2 border-dashed cursor-pointer
        transition-all duration-200 text-center
        ${compact ? "px-3 py-3" : "px-5 py-6"}
        ${disabled
          ? "opacity-40 cursor-not-allowed border-[var(--border-color)]"
          : dragOver
            ? "border-purple-500 bg-purple-500/10 shadow-lg shadow-purple-500/10"
            : "border-[var(--border-color)] hover:border-purple-500/50 hover:bg-purple-500/5"
        }
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple={multiple}
        disabled={disabled}
        className="hidden"
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {/* Upload icon */}
      <div className="flex flex-col items-center gap-1.5">
        <div className={`rounded-lg bg-purple-500/10 flex items-center justify-center ${compact ? "w-8 h-8" : "w-10 h-10"}`}>
          <svg
            className={`text-purple-400 ${compact ? "w-4 h-4" : "w-5 h-5"}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
            />
          </svg>
        </div>
        <p className={`font-medium text-[var(--foreground)] ${compact ? "text-xs" : "text-sm"}`}>
          {label}
        </p>
        <p className={`text-[var(--muted)] ${compact ? "text-[10px]" : "text-xs"}`}>
          {sublabel}
        </p>
      </div>
    </motion.div>
  );
}
