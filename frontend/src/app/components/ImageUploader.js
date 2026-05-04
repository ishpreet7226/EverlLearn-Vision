"use client";

import { useCallback, useRef, useState } from "react";

const ACCEPTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/bmp",
  "image/webp",
];

/**
 * ImageUploader — Drag-and-drop zone with click-to-browse fallback.
 *
 * Props:
 *   onFileSelect(file: File)  — called when a valid image is chosen
 *   disabled: boolean         — disables interaction during prediction
 */
export default function ImageUploader({ onFileSelect, disabled = false }) {
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

      // Client-side type validation
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError(
          `Unsupported format: ${file.type || "unknown"}. Use JPEG, PNG, BMP, or WEBP.`
        );
        return;
      }

      // Size guard (15 MB)
      if (file.size > 15 * 1024 * 1024) {
        setError("File is too large. Maximum size is 15 MB.");
        return;
      }

      setFileName(file.name);
      setFileSize((file.size / 1024).toFixed(1) + " KB");

      // Generate preview
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        setDimensions(`${img.naturalWidth} × ${img.naturalHeight}`);
        URL.revokeObjectURL(url); // free memory after measuring
      };
      img.src = url;

      setPreview(URL.createObjectURL(file));
      onFileSelect(file);
    },
    [onFileSelect]
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
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={onBrowse}
        className={`upload-zone flex flex-col items-center justify-center gap-3 p-10 text-center transition-all ${
          dragOver ? "drag-over" : ""
        } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        {/* Icon */}
        <div className="w-14 h-14 rounded-2xl bg-[rgba(124,92,252,0.1)] flex items-center justify-center">
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
        </div>

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
      </div>

      {/* Error */}
      {error && (
        <div className="text-sm text-red-400 bg-red-400/10 rounded-lg px-4 py-2 animate-fade-in-up">
          ⚠️ {error}
        </div>
      )}

      {/* Preview */}
      {preview && (
        <div className="glass-card p-4 animate-fade-in-up">
          <div className="flex items-start gap-4">
            <img
              src={preview}
              alt="Upload preview"
              className="w-24 h-24 rounded-lg object-cover ring-1 ring-[var(--border)]"
            />
            <div className="flex-1 min-w-0 space-y-1">
              <p className="text-sm font-medium truncate">{fileName}</p>
              <p className="text-xs text-[var(--muted)]">{fileSize}</p>
              {dimensions && (
                <p className="text-xs text-[var(--muted)]">{dimensions} px</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
