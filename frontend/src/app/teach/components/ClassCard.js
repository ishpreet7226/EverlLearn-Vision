"use client";

import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ImageUploader from "./ImageUploader";

// Per-class color palette for visual distinction
const CLASS_COLORS = [
  { bg: "bg-purple-500/12", border: "border-purple-500/25", accent: "text-purple-400", dot: "bg-purple-500" },
  { bg: "bg-blue-500/12", border: "border-blue-500/25", accent: "text-blue-400", dot: "bg-blue-500" },
  { bg: "bg-emerald-500/12", border: "border-emerald-500/25", accent: "text-emerald-400", dot: "bg-emerald-500" },
  { bg: "bg-amber-500/12", border: "border-amber-500/25", accent: "text-amber-400", dot: "bg-amber-500" },
  { bg: "bg-rose-500/12", border: "border-rose-500/25", accent: "text-rose-400", dot: "bg-rose-500" },
  { bg: "bg-cyan-500/12", border: "border-cyan-500/25", accent: "text-cyan-400", dot: "bg-cyan-500" },
  { bg: "bg-indigo-500/12", border: "border-indigo-500/25", accent: "text-indigo-400", dot: "bg-indigo-500" },
  { bg: "bg-pink-500/12", border: "border-pink-500/25", accent: "text-pink-400", dot: "bg-pink-500" },
];

export default function ClassCard({
  cls,
  index,
  onRename,
  onAddImages,
  onRemoveImage,
  onDelete,
  canDelete = true,
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(cls.name);
  const inputRef = useRef(null);
  const colors = CLASS_COLORS[index % CLASS_COLORS.length];

  const handleRenameConfirm = () => {
    const trimmed = editName.trim();
    if (trimmed && trimmed !== cls.name) {
      onRename(cls.id, trimmed);
    } else {
      setEditName(cls.name);
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleRenameConfirm();
    if (e.key === "Escape") {
      setEditName(cls.name);
      setIsEditing(false);
    }
  };

  const previews = cls.images.map((img) => ({
    id: img.id,
    url: img.previewUrl,
    name: img.file.name,
  }));

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9, y: -10 }}
      transition={{ type: "spring", damping: 25, stiffness: 300 }}
      className={`glass-card overflow-hidden`}
    >
      {/* Header */}
      <div className={`flex items-center gap-3 px-4 py-3 ${colors.bg} border-b ${colors.border}`}>
        <div className={`w-3 h-3 rounded-full ${colors.dot} shrink-0`} />

        {isEditing ? (
          <input
            ref={inputRef}
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            onBlur={handleRenameConfirm}
            onKeyDown={handleKeyDown}
            autoFocus
            className="flex-1 bg-transparent border-b border-[var(--foreground)] text-sm font-semibold
                       outline-none text-[var(--foreground)] py-0.5"
          />
        ) : (
          <button
            onClick={() => setIsEditing(true)}
            className="flex-1 text-left text-sm font-semibold text-[var(--foreground)]
                       hover:text-purple-400 transition-colors truncate"
            title="Click to rename"
          >
            {cls.name}
          </button>
        )}

        {/* Image count badge */}
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${colors.bg} ${colors.accent}`}>
          {cls.images.length} img{cls.images.length !== 1 ? "s" : ""}
        </span>

        {/* Edit button */}
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => {
            setIsEditing(true);
            setTimeout(() => inputRef.current?.focus(), 50);
          }}
          className="p-1.5 rounded-lg hover:bg-[var(--surface-hover)] transition-colors text-[var(--muted)]"
          title="Rename class"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
        </motion.button>

        {/* Delete button */}
        {canDelete && (
          <motion.button
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            onClick={() => onDelete(cls.id)}
            className="p-1.5 rounded-lg hover:bg-red-500/15 transition-colors text-[var(--muted)] hover:text-red-400"
            title="Remove class"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </motion.button>
        )}
      </div>

      {/* Body */}
      <div className="p-4 space-y-3">
        {/* Image thumbnails */}
        {previews.length > 0 && (
          <div className="grid grid-cols-5 gap-1.5">
            <AnimatePresence>
              {previews.map((img) => (
                <motion.div
                  key={img.id}
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.5 }}
                  className="relative group aspect-square rounded-lg overflow-hidden bg-[var(--surface)]"
                >
                  <img
                    src={img.url}
                    alt={img.name}
                    className="w-full h-full object-cover"
                  />
                  <motion.button
                    initial={{ opacity: 0 }}
                    whileHover={{ opacity: 1 }}
                    onClick={() => onRemoveImage(cls.id, img.id)}
                    className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100
                               flex items-center justify-center transition-opacity"
                  >
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </motion.button>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}

        {/* Upload actions */}
        <div className="flex gap-2">
          <div className="flex-1">
            <ImageUploader
              onFilesSelect={(files) => onAddImages(cls.id, files)}
              multiple={true}
              compact={true}
              label="Upload"
              sublabel="drag & drop"
            />
          </div>
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            disabled
            className="px-3 py-3 rounded-xl border-2 border-dashed border-[var(--border-color)]
                       text-[var(--muted)] text-xs flex flex-col items-center justify-center gap-1
                       opacity-40 cursor-not-allowed"
            title="Webcam capture (coming soon)"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span>Webcam</span>
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
}
