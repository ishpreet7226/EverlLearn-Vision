"use client";

import { AnimatePresence, motion } from "framer-motion";
import ClassCard from "./ClassCard";

let nextClassId = 3;
let nextImgId = 1;

export function createClass(name) {
  return { id: nextClassId++, name, images: [] };
}

export function createImageEntry(file) {
  return {
    id: nextImgId++,
    file,
    previewUrl: URL.createObjectURL(file),
  };
}

export const DEFAULT_CLASSES = [
  { id: 1, name: "Class 1", images: [] },
  { id: 2, name: "Class 2", images: [] },
];

export default function DatasetBuilder({ classes, setClasses, disabled = false }) {
  const addClass = () => {
    const num = classes.length + 1;
    setClasses((prev) => [...prev, createClass(`Class ${num}`)]);
  };

  const removeClass = (id) => {
    setClasses((prev) => {
      const cls = prev.find((c) => c.id === id);
      if (cls) {
        cls.images.forEach((img) => URL.revokeObjectURL(img.previewUrl));
      }
      return prev.filter((c) => c.id !== id);
    });
  };

  const renameClass = (id, newName) => {
    setClasses((prev) =>
      prev.map((c) => (c.id === id ? { ...c, name: newName } : c))
    );
  };

  const addImages = (classId, files) => {
    const entries = files.map(createImageEntry);
    setClasses((prev) =>
      prev.map((c) =>
        c.id === classId ? { ...c, images: [...c.images, ...entries] } : c
      )
    );
  };

  const removeImage = (classId, imgId) => {
    setClasses((prev) =>
      prev.map((c) => {
        if (c.id !== classId) return c;
        const img = c.images.find((i) => i.id === imgId);
        if (img) URL.revokeObjectURL(img.previewUrl);
        return { ...c, images: c.images.filter((i) => i.id !== imgId) };
      })
    );
  };

  const totalImages = classes.reduce((sum, c) => sum + c.images.length, 0);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-[var(--foreground)]">Dataset</h3>
          <p className="text-xs text-[var(--muted)]">
            {classes.length} class{classes.length !== 1 ? "es" : ""} · {totalImages} image{totalImages !== 1 ? "s" : ""}
          </p>
        </div>
      </div>

      {/* Class cards */}
      <div className="space-y-3">
        <AnimatePresence mode="popLayout">
          {classes.map((cls, i) => (
            <ClassCard
              key={cls.id}
              cls={cls}
              index={i}
              onRename={renameClass}
              onAddImages={addImages}
              onRemoveImage={removeImage}
              onDelete={removeClass}
              canDelete={!disabled && classes.length > 2}
            />
          ))}
        </AnimatePresence>
      </div>

      {/* Add class button */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={addClass}
        disabled={disabled}
        className="w-full py-3 rounded-xl border-2 border-dashed border-[var(--border-color)]
                   text-sm font-medium text-[var(--muted)] hover:text-purple-400
                   hover:border-purple-500/40 hover:bg-purple-500/5
                   disabled:opacity-40 disabled:cursor-not-allowed
                   transition-all flex items-center justify-center gap-2"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
        </svg>
        Add Class
      </motion.button>
    </div>
  );
}
