"use client";

import { createContext, useCallback, useContext, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

const ToastContext = createContext({ toast: () => {} });

export function useToast() {
  return useContext(ToastContext);
}

const ICONS = {
  success: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  ),
  error: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  info: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M12 2a10 10 0 100 20 10 10 0 000-20z" />
    </svg>
  ),
};

const COLORS = {
  success: {
    bg: "bg-emerald-500/15",
    border: "border-emerald-500/30",
    text: "text-emerald-400",
    icon: "bg-emerald-500/20",
  },
  error: {
    bg: "bg-red-500/15",
    border: "border-red-500/30",
    text: "text-red-400",
    icon: "bg-red-500/20",
  },
  info: {
    bg: "bg-blue-500/15",
    border: "border-blue-500/30",
    text: "text-blue-400",
    icon: "bg-blue-500/20",
  },
};

let toastId = 0;

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([]);

  const toast = useCallback((message, type = "info") => {
    const id = ++toastId;
    setToasts((prev) => [...prev, { id, message, type }]);

    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const dismiss = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}

      {/* Toast container — top-right */}
      <div className="fixed top-6 right-6 z-50 flex flex-col gap-3 pointer-events-none">
        <AnimatePresence mode="popLayout">
          {toasts.map((t) => {
            const colors = COLORS[t.type] || COLORS.info;
            return (
              <motion.div
                key={t.id}
                initial={{ opacity: 0, x: 80, scale: 0.9 }}
                animate={{ opacity: 1, x: 0, scale: 1 }}
                exit={{ opacity: 0, x: 80, scale: 0.9 }}
                transition={{ type: "spring", damping: 25, stiffness: 300 }}
                className={`pointer-events-auto flex items-center gap-3 px-5 py-3 rounded-xl border backdrop-blur-xl
                  ${colors.bg} ${colors.border} shadow-lg max-w-sm cursor-pointer`}
                onClick={() => dismiss(t.id)}
              >
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${colors.icon} ${colors.text}`}
                >
                  {ICONS[t.type]}
                </div>
                <p className={`text-sm font-medium ${colors.text}`}>
                  {t.message}
                </p>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}
