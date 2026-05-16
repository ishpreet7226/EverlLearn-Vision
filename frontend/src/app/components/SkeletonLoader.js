"use client";

/**
 * SkeletonLoader — Shimmer placeholder shown while predicting.
 */
export default function SkeletonLoader() {
  return (
    <div className="space-y-6">
      {/* Confidence + label skeleton */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-6">
          <div className="skeleton w-[120px] h-[120px] rounded-full shrink-0" />
          <div className="flex-1 space-y-3">
            <div className="skeleton h-3 w-20" />
            <div className="skeleton h-8 w-40" />
            <div className="skeleton h-3 w-56" />
          </div>
        </div>
      </div>
      {/* Probability bars skeleton */}
      <div className="glass-card p-6">
        <div className="skeleton h-3 w-36 mb-5" />
        <div className="space-y-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-between">
                <div className="skeleton h-3 w-16" />
                <div className="skeleton h-3 w-10" />
              </div>
              <div className="skeleton h-2.5 w-full" style={{ borderRadius: "999px" }} />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
