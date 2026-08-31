"use client";

import { useMemo } from "react";
import type { OcrLine } from "@/lib/api";

/**
 * The page image with a highlight box over the field currently being reviewed.
 *
 * The boxes come from a separate OCR pass, NOT from the extraction model - Donut
 * emits no coordinates. They exist so a reviewer can see where on the page a
 * value came from. They are never fed back into the model and never used to
 * compute a metric.
 */
export default function PageViewer({
  imageUrl,
  ocrWidth,
  ocrHeight,
  highlight,
  allLines,
  showAll,
}: {
  imageUrl: string | null;
  ocrWidth: number;
  ocrHeight: number;
  highlight: OcrLine | null;
  allLines: OcrLine[];
  showAll: boolean;
}) {
  // The OCR ran on the original pixels; the <img> is scaled to fit. Using a
  // viewBox in OCR coordinates makes the overlay track the image at any size,
  // which a pixel-positioned overlay would not.
  const viewBox = useMemo(
    () => `0 0 ${ocrWidth || 1} ${ocrHeight || 1}`,
    [ocrWidth, ocrHeight]
  );

  if (!imageUrl) {
    return (
      <div className="muted" style={{ padding: 30, textAlign: "center" }}>
        No page loaded.
      </div>
    );
  }

  return (
    <div className="viewer">
      <img src={imageUrl} alt="Bill page under review" />
      {ocrWidth > 0 && (
        <svg viewBox={viewBox} preserveAspectRatio="none">
          {showAll &&
            allLines.map((line, index) => (
              <rect
                key={index}
                x={line.box[0]}
                y={line.box[1]}
                width={line.box[2] - line.box[0]}
                height={line.box[3] - line.box[1]}
                fill="none"
                stroke="#4a9eff"
                strokeWidth={1}
                opacity={0.28}
              />
            ))}
          {highlight && (
            <>
              <rect
                x={highlight.box[0] - 3}
                y={highlight.box[1] - 3}
                width={highlight.box[2] - highlight.box[0] + 6}
                height={highlight.box[3] - highlight.box[1] + 6}
                fill="#ffd60a"
                opacity={0.22}
              />
              <rect
                x={highlight.box[0] - 3}
                y={highlight.box[1] - 3}
                width={highlight.box[2] - highlight.box[0] + 6}
                height={highlight.box[3] - highlight.box[1] + 6}
                fill="none"
                stroke="#ffd60a"
                strokeWidth={2}
              />
            </>
          )}
        </svg>
      )}
    </div>
  );
}
