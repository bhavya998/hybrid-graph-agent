"use client";

import { useState } from "react";
import { ChevronDown, FileText, Hash } from "lucide-react";
import type { Citation } from "@/lib/types";

export function CitationCard({
  index,
  citation,
}: {
  index: number;
  citation: Citation;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="rounded-lg border border-border bg-surface-2 overflow-hidden transition-colors hover:border-border-hover"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2.5 px-3 py-2 text-left"
      >
        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-accent-subtle text-[11px] font-semibold text-accent">
          {index}
        </span>
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <FileText className="h-3.5 w-3.5 shrink-0 text-foreground-subtle" />
          <span className="truncate font-mono text-xs text-foreground-muted">
            {citation.doc_id || "unknown"}
          </span>
          {citation.page != null && (
            <span className="shrink-0 text-xs text-foreground-subtle">
              p.{citation.page}
            </span>
          )}
        </div>
        {citation.score > 0 && (
          <span className="shrink-0 text-[11px] tabular-nums text-foreground-subtle">
            {citation.score.toFixed(3)}
          </span>
        )}
        <ChevronDown
          className={`h-3.5 w-3.5 shrink-0 text-foreground-subtle transition-transform ${
            expanded ? "rotate-180" : ""
          }`}
        />
      </button>
      {expanded && (
        <div className="border-t border-border px-3 py-2.5 text-xs">
          <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-foreground-subtle">
            <span className="flex items-center gap-1">
              <Hash className="h-3 w-3" />
              <span className="font-mono">{citation.chunk_id}</span>
            </span>
          </div>
          {citation.span && (
            <p className="text-foreground-muted leading-relaxed">
              {citation.span}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
