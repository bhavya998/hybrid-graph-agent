"use client";

import { Activity, RotateCcw, Sparkles } from "lucide-react";
import type { AskResponse } from "@/lib/types";
import { RouteBadge } from "./route-badge";

export function TracePanel({ response }: { response: AskResponse }) {
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
      <div className="flex items-center gap-1.5 text-foreground-subtle">
        <Activity className="h-3.5 w-3.5" />
        <span className="text-[11px] font-medium uppercase tracking-wide">
          Trace
        </span>
      </div>

      <div className="h-3.5 w-px bg-border" />

      <RouteBadge route={response.route} size="sm" />

      <div className="flex items-center gap-1 text-xs text-foreground-muted">
        <RotateCcw className="h-3 w-3" />
        <span className="tabular-nums">{response.retry_count}</span>
        <span className="text-foreground-subtle">retries</span>
      </div>

      {response.citations.length > 0 && (
        <>
          <div className="h-3.5 w-px bg-border" />
          <div className="flex items-center gap-1 text-xs text-foreground-muted">
            <Sparkles className="h-3 w-3" />
            <span className="tabular-nums">{response.citations.length}</span>
            <span className="text-foreground-subtle">sources</span>
          </div>
        </>
      )}
    </div>
  );
}
