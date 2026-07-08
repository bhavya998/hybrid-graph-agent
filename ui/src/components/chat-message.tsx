"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, AlertCircle } from "lucide-react";
import type { ChatMessage } from "@/lib/types";
import { TracePanel } from "./trace-panel";
import { CitationCard } from "./citation-card";

export function ChatMessageItem({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex gap-3 animate-fade-in-up ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
          isUser
            ? "bg-accent-subtle text-accent"
            : "bg-surface-3 text-foreground-muted"
        }`}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      <div className={`flex min-w-0 max-w-[85%] flex-col gap-2 ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? "bg-accent text-white rounded-tr-sm"
              : "bg-surface-2 border border-border rounded-tl-sm"
          }`}
        >
          {message.error ? (
            <div className="flex items-center gap-2 text-error">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span className="text-sm">{message.error}</span>
            </div>
          ) : isUser ? (
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="markdown-body">
              {message.loading && message.content === "" ? (
                <TypingIndicator />
              ) : (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content}
                </ReactMarkdown>
              )}
            </div>
          )}
        </div>

        {!isUser && !message.loading && message.response && !message.error && (
          <>
            <TracePanel response={message.response} />
            {message.response.citations.length > 0 && (
              <div className="w-full space-y-1.5">
                <p className="px-1 text-[11px] font-medium uppercase tracking-wide text-foreground-subtle">
                  Sources
                </p>
                {message.response.citations.map((c, i) => (
                  <CitationCard key={c.chunk_id || i} index={i + 1} citation={c} />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-2 w-2 rounded-full bg-foreground-muted animate-pulse-dot"
          style={{ animationDelay: `${i * 0.2}s` }}
        />
      ))}
    </div>
  );
}
