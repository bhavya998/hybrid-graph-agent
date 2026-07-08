"use client";

import { useState, type FormEvent } from "react";
import { Send, Square } from "lucide-react";

export function ChatInput({
  onSend,
  disabled,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
}) {
  const [text, setText] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="flex items-end gap-2 rounded-2xl border border-border bg-surface-2 px-4 py-3 transition-colors focus-within:border-accent/50">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          placeholder="Ask about the papers or code…"
          rows={1}
          className="max-h-32 flex-1 resize-none bg-transparent text-sm leading-relaxed text-foreground placeholder:text-foreground-subtle focus:outline-none"
          disabled={disabled}
        />
        <button
          type="submit"
          disabled={!text.trim() || disabled}
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent text-white transition-colors hover:bg-accent-hover disabled:opacity-40 disabled:hover:bg-accent"
        >
          {disabled ? (
            <Square className="h-3.5 w-3.5" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </button>
      </div>
      <p className="mt-1.5 px-2 text-[11px] text-foreground-subtle">
        Enter to send · Shift+Enter for newline
      </p>
    </form>
  );
}
