"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { askQuery } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";
import { ChatMessageItem } from "./chat-message";
import { ChatInput } from "./chat-input";
import { Welcome } from "./welcome";

function newId(): string {
  return Math.random().toString(36).slice(2, 11);
}

export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = useCallback(
    async (text: string) => {
      setError(null);
      const userMsg: ChatMessage = {
        id: newId(),
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      const assistantId = newId();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        loading: true,
      };
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setLoading(true);

      try {
        const response = await askQuery(text);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? {
                  ...m,
                  content: response.answer,
                  response,
                  loading: false,
                }
              : m,
          ),
        );
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, loading: false, error: msg }
              : m,
          ),
        );
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <Welcome onExample={handleSend} />
        ) : (
          <div className="mx-auto max-w-3xl space-y-6 px-4 py-6">
            {messages.map((m) => (
              <ChatMessageItem key={m.id} message={m} />
            ))}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <div className="border-t border-border bg-surface/50 backdrop-blur-sm">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <ChatInput onSend={handleSend} disabled={loading} />
        </div>
      </div>
    </div>
  );
}
