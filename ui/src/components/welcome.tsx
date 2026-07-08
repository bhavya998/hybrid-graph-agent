"use client";

import {
  Sparkles,
  Search,
  Network,
  Code2,
  GitGraph,
} from "lucide-react";

const EXAMPLES = [
  {
    icon: Search,
    label: "What is FLARE?",
    query: "What is FLARE and how does it work?",
  },
  {
    icon: Network,
    label: "How does GraphRAG compare to vector RAG?",
    query: "How does GraphRAG compare to vector RAG for multi-hop reasoning?",
  },
  {
    icon: Code2,
    label: "Show me the retrieval code",
    query: "Show me how the FLARE repo implements retrieval",
  },
  {
    icon: GitGraph,
    label: "Who built on RAG?",
    query: "Which authors introduced retrieval-augmented generation?",
  },
];

export function Welcome({ onExample }: { onExample: (q: string) => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-4">
      <div className="w-full max-w-2xl text-center">
        <div className="mb-5 flex justify-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-accent-subtle">
            <Sparkles className="h-7 w-7 text-accent" />
          </div>
        </div>
        <h1 className="mb-2 text-2xl font-semibold tracking-tight">
          Hybrid RAG + GraphRAG Agent
        </h1>
        <p className="mb-8 text-sm text-foreground-muted">
          Ask questions across the frontier AI literature and the codebases
          that implement it — with provenance and a knowledge graph.
        </p>

        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {EXAMPLES.map((ex) => {
            const Icon = ex.icon;
            return (
              <button
                key={ex.label}
                onClick={() => onExample(ex.query)}
                className="group flex items-center gap-3 rounded-xl border border-border bg-surface-2 px-4 py-3 text-left transition-colors hover:border-border-hover hover:bg-surface-3"
              >
                <Icon className="h-4 w-4 shrink-0 text-foreground-subtle transition-colors group-hover:text-accent" />
                <span className="text-sm text-foreground-muted transition-colors group-hover:text-foreground">
                  {ex.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
