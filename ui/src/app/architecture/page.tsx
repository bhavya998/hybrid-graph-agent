import {
  FileInput,
  Scissors,
  Search,
  Layers,
  Network,
  Bot,
  Wrench,
  Server,
  ShieldCheck,
  ArrowRight,
} from "lucide-react";
import { RouteBadge } from "@/components/route-badge";

const PIPELINE = [
  {
    icon: FileInput,
    title: "Multi-format Ingestion",
    desc: "arXiv PDFs + linked GitHub repos. Docling for layout/tables, tree-sitter for code, Voyage multimodal for figures.",
    tags: ["Docling", "tree-sitter", "Voyage MM"],
  },
  {
    icon: Scissors,
    title: "Chunking + Contextual Retrieval",
    desc: "Layout-aware semantic chunking. Haiku generates a doc-grounding summary per chunk (Anthropic technique) prepended before embedding.",
    tags: ["400-800 tok", "Contextual", "Deduped"],
  },
  {
    icon: Layers,
    title: "Hybrid Retrieval",
    desc: "Voyage dense + BM25 sparse, fused via Reciprocal Rank Fusion. Cohere cross-encoder rerank narrows to top-8.",
    tags: ["Voyage-3-large", "BM25", "RRF k=60", "Cohere Rerank"],
  },
  {
    icon: Network,
    title: "GraphRAG Layer",
    desc: "Microsoft GraphRAG extracts entities/relations → Leiden communities. Kùzu stores the Cypher-queryable graph for local/global/drift search.",
    tags: ["MS GraphRAG", "Kùzu", "Leiden", "Cypher"],
  },
];

const AGENT_NODES = [
  { name: "Router", desc: "Classifies query → picks retrieval path", route: "agentic" },
  { name: "Retriever", desc: "Executes dense / hybrid / graph search", route: "hybrid" },
  { name: "Critic", desc: "CRAG: checks faithfulness, triggers retry", route: "vector" },
  { name: "Synthesizer", desc: "Generates answer with citations", route: "graph_local" },
];

const STACK = [
  { icon: Search, name: "Dense", model: "voyage-3-large" },
  { icon: Layers, name: "Sparse", model: "fastembed BM25" },
  { icon: Wrench, name: "Rerank", model: "Cohere rerank-v3.5" },
  { icon: Bot, name: "Planner", model: "Claude Sonnet 4.5" },
  { icon: Bot, name: "Extractor", model: "Claude Haiku 4" },
];

export default function ArchitecturePage() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-4xl px-4 py-8">
        <div className="mb-10">
          <h1 className="mb-2 text-2xl font-semibold tracking-tight">
            System Architecture
          </h1>
          <p className="text-sm text-foreground-muted">
            End-to-end pipeline: ingestion → chunking → hybrid retrieval →
            reranking → GraphRAG → agent orchestration → serving.
          </p>
        </div>

        <section className="mb-12">
          <h2 className="mb-5 text-xs font-semibold uppercase tracking-wide text-foreground-subtle">
            Ingestion → Retrieval Pipeline
          </h2>
          <div className="space-y-3">
            {PIPELINE.map((stage, i) => {
              const Icon = stage.icon;
              return (
                <div key={stage.title}>
                  <div className="flex gap-4 rounded-xl border border-border bg-surface-2 p-4">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent-subtle">
                      <Icon className="h-4.5 w-4.5 text-accent" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <h3 className="text-sm font-semibold text-foreground">
                        {stage.title}
                      </h3>
                      <p className="mt-1 text-sm text-foreground-muted leading-relaxed">
                        {stage.desc}
                      </p>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {stage.tags.map((tag) => (
                          <span
                            key={tag}
                            className="rounded-md bg-surface-3 px-2 py-0.5 font-mono text-[11px] text-foreground-subtle"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                  {i < PIPELINE.length - 1 && (
                    <div className="flex justify-center py-1">
                      <ArrowRight className="h-4 w-4 rotate-90 text-foreground-subtle" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        <section className="mb-12">
          <h2 className="mb-5 text-xs font-semibold uppercase tracking-wide text-foreground-subtle">
            LangGraph Agent Graph
          </h2>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {AGENT_NODES.map((node, i) => (
              <div key={node.name}>
                <div className="rounded-xl border border-border bg-surface-2 p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-xs font-mono text-foreground-subtle">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                  </div>
                  <h3 className="mb-1 text-sm font-semibold">{node.name}</h3>
                  <p className="text-[11px] leading-relaxed text-foreground-muted">
                    {node.desc}
                  </p>
                  <div className="mt-2.5">
                    <RouteBadge route={node.route} size="sm" />
                  </div>
                </div>
              </div>
            ))}
          </div>
          <p className="mt-3 px-1 text-xs text-foreground-subtle">
            Adaptive routing: a cheap classifier (Haiku) picks one of 6 paths per
            query. The Critic can escalate with up to 2 retry loops (CRAG pattern).
          </p>
        </section>

        <section className="mb-12">
          <h2 className="mb-5 text-xs font-semibold uppercase tracking-wide text-foreground-subtle">
            Model Stack (All-API)
          </h2>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
            {STACK.map((item) => {
              const Icon = item.icon;
              return (
                <div
                  key={item.name}
                  className="rounded-xl border border-border bg-surface-2 p-3 text-center"
                >
                  <Icon className="mx-auto mb-2 h-4 w-4 text-foreground-subtle" />
                  <div className="text-xs font-medium">{item.name}</div>
                  <div className="mt-0.5 truncate font-mono text-[10px] text-foreground-subtle">
                    {item.model}
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        <section className="mb-8">
          <h2 className="mb-5 text-xs font-semibold uppercase tracking-wide text-foreground-subtle">
            Cross-Cutting
          </h2>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <CrossCuttingCard
              icon={Server}
              title="Serving"
              desc="FastAPI REST + Chainlit chat. Provider-profile abstraction for Databricks swap."
            />
            <CrossCuttingCard
              icon={ShieldCheck}
              title="Observability"
              desc="Langfuse spans on every node. PII redaction before any payload leaves the process."
            />
            <CrossCuttingCard
              icon={FileInput}
              title="Eval Harness"
              desc="Custom LLM-as-judge: faithfulness, answer relevancy, context precision/recall + nDCG/MRR/Hit@k."
            />
          </div>
        </section>
      </div>
    </div>
  );
}

function CrossCuttingCard({
  icon: Icon,
  title,
  desc,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  desc: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-surface-2 p-4">
      <Icon className="mb-2.5 h-4 w-4 text-accent" />
      <h3 className="mb-1 text-sm font-semibold">{title}</h3>
      <p className="text-xs leading-relaxed text-foreground-muted">{desc}</p>
    </div>
  );
}
