import {
  Search,
  Layers,
  Network,
  Globe,
  Wrench,
  Brain,
} from "lucide-react";
import type { Route } from "@/lib/types";

const ROUTE_CONFIG: Record<
  string,
  {
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    color: string;
    bg: string;
    description: string;
  }
> = {
  vector: {
    label: "Vector",
    icon: Search,
    color: "var(--route-vector)",
    bg: "rgba(59, 130, 246, 0.12)",
    description: "Dense semantic search",
  },
  hybrid: {
    label: "Hybrid",
    icon: Layers,
    color: "var(--route-hybrid)",
    bg: "rgba(139, 92, 246, 0.12)",
    description: "Dense + BM25 sparse, RRF fusion + rerank",
  },
  graph_local: {
    label: "Graph Local",
    icon: Network,
    color: "var(--route-graph)",
    bg: "rgba(34, 197, 94, 0.12)",
    description: "Entity-centric knowledge graph traversal",
  },
  graph_global: {
    label: "Graph Global",
    icon: Globe,
    color: "var(--route-graph)",
    bg: "rgba(34, 197, 94, 0.12)",
    description: "Community-summary map-reduce",
  },
  tool: {
    label: "Tool",
    icon: Wrench,
    color: "var(--route-tool)",
    bg: "rgba(245, 158, 11, 0.12)",
    description: "MCP tool execution",
  },
  agentic: {
    label: "Agentic",
    icon: Brain,
    color: "var(--route-agentic)",
    bg: "rgba(239, 68, 68, 0.12)",
    description: "Multi-step planning loop",
  },
};

export function RouteBadge({
  route,
  size = "md",
}: {
  route: string;
  size?: "sm" | "md";
}) {
  const config = ROUTE_CONFIG[route] ?? ROUTE_CONFIG.hybrid;
  const Icon = config.icon;
  const pad = size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-2.5 py-1 text-xs";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full font-medium ${pad}`}
      style={{ color: config.color, backgroundColor: config.bg }}
      title={config.description}
    >
      <Icon className={size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5"} />
      {config.label}
    </span>
  );
}

export { ROUTE_CONFIG };
