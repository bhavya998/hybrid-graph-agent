export type Route =
  | "vector"
  | "hybrid"
  | "graph_local"
  | "graph_global"
  | "tool"
  | "agentic";

export interface Citation {
  chunk_id: string;
  doc_id: string;
  page: number | null;
  span: string;
  score: number;
}

export interface AskResponse {
  answer: string;
  citations: Citation[];
  route: string;
  retry_count: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: AskResponse;
  timestamp: number;
  loading?: boolean;
  error?: string;
}
