export type ViewId = 'overview' | 'official' | 'diagnostics' | 'ablations' | 'generalization' | 'robustness' | 'engineering' | 'sessions' | 'live'

export type MetricRun = { label: string; hr: number; mrr: number; mttc: number; efficiency: number; score: number }
export type Session = { sample_id: string; scenario_type: string; hit: boolean; first_hit_turn: number; best_rank: number; reciprocal_rank: number }

export type UsageEvent = {
  session_id: string
  turn: number
  provider: string
  model: string
  retrieval_backend: string
  rerank_backend: string
  output_strategy: string
  prompt_tokens: number
  completion_tokens: number
  cost_usd: number
  online: boolean
  latency_ms: number
  ts?: number
}

export type UsageSummary = {
  total_turns: number
  online_turns: number
  offline_turns: number
  total_prompt_tokens: number
  total_completion_tokens: number
  total_tokens: number
  total_cost_usd: number
  per_provider: { provider: string; turns: number; prompt_tokens: number; completion_tokens: number; cost_usd: number }[]
}

export type LiveMetrics = { summary: UsageSummary; recent: UsageEvent[] }
