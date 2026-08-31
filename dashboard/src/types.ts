export type ViewId = 'overview' | 'official' | 'diagnostics' | 'ablations' | 'generalization' | 'robustness' | 'engineering' | 'sessions'

export type MetricRun = { label: string; hr: number; mrr: number; mttc: number; efficiency: number; score: number }
export type Session = { sample_id: string; scenario_type: string; hit: boolean; first_hit_turn: number; best_rank: number; reciprocal_rank: number }
