const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

async function apiFetch<T>(path: string, params?: Record<string, string | number | boolean>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`)
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString())
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `API error ${res.status}`)
  }
  return res.json()
}

export interface QuizQuestion {
  type: string
  question: string
  options: string[]
  correct_answer: string
  explanation: string
  difficulty: string
}

export interface LeaderEntry {
  metric: string
  player: string
  team: string
  value: string
}

export interface HomeData {
  season: string
  quiz: QuizQuestion | null
  fun_fact: string
  leaders: LeaderEntry[]
  season_progress: number | null
  charts: Record<string, unknown>
}

export interface LeadersData {
  season: string
  leaders: Record<string, Record<string, Record<string, unknown>[]>>
  charts: Record<string, unknown>
}

export interface MvpEntry {
  Rank: number
  PLAYER_NAME: string
  TEAM_ABBREVIATION: string
  MVP_SCORE: number
  RAW_VALUE: number
  RECORD: string
  WIN_PCT: number
  PTS: number
  REB: number
  AST: number
}

export interface MvpData {
  season: string
  ladder: MvpEntry[]
  charts: Record<string, unknown>
}

export interface PlayerMetrics {
  attempts: number
  fg_pct: number
  efg: number
  gsaa: number
  threes_attempted: number
}

export interface ZoneStat {
  zone: string
  attempts: number
  makes: number
  fg_pct: number
  freq_pct: number
}

export interface PlayerShotsData {
  player_name: string
  match_message: string | null
  season: string
  clutch: boolean
  metrics: PlayerMetrics
  zone_stats: ZoneStat[]
  zone_breakdown: Record<string, unknown>[]
  charts: Record<string, unknown>
}

export interface CompareData {
  player_a: string
  player_b: string
  match_message_a: string | null
  match_message_b: string | null
  season: string
  clutch: boolean
  metrics_a: PlayerMetrics
  metrics_b: PlayerMetrics
  charts: Record<string, unknown>
}

export interface SimilarPlayer {
  Rank: number
  Player: string
  Team: string
  Similarity: string
  'USG%': string
  'TS%': string
  '3P Rate': string
}

export interface SimilarityData {
  player_name: string
  season: string
  top_match: string
  similar_players: SimilarPlayer[]
  charts: Record<string, unknown>
}

export const api = {
  home: (season: string) =>
    apiFetch<HomeData>('/api/home', { season }),

  seasons: () =>
    apiFetch<{ seasons: string[] }>('/api/seasons'),

  leaders: (season: string) =>
    apiFetch<LeadersData>('/api/leaders', { season }),

  mvp: (season: string, top_n = 20) =>
    apiFetch<MvpData>('/api/mvp', { season, top_n }),

  playerShots: (name: string, season: string, clutch = false) =>
    apiFetch<PlayerShotsData>('/api/player/shots', { name, season, clutch }),

  playerCompare: (name_a: string, name_b: string, season: string, clutch = false) =>
    apiFetch<CompareData>('/api/player/compare', { name_a, name_b, season, clutch }),

  similarity: (name: string, season: string) =>
    apiFetch<SimilarityData>('/api/similarity', { name, season }),

  autocomplete: () =>
    apiFetch<{ players: string[] }>('/api/players/autocomplete'),
}
