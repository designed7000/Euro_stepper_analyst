'use client'
import useSWR from 'swr'
import { api } from '@/lib/api'

const FALLBACK_SEASONS = ['2025-26', '2024-25', '2023-24', '2022-23', '2021-22', '2015-16']

interface SeasonSelectorProps {
  value: string
  onChange: (season: string) => void
}

export default function SeasonSelector({ value, onChange }: SeasonSelectorProps) {
  const { data } = useSWR('seasons', api.seasons, { revalidateOnFocus: false })
  const seasons = data?.seasons || FALLBACK_SEASONS

  return (
    <div>
      <label className="block text-xs text-muted mb-1 font-medium">Season</label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent transition-colors"
      >
        {seasons.map(s => (
          <option key={s} value={s}>{s}</option>
        ))}
      </select>
    </div>
  )
}
