'use client'
import { useState, FormEvent } from 'react'
import { useRouter } from 'next/router'
import Head from 'next/head'
import PlayerSearch from '@/components/PlayerSearch'
import SeasonSelector from '@/components/SeasonSelector'

export default function DoppelgangersIndexPage() {
  const router = useRouter()
  const [name, setName] = useState('')
  const [season, setSeason] = useState('2024-25')

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!name.trim()) return
    router.push(`/doppelgangers/${encodeURIComponent(name)}?season=${season}`)
  }

  return (
    <>
      <Head>
        <title>Doppelgangers — Euro Stepper Analyst</title>
      </Head>

      <div className="flex items-center gap-3 mb-6">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/images/similar_players.png" alt="" className="w-10 h-10 object-contain" />
        <div>
          <h1 className="text-2xl font-bold text-white">Statistical Doppelgangers</h1>
          <p className="text-muted text-sm">ML-powered player similarity · K-Nearest Neighbors on style vectors</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="max-w-md stat-card space-y-4">
        <PlayerSearch value={name} onChange={setName} label="Find players similar to…" placeholder="e.g. Nikola Jokic" />
        <SeasonSelector value={season} onChange={setSeason} />
        <button
          type="submit"
          disabled={!name.trim()}
          className="w-full bg-accent text-white rounded-lg py-2.5 text-sm font-medium disabled:opacity-40 hover:bg-accent-hover transition-colors"
        >
          Find Doppelgangers →
        </button>
      </form>

      <div className="mt-8 stat-card max-w-md">
        <h3 className="text-sm font-semibold text-white mb-2">How it works</h3>
        <p className="text-xs text-muted">Uses K-Nearest Neighbors on 6 style features: Usage %, True Shooting %, Assist Rate, Rebound Rate, Pace, and 3P Attempt Rate. All features are normalized before comparison.</p>
      </div>
    </>
  )
}
