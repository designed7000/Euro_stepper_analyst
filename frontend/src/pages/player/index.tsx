'use client'
import { useState, FormEvent } from 'react'
import { useRouter } from 'next/router'
import Head from 'next/head'
import PlayerSearch from '@/components/PlayerSearch'
import SeasonSelector from '@/components/SeasonSelector'

export default function PlayerIndexPage() {
  const router = useRouter()
  const [name, setName] = useState('')
  const [season, setSeason] = useState('2024-25')

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!name.trim()) return
    router.push(`/player/${encodeURIComponent(name)}?season=${season}`)
  }

  return (
    <>
      <Head>
        <title>Player Analysis — Euro Stepper Analyst</title>
      </Head>

      <div className="flex items-center gap-3 mb-6">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/images/player_analysis.png" alt="" className="w-10 h-10 object-contain" />
        <div>
          <h1 className="text-2xl font-bold text-white">Player Analysis</h1>
          <p className="text-muted text-sm">Shot charts, zone efficiency, and career trends</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="max-w-md stat-card space-y-4">
        <PlayerSearch value={name} onChange={setName} label="Player Name" placeholder="e.g. LeBron James" />
        <SeasonSelector value={season} onChange={setSeason} />
        <button
          type="submit"
          disabled={!name.trim()}
          className="w-full bg-accent text-white rounded-lg py-2.5 text-sm font-medium disabled:opacity-40 hover:bg-accent-hover transition-colors"
        >
          Analyze Player →
        </button>
      </form>
    </>
  )
}
