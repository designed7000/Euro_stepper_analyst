'use client'
import { useState } from 'react'
import useSWR from 'swr'
import Head from 'next/head'
import { api, MvpEntry } from '@/lib/api'
import PlotlyChart from '@/components/PlotlyChart'
import SeasonSelector from '@/components/SeasonSelector'
import clsx from 'clsx'

const TOP_N_OPTIONS = [5, 10, 20]

export default function MvpPage() {
  const [season, setSeason] = useState('2025-26')
  const [topN, setTopN] = useState(10)
  const [showMethodology, setShowMethodology] = useState(false)

  const { data, isLoading, error } = useSWR(
    ['mvp', season],
    () => api.mvp(season, 20),
    { revalidateOnFocus: false }
  )

  const ladder = data?.ladder?.slice(0, topN) || []

  return (
    <>
      <Head>
        <title>MVP Ladder — Euro Stepper Analyst</title>
      </Head>

      <div className="flex items-center gap-3 mb-6">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/images/mvp_ladder.png" alt="" className="w-10 h-10 object-contain" />
        <div>
          <h1 className="text-2xl font-bold text-white">MVP Ladder</h1>
          <p className="text-muted text-sm">DNA Production Index · Scarcity-weighted statistics × team success</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-4 items-end mb-6">
        <div className="max-w-xs">
          <SeasonSelector value={season} onChange={setSeason} />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted">Show:</span>
          {TOP_N_OPTIONS.map(n => (
            <button
              key={n}
              onClick={() => setTopN(n)}
              className={clsx('px-3 py-1 text-sm rounded-lg border transition-colors', topN === n ? 'border-accent bg-accent/20 text-white' : 'border-border text-muted hover:text-white')}
            >
              Top {n}
            </button>
          ))}
        </div>
      </div>

      {isLoading && <div className="h-64 bg-surface border border-border rounded-xl animate-pulse" />}

      {error && (
        <div className="bg-red-900/30 border border-red-500/30 rounded-xl p-4 text-red-300 text-sm">
          Failed to load MVP data.
        </div>
      )}

      {data && !isLoading && (
        <>
          {/* Ladder table */}
          <div className="stat-card mb-6">
            <h2 className="section-title">Top {topN} MVP Candidates — {season}</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left">
                    {['Rank', 'Player', 'Team', 'MVP Score', 'Raw Value', 'Record', 'Win %', 'PPG', 'RPG', 'APG'].map(h => (
                      <th key={h} className="pb-2 pr-4 text-muted font-medium text-xs whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ladder.map((row: MvpEntry) => (
                    <tr key={row.Rank} className="border-b border-border/50 hover:bg-border/20 transition-colors">
                      <td className="py-2 pr-4 text-muted font-mono text-xs">{row.Rank}</td>
                      <td className="py-2 pr-4 text-white font-medium whitespace-nowrap">{row.PLAYER_NAME}</td>
                      <td className="py-2 pr-4 text-muted">{row.TEAM_ABBREVIATION}</td>
                      <td className="py-2 pr-4 text-accent font-bold font-mono">{row.MVP_SCORE?.toFixed(1)}</td>
                      <td className="py-2 pr-4 text-white font-mono">{row.RAW_VALUE?.toFixed(1)}</td>
                      <td className="py-2 pr-4 text-muted">{row.RECORD}</td>
                      <td className="py-2 pr-4 text-muted">{(row.WIN_PCT * 100).toFixed(1)}%</td>
                      <td className="py-2 pr-4 text-white">{row.PTS?.toFixed(1)}</td>
                      <td className="py-2 pr-4 text-white">{row.REB?.toFixed(1)}</td>
                      <td className="py-2 pr-4 text-white">{row.AST?.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="text-xs text-muted mt-3">
              <strong className="text-white">MVP Score</strong> = Raw Production Value × √(Team Win %) | Higher = stronger MVP case
            </p>
          </div>

          {/* Charts */}
          <h2 className="section-title">DNA Production Analysis</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            {data.charts.breakdown && (
              <div className="stat-card">
                <PlotlyChart chartJson={data.charts.breakdown} />
                <p className="text-xs text-muted mt-2">Stacked bars show contribution from each stat category</p>
              </div>
            )}
            {data.charts.scatter && (
              <div className="stat-card">
                <PlotlyChart chartJson={data.charts.scatter} />
                <p className="text-xs text-muted mt-2">Top-right quadrant = elite production + winning</p>
              </div>
            )}
          </div>

          {/* Methodology */}
          <div className="stat-card">
            <button
              onClick={() => setShowMethodology(!showMethodology)}
              className="flex items-center justify-between w-full text-left"
            >
              <span className="text-sm font-semibold text-white">📖 About DNA Production Index</span>
              <span className="text-muted">{showMethodology ? '▲' : '▼'}</span>
            </button>
            {showMethodology && (
              <div className="mt-4 text-sm text-muted space-y-3">
                <p>The DNA Production Index uses <strong className="text-white">scarcity-weighted statistics</strong> to measure player value.</p>
                <div className="overflow-x-auto">
                  <table className="text-xs w-full">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left text-white pb-1 pr-4">Stat</th>
                        <th className="text-left text-white pb-1">Formula</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        ['Points', 'Total League Points / League PTS Sum'],
                        ['Assists', '(Total League Points / League AST Sum) × 1.5'],
                        ['Rebounds', '(Total League Points / League REB Sum) × 0.7'],
                        ['Steals', 'Total League Points / League STL Sum'],
                        ['Blocks', '(Total League Points / League BLK Sum) × 0.6'],
                      ].map(([s, f]) => (
                        <tr key={s} className="border-b border-border/50">
                          <td className="py-1 pr-4 text-white">{s}</td>
                          <td className="py-1 font-mono text-xs">{f}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="font-mono text-xs bg-background rounded px-3 py-2">
                  MVP Score = Raw Value × √(Team Win Percentage)
                </p>
                <p>Min. 15 games played, 20 min/game</p>
              </div>
            )}
          </div>
        </>
      )}
    </>
  )
}
