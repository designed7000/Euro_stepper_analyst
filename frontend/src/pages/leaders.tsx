'use client'
import { useState } from 'react'
import useSWR from 'swr'
import Head from 'next/head'
import { api } from '@/lib/api'
import PlotlyChart from '@/components/PlotlyChart'
import SeasonSelector from '@/components/SeasonSelector'
import clsx from 'clsx'

const TABS = [
  { key: 'scoring', label: 'Scoring Impact', caption: 'Ranked by USG% × TS% (load × efficiency)' },
  { key: 'playmaking', label: 'Playmaking', caption: 'Ranked by AST/100 poss weighted by AST/TO ratio' },
  { key: 'impact', label: 'Two-Way Impact', caption: 'Ranked by Net Rating (Offense − Defense)' },
]
const POSITIONS = ['Guard', 'Forward', 'Center']
const TOP_N_OPTIONS = [5, 10, 20]

export default function LeadersPage() {
  const [season, setSeason] = useState('2024-25')
  const [tab, setTab] = useState('scoring')
  const [topN, setTopN] = useState(5)

  const { data, isLoading, error } = useSWR(
    ['leaders', season],
    () => api.leaders(season),
    { revalidateOnFocus: false }
  )

  const activeTabInfo = TABS.find(t => t.key === tab)!

  return (
    <>
      <Head>
        <title>League Leaders — Euro Stepper Analyst</title>
      </Head>

      <div className="flex items-center gap-3 mb-6">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/images/league_leaders.png" alt="" className="w-10 h-10 object-contain" />
        <div>
          <h1 className="text-2xl font-bold text-white">League Leaders by Position</h1>
          <p className="text-muted text-sm">Top performers by scoring, playmaking, and two-way impact</p>
        </div>
      </div>

      <div className="max-w-xs mb-6">
        <SeasonSelector value={season} onChange={setSeason} />
      </div>

      {/* Tab bar */}
      <div className="flex gap-2 mb-2 flex-wrap">
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)} className={clsx('tab-btn', tab === t.key && 'active')}>
            {t.label}
          </button>
        ))}
      </div>
      <p className="text-xs text-muted mb-4">{activeTabInfo.caption} | {season}</p>

      {/* Top N selector */}
      <div className="flex items-center gap-2 mb-4">
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

      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map(i => <div key={i} className="h-64 bg-surface border border-border rounded-xl animate-pulse" />)}
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-500/30 rounded-xl p-4 text-red-300 text-sm">
          Failed to load leaders data.
        </div>
      )}

      {data && !isLoading && (
        <>
          {/* Position tables */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {POSITIONS.map(pos => {
              const rows = (data.leaders[tab]?.[pos] || []).slice(0, topN)
              return (
                <div key={pos} className="stat-card">
                  <h3 className="text-sm font-semibold text-white mb-3">{pos}s</h3>
                  {rows.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-border">
                            <th className="text-left text-muted pb-2 pr-2">#</th>
                            <th className="text-left text-muted pb-2 pr-2">Player</th>
                            <th className="text-right text-muted pb-2">Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((row: Record<string, unknown>, i: number) => (
                            <tr key={i} className="border-b border-border/50 hover:bg-border/20 transition-colors">
                              <td className="py-1.5 pr-2 text-muted">{i + 1}</td>
                              <td className="py-1.5 pr-2">
                                <div className="font-medium text-white">{row.PLAYER_NAME as string}</div>
                                <div className="text-muted">{row.TEAM_ABBREVIATION as string}</div>
                              </td>
                              <td className="py-1.5 text-right text-accent font-mono">
                                {typeof row.COMPOSITE_SCORE === 'number'
                                  ? row.COMPOSITE_SCORE.toFixed(1)
                                  : typeof row.NET_RATING === 'number'
                                    ? row.NET_RATING.toFixed(1)
                                    : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="text-xs text-muted">No data</p>
                  )}
                </div>
              )
            })}
          </div>

          {/* Charts */}
          <h2 className="section-title">Advanced Analytics</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            {data.charts.scoring && (
              <div className="stat-card">
                <PlotlyChart chartJson={data.charts.scoring} />
                <p className="text-xs text-muted mt-2">Bubble size = PPG • Above line = efficient scorers</p>
              </div>
            )}
            {data.charts.playmaking && (
              <div className="stat-card">
                <PlotlyChart chartJson={data.charts.playmaking} />
                <p className="text-xs text-muted mt-2">Bubble size = AST/TO ratio | Top-right = elite playmakers</p>
              </div>
            )}
          </div>
          {data.charts.twoway_quadrant && (
            <div className="stat-card mb-4">
              <h3 className="text-sm font-semibold text-white mb-2">Two-Way Impact Quadrant</h3>
              <PlotlyChart chartJson={data.charts.twoway_quadrant} />
              <p className="text-xs text-muted mt-2">Crosshairs = league average | Top-right = elite two-way players</p>
            </div>
          )}
          {data.charts.top_scorers && (
            <div className="stat-card">
              <h3 className="text-sm font-semibold text-white mb-2">Top 10 Scorers</h3>
              <PlotlyChart chartJson={data.charts.top_scorers} />
              <p className="text-xs text-muted mt-2">Guard (cyan) | Forward (red) | Center (green)</p>
            </div>
          )}
        </>
      )}
    </>
  )
}
