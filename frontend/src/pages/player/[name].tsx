'use client'
import { useState } from 'react'
import { useRouter } from 'next/router'
import useSWR from 'swr'
import Head from 'next/head'
import Link from 'next/link'
import { api } from '@/lib/api'
import PlotlyChart from '@/components/PlotlyChart'
import SeasonSelector from '@/components/SeasonSelector'
import PlayerSearch from '@/components/PlayerSearch'
import clsx from 'clsx'

export default function PlayerPage() {
  const router = useRouter()
  const { name, season: seasonParam, clutch: clutchParam } = router.query

  const playerName = decodeURIComponent((name as string) || '')
  const [season, setSeason] = useState((seasonParam as string) || '2025-26')
  const [clutch, setClutch] = useState(clutchParam === 'true')
  const [chartView, setChartView] = useState<'scatter' | 'heatmap'>('scatter')
  const [compareMode, setCompareMode] = useState(false)
  const [playerB, setPlayerB] = useState('')

  const { data, isLoading, error } = useSWR(
    playerName ? ['player-shots', playerName, season, clutch] : null,
    () => api.playerShots(playerName, season, clutch),
    { revalidateOnFocus: false }
  )

  const { data: compareData, isLoading: compareLoading, error: compareError } = useSWR(
    compareMode && playerB && playerName ? ['player-compare', playerName, playerB, season, clutch] : null,
    () => api.playerCompare(playerName, playerB, season, clutch),
    { revalidateOnFocus: false }
  )

  if (!playerName) {
    return (
      <div className="text-center py-20">
        <p className="text-muted mb-4">No player selected.</p>
        <Link href="/player" className="text-accent hover:underline">← Back to search</Link>
      </div>
    )
  }

  const displayName = data?.player_name || playerName

  return (
    <>
      <Head>
        <title>{displayName} — Euro Stepper Analyst</title>
      </Head>

      {/* Header */}
      <div className="flex items-start justify-between mb-6 flex-wrap gap-4">
        <div className="flex items-center gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/images/player_analysis.png" alt="" className="w-10 h-10 object-contain" />
          <div>
            <h1 className="text-2xl font-bold text-white">{compareData ? `${compareData.player_a} vs ${compareData.player_b}` : displayName}</h1>
            {data?.match_message && <p className="text-xs text-teal">{data.match_message}</p>}
            {clutch && <span className="inline-block mt-1 text-xs bg-amber-900/40 text-amber-300 border border-amber-500/30 rounded px-2 py-0.5">Clutch Time</span>}
          </div>
        </div>
        <Link href="/player" className="text-sm text-muted hover:text-white transition-colors">← New search</Link>
      </div>

      {/* Controls */}
      <div className="stat-card mb-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <SeasonSelector value={season} onChange={s => { setSeason(s); router.replace({ query: { ...router.query, season: s } }) }} />
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={clutch}
                onChange={e => setClutch(e.target.checked)}
                className="w-4 h-4 accent-accent"
              />
              <span className="text-sm text-muted">Clutch Only</span>
            </label>
          </div>
          <div className="flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={compareMode}
                onChange={e => setCompareMode(e.target.checked)}
                className="w-4 h-4 accent-accent"
              />
              <span className="text-sm text-muted">Compare Players</span>
            </label>
          </div>
          {compareMode && (
            <PlayerSearch value={playerB} onChange={setPlayerB} label="Player B" placeholder="e.g. Jayson Tatum" />
          )}
        </div>
      </div>

      {isLoading && (
        <div className="space-y-4">
          {[1, 2, 3].map(i => <div key={i} className="h-32 bg-surface border border-border rounded-xl animate-pulse" />)}
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-500/30 rounded-xl p-4 text-red-300 text-sm mb-6">
          {error.message}
        </div>
      )}

      {data && !isLoading && !compareMode && (
        <>
          {/* Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-6">
            {[
              { label: 'FGA', value: data.metrics.attempts },
              { label: 'FG%', value: `${data.metrics.fg_pct}%` },
              { label: 'eFG%', value: `${data.metrics.efg}%` },
              { label: '3PT Attempts', value: data.metrics.threes_attempted },
              { label: 'GSAA', value: `${data.metrics.gsaa > 0 ? '+' : ''}${data.metrics.gsaa}` },
            ].map(m => (
              <div key={m.label} className="stat-card text-center">
                <div className="text-xs text-muted mb-1">{m.label}</div>
                <div className="text-lg font-bold text-white">{m.value}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-muted mb-6">
            <strong className="text-white">GSAA</strong>: Points generated by shooting skill above league average expectation
          </p>

          {/* Chart toggle */}
          <div className="flex gap-2 mb-4">
            {(['scatter', 'heatmap'] as const).map(v => (
              <button
                key={v}
                onClick={() => setChartView(v)}
                className={clsx('tab-btn', chartView === v && 'active')}
              >
                {v === 'scatter' ? 'Scatter (Individual Shots)' : 'Heatmap (Zone Aggregated)'}
              </button>
            ))}
          </div>

          {/* Shot chart */}
          <div className="stat-card mb-6">
            <PlotlyChart chartJson={chartView === 'scatter' ? data.charts.scatter : (data.charts.heatmap || data.charts.scatter)} />
          </div>

          {/* Zone breakdown */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <div className="stat-card">
              <h3 className="section-title text-base">Zone Breakdown vs League Average</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border">
                      {['Zone', 'Attempts', 'Player FG%', 'League FG%', 'Relative'].map(h => (
                        <th key={h} className="text-left text-muted pb-2 pr-3">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.zone_breakdown.map((row, i) => {
                      const rel = row.relative_fg_pct as number
                      return (
                        <tr key={i} className="border-b border-border/50">
                          <td className="py-1.5 pr-3 text-white">{row.SHOT_ZONE_BASIC as string}</td>
                          <td className="py-1.5 pr-3 text-muted">{row.player_attempts as number}</td>
                          <td className="py-1.5 pr-3">{((row.player_fg_pct as number) * 100).toFixed(1)}%</td>
                          <td className="py-1.5 pr-3">{((row.league_fg_pct as number) * 100).toFixed(1)}%</td>
                          <td className={clsx('py-1.5 font-mono', rel > 0 ? 'text-green-400' : rel < 0 ? 'text-red-400' : 'text-muted')}>
                            {rel > 0 ? '+' : ''}{rel.toFixed(1)}%
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
            {data.charts.zone_profile && (
              <div className="stat-card">
                <h3 className="section-title text-base">Shot Frequency Profile</h3>
                <PlotlyChart chartJson={data.charts.zone_profile} />
              </div>
            )}
          </div>

          {/* Doppelganger CTA */}
          <div className="stat-card text-center py-6">
            <p className="text-muted text-sm mb-3">Find players who play just like {displayName}</p>
            <Link
              href={`/doppelgangers/${encodeURIComponent(displayName)}?season=${season}`}
              className="inline-block bg-accent text-white rounded-lg px-6 py-2.5 text-sm font-medium hover:bg-accent-hover transition-colors"
            >
              🔍 Find Doppelgangers
            </Link>
          </div>
        </>
      )}

      {/* Compare mode */}
      {compareMode && compareData && !compareLoading && (
        <>
          <div className="grid grid-cols-2 gap-3 mb-6">
            {(['a', 'b'] as const).map(side => {
              const metrics = side === 'a' ? compareData.metrics_a : compareData.metrics_b
              const pName = side === 'a' ? compareData.player_a : compareData.player_b
              return (
                <div key={side} className="stat-card">
                  <h3 className="text-sm font-semibold text-white mb-3">{pName}</h3>
                  {[
                    ['FGA', metrics.attempts],
                    ['FG%', `${metrics.fg_pct}%`],
                    ['eFG%', `${metrics.efg}%`],
                    ['GSAA', `${metrics.gsaa > 0 ? '+' : ''}${metrics.gsaa}`],
                  ].map(([l, v]) => (
                    <div key={l as string} className="flex justify-between text-xs py-1 border-b border-border/50">
                      <span className="text-muted">{l}</span>
                      <span className="text-white font-medium">{v}</span>
                    </div>
                  ))}
                </div>
              )
            })}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            {compareData.charts.scatter_a && <div className="stat-card"><PlotlyChart chartJson={compareData.charts.scatter_a} /></div>}
            {compareData.charts.scatter_b && <div className="stat-card"><PlotlyChart chartJson={compareData.charts.scatter_b} /></div>}
          </div>
          {compareData.charts.radar && (
            <div className="stat-card mb-4">
              <h3 className="section-title text-base">Player Profile Comparison</h3>
              <PlotlyChart chartJson={compareData.charts.radar} />
            </div>
          )}
          {compareData.charts.zone_freq && (
            <div className="stat-card mb-4">
              <h3 className="section-title text-base">Shot Distribution &amp; Accuracy</h3>
              <PlotlyChart chartJson={compareData.charts.zone_freq} />
            </div>
          )}
          {compareData.charts.efg_trend && compareData.charts.gsaa_trend && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="stat-card"><PlotlyChart chartJson={compareData.charts.efg_trend} /></div>
              <div className="stat-card"><PlotlyChart chartJson={compareData.charts.gsaa_trend} /></div>
            </div>
          )}
        </>
      )}

      {compareMode && compareLoading && (
        <div className="h-64 bg-surface border border-border rounded-xl animate-pulse" />
      )}
      {compareMode && compareError && (
        <div className="bg-red-900/30 border border-red-500/30 rounded-xl p-4 text-red-300 text-sm">
          {compareError.message}
        </div>
      )}
    </>
  )
}
