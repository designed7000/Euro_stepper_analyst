'use client'
import { useState } from 'react'
import { useRouter } from 'next/router'
import useSWR from 'swr'
import Head from 'next/head'
import Link from 'next/link'
import { api, SimilarPlayer } from '@/lib/api'
import PlotlyChart from '@/components/PlotlyChart'
import SeasonSelector from '@/components/SeasonSelector'

export default function DoppelgangersPage() {
  const router = useRouter()
  const { name, season: seasonParam } = router.query

  const playerName = decodeURIComponent((name as string) || '')
  const [season, setSeason] = useState((seasonParam as string) || '2024-25')
  const [showTable, setShowTable] = useState(false)
  const [showMethodology, setShowMethodology] = useState(false)

  const { data, isLoading, error } = useSWR(
    playerName ? ['similarity', playerName, season] : null,
    () => api.similarity(playerName, season),
    { revalidateOnFocus: false }
  )

  if (!playerName) {
    return (
      <div className="text-center py-20">
        <p className="text-muted mb-4">No player selected.</p>
        <Link href="/doppelgangers" className="text-accent hover:underline">← Back to search</Link>
      </div>
    )
  }

  return (
    <>
      <Head>
        <title>{playerName} Doppelgangers — Euro Stepper Analyst</title>
      </Head>

      <div className="flex items-start justify-between mb-6 flex-wrap gap-4">
        <div className="flex items-center gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/images/similar_players.png" alt="" className="w-10 h-10 object-contain" />
          <div>
            <h1 className="text-2xl font-bold text-white">Statistical Doppelgangers</h1>
            <p className="text-muted text-sm">Finding players with similar playing styles to <span className="text-white">{playerName}</span></p>
          </div>
        </div>
        <Link href="/doppelgangers" className="text-sm text-muted hover:text-white transition-colors">← New search</Link>
      </div>

      <div className="max-w-xs mb-6">
        <SeasonSelector value={season} onChange={setSeason} />
      </div>

      {isLoading && (
        <div className="space-y-4">
          {[1, 2, 3].map(i => <div key={i} className="h-48 bg-surface border border-border rounded-xl animate-pulse" />)}
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-500/30 rounded-xl p-4 text-red-300 text-sm">
          {error.message}
        </div>
      )}

      {data && !isLoading && (
        <>
          {/* Similarity bar */}
          {data.charts.similarity_bar && (
            <div className="stat-card mb-6">
              <h2 className="section-title">Match Rankings</h2>
              <PlotlyChart chartJson={data.charts.similarity_bar} />
            </div>
          )}

          {/* Percentile + Box score */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            {data.charts.percentile && (
              <div className="stat-card">
                <h3 className="section-title text-base">League Percentile Rankings</h3>
                <PlotlyChart chartJson={data.charts.percentile} />
              </div>
            )}
            {data.charts.stat_comparison && (
              <div className="stat-card">
                <h3 className="section-title text-base">Box Score Comparison</h3>
                <PlotlyChart chartJson={data.charts.stat_comparison} />
              </div>
            )}
          </div>

          {/* Style DNA */}
          {data.charts.style_profile && (
            <div className="stat-card mb-6">
              <h3 className="section-title">Playing Style DNA</h3>
              <PlotlyChart chartJson={data.charts.style_profile} />
            </div>
          )}

          {/* Radar */}
          {data.charts.radar && (
            <div className="stat-card mb-6">
              <h3 className="section-title">Detailed Style Comparison</h3>
              <p className="text-xs text-muted mb-3">
                {playerName} vs <span className="text-white">{data.top_match}</span>
              </p>
              <PlotlyChart chartJson={data.charts.radar} />
            </div>
          )}

          {/* Match data table */}
          <div className="stat-card mb-4">
            <button
              onClick={() => setShowTable(!showTable)}
              className="flex items-center justify-between w-full text-left"
            >
              <span className="text-sm font-semibold text-white">📊 View Full Match Data</span>
              <span className="text-muted">{showTable ? '▲' : '▼'}</span>
            </button>
            {showTable && (
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border">
                      {['Rank', 'Player', 'Team', 'Similarity', 'USG%', 'TS%', '3P Rate'].map(h => (
                        <th key={h} className="text-left text-muted pb-2 pr-4">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.similar_players.map((p: SimilarPlayer) => (
                      <tr key={p.Rank} className="border-b border-border/50">
                        <td className="py-1.5 pr-4 text-muted">{p.Rank}</td>
                        <td className="py-1.5 pr-4 text-white font-medium">{p.Player}</td>
                        <td className="py-1.5 pr-4 text-muted">{p.Team}</td>
                        <td className="py-1.5 pr-4 text-accent font-mono">{p.Similarity}</td>
                        <td className="py-1.5 pr-4 text-muted">{p['USG%']}</td>
                        <td className="py-1.5 pr-4 text-muted">{p['TS%']}</td>
                        <td className="py-1.5 pr-4 text-muted">{p['3P Rate']}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Methodology */}
          <div className="stat-card">
            <button
              onClick={() => setShowMethodology(!showMethodology)}
              className="flex items-center justify-between w-full text-left"
            >
              <span className="text-sm font-semibold text-white">📖 About the Similarity Model</span>
              <span className="text-muted">{showMethodology ? '▲' : '▼'}</span>
            </button>
            {showMethodology && (
              <div className="mt-4 text-xs text-muted space-y-3">
                <p>Uses <strong className="text-white">K-Nearest Neighbors</strong> (ball tree, 6 neighbors) to find players with similar style vectors.</p>
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left text-white pb-1 pr-4">Metric</th>
                      <th className="text-left text-white pb-1">What it measures</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ['Usage Rate (USG%)', 'How often the player uses possessions'],
                      ['True Shooting (TS%)', 'Overall scoring efficiency'],
                      ['Assist Rate (AST%)', '% of teammate FGs assisted'],
                      ['Rebound Rate (REB%)', '% of available rebounds grabbed'],
                      ['Pace', "Team's possessions per 48 minutes"],
                      ['3-Point Rate (3P_AR)', '% of shots from 3-point range'],
                    ].map(([m, d]) => (
                      <tr key={m} className="border-b border-border/50">
                        <td className="py-1 pr-4 text-white">{m}</td>
                        <td className="py-1">{d}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p>All features are normalized with StandardScaler before comparison. Min. 15 games, 10+ min/game.</p>
              </div>
            )}
          </div>
        </>
      )}
    </>
  )
}
