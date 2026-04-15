'use client'
import { useState } from 'react'
import useSWR from 'swr'
import Head from 'next/head'
import { api } from '@/lib/api'
import PlotlyChart from '@/components/PlotlyChart'
import SeasonSelector from '@/components/SeasonSelector'

const DEFAULT_SEASON = '2024-25'

export default function HomePage() {
  const [season, setSeason] = useState(DEFAULT_SEASON)
  const [quizAnswer, setQuizAnswer] = useState<string | null>(null)
  const [quizSubmitted, setQuizSubmitted] = useState(false)

  const { data, isLoading, error } = useSWR(
    ['home', season],
    () => api.home(season),
    { revalidateOnFocus: false }
  )

  return (
    <>
      <Head>
        <title>Euro Stepper Analyst — NBA Shot DNA</title>
        <meta name="description" content="Advanced NBA player analytics — shot charts, MVP ladder, doppelgangers." />
      </Head>

      {/* Hero */}
      <div className="relative text-center py-12 mb-8 rounded-2xl overflow-hidden border border-border bg-surface">
        <div
          className="absolute inset-0 opacity-10 bg-cover bg-center"
          style={{ backgroundImage: 'url(/images/banner.jpg)' }}
        />
        <div className="relative z-10">
          <h1 className="text-4xl font-bold text-white mb-2">Euro Stepper Analyst</h1>
          <p className="text-muted text-lg">NBA Players DNA Analysis</p>
        </div>
      </div>

      {/* Season selector */}
      <div className="max-w-xs mb-8">
        <SeasonSelector value={season} onChange={s => { setSeason(s); setQuizAnswer(null); setQuizSubmitted(false) }} />
      </div>

      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-48 bg-surface border border-border rounded-xl animate-pulse" />
          ))}
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-500/30 rounded-xl p-4 text-red-300 text-sm mb-8">
          Failed to load home data. Make sure the backend is running.
        </div>
      )}

      {data && !isLoading && (
        <>
          {/* Highlights row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            {/* Daily Quiz */}
            <div className="stat-card">
              <h3 className="text-sm font-semibold text-white mb-1">Daily Quiz</h3>
              <p className="text-xs text-muted mb-3">One question per day</p>
              {data.quiz ? (
                <>
                  <p className="text-sm text-white mb-3">{data.quiz.question}</p>
                  <div className="space-y-2 mb-3">
                    {data.quiz.options.map(opt => (
                      <button
                        key={opt}
                        onClick={() => setQuizAnswer(opt)}
                        className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors border ${
                          quizAnswer === opt
                            ? 'border-accent bg-accent/20 text-white'
                            : 'border-border text-muted hover:text-white hover:border-border/80'
                        }`}
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                  {!quizSubmitted ? (
                    <button
                      onClick={() => setQuizSubmitted(true)}
                      disabled={!quizAnswer}
                      className="w-full bg-accent text-white rounded-lg py-2 text-sm font-medium disabled:opacity-40 hover:bg-accent-hover transition-colors"
                    >
                      Check answer
                    </button>
                  ) : (
                    <div className={`rounded-lg px-3 py-2 text-sm ${
                      quizAnswer === data.quiz.correct_answer
                        ? 'bg-green-900/40 text-green-300 border border-green-500/30'
                        : 'bg-red-900/40 text-red-300 border border-red-500/30'
                    }`}>
                      {quizAnswer === data.quiz.correct_answer ? '✓ Correct! ' : '✗ Not quite. '}
                      {data.quiz.explanation}
                    </div>
                  )}
                </>
              ) : (
                <p className="text-sm text-muted">Quiz loading…</p>
              )}
            </div>

            {/* Season at a Glance */}
            <div className="stat-card">
              <h3 className="text-sm font-semibold text-white mb-1">Season at a Glance</h3>
              <p className="text-xs text-muted mb-3">{season}</p>
              {data.season_progress !== null && (
                <div className="mb-3">
                  <div className="flex justify-between text-xs text-muted mb-1">
                    <span>Season Progress</span>
                    <span>{data.season_progress}%</span>
                  </div>
                  <div className="h-1.5 bg-border rounded-full">
                    <div
                      className="h-full bg-teal rounded-full transition-all"
                      style={{ width: `${data.season_progress}%` }}
                    />
                  </div>
                </div>
              )}
              <div className="space-y-2">
                {data.leaders.map(l => (
                  <div key={l.metric} className="flex justify-between items-center">
                    <span className="text-xs text-muted">{l.metric}</span>
                    <div className="text-right">
                      <span className="text-xs font-semibold text-white">{l.player}</span>
                      <span className="text-xs text-accent ml-2">{l.value}</span>
                    </div>
                  </div>
                ))}
              </div>
              {data.fun_fact && (
                <p className="text-xs text-muted mt-3 pt-3 border-t border-border">{data.fun_fact}</p>
              )}
            </div>

            {/* Who&apos;s Hot placeholder */}
            <div className="stat-card flex flex-col items-center justify-center text-center">
              <span className="text-3xl mb-2">🔥</span>
              <h3 className="text-sm font-semibold text-white mb-1">Who&apos;s Hot Right Now</h3>
              <p className="text-xs text-muted">Hot streaks &amp; MVP movers — coming soon</p>
            </div>
          </div>

          {/* Charts row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {data.charts.volume_efficiency && (
              <div className="stat-card">
                <h3 className="section-title text-base">Volume vs Efficiency</h3>
                <p className="text-xs text-muted mb-3">Top 30 usage players — are they efficient?</p>
                <PlotlyChart chartJson={data.charts.volume_efficiency} />
              </div>
            )}
            {data.charts.win_distribution && (
              <div className="stat-card">
                <h3 className="section-title text-base">Win Distribution</h3>
                <p className="text-xs text-muted mb-3">How teams are spread across win ranges</p>
                <PlotlyChart chartJson={data.charts.win_distribution} />
              </div>
            )}
          </div>
        </>
      )}
    </>
  )
}
