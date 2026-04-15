'use client'
import dynamic from 'next/dynamic'

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false, loading: () => <ChartSkeleton /> })

interface PlotlyChartProps {
  chartJson: unknown
  className?: string
}

function ChartSkeleton() {
  return (
    <div className="w-full h-64 bg-surface rounded-xl animate-pulse flex items-center justify-center">
      <span className="text-muted text-sm">Loading chart…</span>
    </div>
  )
}

export default function PlotlyChart({ chartJson, className = '' }: PlotlyChartProps) {
  if (!chartJson) return <ChartSkeleton />

  const { data, layout } = chartJson as { data: Plotly.Data[]; layout: Partial<Plotly.Layout> }

  const mergedLayout: Partial<Plotly.Layout> = {
    ...layout,
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#f8fafc', ...(layout?.font || {}) },
    margin: { l: 50, r: 20, t: 50, b: 50, ...(layout?.margin || {}) },
    autosize: true,
  }

  return (
    <div className={`w-full ${className}`}>
      <Plot
        data={data}
        layout={mergedLayout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
        useResizeHandler
      />
    </div>
  )
}
