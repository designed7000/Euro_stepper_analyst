'use client'
import Link from 'next/link'
import { useRouter } from 'next/router'
import { useState } from 'react'
import clsx from 'clsx'

const NAV_ITEMS = [
  { href: '/', label: 'Home', icon: '🏠' },
  { href: '/leaders', label: 'League Leaders', icon: '📊' },
  { href: '/mvp', label: 'MVP Ladder', icon: '🏆' },
  { href: '/player', label: 'Player Analysis', icon: '🎯' },
  { href: '/doppelgangers', label: 'Doppelgangers', icon: '🔍' },
]

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const { pathname } = useRouter()
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="flex min-h-screen bg-background">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed top-0 left-0 h-full w-64 bg-surface border-r border-border z-30 flex flex-col transition-transform duration-200',
          sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        )}
      >
        {/* Logo */}
        <div className="p-4 border-b border-border flex justify-center">
          <Link href="/">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/images/logo.png" alt="Euro Stepper Analyst" className="w-28 h-28 object-contain" />
          </Link>
        </div>

        {/* Nav */}
        <nav className="flex-1 p-4 space-y-1">
          {NAV_ITEMS.map(({ href, label, icon }) => {
            const active = pathname === href || (href !== '/' && pathname.startsWith(href))
            return (
              <Link
                key={href}
                href={href}
                onClick={() => setSidebarOpen(false)}
                className={clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                  active
                    ? 'bg-accent/20 text-accent border border-accent/30'
                    : 'text-muted hover:text-white hover:bg-border/50'
                )}
              >
                <span>{icon}</span>
                <span>{label}</span>
              </Link>
            )
          })}
        </nav>

        <div className="p-4 border-t border-border">
          <p className="text-xs text-muted text-center">NBA Shot DNA v2.0</p>
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 lg:ml-64 flex flex-col min-h-screen">
        {/* Mobile topbar */}
        <header className="lg:hidden flex items-center gap-3 p-4 border-b border-border bg-surface sticky top-0 z-10">
          <button onClick={() => setSidebarOpen(true)} className="text-white p-1">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <span className="text-white font-semibold text-sm">Euro Stepper Analyst</span>
        </header>

        <main className="flex-1 p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
