'use client'
import { useState, useEffect, useRef } from 'react'
import useSWR from 'swr'
import { api } from '@/lib/api'

interface PlayerSearchProps {
  value: string
  onChange: (name: string) => void
  placeholder?: string
  label?: string
}

export default function PlayerSearch({ value, onChange, placeholder = 'Search player…', label }: PlayerSearchProps) {
  const { data } = useSWR('autocomplete', api.autocomplete, { revalidateOnFocus: false })
  const players = data?.players || []

  const [query, setQuery] = useState(value)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  // Sync external value
  useEffect(() => { setQuery(value) }, [value])

  // Close on outside click
  useEffect(() => {
    function handler(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const filtered = query.length >= 2
    ? players.filter(p => p.toLowerCase().includes(query.toLowerCase())).slice(0, 8)
    : []

  function select(name: string) {
    setQuery(name)
    onChange(name)
    setOpen(false)
  }

  return (
    <div ref={ref} className="relative w-full">
      {label && <label className="block text-xs text-muted mb-1 font-medium">{label}</label>}
      <input
        type="text"
        value={query}
        onChange={e => { setQuery(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
        placeholder={placeholder}
        className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-white placeholder-muted focus:outline-none focus:border-accent transition-colors"
      />
      {open && filtered.length > 0 && (
        <ul className="absolute z-50 w-full mt-1 bg-surface border border-border rounded-lg shadow-xl overflow-hidden max-h-64 overflow-y-auto">
          {filtered.map(name => (
            <li
              key={name}
              onMouseDown={() => select(name)}
              className="px-3 py-2 text-sm text-white hover:bg-border cursor-pointer transition-colors"
            >
              {name}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
