type Props = { name: string; size?: number }

export function Icon({ name, size = 20 }: Props) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true }
  const paths: Record<string, React.ReactNode> = {
    overview: <><path d="M4 19V10M9 19V5M14 19v-7M19 19V8"/><path d="M3 19h18"/></>,
    official: <><path d="M12 3l8 4v5c0 5-3.4 8-8 9-4.6-1-8-4-8-9V7l8-4z"/><path d="M9 12l2 2 4-5"/></>,
    diagnostics: <><path d="M6 3v5a4 4 0 0 0 8 0V3"/><path d="M5 3h2M13 3h2M14 16a4 4 0 1 0 4-4"/></>,
    ablations: <><path d="M4 7h10M18 7h2M4 17h2M10 17h10M8 14v6M16 4v6"/></>,
    generalization: <><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 0 1 0 18M12 3a15 15 0 0 0 0 18"/></>,
    robustness: <><path d="M12 3l8 4v5c0 5-3.4 8-8 9-4.6-1-8-4-8-9V7l8-4z"/><path d="M8 12h8"/></>,
    engineering: <><path d="M8 8l-4 4 4 4M16 8l4 4-4 4"/><path d="M14 5l-4 14"/></>,
    sessions: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    check: <><circle cx="12" cy="12" r="10" fill="currentColor" stroke="none"/><path d="M7.5 12.5l3 3 6-7" stroke="white" strokeWidth="2"/></>,
    warning: <><path d="M12 3L2.7 20h18.6L12 3z" fill="currentColor" stroke="none"/><path d="M12 8v5M12 17h.01" stroke="white" strokeWidth="2"/></>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></>,
    search: <><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/></>,
    menu: <><path d="M4 7h16M4 12h16M4 17h16"/></>,
    close: <><path d="M6 6l12 12M18 6L6 18"/></>,
  }
  return <svg {...common}>{paths[name] ?? paths.info}</svg>
}
