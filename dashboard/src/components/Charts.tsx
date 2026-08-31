type BarDatum = { label: string; values: { label: string; value: number; color: string }[] }

export function GroupedBars({ data }: { data: BarDatum[] }) {
  return <div className="grouped-bars" role="img" aria-label="Grouped metric comparison">
    {data.map(group => <div className="bar-group" key={group.label}>
      <div className="bars">{group.values.map(item => <div className="bar-slot" key={item.label}>
        <span className="bar-value">{item.value.toFixed(3)}</span>
        <span className="bar" style={{ height: `${Math.max(item.value * 100, 5)}%`, background: item.color }} title={`${item.label}: ${item.value.toFixed(3)}`}/>
      </div>)}</div>
      <span className="axis-label">{group.label}</span>
    </div>)}
  </div>
}

export function HorizontalBars({ data, max }: { data: { label: string; value: number; color?: string }[]; max?: number }) {
  const ceiling = max ?? Math.max(...data.map(item => item.value))
  return <div className="horizontal-bars">{data.map(item => <div className="hbar-row" key={item.label}>
    <span>{item.label}</span><div className="hbar-track"><i style={{ width: `${(item.value / ceiling) * 100}%`, background: item.color }}/></div><strong>{item.value}</strong>
  </div>)}</div>
}

export function ScoreStrip({ items }: { items: { label: string; value: number; tone: string }[] }) {
  return <div className="score-strip">{items.map(item => <div className="score-column" key={item.label}>
    <div className="score-bars">{Array.from({ length: 18 }, (_, index) => <i key={index} style={{ height: `${18 + ((index * 13) % 22)}px`, background: item.tone, opacity: index / 18 <= item.value ? 1 : .16 }}/>)}</div>
    <span>{item.label}</span><strong>{item.value.toFixed(3)}</strong>
  </div>)}</div>
}
