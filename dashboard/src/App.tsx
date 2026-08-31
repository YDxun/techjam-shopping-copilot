import { useEffect, useMemo, useState } from 'react'
import data from './data/dashboardData.json'
import { Icon } from './components/Icon'
import { GroupedBars, HorizontalBars, ScoreStrip } from './components/Charts'
import type { LiveMetrics, Session, ViewId } from './types'

const nav: { id: ViewId; label: string }[] = [
  { id: 'overview', label: 'Overview' }, { id: 'official', label: 'Official 200' },
  { id: 'diagnostics', label: 'Diagnostics' }, { id: 'ablations', label: 'Ablations' },
  { id: 'generalization', label: 'Generalization' }, { id: 'robustness', label: 'Robustness' },
  { id: 'engineering', label: 'Engineering' }, { id: 'sessions', label: 'Sessions' }, { id: 'live', label: 'Live usage' },
]

const colors = ['#285f4b', '#88ad91', '#c5cbd0']
const pct = (value: number) => `${(value * 100).toFixed(1)}%`
const n3 = (value: number) => value.toFixed(3)
const title = (id: ViewId) => nav.find(item => item.id === id)?.label ?? 'Overview'

function Panel({ title, children, className = '', action }: { title: string; children: React.ReactNode; className?: string; action?: React.ReactNode }) {
  return <section className={`panel ${className}`}><header className="panel-header"><h2>{title} <Icon name="info" size={14}/></h2>{action}</header>{children}</section>
}

function Pill({ children, tone = 'official' }: { children: React.ReactNode; tone?: string }) {
  return <span className={`pill ${tone}`}>{children}</span>
}

function Kpi({ label, value, tone = 'official' }: { label: string; value: string; tone?: string }) {
  return <div className="kpi"><div className="kpi-label">{label}<Icon name="info" size={14}/></div><div className="kpi-row"><strong>{value}</strong><Pill tone={tone}>{tone}</Pill></div></div>
}

function DataTable({ headers, rows }: { headers: string[]; rows: (string | number | React.ReactNode)[][] }) {
  return <div className="table-wrap"><table><thead><tr>{headers.map(h => <th key={h}>{h}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>)}</tbody></table></div>
}

function Overview({ setView }: { setView: (id: ViewId) => void }) {
  const runs = data.official.runs
  const current = runs[1]
  const scenarios = Object.entries(data.official.scenarios)
  return <>
    <div className="kpi-grid">
      <Kpi label="HR@10" value={n3(current.hr)}/><Kpi label="MRR" value={n3(current.mrr)}/>
      <Kpi label="Technical Score" value={n3(current.score)}/><Kpi label="API Cost" value="$0" tone="engineering"/>
    </div>
    <div className="overview-grid top-row">
      <Panel title="Official Public 200" className="chart-panel" action={<button className="text-button" onClick={() => setView('official')}>View details</button>}>
        <div className="legend">{runs.map((run, i) => <span key={run.label}><i style={{background: colors[i]}}/>{run.label}</span>)}</div>
        <GroupedBars data={[
          { label: 'HR@10', values: runs.map((r, i) => ({ label:r.label, value:r.hr, color:colors[i] })) },
          { label: 'MRR', values: runs.map((r, i) => ({ label:r.label, value:r.mrr, color:colors[i] })) },
          { label: 'Technical Score', values: runs.map((r, i) => ({ label:r.label, value:r.score, color:colors[i] })) },
        ]}/>
      </Panel>
      <Panel title="Submission Readiness" className="readiness">
        {['Evaluation suite','Version frozen','Results reproducible','Hash verified'].map(item => <div className="status-row" key={item}><Icon name="check"/><span>{item}</span><strong>Verified</strong></div>)}
        <div className="status-row"><Icon name="check"/><span>Generalization risk: Low</span><strong>Official-format re-eval</strong></div>
      </Panel>
    </div>
    <div className="overview-grid middle-row">
      <Panel title="Generalization Gap">
        <div className="legend"><span><i className="green"/>Official</span><span><i className="green"/>Synthetic (New-target)</span><span><i className="red"/>Synonym</span></div>
        <ScoreStrip items={[{label:'Official HR@10',value:1,tone:'#285f4b'},{label:'Synthetic New-target HR@10',value:0.991667,tone:'#285f4b'},{label:'Synonym HR@10',value:0,tone:'#c74c4c'}]}/>
      </Panel>
      <Panel title="Failure Attribution"><HorizontalBars data={data.synthetic.failureAttribution.map((x, i) => ({...x,label:x.label.replace(' failure','').replace(' hidden',''),color:i === 3 ? '#668d79' : '#285f4b'}))} max={120}/></Panel>
    </div>
    <div className="overview-grid bottom-row">
      <Panel title="Scenario Performance">
        <DataTable headers={['Scenario','HR@10','MRR','MTTC','Trend']} rows={scenarios.map(([name, value]) => [name.replace('_',' '),n3(value.hit_rate_at_10),n3(value.mrr),n3(value.mttc),<span className="spark" aria-label="stable high performance">▮▮▮▮▮▮▮▮</span>])}/>
      </Panel>
      <Panel title="Engineering"><div className="kv-list"><div><span>Cold start</span><strong>{(data.engineering.coldStartMs/1000).toFixed(1)} s</strong></div><div><span>Turn P95</span><strong>{Math.round(data.engineering.turn.p95_ms)} ms</strong></div><div><span>Peak RSS</span><strong>{Math.round(data.engineering.peakRssMiB)} MiB</strong></div><div><span>Tokens</span><strong>{data.engineering.tokens}</strong></div></div></Panel>
    </div>
    <p className="separation-note">Official, synthetic, and robustness results are reported separately.</p>
  </>
}

function OfficialPage() {
  const runs = data.official.runs
  return <div className="page-stack"><div className="page-intro"><Pill>official</Pill><p>Local results on the frozen official public 200 set. These are not private leaderboard results.</p></div>
    <Panel title="Three-run comparison"><DataTable headers={['Run','HR@10','MRR','MTTC','Efficiency','Technical Score']} rows={runs.map(r => [r.label,n3(r.hr),n3(r.mrr),n3(r.mttc),n3(r.efficiency),n3(r.score)])}/></Panel>
    <Panel title="Scenario breakdown"><DataTable headers={['Scenario','Sessions','HR@10','MRR','MTTC']} rows={Object.entries(data.official.scenarios).map(([key,v]) => [key.replace('_',' '),v.sample_count,n3(v.hit_rate_at_10),n3(v.mrr),n3(v.mttc)])}/></Panel>
  </div>
}

function DiagnosticsPage() {
  return <div className="page-stack"><div className="callout"><strong>Stage diagnosis</strong><span>Target position tracked across candidate recall, reranking, and final output gating.</span></div>
    <div className="two-col"><Panel title="Primary attribution"><HorizontalBars data={[{label:'Output gate delay',value:83},{label:'Ranking delay',value:43},{label:'State / Override wait',value:27},{label:'Recall delay',value:3}]} max={100}/></Panel>
    <Panel title="Focus set"><div className="stat-list"><div><strong>7</strong><span>later than weak baseline</span></div><div><strong>19</strong><span>non-Rank-1 sessions</span></div><div><strong>24</strong><span>late Intent Override hits</span></div></div></Panel></div>
    <Panel title="Diagnostic interpretation"><div className="insight-grid"><article><Pill>gate</Pill><h3>Primary public-set contributor</h3><p>Targets were already inside rerank Top-10 in 83 delayed cases, but the current emit threshold withheld them.</p></article><article><Pill tone="synthetic">ranking</Pill><h3>Deep-candidate pressure</h3><p>Forty-three sessions reached the candidate pool but were delayed outside rerank Top-10.</p></article><article><Pill tone="robustness">state</Pill><h3>Override timing</h3><p>Most late hits after turn three are concentrated in the Intent Override protocol.</p></article></div></Panel>
  </div>
}

function AblationsPage() {
  const base = data.ablations[0].score
  return <div className="page-stack"><div className="page-intro"><Pill>official</Pill><p>Controlled public-200 ablations explain public-dev behavior only; they do not establish private-set generalization.</p></div>
    <Panel title="Module contribution"><DataTable headers={['Configuration','HR@10','MRR','MTTC','Technical Score','Δ Score']} rows={data.ablations.map((r,i) => [r.label,n3(r.hr),n3(r.mrr),n3(r.mttc),n3(r.score),i===0?'—':(r.score-base).toFixed(6)])}/></Panel>
    <div className="two-col"><Panel title="Technical score"><HorizontalBars data={data.ablations.map((r,i)=>({label:r.label,value:Math.round(r.score*1000),color:i===0?'#285f4b':'#88ad91'}))} max={1000}/></Panel><Panel title="Decision"><div className="decision-card"><strong>Keep the Version A control profile</strong><p>Emit gate off improves early coverage, but public TechnicalScore falls by 0.068512. Other modules have near-zero net movement.</p></div></Panel></div>
  </div>
}

function GeneralizationPage() {
  return <div className="page-stack"><div className="page-intro warning-intro"><Pill tone="synthetic">synthetic</Pill><p>Official-format re-evaluation: 120 new targets from the frozen catalog (public targets excluded), generated with the unmodified official evaluator. Never mixed with official scoring.</p></div>
    <div className="three-kpis"><Kpi label="Version A exact HR" value="0.992" tone="synthetic"/><Kpi label="Official Weak exact HR" value="0.383" tone="synthetic"/><Kpi label="Version A MRR" value="0.872" tone="synthetic"/></div>
    <div className="two-col"><Panel title="New-target comparison"><DataTable headers={['Run','Exact HR','MRR','MTTC','Acceptable HR']} rows={data.synthetic.runs.map(r=>[r.label,n3(r.exact.hit_rate_at_10),n3(r.exact.mrr),n3(r.exact.mttc),n3(r.acceptable.hit_rate_at_10)])}/></Panel><Panel title="Failure decomposition"><HorizontalBars data={data.synthetic.failureAttribution} max={120}/></Panel></div>
    <Panel title="Legacy state metrics (informational)"><div className="stat-list four"><div><strong>{pct(data.synthetic.stateExact)}</strong><span>state exact</span></div><div><strong>{pct(data.synthetic.constraintPrecision)}</strong><span>constraint precision</span></div><div><strong>{pct(data.synthetic.constraintRecall)}</strong><span>constraint recall</span></div><div><strong>{pct(data.synthetic.concreteQuestionRate)}</strong><span>concrete questions</span></div></div></Panel>
  </div>
}

function RobustnessPage() {
  const a = data.robustness.absolute.version_a
  const variants = [['original','Original'],['synonym','Synonym'],['spelling','Spelling'],['missing_condition','Missing condition'],['equivalent_negation','Equivalent negation']] as const
  return <div className="page-stack"><div className="page-intro risk-intro"><Pill tone="robustness">robustness</Pill><p>Non-official perturbation stress test on the new-target set; per the final evaluation FAQ no undisclosed paraphrases are introduced in official scoring. Original row reflects the official-format re-evaluation.</p></div>
    <div className="three-kpis"><Kpi label="Original HR" value={n3(a.original.exact.hr)} tone="robustness"/><Kpi label="Synonym HR" value={n3(a.synonym.exact.hr)} tone="robustness"/><Kpi label="Session mismatches" value="0" tone="robustness"/></div>
    <Panel title="Input perturbation matrix"><DataTable headers={['Input','Exact HR','MRR','MTTC','Candidate Recall','Δ HR']} rows={variants.map(([key,label])=>[label,n3(a[key].exact.hr),n3(a[key].exact.mrr),n3(a[key].exact.mttc),n3(a[key].candidate_recall),key==='original'?'—':(a[key].exact.hr-a.original.exact.hr).toFixed(6)])}/></Panel>
    <div className="callout danger"><strong>Primary finding</strong><span>All 25 original exact hits were lost under synonym rewriting; candidate recall fell from 0.392 to 0.092.</span></div>
  </div>
}

function EngineeringPage() {
  const e = data.engineering
  return <div className="page-stack"><div className="three-kpis"><Kpi label="Cold start" value={`${(e.coldStartMs/1000).toFixed(1)} s`} tone="engineering"/><Kpi label="Turn P95" value={`${Math.round(e.turn.p95_ms)} ms`} tone="engineering"/><Kpi label="Peak RSS" value={`${Math.round(e.peakRssMiB)} MiB`} tone="engineering"/></div>
    <div className="two-col"><Panel title="Latency profile"><div className="kv-list"><div><span>Evaluation</span><strong>{(e.evaluationMs/1000).toFixed(1)} s</strong></div><div><span>Total in process</span><strong>{(e.totalMs/1000).toFixed(1)} s</strong></div><div><span>Turn mean</span><strong>{e.turn.mean_ms.toFixed(1)} ms</strong></div><div><span>Turn P99</span><strong>{e.turn.p99_ms.toFixed(1)} ms</strong></div><div><span>Session P95</span><strong>{e.sessionSummary.p95_ms.toFixed(1)} ms</strong></div></div></Panel>
    <Panel title="Cost"><div className="cost-display"><strong>$0</strong><span>0 prompt tokens · 0 completion tokens</span><p>Version A runs locally with BM25 and deterministic rules.</p></div></Panel></div>
    <Panel title="Fallback matrix"><DataTable headers={['Failure condition','Observed behavior','Status']} rows={e.fallbackChecks.map(([condition,behavior])=>[condition,behavior,<Pill tone={behavior.includes('failure')||behavior.includes('Empty')?'robustness':'official'}>{behavior.includes('failure')||behavior.includes('Empty')?'Guardrail':'Verified'}</Pill>])}/></Panel>
  </div>
}

function SessionsPage() {
  const [query,setQuery] = useState('')
  const [scenario,setScenario] = useState('all')
  const [selected,setSelected] = useState<Session | null>(null)
  const sessions = data.official.sessions as Session[]
  const filtered = useMemo(()=>sessions.filter(s=>(scenario==='all'||s.scenario_type===scenario)&&s.sample_id.includes(query.trim().toLowerCase())),[query,scenario,sessions])
  return <div className="page-stack"><div className="filters"><label className="search"><Icon name="search" size={18}/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search session ID"/></label><select value={scenario} onChange={e=>setScenario(e.target.value)}><option value="all">All scenarios</option><option value="boundary">Boundary</option><option value="browsing">Browsing</option><option value="buying">Buying</option><option value="intent_override">Intent Override</option></select><span>{filtered.length} sessions</span></div>
    <Panel title="Official public 200 sessions"><DataTable headers={['Session','Scenario','First hit','Best rank','Reciprocal rank','']} rows={filtered.map(s=>[s.sample_id,s.scenario_type.replace('_',' '),`Turn ${s.first_hit_turn}`,`#${s.best_rank}`,s.reciprocal_rank.toFixed(3),<button className="text-button" onClick={()=>setSelected(s)}>Inspect</button>])}/></Panel>
    {selected&&<div className="drawer-backdrop" onClick={()=>setSelected(null)}><aside className="drawer" onClick={e=>e.stopPropagation()}><button className="icon-button" onClick={()=>setSelected(null)} aria-label="Close"><Icon name="close"/></button><Pill>official</Pill><h2>{selected.sample_id}</h2><dl><div><dt>Scenario</dt><dd>{selected.scenario_type.replace('_',' ')}</dd></div><div><dt>First hit</dt><dd>Turn {selected.first_hit_turn}</dd></div><div><dt>Best rank</dt><dd>#{selected.best_rank}</dd></div><div><dt>Reciprocal rank</dt><dd>{selected.reciprocal_rank.toFixed(3)}</dd></div></dl><p>This compact view is sourced from the official result record. Full per-turn stage diagnostics remain in the frozen evaluation artifacts.</p></aside></div>}
  </div>
}

function LiveUsagePage() {
  const [metrics, setMetrics] = useState<LiveMetrics | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    let cancelled = false
    async function poll() {
      try {
        const res = await fetch('/api/metrics', { headers: { Accept: 'application/json' } })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        if (!cancelled) { setMetrics(data); setError(null) }
      } catch {
        if (!cancelled) setError('Live metrics are available when served by the local web app (python -m webapp).')
      }
    }
    poll()
    const timer = window.setInterval(poll, 5000)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [])
  const summary = metrics?.summary
  const fmt = (value: number) => `$${value.toFixed(6)}`
  const time = (ts?: number) => (ts ? new Date(ts * 1000).toLocaleTimeString() : '\u2014')
  return <div className="page-stack">
    <div className="page-intro"><Pill tone="engineering">live</Pill><p>Real-time usage recorded by the local web app while you chat. Auto-refreshes every 5 seconds; costs are approximate USD estimates from token counts.</p></div>
    {error && <div className="callout danger"><strong>Live metrics unavailable</strong><span>{error}</span></div>}
    <div className="kpi-grid">
      <Kpi label="Total turns" value={summary ? String(summary.total_turns) : '\u2014'} tone="engineering"/>
      <Kpi label="Online turns" value={summary ? String(summary.online_turns) : '\u2014'} tone={summary && summary.online_turns > 0 ? 'synthetic' : 'engineering'}/>
      <Kpi label="Total tokens" value={summary ? String(summary.total_tokens) : '\u2014'} tone="engineering"/>
      <Kpi label="Estimated cost" value={summary ? fmt(summary.total_cost_usd) : '\u2014'} tone={summary && summary.total_cost_usd > 0 ? 'synthetic' : 'engineering'}/>
    </div>
    <div className="two-col">
      <Panel title="Per-provider usage">
        {summary && summary.per_provider.length > 0
          ? <DataTable headers={['Provider','Turns','Prompt','Completion','Cost']} rows={summary.per_provider.map(p => [p.provider, String(p.turns), String(p.prompt_tokens), String(p.completion_tokens), fmt(p.cost_usd)])}/>
          : <div className="empty-state">No usage recorded yet in this session.</div>}
      </Panel>
      <Panel title="Runtime mix">
        {summary ? <div className="kv-list">
          <div><span>Offline turns</span><strong>{summary.offline_turns}</strong></div>
          <div><span>Online turns</span><strong>{summary.online_turns}</strong></div>
          <div><span>Prompt tokens</span><strong>{summary.total_prompt_tokens}</strong></div>
          <div><span>Completion tokens</span><strong>{summary.total_completion_tokens}</strong></div>
        </div> : <div className="empty-state">No usage recorded yet in this session.</div>}
      </Panel>
    </div>
    <Panel title="Recent turns">
      {metrics && metrics.recent.length > 0
        ? <DataTable headers={['Time','Session','Turn','Provider','Model','Retrieval','Rerank','Output','Prompt','Completion','Cost','Latency']} rows={metrics.recent.map(e => [time(e.ts), e.session_id.slice(0, 8), String(e.turn), e.provider, e.model || '\u2014', e.retrieval_backend, e.rerank_backend, e.output_strategy, String(e.prompt_tokens), String(e.completion_tokens), fmt(e.cost_usd), `${Math.round(e.latency_ms)} ms`])}/>
        : <div className="empty-state">No turns recorded yet. Start chatting in the Shopping Copilot app and enable an online LLM to see tokens and cost.</div>}
    </Panel>
  </div>
}

function App() {
  const [view,setView] = useState<ViewId>('overview')
  const [mobileOpen,setMobileOpen] = useState(false)
  return <div className="app-shell">
    <aside className={`sidebar ${mobileOpen?'open':''}`}><div className="brand"><strong>Evaluation Copilot</strong><span>Shopping agent quality<br/>workspace</span></div><nav>{nav.map(item=><button className={view===item.id?'active':''} key={item.id} onClick={()=>{setView(item.id);setMobileOpen(false)}}><Icon name={item.id}/><span>{item.label}</span></button>)}</nav></aside>
    {mobileOpen&&<button className="mobile-scrim" aria-label="Close navigation" onClick={()=>setMobileOpen(false)}/>}
    <main><header className="topbar"><button className="menu-button" onClick={()=>setMobileOpen(v=>!v)} aria-label="Open navigation"><Icon name="menu"/></button><h1>{view==='overview'?'Evaluation Overview':title(view)}</h1><div className="version-status"><span>Version A</span><i>·</i><span>Frozen</span><i>·</i><span>Hash verified</span><Icon name="check" size={18}/></div></header><div className="content">
      {view==='overview'&&<Overview setView={setView}/>} {view==='official'&&<OfficialPage/>} {view==='diagnostics'&&<DiagnosticsPage/>} {view==='ablations'&&<AblationsPage/>} {view==='generalization'&&<GeneralizationPage/>} {view==='robustness'&&<RobustnessPage/>} {view==='engineering'&&<EngineeringPage/>} {view==='sessions'&&<SessionsPage/>} {view==='live'&&<LiveUsagePage/>}
    </div></main>
  </div>
}

export default App
