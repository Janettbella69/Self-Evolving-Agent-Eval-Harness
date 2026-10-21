import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, Link, useSearchParams, useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getTrace, annotateGrader, annotateHumanPass, updateOpenCodes, startTraceAnalysis, getAnalysisResult, getClassifiedErrors, updateReviewStatus, backflowTrace, listDatasets } from '../lib/api.ts'
import ScoreBadge from '../components/ScoreBadge.tsx'
import GraderBreakdown from '../components/GraderBreakdown.tsx'
import TraceTimeline from '../components/TraceTimeline.tsx'
import FailureFunnel from '../components/FailureFunnel.tsx'
import type { Trace, GradingLogEntry, AnalysisResult } from '../types.ts'

type TabKey = 'guide' | 'products' | 'sources' | 'events' | 'tools' | 'scores' | 'logs' | 'config' | 'rubric' | 'codes' | 'ai'

export default function TracePage() {
  const { id } = useParams<{ id: string }>()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const isReviewMode = searchParams.get('from') === 'review'
  const [trace, setTrace] = useState<Trace | null>(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab] = useState<TabKey>('guide')
  const [reviewAction, setReviewAction] = useState<string | null>(null)
  // Blind labeling: machine verdicts stay hidden until a human labels the trace
  // (or reveals them), so the automated result can't anchor the human's judgment.
  const [revealed, setRevealed] = useState(false)

  useEffect(() => {
    if (!id) return
    setRevealed(false)
    getTrace(Number(id)).then(setTrace).finally(() => setLoading(false))
  }, [id])

  const handleReviewAction = async (status: 'reviewed' | 'flagged') => {
    if (!trace) return
    setReviewAction(status)
    await updateReviewStatus(trace.id, status).catch(() => {})
    navigate('/overview')
  }

  if (loading) return <div className="text-slate-400">Loading...</div>
  if (!trace) return <div className="text-red-500">Trace not found.</div>

  const toolHistory = (trace.hook_metrics as Record<string, unknown>)?.tool_call_history as Array<{tool: string; args: Record<string, string>; ts: number; seq: number}> | undefined
  const blind = (trace.human_pass === null || trace.human_pass === undefined) && !revealed
  const machineTabs: TabKey[] = ['scores', 'logs', 'ai']
  const tabs: TabKey[] = ['guide', 'products', 'sources', 'events', 'tools', 'scores', 'logs', 'config', 'codes', 'ai']
    .filter(t => !blind || !machineTabs.includes(t as TabKey)) as TabKey[]
  if (trace.case_type === 'shoppingcomp' || trace.case_type === 'trap') {
    tabs.push('rubric')
  }
  const activeTab: TabKey = tabs.includes(tab) ? tab : 'guide'

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Review Mode Banner */}
      {isReviewMode && (
        <div className="rounded-xl bg-blue-50 border border-blue-200 px-5 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-blue-500" style={{ fontSize: '18px' }}>rate_review</span>
            <span className="text-sm font-medium text-blue-700">Review Mode</span>
            <span className="text-xs text-blue-500">Review this trace, then mark it</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleReviewAction('reviewed')}
              disabled={reviewAction !== null}
              className="px-3 py-1.5 rounded-lg bg-emerald-600 text-white text-xs font-medium hover:bg-emerald-700 disabled:opacity-50"
            >
              {reviewAction === 'reviewed' ? 'Marking...' : '✓ Mark Reviewed'}
            </button>
            <button
              onClick={() => handleReviewAction('flagged')}
              disabled={reviewAction !== null}
              className="px-3 py-1.5 rounded-lg bg-red-600 text-white text-xs font-medium hover:bg-red-700 disabled:opacity-50"
            >
              {reviewAction === 'flagged' ? 'Flagging...' : '⚑ Flag'}
            </button>
            <Link to="/overview" className="px-3 py-1.5 rounded-lg border border-blue-200 text-xs text-blue-600 hover:bg-blue-100">
              Back to Queue
            </Link>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-3">
            <Link to={`/experiments/${trace.experiment_id}`} className="text-sm text-blue-600 hover:underline">
              Experiment #{trace.experiment_id}
            </Link>
            <span className="text-slate-300">/</span>
            <h1 className="text-xl font-bold text-slate-900">{trace.case_key}</h1>
            <span className="text-sm text-slate-400">Trial #{trace.trial_num}</span>
          </div>
          <div className="text-sm text-slate-500 mt-1 max-w-2xl">{trace.query}</div>
          <div className="flex items-center gap-3 mt-2 text-xs text-slate-400">
            <span className={`px-1.5 py-0.5 rounded font-medium ${
              trace.status === 'graded' ? 'bg-emerald-50 text-emerald-600'
                : trace.status === 'done' ? 'bg-blue-50 text-blue-600'
                : trace.status === 'running' ? 'bg-amber-50 text-amber-600'
                : 'bg-slate-100 text-slate-500'
            }`}>{trace.status}</span>
            {trace.review_status && trace.review_status !== 'pending' && (
              <span className={`px-1.5 py-0.5 rounded font-medium ${
                trace.review_status === 'reviewed' ? 'bg-emerald-50 text-emerald-600'
                  : 'bg-red-50 text-red-600'
              }`}>{trace.review_status}</span>
            )}
            <span className="px-1.5 py-0.5 rounded bg-slate-100">{trace.case_type}</span>
            {trace.model && <span className="font-mono">{trace.model}</span>}
            <span>{trace.duration_s.toFixed(1)}s</span>
            {trace.turn_count > 0 && <span>{trace.turn_count} turns</span>}
            <span>{trace.products.length} products</span>
            <span>{trace.sources.length} sources</span>
            {trace.grading_duration_s > 0 && (
              <span>grading: {trace.grading_duration_s.toFixed(1)}s</span>
            )}
          </div>
          {/* Token usage + prompt versions */}
          <div className="flex items-center gap-3 mt-1.5 text-[10px] text-slate-400 font-mono flex-wrap">
            {(trace.input_tokens > 0 || trace.output_tokens > 0) && (
              <span>
                tokens: {(trace.input_tokens / 1000).toFixed(1)}k in / {(trace.output_tokens / 1000).toFixed(1)}k out
              </span>
            )}
            {trace.prompt_version && <span>prompt: {trace.prompt_version}</span>}
            {trace.judge_prompt_version && <span>judge: {trace.judge_prompt_version}</span>}
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          {blind ? (
            <button
              onClick={() => setRevealed(true)}
              className="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-400 hover:text-slate-600"
            >
              <span className="material-symbols-outlined" style={{ fontSize: '20px' }}>visibility_off</span>
            </button>
          ) : (
            <ScoreBadge score={trace.final_score} pass={trace.final_pass} size="lg" />
          )}
          <HumanPassButton trace={trace} onUpdated={setTrace} />
          <SaveToDatasetButton trace={trace} />
          <BackflowButton trace={trace} />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-slate-200">
        {tabs.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === t
                ? 'border-emerald-600 text-emerald-700'
                : 'border-transparent text-slate-500 hover:text-slate-700'
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
            {t === 'products' && ` (${trace.products.length})`}
            {t === 'sources' && ` (${trace.sources.length})`}
            {t === 'events' && ` (${trace.events.length})`}
            {t === 'tools' && toolHistory && ` (${toolHistory.length})`}
            {t === 'logs' && trace.grading_log.length > 0 && ` (${trace.grading_log.length})`}
            {t === 'codes' && trace.open_codes && trace.open_codes.length > 0 && ` (${trace.open_codes.length})`}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main content */}
        <div className="lg:col-span-2">
          {activeTab === 'guide' && (
            <div className="rounded-xl bg-white border border-slate-200 p-6 prose prose-slate max-w-none">
              {trace.guide_text ? (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {trace.guide_text}
                </ReactMarkdown>
              ) : (
                <div className="text-slate-400 text-center py-8">No guide text generated.</div>
              )}
            </div>
          )}

          {activeTab === 'products' && (
            <div className="space-y-3">
              {trace.products.map((p, i) => {
                const product = p as Record<string, string>
                return (
                  <div key={i} className="rounded-xl bg-white border border-slate-200 p-4 flex gap-4">
                    {product.imageUrl && (
                      <img
                        src={product.imageUrl}
                        alt=""
                        className="w-16 h-16 rounded-lg object-cover bg-slate-100"
                      />
                    )}
                    <div>
                      <div className="font-medium text-slate-900">
                        {product.name || 'Unknown'}
                      </div>
                      <div className="text-sm text-slate-500">
                        {product.brand || ''} {product.price || ''}
                      </div>
                    </div>
                  </div>
                )
              })}
              {trace.products.length === 0 && (
                <div className="text-slate-400 text-center py-8">No products found.</div>
              )}
            </div>
          )}

          {activeTab === 'sources' && (
            <div className="space-y-2">
              {trace.sources.map((s, i) => {
                const source = s as Record<string, string>
                return (
                  <div key={i} className="rounded-lg bg-white border border-slate-200 p-3">
                    <a
                      href={source.url || '#'}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 hover:underline font-medium"
                    >
                      {source.title || source.url || 'Source'}
                    </a>
                    <div className="text-xs text-slate-400 mt-0.5">
                      {source.domain || ''}
                    </div>
                  </div>
                )
              })}
              {trace.sources.length === 0 && (
                <div className="text-slate-400 text-center py-8">No sources found.</div>
              )}
            </div>
          )}

          {activeTab === 'events' && <TraceTimeline events={trace.events} />}

          {activeTab === 'tools' && (
            <div className="rounded-xl bg-white border border-slate-200 overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-100 flex items-center justify-between">
                <h4 className="text-sm font-semibold text-slate-700">Tool Call History</h4>
                <span className="text-xs text-slate-400">{toolHistory?.length ?? 0} calls</span>
              </div>
              {toolHistory && toolHistory.length > 0 ? (
                <div className="divide-y divide-slate-100/80">
                  {toolHistory.map((call, i) => (
                    <div key={i} className="px-4 py-2.5 flex items-start gap-3 hover:bg-slate-50/50">
                      <span className="text-[10px] text-slate-400 tabular-nums w-5 pt-0.5 text-right shrink-0">
                        {call.seq}
                      </span>
                      <span className={`text-xs font-mono font-medium px-1.5 py-0.5 rounded shrink-0 ${
                        call.tool.includes('WebSearch') || call.tool.includes('brave') || call.tool.includes('exa')
                          ? 'bg-blue-50 text-blue-700'
                          : call.tool.includes('WebFetch') || call.tool.includes('firecrawl')
                          ? 'bg-emerald-50 text-emerald-700'
                          : call.tool.includes('serpapi')
                          ? 'bg-amber-50 text-amber-700'
                          : 'bg-slate-100 text-slate-600'
                      }`}>
                        {call.tool.replace('mcp__', '').replace('__', ':')}
                      </span>
                      <div className="flex-1 min-w-0">
                        {call.args.query && (
                          <div className="text-sm text-slate-700 truncate">{call.args.query}</div>
                        )}
                        {call.args.url && (
                          <div className="text-xs text-blue-600 truncate">{call.args.url}</div>
                        )}
                        {!call.args.query && !call.args.url && (
                          <div className="text-xs text-slate-400 italic">no args</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-8 text-center text-slate-400 text-sm">
                  No tool call history. Run a new trace to capture tool calls.
                </div>
              )}
            </div>
          )}

          {activeTab === 'scores' && <ScoresTab trace={trace} />}

          {activeTab === 'logs' && <LogsTab trace={trace} />}

          {activeTab === 'config' && <ConfigTab trace={trace} />}

          {activeTab === 'codes' && <CodesTab trace={trace} onUpdated={setTrace} />}

          {activeTab === 'ai' && <AIAnalysisTab trace={trace} />}

          {activeTab === 'rubric' && <RubricTab trace={trace} />}
        </div>

        {/* Score breakdown sidebar */}
        <div className="space-y-4">
          {!blind && (
            <GraderBreakdown
              l0={trace.l0_scores}
              l1={trace.l1_scores}
              l2={trace.l2_scores}
              compositeScores={trace.composite_scores}
              failureFunnel={trace.failure_funnel}
              gate={(trace as unknown as Record<string, unknown>).gate as { results: Record<string, boolean>; passed: boolean } | undefined}
            />
          )}

          {/* Human Annotation Panel */}
          {trace.composite_scores && Object.keys(trace.composite_scores).length > 0 && (
            <HumanAnnotationPanel trace={trace} onAnnotated={setTrace} />
          )}
        </div>
      </div>
    </div>
  )
}

/* ── Grading Logs Tab ── */
function LogsTab({ trace }: { trace: Trace }) {
  const logs = trace.grading_log || []

  if (logs.length === 0) {
    return (
      <div className="rounded-xl bg-white border border-slate-200 p-6 text-center text-slate-400">
        No grading logs available. Run or re-grade this trace to generate logs.
      </div>
    )
  }

  const codeSteps = logs.filter(l => l.category === 'code')
  const llmSteps = logs.filter(l => l.category === 'llm')
  const systemSteps = logs.filter(l => l.category === 'system')
  const totalEntry = systemSteps.find(s => s.step === 'total')
  const gateEntry = systemSteps.find(s => s.step === 'gate')
  const compositeEntry = systemSteps.find(s => s.step === 'composite')

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="rounded-xl bg-white border border-slate-200 p-4">
        <h4 className="text-sm font-semibold text-slate-700 mb-3">Grading Pipeline Log</h4>
        <div className="flex items-center gap-4 text-xs text-slate-500">
          <span>{codeSteps.length} code graders</span>
          <span>{llmSteps.length} LLM graders</span>
          {totalEntry?.duration_s != null && (
            <span>total: {totalEntry.duration_s.toFixed(2)}s</span>
          )}
          {gateEntry && (
            <span className={gateEntry.passed ? 'text-emerald-600' : 'text-red-600'}>
              gate: {gateEntry.passed ? 'PASS' : 'FAIL'}
            </span>
          )}
          {compositeEntry && (
            <span>composite: {compositeEntry.score?.toFixed(1)} ({compositeEntry.is_pass ? 'pass' : 'fail'})</span>
          )}
        </div>
      </div>

      {/* Timeline */}
      <div className="rounded-xl bg-white border border-slate-200 overflow-hidden">
        <div className="divide-y divide-slate-50">
          {logs.filter(l => l.category !== 'system' || l.step === 'gate').map((entry, i) => (
            <LogEntry key={i} entry={entry} />
          ))}
        </div>
      </div>

      {/* Gate Details */}
      {gateEntry?.results && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <h4 className="text-sm font-semibold text-slate-700 mb-2">Gate Checks</h4>
          <div className="flex flex-wrap gap-2">
            {Object.entries(gateEntry.results).map(([k, v]) => (
              <span
                key={k}
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  v ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-700'
                }`}
              >
                {v ? '✓' : '✗'} {k.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function LogEntry({ entry }: { entry: GradingLogEntry }) {
  const isError = entry.status === 'error'
  const isGate = entry.step === 'gate'

  return (
    <div className={`px-4 py-3 ${isError ? 'bg-red-50/50' : ''}`}>
      <div className="flex items-center gap-3">
        {/* Category badge */}
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
          entry.category === 'llm' ? 'bg-purple-50 text-purple-600'
            : entry.category === 'system' ? 'bg-slate-100 text-slate-500'
            : 'bg-blue-50 text-blue-600'
        }`}>
          {entry.category}
        </span>

        {/* Step name */}
        <span className="text-sm font-medium text-slate-700 flex-1">
          {entry.step.replace(/_/g, ' ')}
        </span>

        {/* Score */}
        {entry.score != null && (
          <span className={`text-sm font-bold tabular-nums ${
            entry.score >= 70 ? 'text-emerald-600' : entry.score >= 40 ? 'text-amber-600' : 'text-red-600'
          }`}>
            {entry.score.toFixed(0)}
          </span>
        )}

        {/* Gate pass/fail */}
        {isGate && (
          <span className={entry.passed ? 'text-emerald-600 text-sm font-bold' : 'text-red-600 text-sm font-bold'}>
            {entry.passed ? 'PASS' : 'FAIL'}
          </span>
        )}

        {/* Weight */}
        {entry.weight != null && (
          <span className="text-[10px] text-slate-400 tabular-nums">w={entry.weight.toFixed(2)}</span>
        )}

        {/* Duration */}
        {entry.duration_s != null && (
          <span className="text-[10px] text-slate-400 tabular-nums">{entry.duration_s.toFixed(3)}s</span>
        )}

        {/* Status indicator */}
        <span className={`w-2 h-2 rounded-full ${
          isError ? 'bg-red-500' : 'bg-emerald-500'
        }`} />

        {/* Revision badge */}
        {entry.revisions != null && entry.revisions > 1 && (
          <span className="text-[10px] text-amber-500 font-medium">revised ×{entry.revisions - 1}</span>
        )}
      </div>

      {/* Error message */}
      {isError && entry.error && (
        <div className="mt-1.5 text-xs text-red-600 font-mono bg-red-50 rounded p-2">
          {entry.error}
        </div>
      )}

      {/* Error types */}
      {entry.error_types && entry.error_types.length > 0 && (
        <div className="mt-1.5 flex gap-1">
          {entry.error_types.map((et, i) => (
            <span key={i} className="px-1.5 py-0.5 rounded bg-red-50 text-red-600 text-[10px]">{et}</span>
          ))}
        </div>
      )}

      {/* LLM reasoning preview */}
      {entry.reasoning_preview && (
        <div className="mt-1.5 text-[11px] text-slate-500 bg-slate-50 rounded p-2 line-clamp-3">
          {entry.reasoning_preview}
        </div>
      )}
    </div>
  )
}

/* ── Human PASS/FAIL Button ── */
function HumanPassButton({ trace, onUpdated }: { trace: Trace; onUpdated: (t: Trace) => void }) {
  const [saving, setSaving] = useState(false)

  const handleClick = async (passed: boolean) => {
    setSaving(true)
    try {
      await annotateHumanPass(trace.id, passed)
      onUpdated({ ...trace, human_pass: passed })
    } finally {
      setSaving(false)
    }
  }

  if (trace.human_pass !== null && trace.human_pass !== undefined) {
    const agrees = trace.human_pass === trace.final_pass
    return (
      <div className="flex items-center gap-1.5">
        <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
          trace.human_pass ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
        }`}>
          Human: {trace.human_pass ? 'PASS' : 'FAIL'}
        </span>
        {!agrees && (
          <span className="text-[10px] text-amber-600 font-medium">disagrees</span>
        )}
        <button
          onClick={() => handleClick(!trace.human_pass)}
          disabled={saving}
          className="text-[10px] text-slate-400 hover:text-blue-600"
        >
          flip
        </button>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-1">
      <button
        onClick={() => handleClick(true)}
        disabled={saving}
        className="px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-50 text-emerald-700 hover:bg-emerald-100 disabled:opacity-50"
      >
        PASS
      </button>
      <button
        onClick={() => handleClick(false)}
        disabled={saving}
        className="px-2 py-0.5 rounded text-[10px] font-medium bg-red-50 text-red-700 hover:bg-red-100 disabled:opacity-50"
      >
        FAIL
      </button>
    </div>
  )
}


/* ── Human Annotation Panel ── */
function HumanAnnotationPanel({ trace, onAnnotated }: { trace: Trace; onAnnotated: (t: Trace) => void }) {
  const [editingGrader, setEditingGrader] = useState<string | null>(null)
  const [humanScore, setHumanScore] = useState('')
  const [humanReasoning, setHumanReasoning] = useState('')
  const [saving, setSaving] = useState(false)

  const graderNames = Object.keys(trace.composite_scores)
  const humanScores = trace.human_scores || {}

  const handleSave = async () => {
    if (!editingGrader || !humanScore) return
    setSaving(true)
    try {
      const result = await annotateGrader(trace.id, editingGrader, Number(humanScore), humanReasoning)
      // Update local state
      const updated = {
        ...trace,
        human_scores: {
          ...trace.human_scores,
          [editingGrader]: { score: Number(humanScore), reasoning: humanReasoning },
        },
      }
      onAnnotated(updated)
      setEditingGrader(null)
      setHumanScore('')
      setHumanReasoning('')
      if (result.agreement) {
        // Could show a toast here
      }
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="rounded-xl bg-white border border-slate-200 p-4">
      <h4 className="text-sm font-semibold text-slate-700 mb-3">
        Human Calibration
      </h4>
      <p className="text-[11px] text-slate-400 mb-3">
        Score each grader dimension to calibrate LLM judges against human judgment.
      </p>

      <div className="space-y-2">
        {graderNames.map(name => {
          const llmData = trace.composite_scores[name]
          const humanData = humanScores[name]
          const isEditing = editingGrader === name

          return (
            <div key={name} className="rounded-lg border border-slate-100 p-2.5">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-600 flex-1 truncate">{name.replace(/_/g, ' ')}</span>
                {/* The LLM score shows only after the human scores this grader (blind labeling) */}
                {humanData ? (
                  <>
                    <span className="text-xs tabular-nums text-slate-500">LLM: {llmData.score.toFixed(0)}</span>
                    <span className={`text-xs tabular-nums font-medium ${
                      Math.abs(humanData.score - llmData.score) <= 15 ? 'text-emerald-600' : 'text-red-600'
                    }`}>
                      H: {humanData.score}
                    </span>
                    <span className="text-[10px] text-slate-400">
                      Δ{Math.abs(humanData.score - llmData.score).toFixed(0)}
                    </span>
                  </>
                ) : (
                  <button
                    onClick={() => { setEditingGrader(name); setHumanScore(''); setHumanReasoning('') }}
                    className="text-[10px] text-blue-600 hover:underline"
                  >
                    annotate
                  </button>
                )}
              </div>

              {isEditing && (
                <div className="mt-2 space-y-1.5">
                  <input
                    type="number"
                    min="0" max="100"
                    value={humanScore}
                    onChange={e => setHumanScore(e.target.value)}
                    placeholder="Score (0-100)"
                    className="w-full px-2 py-1 rounded border border-slate-200 text-xs"
                  />
                  <textarea
                    value={humanReasoning}
                    onChange={e => setHumanReasoning(e.target.value)}
                    placeholder="Reasoning (optional)"
                    rows={2}
                    className="w-full px-2 py-1 rounded border border-slate-200 text-xs resize-none"
                  />
                  <div className="flex gap-1">
                    <button
                      onClick={handleSave}
                      disabled={saving || !humanScore}
                      className="px-2 py-0.5 rounded bg-emerald-600 text-white text-[10px] font-medium disabled:opacity-50"
                    >
                      {saving ? 'Saving...' : 'Save'}
                    </button>
                    <button
                      onClick={() => setEditingGrader(null)}
                      className="px-2 py-0.5 rounded bg-slate-100 text-slate-500 text-[10px]"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Agreement summary */}
      {Object.keys(humanScores).length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-100">
          <div className="text-[11px] text-slate-500">
            {(() => {
              const diffs = graderNames
                .filter(n => humanScores[n])
                .map(n => Math.abs(humanScores[n].score - trace.composite_scores[n].score))
              const avgDiff = diffs.length > 0 ? diffs.reduce((a, b) => a + b, 0) / diffs.length : 0
              const aligned = diffs.filter(d => d <= 15).length
              return (
                <>
                  <span>Annotated: {Object.keys(humanScores).length}/{graderNames.length}</span>
                  <span className="mx-2">·</span>
                  <span>Avg Δ: {avgDiff.toFixed(1)}</span>
                  <span className="mx-2">·</span>
                  <span className={aligned === diffs.length ? 'text-emerald-600' : 'text-amber-600'}>
                    Aligned: {aligned}/{diffs.length}
                  </span>
                </>
              )
            })()}
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Scores Tab ── */
function ScoresTab({ trace }: { trace: Trace }) {
  const scores = trace.composite_scores
  if (!scores || Object.keys(scores).length === 0) {
    return (
      <div className="rounded-xl bg-white border border-slate-200 p-6 text-center text-slate-400">
        No composite scores available. This trace may use the legacy scoring system.
      </div>
    )
  }

  const entries = Object.entries(scores)
  const totalWeight = entries.reduce((s, [, g]) => s + g.weight, 0)
  const humanScores = trace.human_scores || {}

  return (
    <div className="space-y-4">
      {/* Score Table */}
      <div className="rounded-xl bg-white border border-slate-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-100">
              <th className="px-4 py-2 text-left text-xs font-medium text-slate-500">Grader</th>
              <th className="px-4 py-2 text-right text-xs font-medium text-slate-500">LLM Score</th>
              <th className="px-4 py-2 text-right text-xs font-medium text-slate-500">Human</th>
              <th className="px-4 py-2 text-right text-xs font-medium text-slate-500">Weight</th>
              <th className="px-4 py-2 text-right text-xs font-medium text-slate-500">Contrib</th>
              <th className="px-4 py-2 text-center text-xs font-medium text-slate-500">Type</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-50">
            {entries.map(([name, g]) => {
              const human = humanScores[name]
              const diff = human ? Math.abs(human.score - g.score) : null
              return (
                <tr key={name} className="hover:bg-slate-25">
                  <td className="px-4 py-2.5 font-medium text-slate-700">
                    {name.replace(/_/g, ' ')}
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums">
                    <span className={g.score >= 70 ? 'text-emerald-600' : g.score >= 40 ? 'text-amber-600' : 'text-red-600'}>
                      {g.score.toFixed(0)}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums">
                    {human ? (
                      <span className={diff != null && diff <= 15 ? 'text-emerald-600' : 'text-red-600'}>
                        {human.score.toFixed(0)}
                        <span className="text-[10px] text-slate-400 ml-1">Δ{diff?.toFixed(0)}</span>
                      </span>
                    ) : (
                      <span className="text-slate-300">—</span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-right text-slate-500 tabular-nums">
                    {g.weight.toFixed(2)}
                  </td>
                  <td className="px-4 py-2.5 text-right text-slate-600 tabular-nums font-medium">
                    {(g.score * g.weight / (totalWeight || 1)).toFixed(1)}
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                      g.category === 'llm' ? 'bg-purple-50 text-purple-600' : 'bg-slate-100 text-slate-500'
                    }`}>
                      {g.category}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Error Types */}
      {trace.error_types.length > 0 && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <h4 className="text-sm font-semibold text-slate-700 mb-2">Error Classifications</h4>
          <div className="flex flex-wrap gap-1.5">
            {trace.error_types.map((et, i) => (
              <span key={i} className="px-2 py-0.5 rounded bg-red-50 text-red-700 text-xs">
                {et}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Failure Funnel */}
      <FailureFunnel funnel={trace.failure_funnel} />

      {/* Grader Details */}
      {entries.filter(([, g]) => g.details && Object.keys(g.details).length > 0 && !g.details.skipped).map(([name, g]) => (
        <div key={name} className="rounded-xl bg-white border border-slate-200 p-4">
          <div className="flex items-center gap-2 mb-2">
            <h4 className="text-sm font-semibold text-slate-700">{name.replace(/_/g, ' ')} — Details</h4>
            {g.details.revision_count != null && (g.details.revision_count as number) > 1 && (
              <span className="px-1.5 py-0.5 rounded bg-amber-50 text-amber-600 text-[10px] font-medium">
                revised ×{(g.details.revision_count as number) - 1}
              </span>
            )}
          </div>
          <div className="text-xs text-slate-600 space-y-1 font-mono">
            {Object.entries(g.details).filter(([k]) => !['reasoning', 'revisions', 'revision_count'].includes(k)).map(([k, v]) => (
              <div key={k} className="flex gap-2">
                <span className="text-slate-400 min-w-[120px]">{k}:</span>
                <span>{typeof v === 'object' ? JSON.stringify(v).slice(0, 100) : String(v)}</span>
              </div>
            ))}
            {g.details.reasoning ? (
              <div className="mt-2 text-slate-500 whitespace-pre-wrap font-sans text-xs">
                {String(g.details.reasoning)}
              </div>
            ) : null}
          </div>

          {/* Revision History */}
          {Array.isArray(g.details.revisions) && (g.details.revisions as Array<Record<string, unknown>>).length > 0 && (
            <div className="mt-3 border-t border-slate-100 pt-3">
              <h5 className="text-[11px] font-semibold text-slate-500 mb-2">Score Revision History</h5>
              <div className="space-y-2">
                {(g.details.revisions as Array<{ score: number; reasoning: string }>).map((rev, ri) => (
                  <div key={ri} className="rounded-lg bg-slate-50 p-2.5 border border-slate-100">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] text-slate-400 font-medium">v{ri + 1}</span>
                      <span className={`text-xs font-bold tabular-nums ${
                        rev.score >= 70 ? 'text-emerald-600' : rev.score >= 40 ? 'text-amber-600' : 'text-red-600'
                      }`}>
                        {rev.score.toFixed(0)}
                      </span>
                      <span className="material-symbols-outlined text-slate-300" style={{ fontSize: '12px' }}>
                        arrow_forward
                      </span>
                      <span className="text-[10px] text-slate-400">revised to {g.score.toFixed(0)}</span>
                    </div>
                    {rev.reasoning && (
                      <div className="text-[11px] text-slate-500 whitespace-pre-wrap">
                        {rev.reasoning}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

/* ── Config Tab ── */
function MetricCell({ label, value, unit, warn }: { label: string; value: string | number; unit?: string; warn?: boolean }) {
  return (
    <div>
      <div className="text-[10px] text-slate-400">{label}</div>
      <div className={`text-sm font-medium tabular-nums ${warn ? 'text-amber-600' : 'text-slate-700'}`}>
        {value}{unit && <span className="text-[10px] text-slate-400 ml-0.5">{unit}</span>}
      </div>
    </div>
  )
}

function ProcessMetrics({ hookMetrics, trace }: { hookMetrics: Record<string, unknown>; trace: Trace }) {
  const searchCount = (hookMetrics.search_count as number) || 0
  const productCount = (hookMetrics.product_count as number) || trace.products?.length || 0
  const entityCount = (hookMetrics.entity_count as number) || 0
  const dimsExplored = (hookMetrics.dimensions_explored as number) || 0
  const sourceDomainCount = (hookMetrics.source_domain_count as number) || 0
  const toolCallCount = (hookMetrics.tool_call_count as number) || 0
  const failureCount = (hookMetrics.failure_count as number) || 0
  const avgBatch = (hookMetrics.avg_batch_size as number) || 0
  const maxBatch = (hookMetrics.max_batch_size as number) || 0
  const batchCount = (hookMetrics.batch_count as number) || 0
  const reqTotal = (hookMetrics.requirements_total as number) || 0
  const reqCovered = (hookMetrics.requirements_covered as number) || 0

  // Latency
  const ttfs = (hookMetrics.time_to_first_search as number) || 0
  const ttfp = (hookMetrics.time_to_first_product as number) || 0
  const ttft = (hookMetrics.time_to_first_text as number) || 0
  const duration = trace.duration_s || 0
  const tokensPerSec = duration > 0 ? (trace.output_tokens / duration) : 0

  // Derived efficiency
  const pc = Math.max(productCount, 1)
  const tokensPerProduct = trace.output_tokens > 0 ? Math.round(trace.output_tokens / pc) : 0
  const searchesPerProduct = searchCount > 0 ? (searchCount / pc).toFixed(1) : '—'
  const toolsPerProduct = toolCallCount > 0 ? (toolCallCount / pc).toFixed(1) : '—'

  // Dimension coverage breakdown
  const dimCoverage = (hookMetrics.dimension_coverage as Record<string, number>) || {}

  return (
    <div className="rounded-xl bg-white border border-slate-200 p-4">
      <h4 className="text-sm font-semibold text-slate-700 mb-3">Process Metrics</h4>
      <div className="grid grid-cols-3 gap-4">
        {/* Latency column */}
        <div className="space-y-2.5">
          <div className="text-[10px] font-medium text-slate-500 uppercase tracking-wide">Latency</div>
          <MetricCell label="First search" value={ttfs > 0 ? ttfs.toFixed(1) : '—'} unit="s" />
          <MetricCell label="First product" value={ttfp > 0 ? ttfp.toFixed(1) : '—'} unit="s" />
          <MetricCell label="First text" value={ttft > 0 ? ttft.toFixed(1) : '—'} unit="s" />
          <MetricCell label="Tokens/sec" value={tokensPerSec > 0 ? tokensPerSec.toFixed(1) : '—'} />
        </div>

        {/* Efficiency column */}
        <div className="space-y-2.5">
          <div className="text-[10px] font-medium text-slate-500 uppercase tracking-wide">Efficiency</div>
          <MetricCell label="Tokens/product" value={tokensPerProduct > 0 ? tokensPerProduct.toLocaleString() : '—'} warn={tokensPerProduct > 5000} />
          <MetricCell label="Searches/product" value={searchesPerProduct} warn={Number(searchesPerProduct) > 5} />
          <MetricCell label="Tools/product" value={toolsPerProduct} />
          <MetricCell label="Error rate" value={toolCallCount > 0 ? ((failureCount / toolCallCount) * 100).toFixed(0) : '0'} unit="%" warn={failureCount > 0} />
        </div>

        {/* Research Depth column */}
        <div className="space-y-2.5">
          <div className="text-[10px] font-medium text-slate-500 uppercase tracking-wide">Research Depth</div>
          <MetricCell label="Dimensions" value={`${dimsExplored}/6`} warn={dimsExplored < 4} />
          <MetricCell label="Entities" value={entityCount} warn={entityCount < 2} />
          <MetricCell label="Source domains" value={sourceDomainCount} warn={sourceDomainCount < 3} />
          <MetricCell label="Requirements" value={reqTotal > 0 ? `${reqCovered}/${reqTotal}` : '—'} />
        </div>
      </div>

      {/* Dimension coverage mini-bar */}
      {Object.keys(dimCoverage).length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-100">
          <div className="text-[10px] text-slate-400 mb-1.5">Dimension Coverage</div>
          <div className="flex gap-1.5 flex-wrap">
            {Object.entries(dimCoverage).map(([dim, count]) => (
              <span key={dim} className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                (count as number) > 0 ? 'bg-emerald-50 text-emerald-700' : 'bg-slate-50 text-slate-400'
              }`}>
                {dim}: {count as number}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Parallel execution mini-bar */}
      {batchCount > 0 && (
        <div className="mt-2 pt-2 border-t border-slate-100 flex gap-4 text-[10px] text-slate-500">
          <span>Parallel batches: {batchCount}</span>
          <span>Avg batch size: {avgBatch}</span>
          <span>Max batch: {maxBatch}</span>
        </div>
      )}
    </div>
  )
}

function ConfigTab({ trace }: { trace: Trace }) {
  const [expandedSection, setExpandedSection] = useState<string | null>(null)
  const toggle = (section: string) =>
    setExpandedSection(prev => prev === section ? null : section)

  const hookMetrics = trace.hook_metrics as Record<string, unknown>
  const cacheRead = (hookMetrics?.cache_read_tokens as number) || 0
  const cacheWrite = (hookMetrics?.cache_write_tokens as number) || 0
  const totalInput = trace.input_tokens + cacheRead + cacheWrite
  const cachePct = totalInput > 0 ? (cacheRead / totalInput * 100) : 0

  return (
    <div className="space-y-4">
      {/* Execution Summary */}
      <div className="rounded-xl bg-white border border-slate-200 p-4">
        <h4 className="text-sm font-semibold text-slate-700 mb-3">Execution Summary</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div className="text-[10px] text-slate-400">Model</div>
            <div className="text-sm font-medium text-slate-700 font-mono">{trace.model || '—'}</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400">Duration</div>
            <div className="text-sm font-medium text-slate-700">{trace.duration_s.toFixed(1)}s</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400">Turns</div>
            <div className="text-sm font-medium text-slate-700">{trace.turn_count || '—'}</div>
          </div>
          <div>
            <div className="text-[10px] text-slate-400">Status</div>
            <div className={`text-sm font-medium ${
              trace.status === 'graded' ? 'text-emerald-600'
                : trace.status === 'done' ? 'text-blue-600'
                : trace.status === 'running' ? 'text-amber-600'
                : 'text-slate-500'
            }`}>{trace.status}</div>
          </div>
        </div>
      </div>

      {/* Token Usage */}
      {(trace.input_tokens > 0 || trace.output_tokens > 0) && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <h4 className="text-sm font-semibold text-slate-700 mb-3">Token Usage</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-[10px] text-slate-400">Input</div>
              <div className="text-sm font-medium text-slate-700 tabular-nums">
                {trace.input_tokens.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400">Output</div>
              <div className="text-sm font-medium text-slate-700 tabular-nums">
                {trace.output_tokens.toLocaleString()}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400">Cache Read</div>
              <div className="text-sm font-medium text-slate-700 tabular-nums">
                {cacheRead.toLocaleString()}
                {cachePct > 0 && (
                  <span className="text-[10px] text-emerald-500 ml-1">
                    ({cachePct.toFixed(0)}%)
                  </span>
                )}
              </div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400">Cache Write</div>
              <div className="text-sm font-medium text-slate-700 tabular-nums">
                {cacheWrite.toLocaleString()}
              </div>
            </div>
          </div>
          {/* Token bar visualization */}
          <div className="mt-3 flex h-2 rounded-full overflow-hidden bg-slate-100">
            {totalInput > 0 && (
              <>
                <div
                  className="bg-blue-400"
                  style={{ width: `${(trace.input_tokens / (totalInput + trace.output_tokens)) * 100}%` }}
                  title={`Input: ${trace.input_tokens}`}
                />
                <div
                  className="bg-emerald-400"
                  style={{ width: `${(cacheRead / (totalInput + trace.output_tokens)) * 100}%` }}
                  title={`Cache read: ${cacheRead}`}
                />
                <div
                  className="bg-amber-400"
                  style={{ width: `${(cacheWrite / (totalInput + trace.output_tokens)) * 100}%` }}
                  title={`Cache write: ${cacheWrite}`}
                />
                <div
                  className="bg-purple-400"
                  style={{ width: `${(trace.output_tokens / (totalInput + trace.output_tokens)) * 100}%` }}
                  title={`Output: ${trace.output_tokens}`}
                />
              </>
            )}
          </div>
          <div className="mt-1.5 flex gap-3 text-[9px] text-slate-400">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-400" /> Input</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-400" /> Cache Read</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-400" /> Cache Write</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-purple-400" /> Output</span>
          </div>
        </div>
      )}

      {/* Process Metrics */}
      {hookMetrics?.search_count != null && (
        <ProcessMetrics hookMetrics={hookMetrics} trace={trace} />
      )}

      {/* User Message */}
      <div className="rounded-xl bg-white border border-slate-200 p-4">
        <h4 className="text-sm font-semibold text-slate-700 mb-2">User Message</h4>
        <div className="text-sm text-slate-600 bg-slate-50 rounded-lg p-3 font-mono whitespace-pre-wrap">
          {trace.query}
        </div>
      </div>

      {/* Tool Names */}
      {trace.tool_names && trace.tool_names.length > 0 && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <h4 className="text-sm font-semibold text-slate-700 mb-2">
            Available Tools ({trace.tool_names.length})
          </h4>
          <div className="flex flex-wrap gap-1.5">
            {trace.tool_names.map((tool, i) => (
              <span key={i} className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 text-[11px] font-mono">
                {tool}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* System Prompt */}
      {trace.system_prompt && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <button
            onClick={() => toggle('system_prompt')}
            className="w-full flex items-center justify-between"
          >
            <h4 className="text-sm font-semibold text-slate-700">
              System Prompt
              <span className="ml-2 text-[10px] font-normal text-slate-400">
                {(trace.system_prompt.length / 1000).toFixed(1)}k chars
              </span>
            </h4>
            <span className="material-symbols-outlined text-slate-400" style={{ fontSize: '18px' }}>
              {expandedSection === 'system_prompt' ? 'expand_less' : 'expand_more'}
            </span>
          </button>
          {expandedSection === 'system_prompt' && (
            <pre className="mt-3 text-[11px] text-slate-600 bg-slate-50 rounded-lg p-3 overflow-x-auto max-h-[600px] overflow-y-auto whitespace-pre-wrap">
              {trace.system_prompt}
            </pre>
          )}
        </div>
      )}

      {/* Judge Prompts */}
      {trace.judge_prompts && Object.keys(trace.judge_prompts).length > 0 && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <button
            onClick={() => toggle('judge_prompts')}
            className="w-full flex items-center justify-between"
          >
            <h4 className="text-sm font-semibold text-slate-700">
              Judge Prompts ({Object.keys(trace.judge_prompts).length} graders)
            </h4>
            <span className="material-symbols-outlined text-slate-400" style={{ fontSize: '18px' }}>
              {expandedSection === 'judge_prompts' ? 'expand_less' : 'expand_more'}
            </span>
          </button>
          {expandedSection === 'judge_prompts' && (
            <div className="mt-3 space-y-3">
              {Object.entries(trace.judge_prompts).map(([name, prompt]) => (
                <div key={name} className="rounded-lg border border-slate-100 p-3">
                  <div className="text-xs font-medium text-purple-600 mb-1">
                    {name.replace(/_/g, ' ')}
                  </div>
                  <pre className="text-[11px] text-slate-600 bg-slate-50 rounded p-2 overflow-x-auto max-h-[300px] overflow-y-auto whitespace-pre-wrap">
                    {typeof prompt === 'string' ? prompt : JSON.stringify(prompt, null, 2)}
                  </pre>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Prompt Versions */}
      {(trace.prompt_version || trace.judge_prompt_version) && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <h4 className="text-sm font-semibold text-slate-700 mb-2">Version Hashes</h4>
          <div className="text-xs text-slate-500 font-mono space-y-1">
            {trace.prompt_version && <div>prompt: {trace.prompt_version}</div>}
            {trace.judge_prompt_version && <div>judge: {trace.judge_prompt_version}</div>}
          </div>
        </div>
      )}
    </div>
  )
}

/* ── Open Codes Tab ── */
function CodesTab({ trace, onUpdated }: { trace: Trace; onUpdated: (t: Trace) => void }) {
  const [newCode, setNewCode] = useState('')
  const [saving, setSaving] = useState(false)
  const codes = trace.open_codes || []

  // Common code suggestions based on eval domain
  const suggestions = [
    'search-timeout', 'search-irrelevant', 'search-insufficient',
    'price-missing', 'price-wrong', 'price-outdated',
    'product-hallucinated', 'product-wrong-category', 'product-duplicate',
    'source-dead-link', 'source-low-quality', 'source-missing',
    'format-no-table', 'format-no-citations', 'format-too-short',
    'guide-missing-comparison', 'guide-no-recommendation', 'guide-biased',
    'clarification-unnecessary', 'clarification-missing',
  ].filter(s => !codes.includes(s))

  const handleAdd = async (code: string) => {
    if (!code.trim() || codes.includes(code.trim())) return
    setSaving(true)
    try {
      const result = await updateOpenCodes(trace.id, [code.trim()])
      onUpdated({ ...trace, open_codes: result.open_codes })
      setNewCode('')
    } finally {
      setSaving(false)
    }
  }

  const handleRemove = async (code: string) => {
    setSaving(true)
    try {
      const result = await updateOpenCodes(trace.id, [], [code])
      onUpdated({ ...trace, open_codes: result.open_codes })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-4">
      {/* Current codes */}
      <div className="rounded-xl bg-white border border-slate-200 p-4">
        <h4 className="text-sm font-semibold text-slate-700 mb-3">
          Open Codes
          <span className="text-[10px] font-normal text-slate-400 ml-2">
            qualitative failure labels for pattern analysis
          </span>
        </h4>

        {codes.length > 0 ? (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {codes.map(code => (
              <span
                key={code}
                className="group flex items-center gap-1 px-2.5 py-1 rounded-lg bg-violet-50 text-violet-700 text-xs font-medium"
              >
                {code}
                <button
                  onClick={() => handleRemove(code)}
                  disabled={saving}
                  className="opacity-0 group-hover:opacity-100 transition-opacity text-violet-400 hover:text-red-500"
                >
                  <span className="material-symbols-outlined" style={{ fontSize: '14px' }}>close</span>
                </button>
              </span>
            ))}
          </div>
        ) : (
          <div className="text-xs text-slate-400 mb-3">
            No codes applied yet. Add codes to tag failure patterns.
          </div>
        )}

        {/* Add code input */}
        <div className="flex gap-2">
          <input
            type="text"
            value={newCode}
            onChange={e => setNewCode(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleAdd(newCode) }}
            placeholder="Type a code (e.g. price-missing)"
            className="flex-1 px-3 py-1.5 rounded-lg border border-slate-200 text-sm"
            disabled={saving}
          />
          <button
            onClick={() => handleAdd(newCode)}
            disabled={saving || !newCode.trim()}
            className="px-3 py-1.5 rounded-lg bg-violet-600 text-white text-sm font-medium disabled:opacity-50"
          >
            Add
          </button>
        </div>
      </div>

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <h4 className="text-sm font-semibold text-slate-700 mb-2">Suggested Codes</h4>
          <div className="flex flex-wrap gap-1.5">
            {suggestions.map(code => (
              <button
                key={code}
                onClick={() => handleAdd(code)}
                disabled={saving}
                className="px-2 py-0.5 rounded-lg border border-dashed border-slate-300 text-xs text-slate-500 hover:border-violet-400 hover:text-violet-600 hover:bg-violet-50 transition-colors disabled:opacity-50"
              >
                + {code}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Auto-classified errors */}
      <ClassifiedErrorsSection traceId={trace.id} />

      {/* Grouping hint */}
      <div className="text-[11px] text-slate-400 px-1">
        Codes are auto-grouped by prefix (e.g. "search-*") in the Overview coding analysis.
        Use consistent prefixes for effective axial coding.
      </div>
    </div>
  )
}

function ClassifiedErrorsSection({ traceId }: { traceId: number }) {
  const [errors, setErrors] = useState<Array<{ code: string; name: string; stage: string; severity: string }>>([])
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    getClassifiedErrors(traceId)
      .then(r => setErrors(r.errors))
      .catch(() => {})
      .finally(() => setLoaded(true))
  }, [traceId])

  if (!loaded || errors.length === 0) return null

  return (
    <div className="rounded-xl bg-white border border-slate-200 p-4">
      <h4 className="text-sm font-semibold text-slate-700 mb-3">
        Auto-Classified Errors
        <span className="text-[10px] font-normal text-slate-400 ml-2">
          from error taxonomy
        </span>
      </h4>
      <div className="space-y-1.5">
        {errors.map((e, i) => (
          <div key={i} className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-slate-50">
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
              e.severity === 'critical' ? 'bg-red-100 text-red-700'
                : e.severity === 'high' ? 'bg-amber-100 text-amber-700'
                : 'bg-slate-100 text-slate-600'
            }`}>
              {e.severity}
            </span>
            <span className="font-mono text-[11px] text-slate-500">{e.code}</span>
            <span className="text-sm text-slate-700">{e.name}</span>
            <span className="ml-auto text-[10px] text-slate-400">{e.stage}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── AI Analysis Tab ── */
function AIAnalysisTab({ trace }: { trace: Trace }) {
  const [question, setQuestion] = useState('')
  const [requestId, setRequestId] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Preset questions
  const presets = [
    'Why did this trace fail? What went wrong?',
    'What could be improved in the search strategy?',
    'Are the product recommendations well-supported by sources?',
    'Is the buyer guide comprehensive and accurate?',
  ]

  const handleSubmit = useCallback(async (q: string) => {
    if (!q.trim()) return
    setSubmitting(true)
    setResult(null)
    try {
      const res = await startTraceAnalysis([trace.id], q)
      setRequestId(res.request_id)
    } catch (e) {
      setResult({ status: 'error', result: null, error: String(e) })
    } finally {
      setSubmitting(false)
    }
  }, [trace.id])

  // Poll for results
  useEffect(() => {
    if (!requestId) return
    const poll = async () => {
      try {
        const res = await getAnalysisResult(requestId)
        if (res.status !== 'running') {
          setResult(res)
          if (pollRef.current) clearInterval(pollRef.current)
        }
      } catch {
        // keep polling
      }
    }
    poll() // immediate first check
    pollRef.current = setInterval(poll, 2000)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [requestId])

  return (
    <div className="space-y-4">
      {/* Question input */}
      <div className="rounded-xl bg-white border border-slate-200 p-4">
        <h4 className="text-sm font-semibold text-slate-700 mb-3">
          AI Analysis
          <span className="text-[10px] font-normal text-slate-400 ml-2">
            ask Claude to analyze this trace
          </span>
        </h4>

        {/* Presets */}
        <div className="flex flex-wrap gap-1.5 mb-3">
          {presets.map((p, i) => (
            <button
              key={i}
              onClick={() => { setQuestion(p); handleSubmit(p) }}
              disabled={submitting || (result?.status === 'running')}
              className="px-2.5 py-1 rounded-lg border border-slate-200 text-xs text-slate-600 hover:border-blue-300 hover:text-blue-600 hover:bg-blue-50 transition-colors disabled:opacity-50"
            >
              {p}
            </button>
          ))}
        </div>

        {/* Custom question */}
        <div className="flex gap-2">
          <input
            type="text"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleSubmit(question) }}
            placeholder="Ask a custom question about this trace..."
            className="flex-1 px-3 py-1.5 rounded-lg border border-slate-200 text-sm"
            disabled={submitting}
          />
          <button
            onClick={() => handleSubmit(question)}
            disabled={submitting || !question.trim() || result?.status === 'running'}
            className="px-3 py-1.5 rounded-lg bg-blue-600 text-white text-sm font-medium disabled:opacity-50"
          >
            {submitting ? 'Starting...' : 'Analyze'}
          </button>
        </div>
      </div>

      {/* Result */}
      {result?.status === 'running' && (
        <div className="rounded-xl bg-white border border-slate-200 p-6 text-center">
          <span className="material-symbols-outlined animate-spin text-blue-500 block mb-2" style={{ fontSize: '24px' }}>
            progress_activity
          </span>
          <div className="text-sm text-slate-500">Analyzing trace...</div>
        </div>
      )}

      {result?.status === 'done' && result.result && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-slate-700">Analysis Result</h4>
            {result.trace_count && (
              <span className="text-[10px] text-slate-400">{result.trace_count} trace(s) analyzed</span>
            )}
          </div>
          <div className="prose prose-slate prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {result.result}
            </ReactMarkdown>
          </div>
        </div>
      )}

      {result?.status === 'error' && (
        <div className="rounded-xl bg-red-50 border border-red-200 p-4">
          <div className="text-sm text-red-700 font-medium">Analysis failed</div>
          <div className="text-xs text-red-600 mt-1">{result.error}</div>
        </div>
      )}
    </div>
  )
}

function RubricTab({ trace }: { trace: Trace }) {
  const rubricCompliance = trace.composite_scores?.rubric_compliance
  const rubricCoverage = trace.composite_scores?.rubric_coverage
  const trapDetection = trace.composite_scores?.trap_detection

  return (
    <div className="space-y-4">
      {rubricCoverage && !rubricCoverage.details?.skipped && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <h4 className="text-sm font-semibold text-slate-700 mb-2">
            Rubric Keyword Coverage — {rubricCoverage.score.toFixed(0)}/100
          </h4>
          <div className="text-xs text-slate-500 mb-2">
            {(rubricCoverage.details?.hits as number) || 0} / {(rubricCoverage.details?.total as number) || 0} keywords found
          </div>
          {rubricCoverage.details?.miss_examples ? (
            <div className="mt-2">
              <div className="text-xs font-medium text-red-600 mb-1">Missing keywords:</div>
              <div className="flex flex-wrap gap-1">
                {(rubricCoverage.details.miss_examples as string[]).map((kw, i) => (
                  <span key={i} className="px-1.5 py-0.5 rounded bg-red-50 text-red-600 text-[10px]">{kw}</span>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      )}

      {rubricCompliance && !rubricCompliance.details?.skipped && (
        <div className="rounded-xl bg-white border border-slate-200 p-4">
          <h4 className="text-sm font-semibold text-slate-700 mb-2">
            Rubric Compliance (LLM Judge) — {rubricCompliance.score.toFixed(0)}/100
          </h4>
          {rubricCompliance.details?.reasoning ? (
            <div className="text-xs text-slate-600 whitespace-pre-wrap bg-slate-50 rounded-lg p-3 mt-2">
              {String(rubricCompliance.details.reasoning)}
            </div>
          ) : null}
        </div>
      )}

      {trapDetection && !trapDetection.details?.skipped && (
        <div className="rounded-xl bg-amber-50 border border-amber-200 p-4">
          <h4 className="text-sm font-semibold text-amber-800 mb-2">
            Trap Detection — {trapDetection.score.toFixed(0)}/100
          </h4>
          {trapDetection.details?.reasoning ? (
            <div className="text-xs text-amber-700 whitespace-pre-wrap bg-amber-100/50 rounded-lg p-3 mt-2">
              {String(trapDetection.details.reasoning)}
            </div>
          ) : null}
        </div>
      )}

      {!rubricCoverage && !rubricCompliance && !trapDetection && (
        <div className="text-slate-400 text-center py-8">
          No rubric data available for this trace.
        </div>
      )}
    </div>
  )
}


// ── Save to Dataset (Trace Backflow) ──────────────────────────────────

function SaveToDatasetButton({ trace }: { trace: Trace }) {
  const [open, setOpen] = useState(false)
  const [verdict, setVerdict] = useState<string>('')
  const [saving, setSaving] = useState(false)
  const [result, setResult] = useState<string>('')

  const handleSave = async (v: string) => {
    setSaving(true)
    setVerdict(v)
    try {
      const { traceToTestCase } = await import('../lib/api.ts')
      const res = await traceToTestCase(trace.id, v)
      setResult(`✓ Added to "${res.dataset_name}" as ${res.case_key}`)
      setTimeout(() => { setOpen(false); setResult(''); setVerdict('') }, 2500)
    } catch (e) {
      setResult(`Error: ${e instanceof Error ? e.message : 'failed'}`)
    } finally {
      setSaving(false)
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="px-2 py-1 rounded text-[10px] font-medium bg-blue-50 text-blue-600 hover:bg-blue-100"
      >
        Add to Test Set
      </button>
    )
  }

  if (result) {
    return (
      <span className={`text-[10px] font-medium ${result.startsWith('✓') ? 'text-emerald-600' : 'text-red-500'}`}>
        {result}
      </span>
    )
  }

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] text-slate-500">Verdict:</span>
      <button
        onClick={() => handleSave('pass')}
        disabled={saving}
        className="px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-50 text-emerald-700 hover:bg-emerald-100 disabled:opacity-50"
      >
        {saving && verdict === 'pass' ? '...' : 'Pass'}
      </button>
      <button
        onClick={() => handleSave('fail')}
        disabled={saving}
        className="px-2 py-0.5 rounded text-[10px] font-medium bg-red-50 text-red-700 hover:bg-red-100 disabled:opacity-50"
      >
        {saving && verdict === 'fail' ? '...' : 'Fail'}
      </button>
      <button
        onClick={() => handleSave('')}
        disabled={saving}
        className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600 hover:bg-slate-200 disabled:opacity-50"
      >
        {saving && verdict === '' ? '...' : 'Skip'}
      </button>
      <button onClick={() => setOpen(false)} className="text-[10px] text-slate-400 hover:text-slate-600 ml-1">✕</button>
    </div>
  )
}

function BackflowButton({ trace }: { trace: Trace }) {
  const [open, setOpen] = useState(false)
  const [datasets, setDatasets] = useState<Array<{ id: number; name: string }>>([])
  const [selectedDs, setSelectedDs] = useState<number | null>(null)
  const [saving, setSaving] = useState(false)
  const [result, setResult] = useState('')

  const handleOpen = async () => {
    setOpen(true)
    const ds = await listDatasets().catch(() => [])
    setDatasets(ds)
    if (ds.length > 0) setSelectedDs(ds[0].id)
  }

  const handleBackflow = async () => {
    if (!selectedDs) return
    setSaving(true)
    try {
      const res = await backflowTrace(selectedDs, {
        query: trace.query,
        guide_text: trace.guide_text || '',
        products: trace.products,
        sources: trace.sources,
        events: trace.events || [],
        hook_metrics: trace.hook_metrics || {},
        trace_id: trace.id,
      })
      setResult(`✓ Saved as ${res.case_key}`)
      setTimeout(() => { setOpen(false); setResult('') }, 2500)
    } catch (e) {
      setResult(`Error: ${e instanceof Error ? e.message : 'failed'}`)
    } finally {
      setSaving(false)
    }
  }

  if (result) return <span className={`text-[10px] font-medium ${result.startsWith('✓') ? 'text-emerald-600' : 'text-red-500'}`}>{result}</span>

  if (!open) {
    return (
      <button onClick={handleOpen} className="px-2 py-1 rounded text-[10px] font-medium bg-violet-50 text-violet-600 hover:bg-violet-100">
        Backflow to Dataset
      </button>
    )
  }

  return (
    <div className="flex items-center gap-1.5">
      <select value={selectedDs ?? ''} onChange={e => setSelectedDs(Number(e.target.value))} className="text-[10px] border border-slate-200 rounded px-1.5 py-0.5">
        {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
      </select>
      <button onClick={handleBackflow} disabled={saving || !selectedDs} className="px-2 py-0.5 rounded text-[10px] font-medium bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-50">
        {saving ? '...' : 'Save'}
      </button>
      <button onClick={() => setOpen(false)} className="text-[10px] text-slate-400 hover:text-slate-600">✕</button>
    </div>
  )
}
