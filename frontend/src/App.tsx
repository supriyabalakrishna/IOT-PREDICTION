import { useCallback, useEffect, useState } from "react";
import {
  Activity, AlertTriangle, BarChart3, Check, ChevronRight, CircleGauge,
  Database, Gauge, History, Layers3, LockKeyhole, Play, Radio, RotateCcw,
  Square, Pause, Settings2, ShieldCheck, Thermometer, Wifi, WifiOff,
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip,
  XAxis, YAxis,
} from "recharts";
import { api, type EventRecord, type Prediction } from "./api";

type Page = "Overview" | "Machines" | "Predictions" | "Model lab" | "Reliability" | "Replay" | "Events" | "Dataset";
type ReliabilityStatus = {
  status: string;
  model_version: string | null;
  prediction_count: number;
  decision_counts: Record<string, number>;
  review_required_count: number;
  human_reviewed_count: number;
  drift: { status: string; window_size: number; psi_max: number | null; threshold?: number };
};
type CrossLoadEvaluation = {
  status: string;
  reason?: string;
  protocol?: string;
  folds: Array<{
    held_out_load_nm: number;
    train_recording_count: number;
    calibration_recording_count: number;
    test_recording_count: number;
    baseline_macro_f1: number;
    metrics: { macro_f1: number; accuracy: number; balanced_accuracy: number; selective?: { coverage: number; selective_accuracy: number | null } };
  }>;
};
type DegradationEvaluation = {
  status: string;
  reason?: string;
  protocol?: string;
  scenarios: Array<{
    scenario: string;
    affected_channels: string[];
    metrics: { macro_f1: number; accuracy: number };
    macro_f1_change_vs_clean: number | null;
    selective: { coverage: number; referred_fraction: number };
  }>;
};
type Overview = {
  registered_machines: number;
  prediction_count: number;
  predictions_last_24_hours: number;
  latest_prediction: Prediction | null;
  last_data_stale: boolean;
  stale_after_seconds: number;
  broker_status: string;
  model_version: string | null;
  recent_predictions: Prediction[];
  recent_events: EventRecord[];
};
type Comparison = {
  status: string;
  selected_model?: string;
  selection_metric?: string;
  feature_names?: string[];
  comparison: Array<Record<string, unknown>>;
  test_metrics: {
    macro_f1: number;
    accuracy: number;
    balanced_accuracy: number;
    per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
    confusion_matrix: number[][];
    labels: string[];
    prediction_distribution: Record<string, number>;
    per_recording_accuracy?: Record<string, { condition: string; accuracy: number; window_count: number }>;
    recording_accuracy_mean?: number;
    recording_accuracy_std?: number;
  } | null;
  automl?: { status: string; reason?: string; best_estimator?: string };
  class_collapse_detected?: boolean;
  test_classes_missing_from_predictions?: string[];
  reliability?: {
    calibration_method: string;
    calibration_temperature: number;
    validation_calibration_metrics: Record<string, number>;
    acceptance_policy: {
      threshold: number;
      threshold_found: boolean;
      validation_coverage: number;
      validation_selective_accuracy: number | null;
    };
  } | null;
  cross_load_evaluation?: CrossLoadEvaluation;
  sensor_degradation_evaluation?: DegradationEvaluation;
};
type DatasetManifest = {
  recording_count: number;
  recordings_per_class: Record<string, number>;
  matched_vibration_tdms_recordings: number;
  acoustic_recording_count: number;
  acoustic_class_coverage: string[];
  fingerprint: string;
  label_provenance: string;
  channel_provenance: string;
};
type ReplayState = {
  state: string;
  mode?: "local" | "mqtt";
  continuous?: boolean;
  replay_session_id?: string;
  machine_id?: string;
  published?: number;
  processed?: number;
  failed?: number;
  total?: number;
  cycles_completed?: number;
  cycle_processed?: number;
  classes?: string[];
  missing_classes?: string[];
  prediction_distribution?: Record<string, number>;
  recording_ids?: string[];
  last_error?: string | null;
};
type ModelVersion = { version: string; algorithm: string; status: string; checksum: string; created_at: string };
type MachineStatus = {
  machine_id: string;
  last_seen: string;
  status: string;
  source: "live" | "replay";
  condition: string;
  model_version: string;
};
type TrainingJob = { id: number; status: string; started_at: string; finished_at: string | null; error: string | null };

const pages: Page[] = ["Overview", "Machines", "Predictions", "Model lab", "Reliability", "Replay", "Events", "Dataset"];
const conditionColors: Record<string, string> = {
  Normal: "#46c99a", BPFI: "#ff786f", BPFO: "#f2aa5d", Misalignment: "#ad91ff", Unbalance: "#55b5f2",
};
const istDateFormatter = new Intl.DateTimeFormat("en-IN", { timeZone: "Asia/Kolkata" });
const istClockFormatter = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: true,
});
const formatTime = (value?: string | null) => value ? `${istDateFormatter.format(new Date(value))}, ${istClockFormatter.format(new Date(value))} IST` : "—";
const shortVersion = (value?: string | null) => value ? value.replace("model-", "") : "Not deployed";
const formatBytes = (bytes: number) => bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(2)} MB` : `${(bytes / 1024).toFixed(1)} KB`;

export default function App() {
  const [page, setPage] = useState<Page>("Overview");
  const [adminKey, setAdminKey] = useState("");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [distribution, setDistribution] = useState<Record<string, number>>({});
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [manifest, setManifest] = useState<DatasetManifest | null>(null);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [machines, setMachines] = useState<MachineStatus[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [reliability, setReliability] = useState<ReliabilityStatus | null>(null);
  const [reviewQueue, setReviewQueue] = useState<Prediction[]>([]);
  const [crossLoad, setCrossLoad] = useState<CrossLoadEvaluation | null>(null);
  const [degradation, setDegradation] = useState<DegradationEvaluation | null>(null);
  const [replay, setReplay] = useState<ReplayState>({ state: "idle" });
  const [filter, setFilter] = useState("");
  const [machineInput, setMachineInput] = useState("machine-001");
  const [recordingInput, setRecordingInput] = useState("");
  const [replayMode, setReplayMode] = useState<"local" | "mqtt">("local");
  const [continuousReplay, setContinuousReplay] = useState(true);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [clock, setClock] = useState(() => new Date());

  const refresh = useCallback(async () => {
    try {
      const [summary, counts, history, evaluation, dataset, eventList, modelList, replayState, machineList, jobList, reliabilityState, reviews, crossLoadResults, degradationResults] = await Promise.all([
        api<Overview>("/api/overview"),
        api<{ conditions: Record<string, number> }>("/api/condition-distribution"),
        api<{ items: Prediction[] }>(`/api/predictions?limit=100${filter ? `&condition=${encodeURIComponent(filter)}` : ""}`),
        api<Comparison>("/api/models/comparison"),
        api<DatasetManifest>("/api/dataset"),
        api<{ items: EventRecord[] }>("/api/events?limit=100"),
        api<ModelVersion[]>("/api/models"),
        api<ReplayState>("/api/replay"),
        api<{ items: MachineStatus[] }>("/api/machines"),
        api<{ items: TrainingJob[] }>("/api/training-jobs?limit=50"),
        api<ReliabilityStatus>("/api/reliability/status"),
        api<{ items: Prediction[] }>("/api/reliability/reviews?limit=100"),
        api<CrossLoadEvaluation>("/api/research/cross-load"),
        api<DegradationEvaluation>("/api/research/degradation"),
      ]);
      setOverview(summary);
      setDistribution(counts.conditions);
      setPredictions(history.items);
      setComparison(evaluation);
      setManifest(dataset);
      setEvents(eventList.items);
      setModels(modelList);
      setReplay(replayState);
      setMachines(machineList.items);
      setTrainingJobs(jobList.items);
      setReliability(reliabilityState);
      setReviewQueue(reviews.items);
      setCrossLoad(crossLoadResults);
      setDegradation(degradationResults);
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "The backend request failed");
    }
  }, [filter]);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    const interval = window.setInterval(() => setClock(new Date()), 1000);
    return () => window.clearInterval(interval);
  }, []);

  async function action(path: string, body?: unknown) {
    setBusy(true);
    setNotice("");
    setError("");
    try {
      const result = await api<Record<string, unknown>>(path, {
        method: "POST",
        ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      }, adminKey);
      setNotice(String(result.status ?? "Request completed"));
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Administrative action failed");
    } finally {
      setBusy(false);
    }
  }

  const navIcon = (name: Page) => {
    const icons = { Overview: Gauge, Machines: Radio, Predictions: History, "Model lab": BarChart3, Reliability: ShieldCheck, Replay: Play, Events: AlertTriangle, Dataset: Database };
    const Icon = icons[name];
    return <Icon size={17} strokeWidth={1.8} />;
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Activity size={21} /></div>
          <div><strong>Motor Monitor</strong><span>FAULT INTELLIGENCE</span></div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Primary navigation">
          {pages.map((name) => (
            <button key={name} className={`nav-item ${page === name ? "active" : ""}`} onClick={() => setPage(name)}>
              {navIcon(name)}<span>{name}</span>{page === name && <ChevronRight className="nav-chevron" size={15} />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="connection-row"><span className={`status-dot ${overview?.broker_status === "connected" ? "online" : "offline"}`} />
            <span>MQTT broker</span><b>{overview?.broker_status ?? "unknown"}</b></div>
          <div className="connection-row"><span className={`status-dot ${overview?.model_version ? "online" : "offline"}`} />
            <span>Inference model</span><b>{overview?.model_version ? "ready" : "missing"}</b></div>
          <div className="sidebar-foot"><span>Classification only</span><span>v1.0</span></div>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumbs"><span>Workspace</span><ChevronRight size={14} /><strong>{page}</strong></div>
          <div className="topbar-right">
            <div className={`feed-status ${overview?.last_data_stale ? "stale" : "fresh"}`}>
              <span className="status-dot" />{overview?.last_data_stale ? "No recent data" : "Data current"}
            </div>
            <span className="ist-clock">{istClockFormatter.format(clock)} IST</span>
            <span className="model-chip"><Layers3 size={14} />{shortVersion(overview?.model_version)}</span>
          </div>
        </header>

        <div className="content">
          {error && <div className="notice error"><AlertTriangle size={17} />{error}</div>}
          {notice && <div className="notice success"><Check size={17} />{notice}</div>}
          <div className="page-heading">
            <div><p className="eyebrow">INDUSTRIAL CONDITION MONITORING</p><h1>{page}</h1>
              <p className="subheading">Sensor-derived operating-condition classification · Recorded data replay</p></div>
            <button className="icon-button" onClick={() => void refresh()} aria-label="Refresh data"><RotateCcw size={17} /></button>
          </div>

          {page === "Overview" && <OverviewPage overview={overview} distribution={distribution} onNavigate={setPage} />}
          {page === "Machines" && <MachinesPage machines={machines} predictions={predictions} />}
          {page === "Predictions" && <PredictionsPage rows={predictions} filter={filter} setFilter={setFilter} />}
          {page === "Model lab" && <ModelPage comparison={comparison} models={models} jobs={trainingJobs}
            adminKey={adminKey} setAdminKey={setAdminKey} action={action} busy={busy} />}
          {page === "Reliability" && <ReliabilityPage status={reliability} queue={reviewQueue}
            crossLoad={crossLoad} degradation={degradation} comparison={comparison}
            adminKey={adminKey} setAdminKey={setAdminKey} action={action} busy={busy} />}
          {page === "Replay" && <ReplayPage replay={replay} busy={busy} machineInput={machineInput}
            setMachineInput={setMachineInput} recordingInput={recordingInput} setRecordingInput={setRecordingInput}
            replayMode={replayMode} setReplayMode={setReplayMode}
            continuousReplay={continuousReplay} setContinuousReplay={setContinuousReplay}
            adminKey={adminKey} setAdminKey={setAdminKey} action={action} />}
          {page === "Events" && <EventsPage events={events} adminKey={adminKey} action={action} />}
          {page === "Dataset" && <DatasetPage manifest={manifest} />}
          <footer className="disclaimer"><ShieldCheck size={15} />
            Model outputs indicate a classified condition, not a confirmed diagnosis, failure time, or remaining useful life.
          </footer>
        </div>
      </main>
    </div>
  );
}

function OverviewPage({ overview, distribution, onNavigate }: {
  overview: Overview | null; distribution: Record<string, number>; onNavigate: (page: Page) => void;
}) {
  const chartData = Object.entries(distribution).map(([name, value]) => ({ name, count: value }));
  return <>
    <section className="metric-grid">
      <MetricCard icon={<Radio />} label="Machines observed" value={overview?.registered_machines ?? "—"} detail="From persisted input messages" />
      <MetricCard icon={<Activity />} label="Predictions · 24h" value={overview?.predictions_last_24_hours ?? "—"} detail={`${overview?.prediction_count ?? 0} total records`} />
      <MetricCard icon={<CircleGauge />} label="Latest condition" value={overview?.latest_prediction?.predicted_condition ?? "—"} detail={overview?.latest_prediction ? formatTime(overview.latest_prediction.received_at) : "No predictions received"} accent={overview?.latest_prediction?.predicted_condition} />
      <MetricCard icon={<Layers3 />} label="Active model" value={shortVersion(overview?.model_version)} detail="Validated local artifact" />
    </section>
    <section className="overview-grid">
      <article className="panel chart-panel">
        <div className="panel-heading"><div><h2>Condition distribution</h2><p>Actual persisted model outputs</p></div><button className="text-action" onClick={() => onNavigate("Predictions")}>View history <ChevronRight size={15} /></button></div>
        {chartData.length === 0 || chartData.every((item) => item.count === 0)
          ? <EmptyState icon={<BarChart3 />} text="No prediction records yet" subtext="Run a validated held-out dataset replay to populate this chart." />
          : <div className="chart-wrap"><ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 12, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#233146" />
              <XAxis dataKey="name" tick={{ fill: "#8d9bb0", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis allowDecimals={false} tick={{ fill: "#718096", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: "#142033", border: "1px solid #2b3a4f", borderRadius: 8, color: "#eef3f9" }} />
              <Bar dataKey="count" name="Predictions" radius={[5, 5, 0, 0]} maxBarSize={48}>
                {chartData.map((item) => <Cell key={item.name} fill={conditionColors[item.name] ?? "#60a5fa"} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer></div>}
      </article>
      <article className="panel">
        <div className="panel-heading"><div><h2>System signals</h2><p>Live operational state</p></div><Settings2 size={17} className="muted-icon" /></div>
        <div className="signal-list">
          <Signal icon={overview?.broker_status === "connected" ? <Wifi /> : <WifiOff />} label="Message broker" value={overview?.broker_status ?? "unknown"} good={overview?.broker_status === "connected"} />
          <Signal icon={<Thermometer />} label="Data freshness" value={overview?.last_data_stale ? `Stale · >${overview.stale_after_seconds}s` : "Within freshness threshold"} good={!!overview && !overview.last_data_stale} />
          <Signal icon={<Database />} label="Latest receipt" value={formatTime(overview?.latest_prediction?.received_at)} />
          <Signal icon={<Layers3 />} label="Inference version" value={shortVersion(overview?.model_version)} good={!!overview?.model_version} />
        </div>
      </article>
    </section>
    <section className="panel table-panel">
      <div className="panel-heading"><div><h2>Recent predictions</h2><p>Most recently persisted inference results</p></div><button className="text-action" onClick={() => onNavigate("Predictions")}>All records <ChevronRight size={15} /></button></div>
      <PredictionTable rows={overview?.recent_predictions ?? []} />
    </section>
  </>;
}

function MetricCard({ icon, label, value, detail, accent }: { icon: React.ReactNode; label: string; value: string | number; detail: string; accent?: string }) {
  return <article className="metric-card"><div className="metric-top"><span className={`metric-icon ${accent ? `condition-${accent.toLowerCase()}` : ""}`}>{icon}</span><span className="metric-label">{label}</span></div>
    <strong className="metric-value">{value}</strong><span className="metric-detail">{detail}</span></article>;
}

function Signal({ icon, label, value, good }: { icon: React.ReactNode; label: string; value: string; good?: boolean }) {
  return <div className="signal-row"><span className="signal-icon">{icon}</span><div className="signal-copy"><span>{label}</span><strong>{value}</strong></div>{good !== undefined && <span className={`status-dot ${good ? "online" : "offline"}`} />}</div>;
}

function EmptyState({ icon, text, subtext }: { icon: React.ReactNode; text: string; subtext: string }) {
  return <div className="empty-state"><div>{icon}</div><strong>{text}</strong><span>{subtext}</span></div>;
}

function PredictionsPage({ rows, filter, setFilter }: { rows: Prediction[]; filter: string; setFilter: (value: string) => void }) {
  return <section className="panel table-panel">
    <div className="panel-heading"><div><h2>Prediction history</h2><p>Persisted predictions only · labels are shown in evaluation views, not as predictions</p></div>
      <select aria-label="Filter by predicted condition" className="select-control" value={filter} onChange={(event) => setFilter(event.target.value)}>
        <option value="">All conditions</option>{Object.keys(conditionColors).map((name) => <option key={name}>{name}</option>)}
      </select>
    </div>
    <PredictionTable rows={rows} />
  </section>;
}

function MachinesPage({ machines, predictions }: { machines: MachineStatus[]; predictions: Prediction[] }) {
  const [selected, setSelected] = useState("");
  const selectedMachine = machines.find((machine) => machine.machine_id === selected);
  const visibleRows = predictions.filter((row) => !selected || row.machine_id === selected);
  return <div className="page-stack">
    <section className="panel"><div className="panel-heading"><div><h2>Observed machines</h2><p>Machine identities and status are derived from actual persisted feature messages.</p></div><span className="status-tag neutral">{machines.length} observed</span></div>
      {!machines.length ? <EmptyState icon={<Radio />} text="No observed machines" subtext="A machine appears after its first accepted and persisted feature message." /> :
        <div className="machine-grid">{machines.map((machine) => <button key={machine.machine_id} className={`machine-card ${selected === machine.machine_id ? "selected" : ""}`} onClick={() => setSelected(selected === machine.machine_id ? "" : machine.machine_id)}>
          <div><Radio size={16} /><span className={`status-tag ${machine.status === "active" ? "good" : "neutral"}`}>{machine.status}</span></div>
          <strong>{machine.machine_id}</strong><span className={`condition-pill condition-${machine.condition.toLowerCase()}`}><i />{machine.condition}</span>
          <small>Last input: {formatTime(machine.last_seen)}</small><small>Source: {machine.source} · model {shortVersion(machine.model_version)}</small>
        </button>)}</div>}
    </section>
    <section className="panel table-panel"><div className="panel-heading"><div><h2>{selectedMachine ? `Recent inputs · ${selectedMachine.machine_id}` : "Recent inputs"}</h2><p>Latest actual predictions stored for observed machines</p></div>{selected && <button className="text-action" onClick={() => setSelected("")}>Clear selection</button>}</div>
      <PredictionTable rows={visibleRows} />
    </section>
  </div>;
}

function PredictionTable({ rows }: { rows: Prediction[] }) {
  if (rows.length === 0) return <EmptyState icon={<History />} text="Nothing to display" subtext="Predictions will appear here after verified sensor feature messages are processed." />;
  return <div className="table-scroll"><table><thead><tr><th>Timestamp</th><th>Machine</th><th>Condition</th><th>Source</th><th>Recording / window</th><th>Confidence</th><th>Decision</th><th>Model</th><th>Latency</th><th>Details</th></tr></thead>
    <tbody>{rows.map((row) => <PredictionTableRow key={row.id} row={row} />)}</tbody></table></div>;
}

function PredictionTableRow({ row }: { row: Prediction }) {
  const [expanded, setExpanded] = useState(false);
  const [explanation, setExplanation] = useState(row.explanation);
  const [calibratedProbabilities, setCalibratedProbabilities] = useState(row.calibrated_probabilities);
  const [reliability, setReliability] = useState(row.reliability);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState("");
  const probabilities = calibratedProbabilities && Object.keys(calibratedProbabilities).length
    ? calibratedProbabilities : row.probabilities;
  const isCalibrated = Boolean(calibratedProbabilities && Object.keys(calibratedProbabilities).length);
  const confidence = Object.keys(probabilities).length ? Math.max(...Object.values(probabilities)) : null;
  useEffect(() => {
    if (!expanded || explanation?.top_features?.length) return;
    let active = true;
    setExplanationLoading(true);
    setExplanationError("");
    api<{
      explanation: Prediction["explanation"];
      explanation_status: string;
      calibrated_probabilities: Record<string, number>;
      reliability: Prediction["reliability"];
    }>(
      `/api/predictions/${row.id}/explanation`,
    ).then((result) => {
      if (active) {
        setExplanation(result.explanation);
        setCalibratedProbabilities(result.calibrated_probabilities);
        setReliability(result.reliability);
      }
    }).catch((cause) => {
      if (active) setExplanationError(cause instanceof Error ? cause.message : "Could not load explanation");
    }).finally(() => {
      if (active) setExplanationLoading(false);
    });
    return () => { active = false; };
  }, [expanded, explanation?.top_features?.length, row.id]);
  return <>
    <tr><td>{formatTime(row.received_at)}</td><td className="machine-cell">{row.machine_id}</td>
      <td><span className={`condition-pill condition-${row.predicted_condition.toLowerCase()}`}><i />{row.predicted_condition}</span></td>
      <td><span className={`source-pill source-${row.source}`}>{row.source}</span></td>
      <td>{row.recording_id ? `${row.recording_id} · ${row.window_index ?? "—"}` : "—"}</td>
      <td>{confidence === null ? "—" : `${(confidence * 100).toFixed(1)}%${isCalibrated ? " calibrated" : ""}`}</td>
      <td><span className={`decision-pill decision-${(row.decision_status ?? "unknown").toLowerCase()}`}>{row.decision_status ?? "legacy"}</span></td>
      <td className="mono">{shortVersion(row.model_version)}</td><td>{row.latency_ms.toFixed(2)} ms</td>
      <td><button className="text-action" aria-expanded={expanded} onClick={() => setExpanded(!expanded)}>{expanded ? "Hide" : "Explain"}</button></td></tr>
    {expanded && <tr className="detail-row"><td colSpan={10}>
      <div className="prediction-detail">
        <div><strong>Class probabilities</strong><div className="tag-list">{Object.entries(probabilities).map(([label, probability]) =>
          <span className={`condition-pill condition-${label.toLowerCase()}`} key={label}>{label} · {(probability * 100).toFixed(1)}%</span>)}</div></div>
        <div><strong>Reliability checks</strong><p>Sensor quality: {reliability?.sensor_quality?.status ?? "unavailable"} · Distribution: {reliability?.out_of_distribution?.status ?? "unavailable"} · Drift: {reliability?.drift?.status ?? "unavailable"}</p></div>
        <div><strong>Explanation · {explanation?.method ?? (explanationLoading ? "calculating…" : "not available")}</strong>
          {explanationLoading && <p>Calculating model sensitivity from the original sensor features…</p>}
          {explanationError && <p className="inline-error">{explanationError}</p>}
          {explanation?.top_features?.length ? <ul>{explanation.top_features.slice(0, 5).map((item) =>
            <li key={item.feature}>{item.feature}: {item.probability_delta !== undefined
              ? `${(item.probability_delta * 100).toFixed(2)} percentage-point sensitivity`
              : `attribution ${item.attribution?.toFixed(4) ?? "—"}`}</li>)}</ul> : !explanationLoading && !explanationError && <p>No explanation available for this model version.</p>}
          {explanation?.interpretation && <p>{explanation.interpretation}</p>}
        </div>
      </div>
    </td></tr>}
  </>;
}

function ModelPage({ comparison, models, jobs, adminKey, setAdminKey, action, busy }: {
  comparison: Comparison | null; models: ModelVersion[]; jobs: TrainingJob[];
  adminKey: string; setAdminKey: (value: string) => void;
  action: (path: string, body?: unknown) => Promise<void>; busy: boolean;
}) {
  const metrics = comparison?.test_metrics;
  return <div className="page-stack">
    <section className="panel">
      <div className="panel-heading"><div><h2>How the model makes a prediction</h2><p>Every displayed result comes from the selected trained classifier; replay labels are kept out of inference.</p></div><span className="status-tag neutral">{comparison?.selected_model ?? "No model"}</span></div>
      <div className="coverage-grid">
        <div className="coverage-card"><span>1 · Sensor features</span><strong>{comparison?.feature_names?.length ?? "—"}</strong><small>features per one-second window from vibration, temperature, and current</small></div>
        <div className="coverage-card"><span>2 · Trained classifier</span><strong>{comparison?.selected_model ?? "Not trained"}</strong><small>selected by validation macro-F1, then evaluated on separate recordings</small></div>
        <div className="coverage-card"><span>3 · Condition output</span><strong>{metrics?.labels.length ?? "—"} classes</strong><small>predicted class and probabilities are persisted for each processed window</small></div>
      </div>
      {comparison?.test_classes_missing_from_predictions?.length ? <p className="inline-warning">Measured limitation: the held-out model produced no {comparison.test_classes_missing_from_predictions.join(", ")} predictions. In the current run, Normal samples were classified as Unbalance; the model does not yet reliably detect Normal operation.</p> :
        <p className="muted-copy">The output is a machine-condition classification, not a failure forecast or confirmed maintenance diagnosis.</p>}
    </section>
    <section className="metric-grid model-metrics">
      <MetricCard icon={<BarChart3 />} label="Selected by validation" value={comparison?.selected_model ?? "Not trained"} detail={comparison?.selection_metric ?? "Train and compare models"} />
      <MetricCard icon={<Gauge />} label="Held-out macro-F1" value={metrics ? metrics.macro_f1.toFixed(3) : "—"} detail="Independent recording-level test partition" />
      <MetricCard icon={<Check />} label="Held-out balanced accuracy" value={metrics ? metrics.balanced_accuracy.toFixed(3) : "—"} detail={metrics ? `${(metrics.accuracy * 100).toFixed(1)}% accuracy` : "Measured after training"} />
      <MetricCard icon={<AlertTriangle />} label="Class coverage diagnostic" value={comparison?.class_collapse_detected ? "Single-class output" : comparison?.test_classes_missing_from_predictions?.length ? "Missing outputs" : comparison?.status === "completed" ? "All classes output" : "—"} detail={comparison?.test_classes_missing_from_predictions?.length ? `No predictions: ${comparison.test_classes_missing_from_predictions.join(", ")}` : "Held-out prediction distribution"} />
    </section>
    <section className="panel training-panel">
      <div className="panel-heading"><div><h2>Training and administration</h2><p>Model training runs asynchronously and creates a validated candidate; deployment remains a separate approval step.</p></div></div>
      <div className="admin-controls"><label>Administrator API key <input type="password" autoComplete="off" placeholder="Configured on the backend" value={adminKey} onChange={(event) => setAdminKey(event.target.value)} /></label>
        <button className="button primary" disabled={busy} onClick={() => void action("/api/admin/training-jobs?run_automl=true")}><Activity size={15} />Train candidates + AutoML</button>
        <button className="button" disabled={busy} onClick={() => void action("/api/admin/models/rollback")}><RotateCcw size={15} />Rollback model</button>
      </div>
      {jobs.length === 0 ? <p className="muted-copy">No training jobs have been submitted.</p> :
        <div className="table-scroll"><table><thead><tr><th>Job</th><th>Status</th><th>Started</th><th>Finished</th><th>Error</th></tr></thead><tbody>
          {jobs.map((job) => <tr key={job.id}><td>#{job.id}</td><td><span className={`status-tag ${job.status === "completed" ? "good" : "neutral"}`}>{job.status}</span></td><td>{formatTime(job.started_at)}</td><td>{formatTime(job.finished_at)}</td><td>{job.error ?? "—"}</td></tr>)}
        </tbody></table></div>}
    </section>
    <section className="panel">
      <div className="panel-heading"><div><h2>Candidate comparison</h2><p>Measured validation metrics; sorted by macro-F1</p></div></div>
      {!comparison?.comparison.length ? <EmptyState icon={<BarChart3 />} text="No evaluation report" subtext="Use the CLI or protected training API to evaluate candidate models." /> :
        <div className="table-scroll"><table><thead><tr><th>Algorithm</th><th>Validation macro-F1</th><th>Accuracy</th><th>Balanced accuracy</th><th>Train duration</th><th>Inference / row</th><th>Model size</th></tr></thead>
          <tbody>{comparison.comparison.map((row, index) => <tr key={String(row.algorithm)} className={index === 0 ? "winner-row" : ""}>
            <td>{String(row.algorithm)}{index === 0 && <span className="winner-tag">selected</span>}</td>
            <td>{Number(row.macro_f1).toFixed(4)}</td><td>{Number(row.accuracy).toFixed(4)}</td><td>{Number(row.balanced_accuracy).toFixed(4)}</td>
            <td>{Number(row.training_duration_seconds).toFixed(2)} s</td><td>{Number(row.inference_latency_ms_per_row).toFixed(3)} ms</td><td>{formatBytes(Number(row.model_size_bytes))}</td>
          </tr>)}</tbody></table></div>}
    </section>
    <section className="panel">
      <div className="panel-heading"><div><h2>Per-class test recall</h2><p>Genuine independent held-out recordings</p></div></div>
      {!metrics ? <EmptyState icon={<Activity />} text="No test metrics" subtext="Train a model to generate a group-held-out evaluation report." /> : <>
        {!!comparison?.test_classes_missing_from_predictions?.length && <div className="inline-warning">The selected model produced no predictions for: {comparison.test_classes_missing_from_predictions.join(", ")}. This is a measured test-set limitation, not a forced-label or simulator behaviour.</div>}
        <div className="recall-grid">{metrics.labels.map((label) => <div className="recall-card" key={label}><span className={`condition-pill condition-${label.toLowerCase()}`}><i />{label}</span><strong>{(metrics.per_class[label]?.recall * 100).toFixed(1)}%</strong><small>{metrics.per_class[label]?.support ?? 0} test windows</small></div>)}</div>
        <ConfusionMatrix labels={metrics.labels} matrix={metrics.confusion_matrix} />
        {metrics.per_recording_accuracy && <div className="matrix-wrap"><h3>Recording-level test variability <span>Accuracy per independent held-out recording</span></h3>
          <p className="muted-copy">Mean {((metrics.recording_accuracy_mean ?? 0) * 100).toFixed(1)}% · standard deviation {((metrics.recording_accuracy_std ?? 0) * 100).toFixed(1)} percentage points</p>
          <div className="table-scroll"><table><thead><tr><th>Recording</th><th>Condition</th><th>Windows</th><th>Accuracy</th></tr></thead><tbody>
            {Object.entries(metrics.per_recording_accuracy).map(([recording, value]) => <tr key={recording}><td className="mono">{recording}</td><td>{value.condition}</td><td>{value.window_count}</td><td>{(value.accuracy * 100).toFixed(1)}%</td></tr>)}
          </tbody></table></div>
        </div>}
      </>}
    </section>
    <section className="panel">
      <div className="panel-heading"><div><h2>Model registry</h2><p>Versions built and evaluated by this workspace</p></div><span className={`status-tag ${comparison?.automl?.status === "completed" ? "good" : "neutral"}`}>FLAML: {comparison?.automl?.status ?? "not run"}</span></div>
      {comparison?.automl?.reason && <p className="inline-warning">{comparison.automl.reason}</p>}
      {models.length === 0 ? <EmptyState icon={<Layers3 />} text="Registry is empty" subtext="Validated model bundles are registered after training." /> :
        <div className="table-scroll"><table><thead><tr><th>Version</th><th>Algorithm</th><th>Status</th><th>SHA-256</th><th>Created</th><th>Action</th></tr></thead><tbody>
          {models.map((model) => <tr key={model.version}><td className="mono">{model.version}</td><td>{model.algorithm}</td><td><span className={`status-tag ${model.status === "deployed" ? "good" : "neutral"}`}>{model.status}</span></td><td className="mono checksum">{model.checksum}</td><td>{formatTime(model.created_at)}</td><td>{model.status !== "deployed" && <button className="text-action" disabled={busy} onClick={() => void action(`/api/admin/models/${encodeURIComponent(model.version)}/deploy`)}>Deploy version</button>}</td></tr>)}
        </tbody></table></div>}
    </section>
  </div>;
}

function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const maximum = Math.max(1, ...matrix.flat());
  return <div className="matrix-wrap"><h3>Confusion matrix <span>Rows: actual · columns: predicted</span></h3><div className="matrix-scroll"><table className="matrix"><thead><tr><th></th>{labels.map((label) => <th key={label}>{label}</th>)}</tr></thead><tbody>
    {matrix.map((row, i) => <tr key={labels[i]}><th>{labels[i]}</th>{row.map((count, j) => <td key={`${i}-${j}`} style={{ backgroundColor: `rgba(75, 160, 219, ${0.05 + 0.65 * count / maximum})` }}>{count}</td>)}</tr>)}
  </tbody></table></div></div>;
}

function ReliabilityPage({ status, queue, crossLoad, degradation, comparison, adminKey, setAdminKey, action, busy }: {
  status: ReliabilityStatus | null;
  queue: Prediction[];
  crossLoad: CrossLoadEvaluation | null;
  degradation: DegradationEvaluation | null;
  comparison: Comparison | null;
  adminKey: string;
  setAdminKey: (value: string) => void;
  action: (path: string, body?: unknown) => Promise<void>;
  busy: boolean;
}) {
  const [reviewer, setReviewer] = useState("");
  const policy = comparison?.reliability?.acceptance_policy;
  const calibration = comparison?.reliability?.validation_calibration_metrics;
  return <div className="page-stack">
    <section className="metric-grid">
      <MetricCard icon={<ShieldCheck />} label="Prediction decisions" value={status?.prediction_count ?? "—"} detail={`${status?.review_required_count ?? 0} flagged for review`} />
      <MetricCard icon={<AlertTriangle />} label="Unreviewed queue" value={queue.length} detail="Non-accepted predictions awaiting review" />
      <MetricCard icon={<Activity />} label="Feature drift" value={status?.drift.status ?? "unavailable"} detail={status?.drift.psi_max === null || status?.drift.psi_max === undefined ? `${status?.drift.window_size ?? 0} rows · warming up` : `PSI ${status.drift.psi_max.toFixed(3)} · threshold ${status.drift.threshold?.toFixed(3) ?? "—"}`} />
      <MetricCard icon={<Check />} label="Human reviews recorded" value={status?.human_reviewed_count ?? "—"} detail="Stored separately from model output" />
    </section>
    <section className="panel">
      <div className="panel-heading"><div><h2>Calibration and selective prediction</h2><p>Temperature and acceptance threshold are fit on validation recordings only; test metrics remain separate.</p></div><span className="status-tag neutral">{comparison?.reliability?.calibration_method ?? "Not trained"}</span></div>
      <div className="coverage-grid">
        <div className="coverage-card"><span>Calibration temperature</span><strong>{comparison?.reliability?.calibration_temperature.toFixed(3) ?? "—"}</strong><small>Scalar temperature scaling</small></div>
        <div className="coverage-card"><span>Acceptance threshold</span><strong>{policy?.threshold_found ? policy.threshold.toFixed(3) : "Review all"}</strong><small>{policy?.threshold_found ? `${(policy.validation_coverage * 100).toFixed(1)}% validation coverage` : "Validation target not met"}</small></div>
        <div className="coverage-card"><span>Validation selective accuracy</span><strong>{policy?.validation_selective_accuracy === null || policy?.validation_selective_accuracy === undefined ? "—" : `${(policy.validation_selective_accuracy * 100).toFixed(1)}%`}</strong><small>Threshold-selection data, not a held-out estimate</small></div>
        <div className="coverage-card"><span>Validation ECE</span><strong>{calibration?.expected_calibration_error_10_bins?.toFixed(3) ?? "—"}</strong><small>10-bin expected calibration error</small></div>
        <div className="coverage-card"><span>Validation Brier score</span><strong>{calibration?.brier_score?.toFixed(3) ?? "—"}</strong><small>Lower is better</small></div>
      </div>
      <div className="security-note"><ShieldCheck size={16} />An accepted decision is a confidence-based model output, not a diagnosis. Sensor-quality, distribution, drift, and threshold checks can route a prediction for human review. Explanations report model sensitivity, not causation.</div>
    </section>
    <section className="panel">
      <div className="panel-heading"><div><h2>Leave-one-load-out evaluation</h2><p>Each torque load is held out by recording; calibration uses only the other loads.</p></div><span className="status-tag neutral">{crossLoad?.status ?? "unavailable"}</span></div>
      {!crossLoad || !crossLoad.folds.length ? <EmptyState icon={<BarChart3 />} text="Cross-load report unavailable" subtext={crossLoad?.reason ?? "Train candidates to generate cross-load evaluation."} /> :
        <><p className="muted-copy">{crossLoad.protocol}</p><div className="table-scroll"><table><thead><tr><th>Held-out load</th><th>Train recordings</th><th>Calibration recordings</th><th>Test recordings</th><th>Macro-F1</th><th>Accuracy</th><th>Selective coverage</th><th>Selective accuracy</th></tr></thead><tbody>
          {crossLoad.folds.map((fold) => <tr key={fold.held_out_load_nm}><td>{fold.held_out_load_nm} Nm</td><td>{fold.train_recording_count}</td><td>{fold.calibration_recording_count}</td><td>{fold.test_recording_count}</td><td>{fold.metrics.macro_f1.toFixed(3)}</td><td>{(fold.metrics.accuracy * 100).toFixed(1)}%</td><td>{fold.metrics.selective ? `${(fold.metrics.selective.coverage * 100).toFixed(1)}%` : "—"}</td><td>{fold.metrics.selective?.selective_accuracy === null || fold.metrics.selective?.selective_accuracy === undefined ? "—" : `${(fold.metrics.selective.selective_accuracy * 100).toFixed(1)}%`}</td></tr>)}
        </tbody></table></div></>}
    </section>
    <section className="panel">
      <div className="panel-heading"><div><h2>Sensor degradation experiments</h2><p>Deterministic noise and full-channel dropout applied to held-out feature copies.</p></div><span className="status-tag neutral">{degradation?.status ?? "unavailable"}</span></div>
      {!degradation || !degradation.scenarios.length ? <EmptyState icon={<Activity />} text="Degradation report unavailable" subtext={degradation?.reason ?? "Train candidates to generate controlled degradation results."} /> :
        <><p className="muted-copy">{degradation.protocol}</p><div className="table-scroll"><table><thead><tr><th>Scenario</th><th>Affected channels</th><th>Macro-F1</th><th>Change vs clean</th><th>Coverage</th><th>Referred</th></tr></thead><tbody>
          {degradation.scenarios.map((scenario) => <tr key={scenario.scenario}><td>{scenario.scenario}</td><td>{scenario.affected_channels.join(", ") || "—"}</td><td>{scenario.metrics.macro_f1.toFixed(3)}</td><td>{scenario.macro_f1_change_vs_clean === null ? "—" : `${scenario.macro_f1_change_vs_clean >= 0 ? "+" : ""}${scenario.macro_f1_change_vs_clean.toFixed(3)}`}</td><td>{(scenario.selective.coverage * 100).toFixed(1)}%</td><td>{(scenario.selective.referred_fraction * 100).toFixed(1)}%</td></tr>)}
        </tbody></table></div></>}
    </section>
    <section className="panel">
      <div className="panel-heading"><div><h2>Human review queue</h2><p>Review labels are stored separately and never overwrite model predictions.</p></div><span className="status-tag neutral">{queue.length} pending</span></div>
      <div className="admin-controls">
        <label>Administrator API key <input type="password" autoComplete="off" placeholder="Configured on the backend" value={adminKey} onChange={(event) => setAdminKey(event.target.value)} /></label>
        <label>Reviewer name <input autoComplete="name" value={reviewer} onChange={(event) => setReviewer(event.target.value)} placeholder="Your name or operator ID" /></label>
      </div>
      {!queue.length ? <EmptyState icon={<Check />} text="Review queue is clear" subtext="Predictions requiring review will appear here." /> :
        <div className="review-list">{queue.map((prediction) => <ReviewQueueItem key={prediction.id} prediction={prediction} reviewer={reviewer} busy={busy} action={action} />)}</div>}
    </section>
  </div>;
}

function ReviewQueueItem({ prediction, reviewer, busy, action }: {
  prediction: Prediction;
  reviewer: string;
  busy: boolean;
  action: (path: string, body?: unknown) => Promise<void>;
}) {
  const [label, setLabel] = useState("");
  const [note, setNote] = useState("");
  return <article className="review-item">
    <div><strong>#{prediction.id} · {prediction.machine_id}</strong><span className={`condition-pill condition-${prediction.predicted_condition.toLowerCase()}`}><i />Model: {prediction.predicted_condition}</span>
      <small>{formatTime(prediction.received_at)} · {prediction.decision_status ?? "legacy"}</small></div>
    <label>Confirmed condition<select className="select-control" value={label} onChange={(event) => setLabel(event.target.value)}><option value="">Select a label</option>{Object.keys(conditionColors).map((condition) => <option key={condition}>{condition}</option>)}</select></label>
    <label>Review note<input value={note} onChange={(event) => setNote(event.target.value)} maxLength={2000} placeholder="Optional context" /></label>
    <button className="button primary" disabled={busy || !reviewer.trim() || !label} onClick={() => void action(`/api/admin/predictions/${prediction.id}/review`, {
      review_label: label,
      reviewed_by: reviewer.trim(),
      review_note: note.trim() || null,
    })}>Save review</button>
  </article>;
}

function ReplayPage({ replay, busy, machineInput, setMachineInput, recordingInput, setRecordingInput, adminKey, setAdminKey, replayMode, setReplayMode, continuousReplay, setContinuousReplay, action }: {
  replay: ReplayState; busy: boolean; machineInput: string; setMachineInput: (value: string) => void;
  recordingInput: string; setRecordingInput: (value: string) => void; adminKey: string;
  setAdminKey: (value: string) => void; replayMode: "local" | "mqtt"; setReplayMode: (value: "local" | "mqtt") => void;
  continuousReplay: boolean; setContinuousReplay: (value: boolean) => void;
  action: (path: string, body?: unknown) => Promise<void>;
}) {
  const active = ["starting", "running", "paused", "stopping"].includes(replay.state);
  const missingPredictions = Object.entries(replay.prediction_distribution ?? {})
    .filter(([label, count]) => replay.classes?.includes(label) && count === 0)
    .map(([label]) => label);
  return <div className="page-stack">
    <section className="panel replay-banner"><div className="replay-symbol"><Radio size={24} /></div><div><h2>Held-out dataset replay</h2><p>Runs actual test-split sensor features through the selected model. Ground-truth labels are never included in prediction messages.</p></div><span className={`status-tag ${active ? "good" : "neutral"}`}>{replay.state}</span></section>
    <section className="replay-grid">
      <article className="panel replay-controls"><div className="panel-heading"><div><h2>Replay controls</h2><p>{replayMode === "local" ? "Local inference simulator · no broker required" : "MQTT feature publisher · QoS 1"}</p></div><Play size={18} className="muted-icon" /></div>
        <label>Administrator API key <input type="password" autoComplete="off" placeholder="Configured on the backend" value={adminKey} onChange={(event) => setAdminKey(event.target.value)} /></label>
        <label>Machine identifier <input value={machineInput} onChange={(event) => setMachineInput(event.target.value)} /></label>
        <label>Recording IDs <input placeholder="Leave blank for all held-out recordings" value={recordingInput} onChange={(event) => setRecordingInput(event.target.value)} /></label>
        <label>Simulation mode <select value={replayMode} disabled={active} onChange={(event) => setReplayMode(event.target.value as "local" | "mqtt")}><option value="local">Local simulation (recommended)</option><option value="mqtt">MQTT broker and worker</option></select></label>
        <label className="continuous-toggle"><input type="checkbox" checked={continuousReplay} disabled={active} onChange={(event) => setContinuousReplay(event.target.checked)} />Keep replaying continuously until stopped</label>
        <div className="help-text">Leave recording IDs blank to cycle through the full held-out set containing all five source classes. Each cycle uses real sensor features; labels are not passed into inference. Continuous replay keeps producing new readings until you press Stop. Model predictions can differ from source class where the model misclassifies.</div>
        <div className="button-row">
          {!active ? <button className="button primary" disabled={busy} onClick={() => void action("/api/replay/start", {
            machine_id: machineInput,
            recording_ids: recordingInput.trim() ? recordingInput.split(",").map((value) => value.trim()).filter(Boolean) : null,
            speed: 50,
            mode: replayMode,
            continuous: continuousReplay,
          })}><Play size={15} />Start replay</button> :
            <><button className="button" disabled={busy || replay.state === "paused"} onClick={() => void action("/api/replay/pause")}><Pause size={15} />Pause</button>
              <button className="button" disabled={busy || replay.state !== "paused"} onClick={() => void action("/api/replay/resume")}><Play size={15} />Resume</button>
              <button className="button danger" disabled={busy} onClick={() => void action("/api/replay/stop")}><Square size={15} />Stop</button></>}
        </div>
      </article>
      <article className="panel replay-progress"><div className="panel-heading"><div><h2>Session progress</h2><p className="mono">{replay.replay_session_id ?? "No active session"}</p></div></div>
        <div className="progress-count"><strong>{replay.processed ?? replay.published ?? 0}</strong><span>readings {replay.mode === "mqtt" ? "published" : "processed"} · {replay.cycles_completed ?? 0} full cycles{replay.continuous ? " · continuous until stopped" : ""}</span></div>
        <div className="progress-track"><div style={{ width: `${replay.total ? Math.min(100, 100 * (replay.cycle_processed ?? 0) / replay.total) : 0}%` }} /></div>
        <div className="progress-meta"><span>Cycle: {replay.cycle_processed ?? 0} / {replay.total ?? 0}</span><span>Failed: {replay.failed ?? 0}</span><span>Machine: {replay.machine_id ?? "—"}</span></div>
        {replay.last_error && <div className="inline-error">{replay.last_error}</div>}
        <div className="coverage"><h3>Classes present in source recordings</h3>{replay.classes?.length ? <div className="tag-list">{replay.classes.map((label) => <span key={label} className={`condition-pill condition-${label.toLowerCase()}`}><i />{label}</span>)}</div> : <span className="muted-copy">Source classes appear when replay starts.</span>}
          {!!replay.missing_classes?.length && <p className="inline-warning">Missing in selection: {replay.missing_classes.join(", ")}</p>}</div>
        {replay.mode === "local" && <div className="coverage"><h3>Actual model predictions</h3><div className="coverage-grid">{Object.entries(replay.prediction_distribution ?? {}).map(([label, count]) => <div className="coverage-card" key={label}><span className={`condition-pill condition-${label.toLowerCase()}`}><i />{label}</span><strong>{count}</strong><small>model outputs</small></div>)}</div>
          {!!missingPredictions.length && <p className="inline-warning">No {missingPredictions.join(", ")} predictions so far. Source recordings are replayed for every class, but the classifier can misclassify them.</p>}
        </div>}
      </article>
    </section>
    <div className="security-note"><LockKeyhole size={16} />Administrative actions require a backend-configured key. Keep it private; it is held only in this page's memory.</div>
  </div>;
}

function EventsPage({ events, adminKey, action }: { events: EventRecord[]; adminKey: string; action: (path: string) => Promise<void> }) {
  return <section className="panel table-panel">
    <div className="panel-heading"><div><h2>System events</h2><p>Broker state, validation failures, replay and model lifecycle records</p></div><span className="status-tag neutral">{events.filter((event) => !event.acknowledged).length} unacknowledged</span></div>
    {events.length === 0 ? <EmptyState icon={<AlertTriangle />} text="No events recorded" subtext="Operational events are written when relevant services report them." /> :
      <div className="event-list">{events.map((event) => <div className="event-row" key={event.id}>
        <div className={`event-icon severity-${event.severity}`}><AlertTriangle size={16} /></div><div className="event-copy"><div><strong>{event.event_type.replaceAll("_", " ")}</strong><span className={`severity-label severity-text-${event.severity}`}>{event.severity}</span></div><p>{event.message}</p><time>{formatTime(event.created_at)}</time></div>
        {!event.acknowledged && <button className="text-action" disabled={!adminKey} onClick={() => void action(`/api/admin/events/${event.id}/acknowledge`)}><Check size={15} />Acknowledge</button>}
      </div>)}</div>}
  </section>;
}

function DatasetPage({ manifest }: { manifest: DatasetManifest | null }) {
  if (!manifest) return <EmptyState icon={<Database />} text="Dataset unavailable" subtext="Check the configured dataset path and service logs." />;
  return <div className="page-stack">
    <section className="metric-grid">
      <MetricCard icon={<Database />} label="Vibration recordings" value={manifest.recording_count} detail="Matched to current / temperature recordings" />
      <MetricCard icon={<Activity />} label="Current / temperature pairs" value={manifest.matched_vibration_tdms_recordings} detail="Validated by recording identifier" />
      <MetricCard icon={<Radio />} label="Acoustic recordings" value={manifest.acoustic_recording_count} detail={`Available classes: ${manifest.acoustic_class_coverage.join(", ") || "none"}`} />
      <MetricCard icon={<Layers3 />} label="Feature channels" value="7" detail="4 vibration · 2 temperature · 1 common current" />
    </section>
    <section className="panel"><div className="panel-heading"><div><h2>Source class coverage</h2><p>Recording counts from the local source files</p></div><span className="status-tag neutral">Fingerprint {manifest.fingerprint.slice(0, 12)}</span></div>
      <div className="coverage-grid">{Object.entries(manifest.recordings_per_class).map(([label, count]) => <div className="coverage-card" key={label}><span className={`condition-pill condition-${label.toLowerCase()}`}><i />{label}</span><strong>{count}</strong><small>recordings</small></div>)}</div>
      <div className="provenance-grid"><div><h3>Label provenance</h3><p>{manifest.label_provenance}</p></div><div><h3>Sensor metadata</h3><p>{manifest.channel_provenance}</p></div></div>
      <div className="security-note"><AlertTriangle size={16} />The TDMS current channels ai2 and ai3 are empty in all nine BPFO recordings, so the model uses only the common non-empty current ai0 channel. Acoustic files cover only Normal, BPFI, and BPFO and are excluded. No standalone dataset documentation was present; filename labels are cross-checked against embedded vibration run names where available.</div>
    </section>
  </div>;
}
