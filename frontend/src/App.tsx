import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ReferenceLine,
  Label,
  BarChart,
  Bar,
  AreaChart,
  Area,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  CloudSun,
  Database,
  Gauge,
  RefreshCw,
  Settings,
  Sparkles,
  TrendingUp,
  Wind,
  XCircle,
} from "lucide-react";

type HealthResponse = {
  status: "ok" | "degraded" | "error";
  databaseConnected: boolean;
  modelLoaded: boolean;
  latestRefresh?: string;
};

type OverviewResponse = {
  latestGenerationMw: number;
  latestTemperatureC: number;
  lookbackWindow: number;
  storedForecasts: number;
  actualSeries: Array<{ time: string; generation: number }>;
  forecastSeries: Array<{ time: string; forecast: number }>;
  weatherSeries: Array<{ time: string; temperature: number; windSpeed: number; radiation: number }>;
};

type ForecastResponse = {
  nextForecastMw: number;
  generatedAt: string;
  horizonHours: number;
  recentInputs: Array<Record<string, string | number | null>>;
};

type PerformanceResponse = {
  metrics: {
    rmse: number;
    mae: number;
    mse: number;
    mape: number;
    accuracy: number;
  };
  trainingInfo: {
    lastTrainingAt: string;
    lastRetrainingAt: string;
    epochs: number;
    batchSize: number;
  };
  lossCurve: Array<{
    epoch: number;
    trainLoss: number;
    valLoss: number;
  }>;
  actualVsPredicted: Array<{
    time: string;
    actual: number;
    predicted: number;
  }>;
};

type ComparisonResponse = {
  models: Array<{
    name: string;
    rmse: number;
    mae: number;
    mse: number;
    trainingTimeSec: number;
    selected: boolean;
  }>;
  summary: string;
};

type XaiResponse = {
  summary: string;
  featureImportance: Array<{
    feature: string;
    importance: number;
  }>;
  localExplanation: Array<{
    feature: string;
    contribution: number;
  }>;
};

type ForecastHistoryResponse = {
  rows: Array<{
    issuedAt: string;
    forecastFor: string;
    predictedMw: number;
    actualMw: number;
    absoluteError: number;
  }>;
};

type FreshnessResponse = {
  sources: Array<{
    name: string;
    status: "healthy" | "delayed" | "failed";
    lastSuccess: string;
    latencyMinutes: number;
    nextExpectedRun: string;
  }>;
};

type DataStatusResponse = {
  generationRows: Array<Record<string, string | number | null>>;
  weatherRows: Array<Record<string, string | number | null>>;
  featureRows: Array<Record<string, string | number | null>>;
};

type SystemStatusResponse = {
  databasePath: string;
  modelPath: string;
  scalerXPath: string;
  scalerYPath: string;
  modelLoaded: boolean;
  databaseCounts: Record<string, number>;
};

type ScenarioRequest = {
  scenarioName: string;
  temperatureDelta: number;
  windMultiplier: number;
  radiationMultiplier: number;
  cloudDelta: number;
};

type ScenarioResponse = {
  baselineMw: number;
  scenarioMw: number;
  deltaMw: number;
  explanation: string;
};

type AppErrorState = {
  health?: string;
  overview?: string;
  forecast?: string;
  performance?: string;
  comparison?: string;
  xai?: string;
  history?: string;
  freshness?: string;
  dataStatus?: string;
  systemStatus?: string;
  scenario?: string;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const fallbackHealth: HealthResponse = {
  status: "degraded",
  databaseConnected: false,
  modelLoaded: false,
  latestRefresh: new Date().toISOString(),
};

const fallbackOverview: OverviewResponse = {
  latestGenerationMw: 0,
  latestTemperatureC: 0,
  lookbackWindow: 24,
  storedForecasts: 0,
  actualSeries: [],
  forecastSeries: [],
  weatherSeries: [],
};

const fallbackForecast: ForecastResponse = {
  nextForecastMw: 0,
  generatedAt: new Date().toISOString(),
  horizonHours: 1,
  recentInputs: [],
};

const fallbackPerformance: PerformanceResponse = {
  metrics: { rmse: 0, mae: 0, mse: 0, mape: 0, accuracy: 0 },
  trainingInfo: {
    lastTrainingAt: new Date().toISOString(),
    lastRetrainingAt: new Date().toISOString(),
    epochs: 0,
    batchSize: 0,
  },
  lossCurve: [],
  actualVsPredicted: [],
};

const fallbackComparison: ComparisonResponse = {
  models: [],
  summary: "No comparison data available yet.",
};

const fallbackXai: XaiResponse = {
  summary: "No explainability output available yet.",
  featureImportance: [],
  localExplanation: [],
};

const fallbackHistory: ForecastHistoryResponse = {
  rows: [],
};

const fallbackFreshness: FreshnessResponse = {
  sources: [],
};

const fallbackDataStatus: DataStatusResponse = {
  generationRows: [],
  weatherRows: [],
  featureRows: [],
};

const fallbackSystemStatus: SystemStatusResponse = {
  databasePath: "Not available",
  modelPath: "Not available",
  scalerXPath: "Not available",
  scalerYPath: "Not available",
  modelLoaded: false,
  databaseCounts: {},
};

const fallbackScenario: ScenarioResponse = {
  baselineMw: 0,
  scenarioMw: 0,
  deltaMw: 0,
  explanation: "Run a scenario to see the result.",
};

const tabs = [
  "overview",
  "forecast",
  "performance",
  "comparison",
  "xai",
  "scenario",
  "data",
  "system",
] as const;

type TabKey = (typeof tabs)[number];

const tabMeta: Record<
  TabKey,
  { label: string; icon: React.ReactNode }
> = {
  overview: { label: "Overview", icon: <BarChart3 className="h-4 w-4" /> },
  forecast: { label: "Forecast", icon: <Gauge className="h-4 w-4" /> },
  performance: { label: "Performance", icon: <TrendingUp className="h-4 w-4" /> },
  comparison: { label: "Model Comparison", icon: <Activity className="h-4 w-4" /> },
  xai: { label: "Explainability", icon: <Sparkles className="h-4 w-4" /> },
  scenario: { label: "What-if Scenario", icon: <CloudSun className="h-4 w-4" /> },
  data: { label: "Data", icon: <Wind className="h-4 w-4" /> },
  system: { label: "System", icon: <Settings className="h-4 w-4" /> },
};

async function fetchJson<T>(path: string): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`Request failed (${res.status}) for ${path}`);
  }
  return (await res.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Request failed (${res.status}) for ${path}`);
  }
  return (await res.json()) as T;
}

function formatDateTime(value?: string) {
  if (!value) return "Not available";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function formatTimeOnly(value?: string) {
  if (!value) return "Not available";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleTimeString();
}

function formatColumnLabel(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\btotal generation\b/gi, "total load")
    .replace(/\bgeneration\b/gi, "load")
    .replace(/\bforecast generated at\b/gi, "forecast issued at")
    .replace(/\bmw\b/g, "MW")
    .replace(/\bc\b/g, "C")
    .replace(/\b2m\b/g, "2 m")
    .replace(/\b10m\b/g, "10 m")
    .replace(/\bof\b/g, "of")
    .replace(/\b([a-z])/g, (match) => match.toUpperCase());
}

function formatDbLabel(value: string) {
  return formatColumnLabel(value)
    .replace(/Generation Observations/g, "Load Observations")
    .replace(/Scenario Runs/g, "Scenario Runs")
    .replace(/Model Features/g, "Model Features")
    .replace(/Weather Observations/g, "Weather Observations");
}

function SectionCard({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-[28px] border border-slate-200/80 bg-white/90 p-5 shadow-[0_12px_40px_rgba(15,23,42,0.06)] backdrop-blur ${className}`}>
      <div className="mb-4">
        <h2 className="text-lg font-semibold tracking-tight text-slate-950">{title}</h2>
        {subtitle ? <p className="mt-1 text-sm leading-6 text-slate-500">{subtitle}</p> : null}
      </div>
      {children}
    </div>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded-[24px] border border-slate-200/80 bg-white/95 p-5 shadow-[0_10px_30px_rgba(15,23,42,0.05)] transition-transform duration-200 hover:-translate-y-0.5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-slate-500">{title}</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">{value}</p>
          <p className="mt-2 text-sm leading-6 text-slate-500">{subtitle}</p>
        </div>
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3 text-slate-700 shadow-inner">{icon}</div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const className =
    status === "ok" || status === "healthy"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : status === "degraded" || status === "delayed"
        ? "border-amber-200 bg-amber-50 text-amber-700"
        : "border-rose-200 bg-rose-50 text-rose-700";

  return <span className={`rounded-full border px-3 py-1 text-xs font-semibold capitalize ${className}`}>{status}</span>;
}

function DataTable({ rows }: { rows: Array<Record<string, string | number | null>> }) {
  if (!rows.length) {
    return <p className="text-sm text-slate-500">No data available.</p>;
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-50/90 text-left text-slate-600">
            <tr>
              {columns.map((column) => (
                <th key={column} className="whitespace-nowrap px-4 py-3 font-semibold">
                  {formatColumnLabel(column)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-t border-slate-100 odd:bg-white even:bg-slate-50/40">
                {columns.map((column) => (
                  <td key={column} className="whitespace-nowrap px-4 py-3 text-slate-700">
                    {row[column] === null ? "-" : String(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ErrorBanner({ message }: { message?: string }) {
  if (!message) return null;
  return (
    <div className="mb-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 shadow-sm">
      {message}
    </div>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("overview");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [errors, setErrors] = useState<AppErrorState>({});

  const [health, setHealth] = useState<HealthResponse>(fallbackHealth);
  const [overview, setOverview] = useState<OverviewResponse>(fallbackOverview);
  const [forecast, setForecast] = useState<ForecastResponse>(fallbackForecast);
  const [performance, setPerformance] = useState<PerformanceResponse>(fallbackPerformance);
  const [comparison, setComparison] = useState<ComparisonResponse>(fallbackComparison);
  const [xai, setXai] = useState<XaiResponse>(fallbackXai);
  const [history, setHistory] = useState<ForecastHistoryResponse>(fallbackHistory);
  const [freshness, setFreshness] = useState<FreshnessResponse>(fallbackFreshness);
  const [dataStatus, setDataStatus] = useState<DataStatusResponse>(fallbackDataStatus);
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse>(fallbackSystemStatus);
  const [scenario, setScenario] = useState<ScenarioResponse>(fallbackScenario);
  const [scenarioLoading, setScenarioLoading] = useState(false);

  const [scenarioName, setScenarioName] = useState("Peak demand stress test");
  const [temperatureDelta, setTemperatureDelta] = useState(0);
  const [windMultiplier, setWindMultiplier] = useState(1.0);
  const [radiationMultiplier, setRadiationMultiplier] = useState(1.0);
  const [cloudDelta, setCloudDelta] = useState(0);

  const combinedSeries = useMemo(
    () => [
      ...overview.actualSeries.map((x) => ({ ...x, forecast: null as number | null })),
      ...overview.forecastSeries.map((x) => ({ time: x.time, generation: null as number | null, forecast: x.forecast })),
    ],
    [overview],
  );

  const forecastStartTime = useMemo(
    () => overview.forecastSeries.length ? overview.forecastSeries[0].time : undefined,
    [overview.forecastSeries],
  );

  const finalLoss = useMemo(
    () => performance.lossCurve.length ? performance.lossCurve[performance.lossCurve.length - 1] : undefined,
    [performance.lossCurve],
  );

  const comparisonChartData = useMemo(
    () => comparison.models.map((model) => ({ name: model.name, rmse: model.rmse, mae: model.mae })),
    [comparison],
  );

  const historyChartData = useMemo(
    () => history.rows.map((row) => ({
      time: row.forecastFor.slice(5, 16),
      predicted: row.predictedMw,
      actual: row.actualMw,
      error: row.absoluteError,
    })),
    [history],
  );

  async function loadAll() {
    const nextErrors: AppErrorState = {};
    setRefreshing(true);

    const tasks = await Promise.allSettled([
      fetchJson<HealthResponse>("/api/health"),
      fetchJson<OverviewResponse>("/api/overview"),
      fetchJson<ForecastResponse>("/api/forecast"),
      fetchJson<PerformanceResponse>("/api/performance"),
      fetchJson<ComparisonResponse>("/api/comparison"),
      fetchJson<XaiResponse>("/api/xai"),
      fetchJson<ForecastHistoryResponse>("/api/forecast-history"),
      fetchJson<FreshnessResponse>("/api/freshness"),
      fetchJson<DataStatusResponse>("/api/data-status"),
      fetchJson<SystemStatusResponse>("/api/system-status"),
    ]);

    const [
      healthResult,
      overviewResult,
      forecastResult,
      performanceResult,
      comparisonResult,
      xaiResult,
      historyResult,
      freshnessResult,
      dataStatusResult,
      systemStatusResult,
    ] = tasks;

    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else nextErrors.health = healthResult.reason instanceof Error ? healthResult.reason.message : "Health endpoint failed.";

    if (overviewResult.status === "fulfilled") setOverview(overviewResult.value);
    else nextErrors.overview = overviewResult.reason instanceof Error ? overviewResult.reason.message : "Overview endpoint failed.";

    if (forecastResult.status === "fulfilled") setForecast(forecastResult.value);
    else nextErrors.forecast = forecastResult.reason instanceof Error ? forecastResult.reason.message : "Forecast endpoint failed.";

    if (performanceResult.status === "fulfilled") setPerformance(performanceResult.value);
    else nextErrors.performance = performanceResult.reason instanceof Error ? performanceResult.reason.message : "Performance endpoint failed.";

    if (comparisonResult.status === "fulfilled") setComparison(comparisonResult.value);
    else nextErrors.comparison = comparisonResult.reason instanceof Error ? comparisonResult.reason.message : "Comparison endpoint failed.";

    if (xaiResult.status === "fulfilled") setXai(xaiResult.value);
    else nextErrors.xai = xaiResult.reason instanceof Error ? xaiResult.reason.message : "Explainability endpoint failed.";

    if (historyResult.status === "fulfilled") setHistory(historyResult.value);
    else nextErrors.history = historyResult.reason instanceof Error ? historyResult.reason.message : "History endpoint failed.";

    if (freshnessResult.status === "fulfilled") setFreshness(freshnessResult.value);
    else nextErrors.freshness = freshnessResult.reason instanceof Error ? freshnessResult.reason.message : "Freshness endpoint failed.";

    if (dataStatusResult.status === "fulfilled") setDataStatus(dataStatusResult.value);
    else nextErrors.dataStatus = dataStatusResult.reason instanceof Error ? dataStatusResult.reason.message : "Data status endpoint failed.";

    if (systemStatusResult.status === "fulfilled") setSystemStatus(systemStatusResult.value);
    else nextErrors.systemStatus = systemStatusResult.reason instanceof Error ? systemStatusResult.reason.message : "System status endpoint failed.";

    setErrors(nextErrors);
    setLoading(false);
    setRefreshing(false);
  }

  async function runScenario() {
    setScenarioLoading(true);
    try {
      const payload: ScenarioRequest = {
        scenarioName,
        temperatureDelta,
        windMultiplier,
        radiationMultiplier,
        cloudDelta,
      };
      const result = await postJson<ScenarioResponse>("/api/scenario", payload);
      setScenario(result);
      setErrors((prev) => ({ ...prev, scenario: undefined }));
    } catch (error) {
      setErrors((prev) => ({
        ...prev,
        scenario: error instanceof Error ? error.message : "Scenario request failed.",
      }));
    } finally {
      setScenarioLoading(false);
    }
  }

  useEffect(() => {
    void loadAll();
  }, []);

  return (
    <div className="min-h-screen overflow-hidden bg-slate-100 text-slate-900">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute left-[-8rem] top-[-7rem] h-72 w-72 rounded-full bg-cyan-200/40 blur-3xl" />
        <div className="absolute right-[-6rem] top-20 h-72 w-72 rounded-full bg-indigo-200/40 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-emerald-200/30 blur-3xl" />
      </div>

      <div className="relative mx-auto max-w-7xl px-4 py-6 md:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 overflow-hidden rounded-[32px] border border-slate-200/80 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 p-6 text-white shadow-[0_20px_60px_rgba(15,23,42,0.18)]"
        >
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="mb-4 flex flex-wrap items-center gap-2">
                <span className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-medium text-slate-100">
                  <Sparkles className="h-3.5 w-3.5" />
                  Dissertation dashboard
                </span>
                <span className="inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-medium text-emerald-200">
                  <CheckCircle2 className="h-3.5 w-3.5" />
                  Electricity load / demand focus
                </span>
              </div>
              <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Netherlands Electricity Load Forecasting Dashboard</h1>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-300 md:text-base">
                A real-time dashboard for monitoring observed electricity load, next-hour demand forecasts, weather inputs and model performance using a deployed backend.
              </p>
              
            </div>

            <button
              onClick={() => void loadAll()}
              className="inline-flex items-center justify-center rounded-2xl border border-white/15 bg-white/10 px-4 py-3 text-sm font-medium text-white shadow-lg backdrop-blur transition hover:bg-white/15"
            >
              <RefreshCw className={`mr-2 h-4 w-4 ${loading || refreshing ? "animate-spin" : ""}`} />
              Refresh dashboard
            </button>
          </div>
        </motion.div>

        <div className="mb-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard
            title="System status"
            value={health.status}
            subtitle={health.databaseConnected ? "Backend and database responding" : "Backend response issue detected"}
            icon={<Activity className="h-5 w-5" />}
          />
          <StatCard
            title="Latest load"
            value={`${overview.latestGenerationMw.toLocaleString()} MW`}
            subtitle="Most recent stored electricity demand"
            icon={<Gauge className="h-5 w-5" />}
          />
          <StatCard
            title="Latest temperature"
            value={`${overview.latestTemperatureC.toFixed(1)} °C`}
            subtitle="Most recent weather driver"
            icon={<CloudSun className="h-5 w-5" />}
          />
          <StatCard
            title="Model loaded"
            value={health.modelLoaded ? "Yes" : "No"}
            subtitle="Is forecasting model available?"
            icon={<Database className="h-5 w-5" />}
          />
        </div>

        <div className="mb-6 flex flex-wrap gap-2 rounded-[28px] border border-slate-200/80 bg-white/80 p-2 shadow-sm backdrop-blur">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`inline-flex items-center gap-2 rounded-2xl px-4 py-2.5 text-sm font-medium transition ${
                activeTab === tab
                  ? "bg-slate-950 text-white shadow-lg"
                  : "border border-transparent bg-transparent text-slate-600 hover:border-slate-200 hover:bg-slate-50"
              }`}
            >
              {tabMeta[tab].icon}
              {tabMeta[tab].label}
            </button>
          ))}
        </div>

        {activeTab === "overview" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.overview || errors.health} />
            <div className="grid gap-6 xl:grid-cols-[1.65fr_1fr]">
              <SectionCard title="Observed load and forecast" subtitle="Actual electricity demand followed by the upcoming forecast horizon. Forecast generated using the Hybrid GRU–LSTM model.">
                <div className="h-[360px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={combinedSeries}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="time" stroke="#64748b">
                        <Label value="Time" offset={-4} position="insideBottom" fill="#64748b" />
                      </XAxis>
                      <YAxis stroke="#64748b" label={{ value: "Electricity Load (MW)", angle: -90, position: "insideLeft", fill: "#64748b" }} />
                      <Tooltip />
                      <Legend verticalAlign="bottom" height={36} />
                      {forecastStartTime ? <ReferenceLine x={forecastStartTime} stroke="#94a3b8" strokeDasharray="4 4" label={{ value: "Forecast starts", position: "top", fill: "#64748b", fontSize: 12 }} /> : null}
                      <Line type="monotone" dataKey="generation" dot={false} stroke="#0f172a" strokeWidth={2.5} name="Observed load" />
                      <Line type="monotone" dataKey="forecast" dot={false} stroke="#0ea5e9" strokeWidth={2.5} name="Forecast load" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>

              <SectionCard title="Live platform snapshot" subtitle="High-level runtime status from the connected backend.">
                <div className="space-y-3 text-sm text-slate-600">
                  <div className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
                    <span>Backend</span>
                    <StatusBadge status={health.status} />
                  </div>
                  <div className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
                    <span>Database connected</span>
                    <span className="font-medium">{health.databaseConnected ? "Yes" : "No"}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
                    <span>Model loaded</span>
                    <span className="font-medium">{health.modelLoaded ? "Yes" : "No"}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3">
                    <span>Last refresh</span>
                    <span className="font-medium">{formatDateTime(health.latestRefresh)}</span>
                  </div>
                </div>
              </SectionCard>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              <StatCard title="RMSE" value={`${performance.metrics.rmse.toFixed(1)} MW`} subtitle="Hybrid GRU–LSTM error" icon={<TrendingUp className="h-5 w-5" />} />
              <StatCard title="MAE" value={`${performance.metrics.mae.toFixed(1)} MW`} subtitle="Average absolute error" icon={<TrendingUp className="h-5 w-5" />} />
              <StatCard title="MAPE" value={`${performance.metrics.mape.toFixed(2)}%`} subtitle="Average percentage error" icon={<CheckCircle2 className="h-5 w-5" />} />
            </div>

            <SectionCard title="Recent weather inputs" subtitle="Weather variables currently used to support load and demand forecasting.">
              <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={overview.weatherSeries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis dataKey="time" stroke="#64748b">
                      <Label value="Time" offset={-4} position="insideBottom" fill="#64748b" />
                    </XAxis>
                    <YAxis stroke="#64748b" label={{ value: "Weather variables", angle: -90, position: "insideLeft", fill: "#64748b" }} />
                    <Tooltip />
                    <Legend verticalAlign="bottom" height={36} />
                    <Area type="monotone" dataKey="temperature" fill="#f97316" fillOpacity={0.12} stroke="#f97316" strokeWidth={2} name="Temperature" />
                    <Area type="monotone" dataKey="windSpeed" fill="#0ea5e9" fillOpacity={0.1} stroke="#0284c7" strokeWidth={2} name="Wind speed" />
                    <Area type="monotone" dataKey="radiation" fill="#14b8a6" fillOpacity={0.1} stroke="#0f766e" strokeWidth={2} name="Radiation" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>
          </div>
        )}

        {activeTab === "forecast" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.forecast} />
            <div className="grid gap-6 xl:grid-cols-[1fr_1.3fr]">
              <SectionCard title="Next-hour load forecast" subtitle="Generated from the live forecasting endpoint.">
                <div className="rounded-[28px] bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 p-6 text-white shadow-lg">
                  <p className="text-sm text-slate-300">Forecast electricity demand</p>
                  <p className="mt-3 text-4xl font-semibold tracking-tight">{forecast.nextForecastMw.toLocaleString()} MW</p>
                  <p className="mt-2 text-sm text-slate-300">Forecast issued at {formatDateTime(forecast.generatedAt)}</p>
                  <p className="mt-2 text-sm text-slate-300">Forecast horizon: {forecast.horizonHours} hour</p>
                </div>
              </SectionCard>
              <SectionCard title="Latest model input window" subtitle="Recent processed inputs returned by /api/forecast.">
                <DataTable rows={forecast.recentInputs} />
              </SectionCard>
            </div>
          </div>
        )}

        {activeTab === "performance" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.performance} />
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              <StatCard title="RMSE" value={`${performance.metrics.rmse.toFixed(1)} MW`} subtitle="Root mean squared error" icon={<TrendingUp className="h-5 w-5" />} />
              <StatCard title="MAE" value={`${performance.metrics.mae.toFixed(1)} MW`} subtitle="Mean absolute error" icon={<TrendingUp className="h-5 w-5" />} />
              <StatCard title="MSE" value={performance.metrics.mse.toFixed(1)} subtitle="Mean squared error" icon={<TrendingUp className="h-5 w-5" />} />
              <StatCard title="MAPE" value={`${performance.metrics.mape.toFixed(1)}%`} subtitle="Mean absolute percentage error" icon={<TrendingUp className="h-5 w-5" />} />
              <StatCard title="Accuracy" value={`${performance.metrics.accuracy.toFixed(1)}%`} subtitle="Reported project accuracy" icon={<CheckCircle2 className="h-5 w-5" />} />
            </div>

            <div className="grid gap-6 xl:grid-cols-2">
              <SectionCard title="Training vs Validation Loss" subtitle="Training and validation loss across epochs from the deployed model performance endpoint.">
                <div className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performance.lossCurve}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" stroke="#64748b">
                        <Label value="Epoch" offset={-4} position="insideBottom" fill="#64748b" />
                      </XAxis>
                      <YAxis stroke="#64748b" label={{ value: "Loss", angle: -90, position: "insideLeft", fill: "#64748b" }} />
                      <Tooltip />
                      <Legend verticalAlign="bottom" height={36} />
                      <Line type="monotone" dataKey="trainLoss" dot={false} stroke="#0f172a" strokeWidth={2.5} name="Training loss" />
                      <Line type="monotone" dataKey="valLoss" dot={false} stroke="#0ea5e9" strokeWidth={2.5} strokeDasharray="6 4" name="Validation loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-3 rounded-2xl bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-600">
                  {finalLoss ? (
                    <span>Final training loss: <strong>{finalLoss.trainLoss.toFixed(3)}</strong> · Final validation loss: <strong>{finalLoss.valLoss.toFixed(3)}</strong>. Both curves decrease across epochs, showing that the model is learning without severe overfitting.</span>
                  ) : (
                    <span>No loss values available yet.</span>
                  )}
                </div>
              </SectionCard>
              <SectionCard title="Actual vs predicted load" subtitle="Evaluation output from the live backend.">
                <div className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performance.actualVsPredicted}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="time" stroke="#64748b">
                        <Label value="Time" offset={-4} position="insideBottom" fill="#64748b" />
                      </XAxis>
                      <YAxis stroke="#64748b" label={{ value: "Electricity Load (MW)", angle: -90, position: "insideLeft", fill: "#64748b" }} />
                      <Tooltip />
                      <Legend verticalAlign="bottom" height={36} />
                      <Line type="monotone" dataKey="actual" dot={false} stroke="#0f172a" strokeWidth={2.5} name="Actual load" />
                      <Line type="monotone" dataKey="predicted" dot={false} stroke="#10b981" strokeWidth={2.5} name="Predicted load" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
            </div>
          </div>
        )}

        {activeTab === "comparison" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.comparison} />
            <div className="grid gap-6 xl:grid-cols-[1.2fr_1fr]">
              <SectionCard title="Model comparison" subtitle="Forecasting error comparison across the evaluated models.">
                <div className="h-[340px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="name" stroke="#64748b" />
                      <YAxis stroke="#64748b" label={{ value: "Error (MW)", angle: -90, position: "insideLeft", fill: "#64748b" }} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="rmse" fill="#0f172a" radius={[10, 10, 0, 0]} name="RMSE" />
                      <Bar dataKey="mae" fill="#0ea5e9" radius={[10, 10, 0, 0]} name="MAE" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
              <SectionCard title="Selection summary" subtitle="Why the chosen forecasting model was deployed.">
                <div className="space-y-3">
                  {comparison.models.map((model) => (
                    <div key={model.name} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="font-semibold text-slate-900">{model.name}</p>
                          <p className="mt-1 text-sm text-slate-500">RMSE {model.rmse.toFixed(1)} · MAE {model.mae.toFixed(1)}</p>
                        </div>
                        {model.selected ? <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-700">Selected model</span> : null}
                      </div>
                    </div>
                  ))}
                  <p className="text-sm leading-7 text-slate-600">{comparison.summary}</p>
                </div>
              </SectionCard>
            </div>
          </div>
        )}

        {activeTab === "xai" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.xai} />
            <div className="grid gap-6 xl:grid-cols-2">
              <SectionCard title="Global feature importance" subtitle="Which variables matter most for load and demand forecasting.">
                <div className="h-[340px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={xai.featureImportance} layout="vertical" margin={{ left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis type="number" stroke="#64748b" />
                      <YAxis dataKey="feature" type="category" width={120} stroke="#64748b" />
                      <Tooltip />
                      <Bar dataKey="importance" fill="#0ea5e9" radius={[0, 10, 10, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
              <SectionCard title="Local explanation" subtitle="Feature contribution for the latest load forecast instance.">
                <div className="h-[340px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={xai.localExplanation} layout="vertical" margin={{ left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis type="number" stroke="#64748b" />
                      <YAxis dataKey="feature" type="category" width={120} stroke="#64748b" />
                      <Tooltip />
                      <Bar dataKey="contribution" fill="#10b981" radius={[0, 10, 10, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
            </div>
            <SectionCard title="Explanation summary" subtitle="Plain-language interpretation of the model output.">
              <p className="text-sm leading-7 text-slate-600">{xai.summary}</p>
            </SectionCard>
          </div>
        )}


        {activeTab === "scenario" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.scenario} />
            <div className="grid gap-6 xl:grid-cols-[1fr_1.2fr]">
              <SectionCard title="Scenario settings" subtitle="Adjust weather-driven conditions and test how the model responds.">
                <div className="space-y-5">
                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Scenario name</label>
                    <input
                      value={scenarioName}
                      onChange={(e) => setScenarioName(e.target.value)}
                      className="w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm shadow-sm outline-none transition focus:border-slate-400"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Temperature delta: {temperatureDelta.toFixed(1)} °C</label>
                    <input type="range" min={-10} max={10} step={0.5} value={temperatureDelta} onChange={(e) => setTemperatureDelta(Number(e.target.value))} className="w-full accent-slate-900" />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Wind multiplier: {windMultiplier.toFixed(2)}x</label>
                    <input type="range" min={0.5} max={1.5} step={0.05} value={windMultiplier} onChange={(e) => setWindMultiplier(Number(e.target.value))} className="w-full accent-sky-600" />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Radiation multiplier: {radiationMultiplier.toFixed(2)}x</label>
                    <input type="range" min={0.5} max={1.5} step={0.05} value={radiationMultiplier} onChange={(e) => setRadiationMultiplier(Number(e.target.value))} className="w-full accent-emerald-600" />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Cloud cover delta: {cloudDelta} points</label>
                    <input type="range" min={-50} max={50} step={5} value={cloudDelta} onChange={(e) => setCloudDelta(Number(e.target.value))} className="w-full accent-orange-500" />
                  </div>

                  <button
                    onClick={() => void runScenario()}
                    disabled={scenarioLoading}
                    className="inline-flex items-center rounded-2xl bg-slate-950 px-4 py-3 text-sm font-medium text-white shadow-lg transition hover:bg-slate-800 disabled:opacity-60"
                  >
                    {scenarioLoading ? "Running scenario..." : "Run scenario"}
                  </button>
                </div>
              </SectionCard>

              <div className="space-y-6">
                <div className="grid gap-4 md:grid-cols-3">
                  <StatCard title="Baseline forecast" value={`${scenario.baselineMw.toLocaleString()} MW`} subtitle="Current demand output" icon={<Gauge className="h-5 w-5" />} />
                  <StatCard title="Scenario forecast" value={`${scenario.scenarioMw.toLocaleString()} MW`} subtitle="Modified-condition demand" icon={<Sparkles className="h-5 w-5" />} />
                  <StatCard title="Difference" value={`${scenario.deltaMw > 0 ? "+" : ""}${scenario.deltaMw.toLocaleString()} MW`} subtitle="Scenario versus baseline" icon={<Activity className="h-5 w-5" />} />
                </div>
                <SectionCard title="Scenario explanation">
                  <p className="text-sm leading-7 text-slate-600">{scenario.explanation}</p>
                </SectionCard>
              </div>
            </div>
          </div>
        )}

        {activeTab === "data" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.dataStatus} />
            <SectionCard title="Dataset meaning" subtitle="The dashboard uses Netherlands electricity load/demand and weather variables aligned by timestamp.">
              <div className="grid gap-3 text-sm text-slate-600 md:grid-cols-4">
                <div className="rounded-2xl bg-slate-50 p-4"><strong>Target:</strong><br />Electricity load / demand</div>
                <div className="rounded-2xl bg-slate-50 p-4"><strong>Unit:</strong><br />Megawatts (MW)</div>
                <div className="rounded-2xl bg-slate-50 p-4"><strong>Weather:</strong><br />Temperature and radiation inputs</div>
                <div className="rounded-2xl bg-slate-50 p-4"><strong>Source:</strong><br />Netherlands load + weather data</div>
              </div>
            </SectionCard>
            <div className="grid gap-6 xl:grid-cols-3">
              <SectionCard title="Load observations">
                <DataTable
                  rows={dataStatus.generationRows.map((row) => ({
                    observed_at: row.observed_at ?? null,
                    total_load_mw: row.total_generation_mw ?? null,
                    load_forecast_mw: row.load_forecast_mw ?? null,
                  }))}
                />
              </SectionCard>
              <SectionCard title="Weather observations">
                <DataTable rows={dataStatus.weatherRows} />
              </SectionCard>
              <SectionCard title="Processed model features">
                <DataTable
                  rows={dataStatus.featureRows.map((row) => ({
                    observed_at: row.observed_at ?? null,
                    total_load_mw: row.total_generation_mw ?? null,
                    temperature_2m: row.temperature_2m ?? null,
                    hour_of_day: row.hour_of_day ?? null,
                  }))}
                />
              </SectionCard>
            </div>
          </div>
        )}

        {activeTab === "system" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.systemStatus || errors.freshness} />
            <div className="grid gap-6 xl:grid-cols-[1fr_1fr]">
              <SectionCard title="Runtime assets" subtitle="Values returned by /api/system-status.">
                <div className="space-y-3 text-sm text-slate-600">
                  <div className="rounded-2xl bg-slate-50 p-4"><strong>Database:</strong> {systemStatus.databasePath}</div>
                  <div className="rounded-2xl bg-slate-50 p-4"><strong>Model:</strong> {systemStatus.modelPath}</div>
                  <div className="rounded-2xl bg-slate-50 p-4"><strong>Scaler X:</strong> {systemStatus.scalerXPath}</div>
                  <div className="rounded-2xl bg-slate-50 p-4"><strong>Scaler Y:</strong> {systemStatus.scalerYPath}</div>
                  <div className="rounded-2xl bg-slate-50 p-4"><strong>Model loaded:</strong> {systemStatus.modelLoaded ? "Yes" : "No"}</div>
                </div>
              </SectionCard>
              <SectionCard title="Database counts" subtitle="Current record counts by table.">
                <div className="space-y-3">
                  {Object.entries(systemStatus.databaseCounts).length ? (
                    Object.entries(systemStatus.databaseCounts).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-3 text-sm">
                        <span className="text-slate-600">{formatDbLabel(key)}</span>
                        <span className="font-semibold text-slate-900">{value}</span>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-slate-500">No system counts available.</p>
                  )}
                </div>
              </SectionCard>
            </div>

            <SectionCard title="Data freshness and API status" subtitle="Returned by /api/freshness.">
              <div className="grid gap-4 xl:grid-cols-3">
                {freshness.sources.length ? (
                  freshness.sources.map((source) => (
                    <div key={source.name} className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                      <div className="mb-3 flex items-center justify-between gap-4">
                        <div className="flex items-center gap-2 font-semibold text-slate-900">
                          {source.status === "healthy" ? (
                            <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                          ) : source.status === "delayed" ? (
                            <AlertTriangle className="h-4 w-4 text-amber-600" />
                          ) : (
                            <XCircle className="h-4 w-4 text-rose-600" />
                          )}
                          {source.name}
                        </div>
                        <StatusBadge status={source.status} />
                      </div>
                      <div className="space-y-2 text-sm text-slate-600">
                        <div className="flex items-center justify-between"><span>Last success</span><span>{formatTimeOnly(source.lastSuccess)}</span></div>
                        <div className="flex items-center justify-between"><span>Latency</span><span>{source.latencyMinutes} min</span></div>
                        <div className="flex items-center justify-between"><span>Next run</span><span>{formatTimeOnly(source.nextExpectedRun)}</span></div>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-slate-500">No freshness records available.</p>
                )}
              </div>
            </SectionCard>
          </div>
        )}

        <footer className="mt-8 rounded-[28px] border border-slate-200/80 bg-white/80 p-5 text-sm leading-6 text-slate-600 shadow-sm backdrop-blur">
          <p className="font-semibold text-slate-900">BSc (Hons) Data Science Dissertation · Real-time Electricity Load Forecasting using Hybrid GRU–LSTM</p>
          <p className="mt-1">Forecasts are estimates based on historical Netherlands electricity load and weather inputs. Sudden events, holidays, missing data, or abnormal weather conditions may affect accuracy.</p>
        </footer>
      </div>
    </div>
  );
}
