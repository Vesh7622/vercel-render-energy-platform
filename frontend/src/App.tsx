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
    <div className={`rounded-3xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
        {subtitle ? <p className="mt-1 text-sm text-slate-500">{subtitle}</p> : null}
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
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm text-slate-500">{title}</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{value}</p>
          <p className="mt-2 text-sm text-slate-500">{subtitle}</p>
        </div>
        <div className="rounded-2xl bg-slate-100 p-3 text-slate-700">{icon}</div>
      </div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const className =
    status === "ok" || status === "healthy"
      ? "bg-emerald-100 text-emerald-700"
      : status === "degraded" || status === "delayed"
        ? "bg-amber-100 text-amber-700"
        : "bg-red-100 text-red-700";

  return <span className={`rounded-full px-3 py-1 text-xs font-medium ${className}`}>{status}</span>;
}

function DataTable({ rows }: { rows: Array<Record<string, string | number | null>> }) {
  if (!rows.length) {
    return <p className="text-sm text-slate-500">No data available.</p>;
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-slate-600">
            <tr>
              {columns.map((column) => (
                <th key={column} className="whitespace-nowrap px-3 py-2 font-medium">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-t border-slate-100">
                {columns.map((column) => (
                  <td key={column} className="whitespace-nowrap px-3 py-2 text-slate-700">
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
    <div className="mb-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      {message}
    </div>
  );
}

const tabs = [
  "overview",
  "forecast",
  "performance",
  "comparison",
  "xai",
  "history",
  "scenario",
  "data",
  "system",
] as const;

type TabKey = (typeof tabs)[number];

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

  const [scenarioName, setScenarioName] = useState("Wind increase scenario");
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
    else nextErrors.xai = xaiResult.reason instanceof Error ? xaiResult.reason.message : "XAI endpoint failed.";

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
    loadAll();
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto max-w-7xl px-4 py-6">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm"
        >
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-slate-100 p-3 text-slate-700">
                  <Sparkles className="h-6 w-6" />
                </div>
                <div>
                  <h1 className="text-2xl font-semibold">Netherlands Electricity Forecasting Platform</h1>
                  <p className="mt-1 text-sm text-slate-500">
                    Live dashboard using your Render backend instead of local demo-only frontend data.
                  </p>
                </div>
              </div>
              <p className="mt-4 text-xs text-slate-500">
                API base: <span className="font-medium">{API_BASE || "same-origin / not configured"}</span>
              </p>
            </div>
            <button
              onClick={loadAll}
              className="inline-flex items-center rounded-2xl border border-slate-200 bg-white px-4 py-2 text-sm shadow-sm hover:bg-slate-50"
            >
              <RefreshCw className={`mr-2 h-4 w-4 ${loading || refreshing ? "animate-spin" : ""}`} />
              Refresh all sections
            </button>
          </div>
        </motion.div>

        <div className="mb-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard
            title="System status"
            value={health.status}
            subtitle={health.databaseConnected ? "Backend and DB responding" : "Backend response issue"}
            icon={<Activity className="h-5 w-5" />}
          />
          <StatCard
            title="Latest generation"
            value={`${overview.latestGenerationMw.toLocaleString()} MW`}
            subtitle="Most recent stored generation"
            icon={<Gauge className="h-5 w-5" />}
          />
          <StatCard
            title="Latest temperature"
            value={`${overview.latestTemperatureC.toFixed(1)} °C`}
            subtitle="Latest weather input"
            icon={<CloudSun className="h-5 w-5" />}
          />
          <StatCard
            title="Model loaded"
            value={health.modelLoaded ? "Yes" : "No"}
            subtitle="Real model availability on Render"
            icon={<Database className="h-5 w-5" />}
          />
        </div>

        <div className="mb-6 flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`rounded-2xl px-4 py-2 text-sm font-medium capitalize ${
                activeTab === tab ? "bg-slate-900 text-white" : "bg-white text-slate-700 border border-slate-200"
              }`}
            >
              {tab === "xai" ? "Explainability" : tab.replace("-", " ")}
            </button>
          ))}
        </div>

        {activeTab === "overview" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.overview || errors.health} />
            <div className="grid gap-6 xl:grid-cols-[1.6fr_1fr]">
              <SectionCard title="Generation and forecast" subtitle="Actual observed generation followed by the forecast horizon.">
                <div className="h-[360px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={combinedSeries}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="generation" dot={false} strokeWidth={2.5} name="Actual generation" />
                      <Line type="monotone" dataKey="forecast" dot={false} strokeWidth={2.5} name="Forecast" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>

              <SectionCard title="Live platform snapshot" subtitle="High-level runtime status from the real backend.">
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
                    <span className="font-medium">{new Date(health.latestRefresh || new Date().toISOString()).toLocaleString()}</span>
                  </div>
                </div>
              </SectionCard>
            </div>

            <SectionCard title="Recent weather" subtitle="Weather values coming from the current backend response.">
              <div className="h-[320px]">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={overview.weatherSeries}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="temperature" fillOpacity={0.1} strokeWidth={2} name="Temperature" />
                    <Area type="monotone" dataKey="windSpeed" fillOpacity={0.1} strokeWidth={2} name="Wind speed" />
                    <Area type="monotone" dataKey="radiation" fillOpacity={0.08} strokeWidth={2} name="Radiation" />
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
              <SectionCard title="Automated forecast" subtitle="Generated by the current backend forecast endpoint.">
                <div className="rounded-3xl bg-slate-900 p-6 text-white">
                  <p className="text-sm text-slate-300">Next forecasted generation</p>
                  <p className="mt-3 text-4xl font-semibold">{forecast.nextForecastMw.toLocaleString()} MW</p>
                  <p className="mt-2 text-sm text-slate-300">Generated at {new Date(forecast.generatedAt).toLocaleString()}</p>
                  <p className="mt-2 text-sm text-slate-300">Horizon: {forecast.horizonHours} hour</p>
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
              <SectionCard title="Training and validation loss" subtitle="Live data from /api/performance.">
                <div className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performance.lossCurve}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="epoch" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="trainLoss" dot={false} strokeWidth={2.5} name="Training loss" />
                      <Line type="monotone" dataKey="valLoss" dot={false} strokeWidth={2.5} name="Validation loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
              <SectionCard title="Actual vs predicted" subtitle="Evaluation series from the real backend.">
                <div className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={performance.actualVsPredicted}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="actual" dot={false} strokeWidth={2.5} name="Actual" />
                      <Line type="monotone" dataKey="predicted" dot={false} strokeWidth={2.5} name="Predicted" />
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
              <SectionCard title="Baseline model comparison" subtitle="Error comparison across models from /api/comparison.">
                <div className="h-[340px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="rmse" radius={[10, 10, 0, 0]} name="RMSE" />
                      <Bar dataKey="mae" radius={[10, 10, 0, 0]} name="MAE" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
              <SectionCard title="Selection summary" subtitle="Why the chosen model was deployed.">
                <div className="space-y-3">
                  {comparison.models.map((model) => (
                    <div key={model.name} className="rounded-2xl bg-slate-50 p-4">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <p className="font-semibold text-slate-900">{model.name}</p>
                          <p className="mt-1 text-sm text-slate-500">RMSE {model.rmse.toFixed(1)} · MAE {model.mae.toFixed(1)}</p>
                        </div>
                        {model.selected ? <StatusBadge status="healthy" /> : null}
                      </div>
                    </div>
                  ))}
                  <p className="text-sm leading-6 text-slate-600">{comparison.summary}</p>
                </div>
              </SectionCard>
            </div>
          </div>
        )}

        {activeTab === "xai" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.xai} />
            <div className="grid gap-6 xl:grid-cols-2">
              <SectionCard title="Global feature importance" subtitle="Explainability values from /api/xai.">
                <div className="h-[340px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={xai.featureImportance} layout="vertical" margin={{ left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="feature" type="category" width={120} />
                      <Tooltip />
                      <Bar dataKey="importance" radius={[0, 10, 10, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
              <SectionCard title="Local explanation" subtitle="Feature contribution for the latest forecast instance.">
                <div className="h-[340px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={xai.localExplanation} layout="vertical" margin={{ left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="feature" type="category" width={120} />
                      <Tooltip />
                      <Bar dataKey="contribution" radius={[0, 10, 10, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
            </div>
            <SectionCard title="Explanation summary" subtitle="Plain-language interpretation.">
              <p className="text-sm leading-7 text-slate-600">{xai.summary}</p>
            </SectionCard>
          </div>
        )}

        {activeTab === "history" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.history} />
            <div className="grid gap-6 xl:grid-cols-[1.2fr_1fr]">
              <SectionCard title="Forecast archive" subtitle="Predicted vs actual values from /api/forecast-history.">
                <div className="h-[340px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={historyChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="predicted" dot={false} strokeWidth={2.5} name="Predicted" />
                      <Line type="monotone" dataKey="actual" dot={false} strokeWidth={2.5} name="Actual" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
              <SectionCard title="Recent forecast errors" subtitle="Absolute error from archived results.">
                <div className="h-[340px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={historyChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="error" radius={[10, 10, 0, 0]} name="Absolute error" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </SectionCard>
            </div>
            <SectionCard title="History table">
              <DataTable
                rows={history.rows.map((row) => ({
                  issued_at: row.issuedAt,
                  forecast_for: row.forecastFor,
                  predicted_mw: row.predictedMw,
                  actual_mw: row.actualMw,
                  absolute_error: row.absoluteError,
                }))}
              />
            </SectionCard>
          </div>
        )}

        {activeTab === "scenario" && (
          <div className="space-y-6">
            <ErrorBanner message={errors.scenario} />
            <div className="grid gap-6 xl:grid-cols-[1fr_1.2fr]">
              <SectionCard title="What-if scenario settings" subtitle="These values are sent to /api/scenario.">
                <div className="space-y-5">
                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Scenario name</label>
                    <input
                      value={scenarioName}
                      onChange={(e) => setScenarioName(e.target.value)}
                      className="w-full rounded-2xl border border-slate-200 px-4 py-2 text-sm"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Temperature delta: {temperatureDelta.toFixed(1)} °C</label>
                    <input type="range" min={-10} max={10} step={0.5} value={temperatureDelta} onChange={(e) => setTemperatureDelta(Number(e.target.value))} className="w-full" />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Wind multiplier: {windMultiplier.toFixed(2)}x</label>
                    <input type="range" min={0.5} max={1.5} step={0.05} value={windMultiplier} onChange={(e) => setWindMultiplier(Number(e.target.value))} className="w-full" />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Radiation multiplier: {radiationMultiplier.toFixed(2)}x</label>
                    <input type="range" min={0.5} max={1.5} step={0.05} value={radiationMultiplier} onChange={(e) => setRadiationMultiplier(Number(e.target.value))} className="w-full" />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm font-medium text-slate-700">Cloud cover delta: {cloudDelta} points</label>
                    <input type="range" min={-50} max={50} step={5} value={cloudDelta} onChange={(e) => setCloudDelta(Number(e.target.value))} className="w-full" />
                  </div>

                  <button
                    onClick={runScenario}
                    disabled={scenarioLoading}
                    className="inline-flex items-center rounded-2xl bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-60"
                  >
                    {scenarioLoading ? "Running scenario..." : "Run scenario"}
                  </button>
                </div>
              </SectionCard>

              <div className="space-y-6">
                <div className="grid gap-4 md:grid-cols-3">
                  <StatCard title="Baseline forecast" value={`${scenario.baselineMw.toLocaleString()} MW`} subtitle="Current model output" icon={<Gauge className="h-5 w-5" />} />
                  <StatCard title="Scenario forecast" value={`${scenario.scenarioMw.toLocaleString()} MW`} subtitle="Modified-condition output" icon={<Sparkles className="h-5 w-5" />} />
                  <StatCard title="Difference" value={`${scenario.deltaMw > 0 ? "+" : ""}${scenario.deltaMw.toLocaleString()} MW`} subtitle="Scenario vs baseline" icon={<Activity className="h-5 w-5" />} />
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
            <div className="grid gap-6 xl:grid-cols-3">
              <SectionCard title="Generation observations">
                <DataTable rows={dataStatus.generationRows} />
              </SectionCard>
              <SectionCard title="Weather observations">
                <DataTable rows={dataStatus.weatherRows} />
              </SectionCard>
              <SectionCard title="Processed model features">
                <DataTable rows={dataStatus.featureRows} />
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
                        <span className="text-slate-600">{key}</span>
                        <span className="font-semibold text-slate-900">{value}</span>
                      </div>
                    ))
                  ) : (
                    <p className="text-sm text-slate-500">No system counts available.</p>
                  )}
                </div>
              </SectionCard>
            </div>

            <SectionCard title="Data freshness / API status" subtitle="Returned by /api/freshness.">
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
                            <XCircle className="h-4 w-4 text-red-600" />
                          )}
                          {source.name}
                        </div>
                        <StatusBadge status={source.status} />
                      </div>
                      <div className="space-y-2 text-sm text-slate-600">
                        <div className="flex items-center justify-between"><span>Last success</span><span>{new Date(source.lastSuccess).toLocaleTimeString()}</span></div>
                        <div className="flex items-center justify-between"><span>Latency</span><span>{source.latencyMinutes} min</span></div>
                        <div className="flex items-center justify-between"><span>Next run</span><span>{new Date(source.nextExpectedRun).toLocaleTimeString()}</span></div>
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
      </div>
    </div>
  );
}
