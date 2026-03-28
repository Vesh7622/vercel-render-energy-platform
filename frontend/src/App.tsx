import { useEffect, useState } from "react";
import { Activity, RefreshCw, Wind, Database, Gauge } from "lucide-react";
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
  Bar
} from "recharts";

type HealthResponse = {
  status: string;
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
  recentInputs: Array<Record<string, string | number>>;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

const fallbackHealth: HealthResponse = {
  status: "ok",
  databaseConnected: true,
  modelLoaded: false,
  latestRefresh: new Date().toISOString()
};

const fallbackOverview: OverviewResponse = {
  latestGenerationMw: 10482,
  latestTemperatureC: 17.8,
  lookbackWindow: 24,
  storedForecasts: 48,
  actualSeries: Array.from({ length: 24 }).map((_, i) => ({
    time: `${String(i).padStart(2, "0")}:00`,
    generation: 9800 + Math.round(Math.sin(i / 3) * 350)
  })),
  forecastSeries: Array.from({ length: 12 }).map((_, i) => ({
    time: `${String(i + 24).padStart(2, "0")}:00`,
    forecast: 10150 + Math.round(Math.sin(i / 2.2) * 260)
  })),
  weatherSeries: Array.from({ length: 24 }).map((_, i) => ({
    time: `${String(i).padStart(2, "0")}:00`,
    temperature: 15 + Math.sin(i / 5) * 4,
    windSpeed: 6 + Math.cos(i / 4) * 2,
    radiation: Math.max(0, 520 * Math.sin((i / 24) * Math.PI))
  }))
};

const fallbackForecast: ForecastResponse = {
  nextForecastMw: 10526,
  generatedAt: new Date().toISOString(),
  horizonHours: 1,
  recentInputs: Array.from({ length: 8 }).map((_, i) => ({
    observed_at: `2026-03-27 ${String(8 + i).padStart(2, "0")}:00`,
    total_generation_mw: 9920 + i * 42,
    temperature_2m: 15.1 + i * 0.3,
    wind_speed_10m: 5.9 + i * 0.1
  }))
};

async function fetchJson<T>(path: string, fallback: T): Promise<T> {
  try {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) throw new Error(String(res.status));
    return (await res.json()) as T;
  } catch {
    return fallback;
  }
}

function StatCard({
  title,
  value,
  subtitle,
  icon
}: {
  title: string;
  value: string;
  subtitle: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-slate-500">{title}</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{value}</p>
          <p className="mt-2 text-sm text-slate-500">{subtitle}</p>
        </div>
        <div className="rounded-xl bg-slate-100 p-3 text-slate-700">{icon}</div>
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse>(fallbackHealth);
  const [overview, setOverview] = useState<OverviewResponse>(fallbackOverview);
  const [forecast, setForecast] = useState<ForecastResponse>(fallbackForecast);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    const [healthRes, overviewRes, forecastRes] = await Promise.all([
      fetchJson("/api/health", fallbackHealth),
      fetchJson("/api/overview", fallbackOverview),
      fetchJson("/api/forecast", fallbackForecast)
    ]);
    setHealth(healthRes);
    setOverview(overviewRes);
    setForecast(forecastRes);
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, []);

  const combinedSeries = [
    ...overview.actualSeries.map((x) => ({ ...x, forecast: null as number | null })),
    ...overview.forecastSeries.map((x) => ({ time: x.time, generation: null as number | null, forecast: x.forecast }))
  ];

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
              <h1 className="text-2xl font-semibold">Netherlands Electricity Forecasting Platform</h1>
              <p className="mt-1 text-sm text-slate-500">
                Frontend on Vercel, backend API on Render.
              </p>
            </div>
            <button
              onClick={load}
              className="inline-flex items-center rounded-2xl border border-slate-200 bg-white px-4 py-2 text-sm shadow-sm"
            >
              <RefreshCw className={`mr-2 h-4 w-4 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </button>
          </div>
        </motion.div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard
            title="System status"
            value={health.status}
            subtitle="Health endpoint"
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
            subtitle="Most recent stored weather"
            icon={<Wind className="h-5 w-5" />}
          />
          <StatCard
            title="Stored forecasts"
            value={String(overview.storedForecasts)}
            subtitle="Forecast records in DB"
            icon={<Database className="h-5 w-5" />}
          />
        </div>

        <div className="mt-6 grid gap-6 xl:grid-cols-[1.6fr_1fr]">
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-lg font-semibold">Generation and forecast</h2>
            <div className="mt-4 h-[360px]">
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
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-lg font-semibold">Automated forecast</h2>
            <p className="mt-4 text-sm text-slate-500">Next forecasted generation</p>
            <p className="mt-2 text-4xl font-semibold">{forecast.nextForecastMw.toLocaleString()} MW</p>
            <p className="mt-2 text-sm text-slate-500">
              Generated at {new Date(forecast.generatedAt).toLocaleString()}
            </p>
          </div>
        </div>

        <div className="mt-6 grid gap-6 xl:grid-cols-2">
          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-lg font-semibold">Recent weather</h2>
            <div className="mt-4 h-[320px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={overview.weatherSeries}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="temperature" dot={false} name="Temperature" />
                  <Line type="monotone" dataKey="windSpeed" dot={false} name="Wind speed" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-lg font-semibold">Recent input window</h2>
            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 text-left text-slate-600">
                  <tr>
                    {forecast.recentInputs.length > 0 &&
                      Object.keys(forecast.recentInputs[0]).map((col) => (
                        <th key={col} className="px-3 py-2 font-medium">
                          {col}
                        </th>
                      ))}
                  </tr>
                </thead>
                <tbody>
                  {forecast.recentInputs.map((row, i) => (
                    <tr key={i} className="border-t border-slate-100">
                      {Object.keys(row).map((col) => (
                        <td key={col} className="px-3 py-2">
                          {String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="mt-6 rounded-3xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-lg font-semibold">Weather radiation snapshot</h2>
          <div className="mt-4 h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={overview.weatherSeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="radiation" radius={[10, 10, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
