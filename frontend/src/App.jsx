import { useEffect, useState } from "react";
import { getHealth } from "./services/api";

function StatusCard({ label, value, ok, subtitle }) {
  return (
    <div className="group relative overflow-hidden rounded-2xl border border-slate-200/80 bg-white/80 p-6 shadow-sm backdrop-blur transition-all duration-300 hover:shadow-md hover:border-slate-300">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium tracking-wide text-slate-500 uppercase">{label}</p>
        <span
          className={`inline-flex h-3 w-3 rounded-full ${
            ok ? "bg-emerald-500 shadow-sm shadow-emerald-500/50 animate-pulse" : "bg-rose-500 shadow-sm shadow-rose-500/50"
          }`}
        />
      </div>
      <p className={`mt-3 text-2xl font-bold tracking-tight ${ok ? "text-slate-900" : "text-rose-600"}`}>
        {value}
      </p>
      {subtitle && <p className="mt-1 text-xs text-slate-400 font-mono">{subtitle}</p>}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(null);
  const [lastChecked, setLastChecked] = useState(null);

  const fetchHealth = () => {
    getHealth()
      .then((data) => {
        setHealth(data);
        setError(null);
        setLastChecked(new Date().toLocaleTimeString());
      })
      .catch((err) => {
        setError(err.message);
        setHealth(null);
        setLastChecked(new Date().toLocaleTimeString());
      });
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-slate-100 to-indigo-50/30 p-6 md:p-12 text-slate-800">
      <div className="mx-auto max-w-4xl">
        {/* Header */}
        <header className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between border-b border-slate-200/60 pb-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold tracking-wide uppercase mb-2">
              Platform Status
            </div>
            <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
              AI Operations Intelligence Platform
            </h1>
            <p className="mt-1 text-sm text-slate-600">
              Predictive Operations, Anomaly Detection & AI Reasoning Engine
            </p>
          </div>
          <button
            onClick={fetchHealth}
            className="self-start md:self-auto inline-flex items-center gap-2 rounded-xl bg-slate-900 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-slate-800 active:scale-95 cursor-pointer"
          >
            Refresh Status
          </button>
        </header>

        {/* Status Grid */}
        <section className="mt-8 grid gap-5 sm:grid-cols-3">
          {error ? (
            <>
              <StatusCard label="API Service" value="Unreachable" ok={false} subtitle={error} />
              <StatusCard label="Database" value="Unknown" ok={false} subtitle="Waiting for API connection" />
              <StatusCard label="Environment" value="Offline" ok={false} subtitle="Backend offline" />
            </>
          ) : !health ? (
            <>
              <StatusCard label="API Service" value="Connecting..." ok={true} subtitle="Checking endpoint" />
              <StatusCard label="Database" value="Checking..." ok={true} subtitle="Querying engine" />
              <StatusCard label="Environment" value="Loading..." ok={true} subtitle="Reading config" />
            </>
          ) : (
            <>
              <StatusCard
                label="API Service"
                value={health.api === "up" ? "Healthy" : "Degraded"}
                ok={health.api === "up"}
                subtitle={`FastAPI v${health.version}`}
              />
              <StatusCard
                label="Database"
                value={health.database === "connected" ? "Connected" : "Disconnected"}
                ok={health.database === "connected"}
                subtitle="PostgreSQL 16 Engine"
              />
              <StatusCard
                label="Environment"
                value={health.environment.toUpperCase()}
                ok={true}
                subtitle={`v${health.version} - Day 1 Skeleton`}
              />
            </>
          )}
        </section>

        {/* Footer info */}
        <footer className="mt-8 flex items-center justify-between text-xs text-slate-400">
          <span>Target Stack: React + FastAPI + PostgreSQL + ML Engine</span>
          {lastChecked && <span>Last checked: {lastChecked}</span>}
        </footer>
      </div>
    </main>
  );
}
