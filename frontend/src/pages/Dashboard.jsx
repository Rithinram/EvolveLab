/**
 * EvolveLab — Dashboard Page
 * KPI overview with fitness trends, best genome, and recent activity.
 */

import { usePolling, getStatus, getGenerations, getBestGenome, getEvents } from '../api/client';
import {
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';

export default function Dashboard() {
  const { data: status } = usePolling(getStatus, 2000);
  const { data: generations } = usePolling(getGenerations, 3000);
  const { data: bestGenome } = usePolling(getBestGenome, 5000);
  const { data: events } = usePolling(getEvents, 4000);

  const gens = Array.isArray(generations) ? generations : [];
  const recentEvents = Array.isArray(events) ? events.slice(0, 10) : [];

  const latestGen = gens.length > 0 ? gens[gens.length - 1] : null;
  const bestFit = bestGenome?.fitness_score || latestGen?.best_fitness || 0;
  const bestAcc = bestGenome?.accuracy || latestGen?.best_accuracy || 0;
  const totalGens = gens.length;
  const avgFit = latestGen?.avg_fitness || 0;

  // Compute improvement
  const fitnessImprovement = gens.length >= 2
    ? ((gens[gens.length - 1]?.best_fitness || 0) - (gens[0]?.best_fitness || 0)).toFixed(4)
    : '0.0000';

  const chartData = gens.map(g => ({
    gen: g.number,
    best: g.best_fitness,
    avg: g.avg_fitness,
    accuracy: g.best_accuracy,
    diversity: g.diversity_score,
  }));

  return (
    <>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Overview of the evolutionary process and current performance</p>
      </div>

      <div className="page-body">
        {/* KPI Metrics */}
        <div className="metrics-grid stagger">
          <div className="metric-card indigo animate-in">
            <div className="metric-label">Generations</div>
            <div className="metric-value indigo">{totalGens}</div>
            <div className="metric-change positive">
              {status?.running ? 'Evolving...' : 'Complete'}
            </div>
          </div>
          <div className="metric-card emerald animate-in">
            <div className="metric-label">Best Fitness</div>
            <div className="metric-value emerald">{bestFit.toFixed(4)}</div>
            <div className={`metric-change ${Number(fitnessImprovement) >= 0 ? 'positive' : 'negative'}`}>
              {Number(fitnessImprovement) >= 0 ? '+' : ''}{fitnessImprovement} total improvement
            </div>
          </div>
          <div className="metric-card cyan animate-in">
            <div className="metric-label">Best Accuracy</div>
            <div className="metric-value cyan">{(bestAcc * 100).toFixed(1)}%</div>
          </div>
          <div className="metric-card amber animate-in">
            <div className="metric-label">Avg Fitness</div>
            <div className="metric-value amber">{avgFit.toFixed(4)}</div>
          </div>
          <div className="metric-card rose animate-in">
            <div className="metric-label">Population</div>
            <div className="metric-value rose">{status?.population_size || latestGen?.population_size || 0}</div>
          </div>
        </div>

        <div className="grid-2">
          {/* Fitness Progression Chart */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Fitness Progression</div>
            </div>
            {chartData.length > 0 ? (
              <div className="chart-container">
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="gradBest" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                      </linearGradient>
                      <linearGradient id="gradAvg" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                    <XAxis dataKey="gen" stroke="#6b7280" fontSize={11} />
                    <YAxis stroke="#6b7280" fontSize={11} />
                    <Tooltip
                      contentStyle={{ background: '#1a2035', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, fontSize: 12 }}
                    />
                    <Area type="monotone" dataKey="best" stroke="#6366f1" fill="url(#gradBest)" strokeWidth={2} name="Best Fitness" />
                    <Area type="monotone" dataKey="avg" stroke="#10b981" fill="url(#gradAvg)" strokeWidth={2} name="Avg Fitness" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="empty-state">
                <h3>No data yet</h3>
                <p>Start an evolution run to see fitness progression</p>
              </div>
            )}
          </div>

          {/* Best Genome */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Best Genome</div>
              {bestGenome && <span className="badge badge-indigo">{bestGenome.species}</span>}
            </div>
            {bestGenome && !bestGenome.error ? (
              <div>
                <table className="data-table">
                  <tbody>
                    <tr><td>ID</td><td style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12 }}>{bestGenome.id?.slice(0, 12)}...</td></tr>
                    <tr><td>Generation</td><td>{bestGenome.generation_number}</td></tr>
                    <tr><td>Species</td><td><span className="badge badge-violet">{bestGenome.species}</span></td></tr>
                    <tr><td>Fitness</td><td style={{ color: '#10b981', fontWeight: 600 }}>{bestGenome.fitness_score?.toFixed(4)}</td></tr>
                    <tr><td>Accuracy</td><td>{(bestGenome.accuracy * 100)?.toFixed(1)}%</td></tr>
                    <tr><td>Compute Cost</td><td>{bestGenome.compute_cost?.toFixed(4)}</td></tr>
                    <tr><td>Parameters</td><td>{(bestGenome.param_count || 0).toLocaleString()}</td></tr>
                    <tr><td>Architecture</td><td><span className="badge badge-cyan">{bestGenome.architecture?.type}</span></td></tr>
                    <tr><td>Layers</td><td>{bestGenome.architecture?.layers?.length || 0}</td></tr>
                    <tr><td>Optimizer</td><td>{bestGenome.training_strategy?.optimizer}</td></tr>
                    <tr><td>Method</td><td><span className="badge badge-amber">{bestGenome.creation_method}</span></td></tr>
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">
                <h3>No genomes evaluated yet</h3>
                <p>Run evolution to generate and evaluate architectures</p>
              </div>
            )}
          </div>
        </div>

        {/* Recent Events */}
        <div className="card animate-in" style={{ marginTop: 20 }}>
          <div className="card-header">
            <div className="card-title">Recent Events</div>
            <span className="badge badge-indigo">{recentEvents.length} events</span>
          </div>
          {recentEvents.length > 0 ? (
            <div className="timeline">
              {recentEvents.map((event, i) => (
                <div key={event.id || i} className={`timeline-item ${event.event_type === 'generation_end' ? 'highlight' : ''}`}>
                  <div className="timeline-time">Gen {event.generation_number}</div>
                  <div className="timeline-desc">{event.description || event.event_type}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <h3>No events</h3>
              <p>Events will appear here during evolution</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
