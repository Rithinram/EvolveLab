/**
 * EvolveLab — Analytics Page
 * Survival rates, mutation analytics, fitness distributions, and species trends.
 */

import { usePolling, getSurvivalRates, getMutationAnalytics, getGenerations, getSpeciesDistribution } from '../api/client';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';

const COLORS = ['#6366f1', '#10b981', '#8b5cf6', '#22d3ee', '#f59e0b', '#f43f5e', '#3b82f6'];

export default function Analytics() {
  const { data: survivalData } = usePolling(getSurvivalRates, 5000);
  const { data: mutationData } = usePolling(getMutationAnalytics, 5000);
  const { data: generations } = usePolling(getGenerations, 5000);
  const { data: speciesData } = usePolling(getSpeciesDistribution, 5000);

  const survivalRates = Array.isArray(survivalData) ? survivalData : [];
  const gens = Array.isArray(generations) ? generations : [];

  // Mutation analytics for pie chart
  const mutationChartData = mutationData && typeof mutationData === 'object'
    ? Object.entries(mutationData).map(([type, stats]) => ({
        name: type.replace('_', ' '),
        value: stats.total || 0,
        successRate: ((stats.success_rate || 0) * 100).toFixed(0),
        avgDelta: (stats.avg_fitness_delta || 0).toFixed(4),
      }))
    : [];

  // Species distribution for latest generation
  const speciesChartData = [];
  if (speciesData && typeof speciesData === 'object') {
    const latestGen = Object.keys(speciesData).map(Number).sort((a, b) => b - a)[0];
    if (latestGen !== undefined && speciesData[latestGen]) {
      Object.entries(speciesData[latestGen]).forEach(([species, count]) => {
        speciesChartData.push({ name: species.replace(/_/g, ' '), value: count });
      });
    }
  }

  // Fitness distribution chart
  const fitnessChartData = gens.map(g => ({
    gen: g.number,
    best: g.best_fitness,
    avg: g.avg_fitness,
    diversity: g.diversity_score,
  }));

  return (
    <>
      <div className="page-header">
        <h2>Analytics</h2>
        <p>Survival rates, mutation effectiveness, and evolutionary trends</p>
      </div>

      <div className="page-body">
        <div className="grid-2">
          {/* Survival Rate Chart */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Survival Rates by Generation</div>
            </div>
            {survivalRates.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={survivalRates}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                  <XAxis dataKey="generation" stroke="#6b7280" fontSize={11} />
                  <YAxis stroke="#6b7280" fontSize={11} />
                  <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, fontSize: 12 }} />
                  <Bar dataKey="survived" fill="#10b981" name="Survived" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="total" fill="rgba(99,102,241,0.3)" name="Total" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state"><h3>No data</h3></div>
            )}
          </div>

          {/* Mutation Success by Type */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Mutation Distribution</div>
            </div>
            {mutationChartData.length > 0 ? (
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <ResponsiveContainer width="50%" height={280}>
                  <PieChart>
                    <Pie data={mutationChartData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={90} paddingAngle={3}>
                      {mutationChartData.map((_, idx) => (
                        <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, fontSize: 12 }} />
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ flex: 1, paddingLeft: 16 }}>
                  <table className="data-table">
                    <thead><tr><th>Type</th><th>Count</th><th>Success</th></tr></thead>
                    <tbody>
                      {mutationChartData.map((m, i) => (
                        <tr key={i}>
                          <td style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ width: 8, height: 8, borderRadius: '50%', background: COLORS[i % COLORS.length], display: 'inline-block' }} />
                            {m.name}
                          </td>
                          <td>{m.value}</td>
                          <td>{m.successRate}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="empty-state"><h3>No mutation data</h3></div>
            )}
          </div>

          {/* Diversity Trend */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Fitness and Diversity Trends</div>
            </div>
            {fitnessChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={fitnessChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                  <XAxis dataKey="gen" stroke="#6b7280" fontSize={11} />
                  <YAxis stroke="#6b7280" fontSize={11} />
                  <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, fontSize: 12 }} />
                  <Line type="monotone" dataKey="best" stroke="#6366f1" strokeWidth={2} name="Best Fitness" />
                  <Line type="monotone" dataKey="avg" stroke="#10b981" strokeWidth={2} name="Avg Fitness" />
                  <Line type="monotone" dataKey="diversity" stroke="#f59e0b" strokeWidth={2} name="Diversity" strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state"><h3>No data</h3></div>
            )}
          </div>

          {/* Species Distribution */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Species Distribution (Latest Gen)</div>
            </div>
            {speciesChartData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie data={speciesChartData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100} label={({ name, value }) => `${name}: ${value}`}>
                    {speciesChartData.map((_, idx) => (
                      <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, fontSize: 12 }} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state"><h3>No species data</h3></div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
