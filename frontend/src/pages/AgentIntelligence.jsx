/**
 * EvolveLab — Agent Intelligence Panel
 * Agent personalities, memory, performance, and prompt evolution.
 */

import { usePolling, getAgents, getPrompts } from '../api/client';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

const SPECIES_COLORS = {
  transformer_specialist: '#6366f1',
  efficient_architect: '#10b981',
  hybrid_innovator: '#8b5cf6',
  accuracy_maximizer: '#22d3ee',
  cost_minimizer: '#f59e0b',
};

export default function AgentIntelligence() {
  const { data: agents } = usePolling(getAgents, 4000);
  const { data: prompts } = usePolling(getPrompts, 5000);

  const allAgents = Array.isArray(agents) ? agents : [];
  const allPrompts = Array.isArray(prompts) ? prompts : [];

  return (
    <>
      <div className="page-header">
        <h2>Agent Intelligence</h2>
        <p>Builder agent personalities, memory, performance history, and prompt evolution</p>
      </div>

      <div className="page-body">
        {allAgents.length > 0 ? (
          <div className="stagger">
            {allAgents.map(agent => {
              const agentPrompts = allPrompts.filter(p => p.agent_id === agent.id)
                .sort((a, b) => a.generation_number - b.generation_number);
              const promptChartData = agentPrompts.map(p => ({
                gen: p.generation_number,
                fitness: p.prompt_fitness,
              }));
              const color = SPECIES_COLORS[agent.species] || '#6366f1';

              return (
                <div key={agent.id} className="card animate-in" style={{ marginBottom: 20 }}>
                  <div className="card-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <div style={{
                        width: 40, height: 40, borderRadius: 'var(--radius-md)',
                        background: color, display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontWeight: 700, fontSize: 14, color: 'white',
                      }}>
                        {agent.species?.charAt(0).toUpperCase()}
                      </div>
                      <div>
                        <div className="card-title">{agent.species?.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</div>
                        <div className="card-subtitle">{agent.personality?.slice(0, 80)}...</div>
                      </div>
                    </div>
                    <span className="badge" style={{ background: `${color}22`, color }}>{agent.species}</span>
                  </div>

                  <div className="grid-3" style={{ gap: 16 }}>
                    {/* Stats */}
                    <div>
                      <div className="card-subtitle" style={{ marginBottom: 12 }}>Performance Stats</div>
                      <table className="data-table">
                        <tbody>
                          <tr><td>Genomes Created</td><td style={{ fontWeight: 600 }}>{agent.total_genomes_created}</td></tr>
                          <tr><td>Generations Active</td><td>{agent.generations_active}</td></tr>
                          <tr><td>Best Fitness</td><td style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{agent.best_fitness_achieved?.toFixed(4)}</td></tr>
                          <tr><td>Avg Fitness</td><td>{agent.avg_fitness?.toFixed(4)}</td></tr>
                          <tr><td>Survival Rate</td><td>{(agent.survival_rate * 100).toFixed(0)}%</td></tr>
                          <tr><td>Creativity</td><td>{agent.creativity?.toFixed(2)}</td></tr>
                          <tr><td>Focus</td><td><span className="badge badge-indigo">{agent.focus}</span></td></tr>
                        </tbody>
                      </table>
                    </div>

                    {/* Prompt Strategy */}
                    <div>
                      <div className="card-subtitle" style={{ marginBottom: 12 }}>Current Prompt Strategy</div>
                      <pre className="code-block">{JSON.stringify(agent.prompt_strategy || {}, null, 2)}</pre>

                      <div className="card-subtitle" style={{ marginTop: 16, marginBottom: 8 }}>Memory</div>
                      <pre className="code-block" style={{ maxHeight: 140, overflow: 'auto' }}>
                        {JSON.stringify(agent.memory || {}, null, 2)}
                      </pre>
                    </div>

                    {/* Prompt Fitness Chart */}
                    <div>
                      <div className="card-subtitle" style={{ marginBottom: 12 }}>Prompt Fitness Evolution</div>
                      {promptChartData.length > 1 ? (
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={promptChartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.08)" />
                            <XAxis dataKey="gen" stroke="#6b7280" fontSize={10} />
                            <YAxis stroke="#6b7280" fontSize={10} />
                            <Tooltip contentStyle={{ background: '#1a2035', border: '1px solid rgba(99,102,241,0.2)', borderRadius: 8, fontSize: 11 }} />
                            <Line type="monotone" dataKey="fitness" stroke={color} strokeWidth={2} dot={{ r: 3 }} />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="empty-state" style={{ padding: 20 }}>
                          <p>Not enough data to chart</p>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="empty-state">
            <h3>No agents registered</h3>
            <p>Run an evolution cycle to initialize builder agents</p>
          </div>
        )}
      </div>
    </>
  );
}
