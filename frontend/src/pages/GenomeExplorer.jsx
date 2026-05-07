/**
 * EvolveLab — Genome Explorer Page
 * Searchable genome table with expandable JSON config viewer.
 */

import { useState } from 'react';
import { usePolling, getGenomes, getGenerations } from '../api/client';

export default function GenomeExplorer() {
  const { data: genomes } = usePolling(getGenomes, 4000);
  const { data: generations } = usePolling(getGenerations, 5000);

  const [selectedGen, setSelectedGen] = useState(null);
  const [expandedId, setExpandedId] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  const gens = Array.isArray(generations) ? generations : [];
  let allGenomes = Array.isArray(genomes) ? genomes : [];

  // Filter by generation
  if (selectedGen !== null) {
    allGenomes = allGenomes.filter(g => g.generation_number === selectedGen);
  }

  // Filter by search
  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    allGenomes = allGenomes.filter(g =>
      g.id?.toLowerCase().includes(q) ||
      g.species?.toLowerCase().includes(q) ||
      g.architecture?.type?.toLowerCase().includes(q) ||
      g.creation_method?.toLowerCase().includes(q)
    );
  }

  const speciesColors = {
    transformer_specialist: 'badge-indigo',
    efficient_architect: 'badge-emerald',
    hybrid_innovator: 'badge-violet',
    accuracy_maximizer: 'badge-cyan',
    cost_minimizer: 'badge-amber',
  };

  return (
    <>
      <div className="page-header">
        <h2>Genome Explorer</h2>
        <p>Inspect architectures, training strategies, and prompt configurations</p>
      </div>

      <div className="page-body">
        {/* Filters */}
        <div className="card animate-in" style={{ marginBottom: 20 }}>
          <div className="controls-bar">
            <div className="input-group">
              <label>Generation</label>
              <select value={selectedGen ?? ''} onChange={e => setSelectedGen(e.target.value ? Number(e.target.value) : null)}>
                <option value="">All</option>
                {gens.map(g => <option key={g.number} value={g.number}>Gen {g.number}</option>)}
              </select>
            </div>
            <div className="input-group" style={{ flex: 1 }}>
              <label>Search</label>
              <input
                type="text"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                placeholder="ID, species, type..."
                style={{ width: '100%', maxWidth: 300, background: 'var(--bg-input)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', padding: '6px 10px', color: 'var(--text-primary)', fontSize: 13, fontFamily: 'inherit', outline: 'none' }}
              />
            </div>
            <span className="badge badge-indigo">{allGenomes.length} genomes</span>
          </div>
        </div>

        {/* Genome Table */}
        <div className="card animate-in">
          {allGenomes.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Gen</th>
                    <th>Species</th>
                    <th>Architecture</th>
                    <th>Layers</th>
                    <th>Params</th>
                    <th>Fitness</th>
                    <th>Accuracy</th>
                    <th>Method</th>
                    <th>Elite</th>
                  </tr>
                </thead>
                <tbody>
                  {allGenomes.map(g => (
                    <>
                      <tr key={g.id} onClick={() => setExpandedId(expandedId === g.id ? null : g.id)} style={{ cursor: 'pointer' }}>
                        <td style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>{g.id?.slice(0, 10)}</td>
                        <td>{g.generation_number}</td>
                        <td><span className={`badge ${speciesColors[g.species] || 'badge-indigo'}`}>{g.species?.replace('_', ' ')}</span></td>
                        <td><span className="badge badge-cyan">{g.architecture?.type}</span></td>
                        <td>{g.architecture?.layers?.length || 0}</td>
                        <td>{(g.param_count || 0).toLocaleString()}</td>
                        <td style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{g.fitness_score?.toFixed(4) || '-'}</td>
                        <td>{g.accuracy ? `${(g.accuracy * 100).toFixed(1)}%` : '-'}</td>
                        <td><span className="badge badge-amber">{g.creation_method}</span></td>
                        <td>{g.is_elite ? <span className="badge badge-rose">elite</span> : '-'}</td>
                      </tr>
                      {expandedId === g.id && (
                        <tr key={`${g.id}-detail`}>
                          <td colSpan={10} style={{ padding: 0 }}>
                            <div style={{ padding: 20, background: 'var(--bg-input)' }}>
                              <div className="grid-3" style={{ gap: 12 }}>
                                <div>
                                  <div className="card-subtitle" style={{ marginBottom: 8 }}>Architecture</div>
                                  <pre className="code-block">{JSON.stringify(g.architecture, null, 2)}</pre>
                                </div>
                                <div>
                                  <div className="card-subtitle" style={{ marginBottom: 8 }}>Training Strategy</div>
                                  <pre className="code-block">{JSON.stringify(g.training_strategy, null, 2)}</pre>
                                </div>
                                <div>
                                  <div className="card-subtitle" style={{ marginBottom: 8 }}>Prompt Strategy</div>
                                  <pre className="code-block">{JSON.stringify(g.prompt_strategy, null, 2)}</pre>
                                  {g.parent_a_id && (
                                    <div style={{ marginTop: 12 }}>
                                      <div className="card-subtitle">Lineage</div>
                                      <div style={{ fontSize: 12, marginTop: 4, color: 'var(--text-secondary)' }}>
                                        Parent A: <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>{g.parent_a_id?.slice(0, 10)}</span>
                                      </div>
                                      {g.parent_b_id && (
                                        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                                          Parent B: <span style={{ fontFamily: "'JetBrains Mono', monospace" }}>{g.parent_b_id?.slice(0, 10)}</span>
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <h3>No genomes found</h3>
              <p>Run an evolution cycle to generate genomes</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
