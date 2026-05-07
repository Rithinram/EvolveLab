/**
 * EvolveLab — Lineage Tree Page
 * Interactive genealogy visualization showing parent-child evolution.
 */

import { useState } from 'react';
import { usePolling, getGenomes, getGenomeLineage } from '../api/client';

export default function LineageTree() {
  const { data: genomes } = usePolling(getGenomes, 5000);
  const [selectedId, setSelectedId] = useState('');
  const [lineage, setLineage] = useState(null);
  const [loading, setLoading] = useState(false);

  const allGenomes = Array.isArray(genomes) ? genomes : [];

  // Group by generation for the visual tree
  const genMap = {};
  allGenomes.forEach(g => {
    const gen = g.generation_number;
    if (!genMap[gen]) genMap[gen] = [];
    genMap[gen].push(g);
  });
  const sortedGens = Object.keys(genMap).map(Number).sort((a, b) => a - b);

  const fetchLineage = async (id) => {
    if (!id) return;
    setLoading(true);
    try {
      const data = await getGenomeLineage(id);
      setLineage(data);
      setSelectedId(id);
    } catch (e) {
      setLineage(null);
    }
    setLoading(false);
  };

  const speciesColors = {
    transformer_specialist: '#6366f1',
    efficient_architect: '#10b981',
    hybrid_innovator: '#8b5cf6',
    accuracy_maximizer: '#22d3ee',
    cost_minimizer: '#f59e0b',
  };

  return (
    <>
      <div className="page-header">
        <h2>Lineage Tree</h2>
        <p>Trace the evolutionary ancestry of genomes across generations</p>
      </div>

      <div className="page-body">
        {/* Lineage Lookup */}
        <div className="card animate-in" style={{ marginBottom: 20 }}>
          <div className="card-header">
            <div className="card-title">Ancestry Lookup</div>
          </div>
          <div className="controls-bar">
            <div className="input-group" style={{ flex: 1 }}>
              <label>Genome ID</label>
              <input
                type="text"
                value={selectedId}
                onChange={e => setSelectedId(e.target.value)}
                placeholder="Enter or click a genome ID..."
                style={{ width: '100%', maxWidth: 400, background: 'var(--bg-input)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', padding: '6px 10px', color: 'var(--text-primary)', fontSize: 13, fontFamily: "'JetBrains Mono', monospace", outline: 'none' }}
              />
            </div>
            <button className="btn btn-primary" onClick={() => fetchLineage(selectedId)} disabled={!selectedId || loading}>
              {loading ? 'Tracing...' : 'Trace Ancestry'}
            </button>
          </div>

          {/* Lineage Chain */}
          {lineage && lineage.length > 0 && (
            <div style={{ marginTop: 20 }}>
              <div className="card-subtitle" style={{ marginBottom: 12 }}>Ancestry Chain ({lineage.length} nodes)</div>
              <div style={{ display: 'flex', gap: 0, alignItems: 'center', overflowX: 'auto', padding: '10px 0' }}>
                {lineage.map((node, i) => (
                  <div key={node.id} style={{ display: 'flex', alignItems: 'center' }}>
                    <div
                      className="lineage-node"
                      style={{ borderColor: speciesColors[node.species] || 'var(--border-default)', minWidth: 160 }}
                    >
                      <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginBottom: 4 }}>Gen {node.generation_number}</div>
                      <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: 'var(--text-primary)', marginBottom: 4 }}>
                        {node.id?.slice(0, 10)}
                      </div>
                      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
                        <span className="badge badge-violet" style={{ fontSize: 10 }}>{node.species?.replace('_', ' ')}</span>
                        {node.fitness_score != null && (
                          <span style={{ fontSize: 11, color: 'var(--accent-emerald)', fontWeight: 600 }}>{node.fitness_score.toFixed(3)}</span>
                        )}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-tertiary)', marginTop: 4 }}>{node.creation_method}</div>
                    </div>
                    {i < lineage.length - 1 && (
                      <svg width="40" height="24" style={{ flexShrink: 0 }}>
                        <line x1="0" y1="12" x2="32" y2="12" stroke="var(--border-default)" strokeWidth="2" />
                        <polygon points="32,8 40,12 32,16" fill="var(--border-default)" />
                      </svg>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Generation-based Tree View */}
        <div className="card animate-in">
          <div className="card-header">
            <div className="card-title">Population by Generation</div>
            <span className="badge badge-indigo">{allGenomes.length} total genomes</span>
          </div>

          {sortedGens.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              {sortedGens.map(gen => (
                <div key={gen} style={{ marginBottom: 20 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                    Generation {gen}
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {genMap[gen].sort((a, b) => (b.fitness_score || 0) - (a.fitness_score || 0)).map(g => (
                      <div
                        key={g.id}
                        className="lineage-node"
                        onClick={() => { setSelectedId(g.id); fetchLineage(g.id); }}
                        style={{
                          borderColor: selectedId === g.id ? 'var(--accent-indigo)' : speciesColors[g.species] || 'var(--border-subtle)',
                          borderWidth: selectedId === g.id ? 2 : 1,
                          boxShadow: selectedId === g.id ? 'var(--shadow-glow)' : 'none',
                          opacity: g.is_elite ? 1 : 0.85,
                        }}
                      >
                        <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10, color: 'var(--text-secondary)' }}>
                          {g.id?.slice(0, 8)}
                        </div>
                        <div style={{ fontSize: 11, marginTop: 2, display: 'flex', gap: 4, alignItems: 'center' }}>
                          <span style={{ width: 6, height: 6, borderRadius: '50%', background: speciesColors[g.species] || '#666', display: 'inline-block' }} />
                          <span style={{ color: 'var(--accent-emerald)', fontWeight: 600 }}>{g.fitness_score?.toFixed(3) || '-'}</span>
                        </div>
                        {g.is_elite && <span className="badge badge-rose" style={{ fontSize: 9, marginTop: 4 }}>elite</span>}
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <h3>No genomes yet</h3>
              <p>Run evolution to build the lineage tree</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
