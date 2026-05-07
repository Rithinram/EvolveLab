/**
 * EvolveLab — Evolution Monitor Page
 * Live controls, mutation logs, and generation timeline.
 */

import { useState } from 'react';
import {
  usePolling, getStatus, getEvents, getMutations,
  startEvolution, pauseEvolution, resumeEvolution, stopEvolution
} from '../api/client';

export default function EvolutionMonitor() {
  const { data: status, refetch: refetchStatus } = usePolling(getStatus, 1500);
  const { data: events } = usePolling(getEvents, 2000);
  const { data: mutations } = usePolling(getMutations, 3000);

  const [popSize, setPopSize] = useState(8);
  const [maxGens, setMaxGens] = useState(20);
  const [error, setError] = useState(null);

  const isRunning = status?.running;
  const isPaused = status?.paused;

  const handleStart = async () => {
    try {
      setError(null);
      await startEvolution({ population_size: popSize, max_generations: maxGens });
      setTimeout(refetchStatus, 500);
    } catch (e) {
      setError(e.message);
    }
  };

  const handlePause = async () => {
    try { await pauseEvolution(); setTimeout(refetchStatus, 300); } catch (e) { setError(e.message); }
  };

  const handleResume = async () => {
    try { await resumeEvolution(); setTimeout(refetchStatus, 300); } catch (e) { setError(e.message); }
  };

  const handleStop = async () => {
    try { await stopEvolution(); setTimeout(refetchStatus, 300); } catch (e) { setError(e.message); }
  };

  const allEvents = Array.isArray(events) ? events : [];
  const allMutations = Array.isArray(mutations) ? mutations : [];

  return (
    <>
      <div className="page-header">
        <h2>Evolution Monitor</h2>
        <p>Control and observe the evolutionary process in real time</p>
      </div>

      <div className="page-body">
        {/* Controls */}
        <div className="card animate-in" style={{ marginBottom: 20 }}>
          <div className="card-header">
            <div className="card-title">Evolution Controls</div>
            <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className={`status-dot ${isRunning ? (isPaused ? 'paused' : 'running') : 'stopped'}`} />
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                {isRunning ? (isPaused ? 'Paused' : 'Running') : 'Idle'}
              </span>
            </span>
          </div>

          <div className="controls-bar">
            <div className="input-group">
              <label>Population</label>
              <input type="number" value={popSize} onChange={e => setPopSize(Number(e.target.value))} min={4} max={50} disabled={isRunning} />
            </div>
            <div className="input-group">
              <label>Generations</label>
              <input type="number" value={maxGens} onChange={e => setMaxGens(Number(e.target.value))} min={1} max={100} disabled={isRunning} />
            </div>

            <div style={{ flex: 1 }} />

            {!isRunning && (
              <button className="btn btn-primary" onClick={handleStart}>Start Evolution</button>
            )}
            {isRunning && !isPaused && (
              <>
                <button className="btn btn-secondary" onClick={handlePause}>Pause</button>
                <button className="btn btn-danger" onClick={handleStop}>Stop</button>
              </>
            )}
            {isPaused && (
              <>
                <button className="btn btn-primary" onClick={handleResume}>Resume</button>
                <button className="btn btn-danger" onClick={handleStop}>Stop</button>
              </>
            )}
          </div>

          {error && <div style={{ color: 'var(--accent-rose)', fontSize: 13, marginTop: 8 }}>{error}</div>}

          {/* Live Stats */}
          {status && (
            <div className="metrics-grid" style={{ marginTop: 16 }}>
              <div className="metric-card indigo">
                <div className="metric-label">Current Gen</div>
                <div className="metric-value indigo">{status.current_generation}</div>
              </div>
              <div className="metric-card emerald">
                <div className="metric-label">Best Fitness</div>
                <div className="metric-value emerald">{(status.best_fitness || 0).toFixed(4)}</div>
              </div>
              <div className="metric-card cyan">
                <div className="metric-label">Best Accuracy</div>
                <div className="metric-value cyan">{((status.best_accuracy || 0) * 100).toFixed(1)}%</div>
              </div>
              <div className="metric-card amber">
                <div className="metric-label">Mutation Rate</div>
                <div className="metric-value amber">{(status.mutation_rate || 0).toFixed(3)}</div>
              </div>
            </div>
          )}
        </div>

        <div className="grid-2">
          {/* Event Timeline */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Event Timeline</div>
              <span className="badge badge-indigo">{allEvents.length}</span>
            </div>
            {allEvents.length > 0 ? (
              <div className="timeline" style={{ maxHeight: 500, overflowY: 'auto' }}>
                {allEvents.slice(0, 30).map((event, i) => (
                  <div key={event.id || i} className={`timeline-item ${event.event_type.includes('end') ? 'highlight' : ''}`}>
                    <div className="timeline-time">
                      Gen {event.generation_number} / {event.event_type}
                    </div>
                    <div className="timeline-desc">{event.description}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <h3>No events yet</h3>
                <p>Start an evolution run to see events</p>
              </div>
            )}
          </div>

          {/* Mutation Log */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Mutation Log</div>
              <span className="badge badge-violet">{allMutations.length}</span>
            </div>
            {allMutations.length > 0 ? (
              <div style={{ maxHeight: 500, overflowY: 'auto' }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Gen</th>
                      <th>Type</th>
                      <th>Field</th>
                      <th>Success</th>
                    </tr>
                  </thead>
                  <tbody>
                    {allMutations.slice(0, 50).map((m, i) => (
                      <tr key={m.id || i}>
                        <td>{m.generation_number}</td>
                        <td><span className="badge badge-violet">{m.mutation_type}</span></td>
                        <td style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>{m.field_changed}</td>
                        <td>
                          {m.success === null ? (
                            <span className="badge badge-amber">pending</span>
                          ) : m.success ? (
                            <span className="badge badge-emerald">yes</span>
                          ) : (
                            <span className="badge badge-rose">no</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty-state">
                <h3>No mutations recorded</h3>
                <p>Mutations will appear after the first evolution cycle</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
