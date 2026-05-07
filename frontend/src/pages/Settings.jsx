/**
 * EvolveLab — Settings Page
 * Evolution parameter configuration.
 */

import { useState } from 'react';

export default function Settings() {
  const [popSize, setPopSize] = useState(8);
  const [maxGens, setMaxGens] = useState(20);
  const [eliteCount, setEliteCount] = useState(2);
  const [tournamentSize, setTournamentSize] = useState(3);
  const [accWeight, setAccWeight] = useState(0.7);
  const [costWeight, setCostWeight] = useState(0.3);
  const [mutationRate, setMutationRate] = useState(0.3);
  const [adaptiveMutation, setAdaptiveMutation] = useState(true);
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    // Settings are applied when starting a new evolution via the API
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <>
      <div className="page-header">
        <h2>Settings</h2>
        <p>Configure evolution parameters for the next run</p>
      </div>

      <div className="page-body">
        <div className="grid-2">
          {/* Evolution Parameters */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Evolution Parameters</div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <SettingRow label="Population Size" description="Number of genomes per generation">
                <input type="number" value={popSize} onChange={e => setPopSize(Number(e.target.value))} min={4} max={50} className="setting-input" />
              </SettingRow>
              <SettingRow label="Max Generations" description="Maximum number of evolution cycles">
                <input type="number" value={maxGens} onChange={e => setMaxGens(Number(e.target.value))} min={1} max={200} className="setting-input" />
              </SettingRow>
              <SettingRow label="Elite Count" description="Top N genomes guaranteed to survive">
                <input type="number" value={eliteCount} onChange={e => setEliteCount(Number(e.target.value))} min={1} max={10} className="setting-input" />
              </SettingRow>
              <SettingRow label="Tournament Size" description="Number of contestants in selection tournaments">
                <input type="number" value={tournamentSize} onChange={e => setTournamentSize(Number(e.target.value))} min={2} max={10} className="setting-input" />
              </SettingRow>
            </div>
          </div>

          {/* Fitness Weights */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Fitness Configuration</div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <SettingRow label="Accuracy Weight" description="Weight for accuracy in fitness calculation">
                <input type="number" value={accWeight} onChange={e => { setAccWeight(Number(e.target.value)); setCostWeight(round(1 - Number(e.target.value))); }} min={0} max={1} step={0.05} className="setting-input" />
              </SettingRow>
              <SettingRow label="Cost Weight" description="Weight for compute cost penalty">
                <input type="number" value={costWeight} onChange={e => { setCostWeight(Number(e.target.value)); setAccWeight(round(1 - Number(e.target.value))); }} min={0} max={1} step={0.05} className="setting-input" />
              </SettingRow>
              <div style={{ padding: '12px 16px', background: 'var(--bg-input)', borderRadius: 'var(--radius-md)', fontSize: 13 }}>
                <span style={{ color: 'var(--text-tertiary)' }}>Fitness formula: </span>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", color: 'var(--accent-indigo-light)' }}>
                  ({accWeight} * accuracy) - ({costWeight} * cost)
                </span>
              </div>
            </div>
          </div>

          {/* Mutation Settings */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">Mutation Settings</div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <SettingRow label="Base Mutation Rate" description="Probability of mutation per genome">
                <input type="number" value={mutationRate} onChange={e => setMutationRate(Number(e.target.value))} min={0.05} max={0.8} step={0.05} className="setting-input" />
              </SettingRow>
              <SettingRow label="Adaptive Mutation" description="Automatically adjust rates based on success history">
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  <input
                    type="checkbox" checked={adaptiveMutation}
                    onChange={e => setAdaptiveMutation(e.target.checked)}
                    style={{ width: 16, height: 16, accentColor: 'var(--accent-indigo)' }}
                  />
                  <span style={{ fontSize: 13, color: adaptiveMutation ? 'var(--accent-emerald)' : 'var(--text-tertiary)' }}>
                    {adaptiveMutation ? 'Enabled' : 'Disabled'}
                  </span>
                </label>
              </SettingRow>
            </div>
          </div>

          {/* Info */}
          <div className="card animate-in">
            <div className="card-header">
              <div className="card-title">System Info</div>
            </div>
            <table className="data-table">
              <tbody>
                <tr><td>Version</td><td>1.0.0</td></tr>
                <tr><td>Backend</td><td>FastAPI + SQLite</td></tr>
                <tr><td>Evaluation</td><td>Heuristic (no GPU required)</td></tr>
                <tr><td>Agents</td><td>5 builder species + evaluator + selection + mutation + crossover + meta-prompt</td></tr>
                <tr><td>Novelty Features</td><td>Meta-prompt evolution, adaptive mutation rates</td></tr>
              </tbody>
            </table>

            <div style={{ marginTop: 16 }}>
              <button className="btn btn-primary" onClick={handleSave}>
                {saved ? 'Settings Saved' : 'Save Settings'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function SettingRow({ label, description, children }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
      <div>
        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>{label}</div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)' }}>{description}</div>
      </div>
      {children}
    </div>
  );
}

function round(n) {
  return Math.round(n * 100) / 100;
}
