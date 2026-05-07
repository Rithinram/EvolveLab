/**
 * EvolveLab — App Entry Point
 * React Router configuration with all page routes.
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import EvolutionMonitor from './pages/EvolutionMonitor';
import GenomeExplorer from './pages/GenomeExplorer';
import LineageTree from './pages/LineageTree';
import AgentIntelligence from './pages/AgentIntelligence';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="evolution" element={<EvolutionMonitor />} />
          <Route path="genomes" element={<GenomeExplorer />} />
          <Route path="lineage" element={<LineageTree />} />
          <Route path="agents" element={<AgentIntelligence />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
