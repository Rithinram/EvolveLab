/**
 * EvolveLab — API Client
 * HTTP client and React hooks for backend communication.
 */

const BASE_URL = '/api';

async function fetchAPI(path, options = {}) {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

// ── Status & Health ───────────────────────────────────────────

export const getHealth = () => fetchAPI('/health');
export const getStatus = () => fetchAPI('/status');

// ── Evolution Control ─────────────────────────────────────────

export const startEvolution = (params = {}) =>
  fetchAPI('/evolution/start', {
    method: 'POST',
    body: JSON.stringify(params),
  });

export const pauseEvolution = () =>
  fetchAPI('/evolution/pause', { method: 'POST' });

export const resumeEvolution = () =>
  fetchAPI('/evolution/resume', { method: 'POST' });

export const stopEvolution = () =>
  fetchAPI('/evolution/stop', { method: 'POST' });

// ── Generations ───────────────────────────────────────────────

export const getGenerations = () => fetchAPI('/generations');
export const getGeneration = (number) => fetchAPI(`/generations/${number}`);

// ── Genomes ───────────────────────────────────────────────────

export const getGenomes = (generation = null, limit = 200) => {
  const params = new URLSearchParams();
  if (generation !== null) params.set('generation', generation);
  if (limit) params.set('limit', limit);
  return fetchAPI(`/genomes?${params}`);
};

export const getBestGenome = () => fetchAPI('/genomes/best');
export const getGenome = (id) => fetchAPI(`/genomes/${id}`);
export const getGenomeLineage = (id, depth = 10) =>
  fetchAPI(`/genomes/${id}/lineage?depth=${depth}`);

// ── Agents ────────────────────────────────────────────────────

export const getAgents = () => fetchAPI('/agents');
export const getAgent = (id) => fetchAPI(`/agents/${id}`);

// ── Mutations ─────────────────────────────────────────────────

export const getMutations = (generation = null, limit = 200) => {
  const params = new URLSearchParams();
  if (generation !== null) params.set('generation', generation);
  if (limit) params.set('limit', limit);
  return fetchAPI(`/mutations?${params}`);
};

export const getMutationAnalytics = () => fetchAPI('/mutations/analytics');

// ── Prompts ───────────────────────────────────────────────────

export const getPrompts = (agentId = null, limit = 100) => {
  const params = new URLSearchParams();
  if (agentId) params.set('agent_id', agentId);
  if (limit) params.set('limit', limit);
  return fetchAPI(`/prompts?${params}`);
};

// ── Analytics ─────────────────────────────────────────────────

export const getFitnessTrends = () => fetchAPI('/analytics/fitness');
export const getSpeciesDistribution = () => fetchAPI('/analytics/species');
export const getSurvivalRates = () => fetchAPI('/analytics/survival');

// ── Events ────────────────────────────────────────────────────

export const getEvents = (generation = null, limit = 500) => {
  const params = new URLSearchParams();
  if (generation !== null) params.set('generation', generation);
  if (limit) params.set('limit', limit);
  return fetchAPI(`/events?${params}`);
};

// ── Checkpoints ───────────────────────────────────────────────

export const getCheckpoints = () => fetchAPI('/checkpoints');

// ── Polling Hook ──────────────────────────────────────────────

import { useState, useEffect, useRef } from 'react';

export function usePolling(fetchFn, intervalMs = 3000, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);

  const fetchData = async () => {
    try {
      const result = await fetchFn();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    intervalRef.current = setInterval(fetchData, intervalMs);
    return () => clearInterval(intervalRef.current);
  }, deps);

  return { data, loading, error, refetch: fetchData };
}
