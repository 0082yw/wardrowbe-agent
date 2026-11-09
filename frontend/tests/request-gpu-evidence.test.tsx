import { act, render, screen } from '@testing-library/react';
import { NextIntlClientProvider } from 'next-intl';
import { afterEach, expect, it, vi } from 'vitest';
import { RequestGpuEvidence } from '@/components/request-gpu-evidence';
import { currentRequestGpuState, type RequestExecution } from '@/lib/gpu-evidence-state';
import agent from '../messages/en/agent.json';

vi.unmock('next-intl');
const model = 'qwen3:4b-instruct-2507-q4_K_M';
const request_id = 'a'.repeat(32);
const execution: RequestExecution = {
  request_id, model, execution_mode: 'remote_gpu', gpu_verified: true, gpu_evidence_status: 'verified',
  fallback_fields: [],
  gpu_evidence: {
    model, model_digest: 'b'.repeat(64), process_id: 123, process_started_at: 990,
    gpu_uuid: 'GPU-11111111-1111-1111-1111-111111111111', observed_at: 1000,
    layers_loaded: 37, layers_total: 37, model_size_bytes: 1048576, gpu_size_bytes: 1048576,
    process_gpu_memory_bytes: 2097152,
    memory_samples: [{ observed_at: 1000, process_gpu_memory_bytes: 2097152 }],
    request_id, request_started_at: 999, request_finished_at: 1000.5, evidence_id: 'c'.repeat(64),
  },
};
afterEach(() => vi.useRealTimers());

it('verifies only fresh and matching request proof', () => {
  expect(currentRequestGpuState(execution, 1001000)).toBe('verified');
  expect(currentRequestGpuState(execution, 1010000)).toBe('verified');
  expect(currentRequestGpuState(execution, 1010001)).toBe('unknown');
  expect(currentRequestGpuState(execution, 999999)).toBe('unknown');
  expect(currentRequestGpuState({ ...execution, request_id: 'd'.repeat(32) }, 1001000)).toBe('unknown');
  expect(currentRequestGpuState({ ...execution, gpu_evidence: null }, 1001000)).toBe('unknown');
});

it('shows linked evidence and expires it locally without polling', () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date(1001000));
  vi.mocked(global.fetch).mockClear();
  render(<NextIntlClientProvider locale="en" messages={{ agent }}>
    <RequestGpuEvidence execution={execution} />
  </NextIntlClientProvider>);
  expect(screen.getByText(agent.requestExecution.gpuVerified)).toBeInTheDocument();
  expect(screen.getByText(request_id)).toBeInTheDocument();
  expect(screen.getByText('123')).toBeInTheDocument();
  expect(screen.getByText('37/37')).toBeInTheDocument();
  expect(screen.getByText('2 MiB')).toBeInTheDocument();
  act(() => vi.advanceTimersByTime(9001));
  expect(screen.getByText(agent.requestExecution.gpuUnverified)).toBeInTheDocument();
  expect(screen.getByText(agent.requestExecution.proofExpired)).toBeInTheDocument();
  expect(global.fetch).not.toHaveBeenCalled();
});
