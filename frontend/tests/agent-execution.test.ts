import { describe, expect, it } from 'vitest';
import { agentExecutionSchema, agentPreviewSchema, agentStatusSchema } from '@/lib/agent';

export const executionFixture = {
  request_id: 'b'.repeat(32), execution_mode: 'remote_gpu' as const,
  model: 'qwen3:4b-instruct-2507-q4_K_M', gpu_verified: true as const,
  gpu_evidence_status: 'verified' as const, fallback_fields: [],
  gpu_evidence: {
    model: 'qwen3:4b-instruct-2507-q4_K_M', model_digest: 'a'.repeat(64),
    process_id: 123, process_started_at: 900, gpu_uuid: 'GPU-11111111-1111-1111-1111-111111111111',
    observed_at: 995, layers_loaded: 37, layers_total: 37,
    model_size_bytes: 3000, gpu_size_bytes: 3000, process_gpu_memory_bytes: 4096,
    memory_samples: [{observed_at: 995, process_gpu_memory_bytes: 4096}], request_id: 'b'.repeat(32),
    request_started_at: 994, request_finished_at: 996, evidence_id: 'c'.repeat(64),
  },
};
describe('Request-specific GPU proof contract', () => {
  it('accepts complete evidence for one request, not a global readiness claim', () => {
    expect(agentExecutionSchema.parse(executionFixture).gpu_verified).toBe(true);
    expect(agentStatusSchema.safeParse({ mode: 'remote_ollama', model: executionFixture.model,
      available: true, cpu_only: false, execution_mode: 'remote_gpu',
      tunnel_status: 'connected', gpu_verified: true }).success).toBe(false);
  });
  it.each([
    { gpu_evidence: null }, { request_id: 'd'.repeat(32) }, { model: 'qwen3:4b' },
    { execution_mode: 'local_cpu' }, { gpu_evidence_status: 'unavailable' },
    { gpu_verified: 1 }, { gpu_verified: 'true' }, { model: 'https://external.invalid' },
  ])('rejects unsupported verified flags %#', (change) => {
    expect(agentExecutionSchema.safeParse({ ...executionFixture, ...change }).success).toBe(false);
  });
  it.each([
    { process_id: 0 }, { gpu_uuid: 'GPU-7925e602-ae47-c052-60bb-9b9158b6132c' },
    { layers_loaded: 36 }, { gpu_size_bytes: 0 }, { observed_at: 997 },
    { observed_at: 993 }, { process_started_at: 1000 }, { request_id: 'd'.repeat(32) },
  ])('rejects inconsistent hardware or timing evidence %#', (change) => {
    expect(agentExecutionSchema.safeParse({ ...executionFixture,
      gpu_evidence: { ...executionFixture.gpu_evidence, ...change } }).success).toBe(false);
  });
  it('retains old preview compatibility and explicit absence of evidence', () => {
    expect(agentPreviewSchema.parse({ status: 'no_items', answer: 'No items', outfits: [] }).execution).toBeUndefined();
    expect(agentExecutionSchema.parse({ ...executionFixture, gpu_verified: 'unknown',
      gpu_evidence_status: 'unavailable', gpu_evidence: null }).gpu_verified).toBe('unknown');
  });
});


describe('Request VRAM sample completeness', () => {
 it.each([
   {process_gpu_memory_bytes: null}, {process_gpu_memory_bytes: 0},
   {memory_samples: []},
   {memory_samples: [{observed_at: 993,process_gpu_memory_bytes: 4096}]},
   {memory_samples: [{observed_at: 995,process_gpu_memory_bytes: 0}]},
   {memory_samples: [{observed_at: 995,process_gpu_memory_bytes: 8192}]},
 ])('does not claim GPU with incomplete or inconsistent memory evidence %#', change => {
   expect(agentExecutionSchema.safeParse({...executionFixture,
     gpu_evidence: {...executionFixture.gpu_evidence,...change}}).success).toBe(false);
 });
});
