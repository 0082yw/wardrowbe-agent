import { agentExecutionSchema, type AgentPreviewResponse } from '@/lib/agent';

export const GPU_EVIDENCE_MAX_AGE_MS = 10_000;
export type RequestExecution = NonNullable<AgentPreviewResponse['execution']>;

export function currentRequestGpuState(execution: unknown, nowMs = Date.now()): 'verified' | 'unknown' {
  const parsed = agentExecutionSchema.safeParse(execution);
  if (!parsed.success || !Number.isFinite(nowMs)) return 'unknown';
  const value = parsed.data;
  if (value.execution_mode !== 'remote_gpu' || value.gpu_verified !== true || !value.gpu_evidence) return 'unknown';
  const age = nowMs - value.gpu_evidence.observed_at * 1000;
  return age >= 0 && age <= GPU_EVIDENCE_MAX_AGE_MS ? 'verified' : 'unknown';
}
