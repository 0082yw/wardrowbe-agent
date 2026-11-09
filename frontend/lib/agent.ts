import { z } from 'zod';
import { ITEM_ROLE } from '@/lib/generated/garment-vocabulary';

export const MAX_AGENT_MESSAGE_LENGTH = 10000;

export interface AgentRequest {
  message: string;
}

export type AgentInputError = 'blank' | 'tooLong';

export function getAgentInputError(message: string): AgentInputError | undefined {
  // Python counts Unicode code points rather than UTF-16 code units.
  if (Array.from(message).length > MAX_AGENT_MESSAGE_LENGTH) return 'tooLong';
  const visible = message.replace(
    new RegExp('[\\s\\p{Default_Ignorable_Code_Point}\\p{Mn}\\p{Me}\\u2800\\uFFF9-\\uFFFB\\u{13430}-\\u{1343F}]', 'gu'),
    '',
  );
  if (!visible) return 'blank';
}

// Mirror the API contract: previews have no saved outfit ID; image URLs come from the backend.
function text(max: number) {
  return z.string().trim().min(1).refine((value) => Array.from(value).length <= max);
}

const itemSchema = z.object({
  id: z.string().uuid(),
  name: z.string().refine((value) => Array.from(value).length <= 100).nullable().default(null),
  type: text(50),
  thumbnail_url: z.string().max(2048).regex(
    /^\/api\/v1\/images\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\/[a-zA-Z0-9_-]+\.(?:jpg|jpeg|png|webp)\?expires=[0-9]+&sig=[0-9a-f]{32}$/,
  ).nullable().default(null),
}).strict();

const outfitSchema = z.object({
  title: text(100).optional(),
  item_ids: z.array(z.string().uuid()).min(1).max(20),
  reasoning: text(1000),
  style_notes: text(1000),
  items: z.array(itemSchema).min(1).max(20),
}).strict().refine((outfit) =>
  new Set(outfit.item_ids).size === outfit.item_ids.length
  && outfit.items.length === outfit.item_ids.length
  && outfit.items.every((item, index) => item.id === outfit.item_ids[index]),
);

const modelNameSchema = z.string().max(128).regex(/^[a-zA-Z0-9][a-zA-Z0-9_.-]*(?::[a-zA-Z0-9][a-zA-Z0-9_.-]*)?$/)
  .refine((value) => !value.toLowerCase().includes('cloud'));

const gpuEvidenceSchema = z.object({
  model: modelNameSchema, model_digest: z.string().regex(/^[0-9a-f]{64}$/),
  process_id: z.number().int().positive(), process_started_at: z.number().finite().positive(),
  gpu_uuid: z.literal('GPU-11111111-1111-1111-1111-111111111111'),
  observed_at: z.number().finite().positive(),
  layers_loaded: z.number().int().positive(), layers_total: z.number().int().positive(),
  model_size_bytes: z.number().int().positive(), gpu_size_bytes: z.number().int().positive(),
  process_gpu_memory_bytes: z.number().int().positive(),
  memory_samples: z.array(z.object({
    observed_at: z.number().finite().positive(),
    process_gpu_memory_bytes: z.number().int().positive(),
  }).strict()).min(1).max(32),
  request_id: z.string().regex(/^[0-9a-f]{32}$/),
  request_started_at: z.number().finite().positive(), request_finished_at: z.number().finite().positive(),
  evidence_id: z.string().regex(/^[0-9a-f]{64}$/),
}).strict().refine((proof) => proof.layers_loaded === proof.layers_total
  && proof.gpu_size_bytes === proof.model_size_bytes
  && proof.process_started_at <= proof.observed_at
  && proof.request_started_at <= proof.observed_at && proof.observed_at <= proof.request_finished_at
  && proof.memory_samples.every((sample, index, samples) =>
    sample.observed_at >= proof.request_started_at && sample.observed_at <= proof.observed_at
    && (index === 0 || sample.observed_at > samples[index - 1].observed_at))
  && proof.memory_samples.at(-1)?.observed_at === proof.observed_at
  && proof.memory_samples.at(-1)?.process_gpu_memory_bytes === proof.process_gpu_memory_bytes);

export const agentExecutionSchema = z.object({
  request_id: z.string().regex(/^[0-9a-f]{32}$/),
  execution_mode: z.enum(['local_cpu', 'remote_gpu']), model: modelNameSchema,
  gpu_verified: z.union([z.boolean(), z.literal('unknown')]),
  gpu_evidence_status: z.enum(['unavailable', 'rejected', 'verified']).default('unavailable'),
  gpu_evidence: gpuEvidenceSchema.nullable().default(null),
  fallback_fields: z.array(z.enum(['answer', 'reasoning', 'style_notes'])).max(3).default([]),
}).strict().refine((execution) => {
  if (new Set(execution.fallback_fields).size !== execution.fallback_fields.length) return false;
  if (execution.execution_mode === 'local_cpu' && execution.gpu_verified !== false) return false;
  if (execution.gpu_verified === true) return execution.execution_mode === 'remote_gpu'
    && execution.gpu_evidence_status === 'verified' && execution.gpu_evidence !== null
    && execution.gpu_evidence.request_id === execution.request_id && execution.gpu_evidence.model === execution.model;
  return execution.gpu_evidence === null && execution.gpu_evidence_status !== 'verified';
});

export const agentPreviewSchema = z.object({
  kind: z.enum(['items', 'outfit']).optional(),
  scene: z.enum(['daily', 'commute', 'formal']).optional(),
  status: z.enum(['success', 'needs_clarification', 'no_items', 'invalid_result']),
  answer: text(1000),
  outfits: z.array(outfitSchema).max(3).default([]),
  execution: agentExecutionSchema.optional(),
}).strict().superRefine((preview, context) => {
  const reject = () => context.addIssue({ code: z.ZodIssueCode.custom, message: 'Invalid preview composition' });
  if ((preview.status === 'success') !== (preview.outfits.length > 0)) reject();
  if (preview.kind === 'outfit') {
    if (!preview.scene) reject();
    const sets = new Set<string>();
    for (const outfit of preview.outfits) {
      const roles = outfit.items.map((item) => ITEM_ROLE[item.type]);
      if (!outfit.title || roles.filter((r) => r === 'base_top').length !== 1
        || roles.filter((r) => r === 'bottom').length !== 1
        || roles.some((r) => !['base_top', 'bottom', 'footwear', 'outer_layer'].includes(r))
        || roles.filter((r) => r === 'footwear').length > 1
        || roles.filter((r) => r === 'outer_layer').length > 1) reject();
      const key = [...outfit.item_ids].sort().join(',');
      if (sets.has(key)) reject();
      sets.add(key);
    }
  } else if (preview.scene || preview.outfits.some((o) => o.title !== undefined)) reject();
});

export type AgentPreviewResponse = z.infer<typeof agentPreviewSchema>;

/** Readiness metadata does not prove that model inference has succeeded. */
export const agentStatusSchema = z.object({
  mode: z.enum(['local_ollama', 'remote_ollama', 'unknown']),
  model: z.string().max(128).regex(/^[a-zA-Z0-9][a-zA-Z0-9_.-]*(?::[a-zA-Z0-9][a-zA-Z0-9_.-]*)?$/)
    .refine((value) => !value.toLowerCase().includes('cloud')).nullable(),
  available: z.boolean(),
  cpu_only: z.boolean(),
  execution_mode: z.enum(['local_cpu', 'remote_gpu', 'unknown']).optional(),
  tunnel_status: z.enum(['connected', 'disconnected', 'not_applicable', 'unknown']).optional(),
  gpu_verified: z.union([z.literal(false), z.literal('unknown')]).optional(),
}).strict().refine((status) => {
  if (status.mode === 'unknown') return status.model === null && !status.available && !status.cpu_only
    && (status.execution_mode === undefined || status.execution_mode === 'unknown')
    && (status.tunnel_status === undefined || status.tunnel_status === 'unknown')
    && (status.gpu_verified === undefined || status.gpu_verified === 'unknown');
  if (status.mode === 'local_ollama') return status.model === 'qwen3:0.6b-cpu'
    && (!status.available || status.cpu_only)
    && (status.execution_mode === undefined || status.execution_mode === 'local_cpu')
    && (status.tunnel_status === undefined || status.tunnel_status === 'not_applicable')
    && (status.gpu_verified === undefined || status.gpu_verified === false);
  return status.model !== null && !status.cpu_only && status.execution_mode === 'remote_gpu'
    && ['connected', 'disconnected', 'unknown'].includes(status.tunnel_status ?? '')
    && (!status.available || status.tunnel_status === 'connected')
    && status.gpu_verified !== undefined;
});

export type AgentStatusResponse = z.infer<typeof agentStatusSchema>;
