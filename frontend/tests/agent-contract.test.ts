import { describe, expect, it } from 'vitest';
import { agentPreviewSchema, agentStatusSchema, getAgentInputError, MAX_AGENT_MESSAGE_LENGTH } from '@/lib/agent';

const id = '00000000-0000-4000-8000-000000000001';
const outfit = {
  item_ids: [id],
  items: [{ id, name: null, type: 'shirt' }],
  reasoning: 'A confirmed wardrobe item.',
  style_notes: 'A preview only.',
};

const completeOutfit = {
  ...outfit, title: 'Everyday outfit 1',
  item_ids: [id, '00000000-0000-4000-8000-000000000002'],
  items: [outfit.items[0], { id: '00000000-0000-4000-8000-000000000002', name: 'Jeans', type: 'jeans' }],
};

describe('Complete outfit response contract', () => {
  it('accepts formal outfits only with the same complete composition contract', () => {
    expect(agentPreviewSchema.parse({ kind: 'outfit', scene: 'formal', status: 'success',
      answer: 'Formal preview', outfits: [completeOutfit] }).scene).toBe('formal');
  });
  it('accepts one outfit containing a verified top and bottom', () => {
    const value = agentPreviewSchema.parse({ kind: 'outfit', scene: 'daily', status: 'success',
      answer: 'One complete outfit', outfits: [completeOutfit] });
    expect(value.outfits).toHaveLength(1);
    expect(value.outfits[0].items.map((i) => i.type)).toEqual(['shirt', 'jeans']);
  });

  it.each([
    { scene: undefined }, { scene: 'rain' }, { outfits: [{ ...completeOutfit, title: undefined }] },
    { outfits: [{ ...outfit, title: 'Only a shirt' }] },
    { outfits: [completeOutfit, completeOutfit] },
    { outfits: [{ ...completeOutfit, items: [outfit.items[0], { ...completeOutfit.items[1], type: 'shoes' }] }] },
    { outfits: [{ ...completeOutfit, items: [outfit.items[0], { ...completeOutfit.items[1], type: 'unknown' }] }] },
    { outfits: [{ ...completeOutfit, item_ids: [id, id] }] },
    { outfits: [{ ...completeOutfit, items: [outfit.items[0], { ...completeOutfit.items[1], id }] }] },
    { kind: 'items' }, { kind: undefined },
  ])('rejects incomplete or inconsistent combinations %#', (changes) => {
    expect(agentPreviewSchema.safeParse({ kind: 'outfit', scene: 'daily', status: 'success',
      answer: 'Preview', outfits: [completeOutfit], ...changes }).success).toBe(false);
  });

  it('accepts a missing-core explanation without fake outfits', () => {
    expect(agentPreviewSchema.parse({ kind: 'outfit', scene: 'commute', status: 'no_items',
      answer: 'Missing a wearable bottom', outfits: [] }).outfits).toEqual([]);
  });
});

describe('Agent runtime metadata contract', () => {
  const remote = { mode: 'remote_ollama', model: 'qwen3:4b', available: true, cpu_only: false,
    execution_mode: 'remote_gpu', tunnel_status: 'connected', gpu_verified: 'unknown' };
  it('accepts a reachable remote endpoint without claiming verified GPU execution', () => {
    expect(agentStatusSchema.parse(remote)).toEqual(remote);
  });
  it('accepts the approved Instruct model without changing the runtime contract', () => {
    const model = 'qwen3:4b-instruct-2507-q4_K_M';
    const value = agentStatusSchema.parse({ ...remote, model });
    expect(value.model).toBe(model);
    expect(value.gpu_verified).toBe('unknown');
  });
  it.each([
    { gpu_verified: true }, { cpu_only: true }, { execution_mode: 'local_cpu' },
    { tunnel_status: 'disconnected' }, { model: 'https://external.invalid' },
    { execution_mode: undefined }, { gpu_verified: undefined },
  ])('rejects contradictory remote metadata or unsupported GPU claims %#', (changes) => {
    expect(agentStatusSchema.safeParse({ ...remote, ...changes }).success).toBe(false);
  });
  it('accepts an explicit unavailable remote endpoint', () => {
    expect(agentStatusSchema.parse({ ...remote, available: false, tunnel_status: 'disconnected' }).available).toBe(false);
  });
  it('accepts an incompatible model on a connected tunnel without claiming GPU success', () => {
    const value = agentStatusSchema.parse({ ...remote, available: false, tunnel_status: 'connected' });
    expect(value.available).toBe(false);
    expect(value.tunnel_status).toBe('connected');
    expect(value.gpu_verified).toBe('unknown');
  });
  it.each([
    { mode: 'unknown', model: null, available: false, cpu_only: false },
    { mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: true, cpu_only: true },
    { mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: false, cpu_only: false },
    { mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: false, cpu_only: true },
  ])('accepts explicit readiness metadata without adding inference claims %#', (value) => {
    expect(agentStatusSchema.parse(value)).toEqual(value);
  });

  it.each([
    { mode: 'unknown', model: 'qwen3:0.6b-cpu', available: false, cpu_only: false },
    { mode: 'unknown', model: null, available: true, cpu_only: false },
    { mode: 'unknown', model: null, available: false, cpu_only: true },
    { mode: 'local_ollama', model: null, available: true, cpu_only: true },
    { mode: 'local_ollama', model: 'cloud-model', available: true, cpu_only: true },
    { mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: true, cpu_only: false },
    { mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: 'true', cpu_only: true },
    { mode: 'unknown', model: null, available: false, cpu_only: false, endpoint: 'https://external.invalid' },
  ])('rejects unsupported or contradictory model metadata %#', (value) => {
    expect(agentStatusSchema.safeParse(value).success).toBe(false);
  });
});

describe('Agent API response contract', () => {
  it('accepts a structured preview without pretending it is a saved outfit', () => {
    const result = agentPreviewSchema.parse({ status: 'success', answer: 'Preview', outfits: [outfit] });
    expect(result.outfits[0].items[0].name).toBeNull();
    expect(result.outfits[0].items[0].thumbnail_url).toBeNull();
    expect(result.outfits[0]).not.toHaveProperty('id');
    expect(result.outfits[0].items[0]).not.toHaveProperty('image_url');
  });

  it('accepts up to three independent item previews from combined item types', () => {
    const outfits = ['shirt', 'pants', 'shoes'].map((type, index) => {
      const itemId = `00000000-0000-4000-8000-00000000000${index + 1}`;
      return { ...outfit, item_ids: [itemId], items: [{ id: itemId, name: null, type }] };
    });
    const result = agentPreviewSchema.parse({ status: 'success', answer: 'Combined query results', outfits });
    expect(result.outfits).toHaveLength(3);
    expect(result.outfits.map((entry) => entry.items.map((item) => item.type))).toEqual([
      ['shirt'], ['pants'], ['shoes'],
    ]);
  });

  it.each(['needs_clarification', 'no_items', 'invalid_result'])('accepts an empty %s result', (status) => {
    expect(agentPreviewSchema.parse({ status, answer: 'Explanation', outfits: [] }).outfits).toEqual([]);
  });

  it('honours the backend defaults for omitted outfits and item names', () => {
    expect(agentPreviewSchema.parse({ status: 'no_items', answer: 'Empty' }).outfits).toEqual([]);
    const withoutName = { ...outfit, items: [{ id, type: 'shirt' }] };
    expect(agentPreviewSchema.parse({ status: 'success', answer: 'Preview', outfits: [withoutName] })
      .outfits[0].items[0].name).toBeNull();
  });

  it.each([
    { status: 'success', answer: 'Empty success', outfits: [] },
    { status: 'no_items', answer: 'Unexpected outfits', outfits: [outfit] },
    { status: 'unknown', answer: 'Unknown', outfits: [] },
    { status: 'no_items', answer: '', outfits: [] },
    { status: 'success', answer: 'Invalid', outfits: [{ ...outfit, item_ids: ['not-a-uuid'] }] },
    { status: 'success', answer: 'Invalid', outfits: [{ ...outfit, items: [] }] },
    { status: 'success', answer: 'Invalid', outfits: [{ ...outfit, item_ids: [id, id], items: [outfit.items[0], outfit.items[0]] }] },
    { status: 'success', answer: 'Invalid', outfits: [{ ...outfit, items: [{ ...outfit.items[0], id: '00000000-0000-4000-8000-000000000002' }] }] },
    { status: 'success', answer: 'Invalid', outfits: [{ ...outfit, items: [{ ...outfit.items[0], user_id: id }] }] },
    { status: 'success', answer: 'Invalid', outfits: [outfit, outfit, outfit, outfit] },
    { status: 'no_items', answer: 'Empty', outfits: [], endpoint: 'unexpected' },
  ])('rejects malformed or inconsistent responses %#', (value) => {
    expect(agentPreviewSchema.safeParse(value).success).toBe(false);
  });

  it('checks item order as well as membership', () => {
    const second = { id: '00000000-0000-4000-8000-000000000002', name: 'Trousers', type: 'pants' };
    expect(agentPreviewSchema.safeParse({
      status: 'success', answer: 'Preview',
      outfits: [{ ...outfit, item_ids: [id, second.id], items: [second, outfit.items[0]] }],
    }).success).toBe(false);
  });
});

describe('Agent request validation', () => {
  it.each(['', ' \n\t ', '\u200B\u2060', '\u0301', '\u2800'])('blocks blank input %#', (message) => {
    expect(getAgentInputError(message)).toBe('blank');
  });

  it('allows arbitrary meaningful natural language and multiple lines', () => {
    expect(getAgentInputError('明天想去散步\n帮我看看衣橱')).toBeUndefined();
  });

  it('counts Unicode characters consistently with the backend limit', () => {
    expect(getAgentInputError('😀'.repeat(MAX_AGENT_MESSAGE_LENGTH))).toBeUndefined();
    expect(getAgentInputError('😀'.repeat(MAX_AGENT_MESSAGE_LENGTH + 1))).toBe('tooLong');
  });
});


describe('Agent image response contract', () => {
  const signedImage = '/api/v1/images/00000000-0000-4000-8000-000000000099/shirt_thumb.jpg?expires=9999999999&sig=' + 'a'.repeat(32);

  it('accepts a local signed thumbnail without exposing storage paths', () => {
    const item = { ...outfit.items[0], thumbnail_url: signedImage };
    const result = agentPreviewSchema.parse({
      status: 'success', answer: 'Preview', outfits: [{ ...outfit, items: [item] }],
    });
    expect(result.outfits[0].items[0].thumbnail_url).toBe(signedImage);
  });

  it.each([
    'https://example.com/private.jpg',
    '//example.com/private.jpg',
    '/home/developer/private.jpg',
    '/api/v1/images/00000000-0000-4000-8000-000000000099/../private.jpg',
    '/api/v1/images/00000000-0000-4000-8000-000000000099/shirt.jpg',
  ])('rejects external, internal, traversing or unsigned locations: %s', (url) => {
    expect(agentPreviewSchema.safeParse({
      status: 'success', answer: 'Preview',
      outfits: [{ ...outfit, items: [{ ...outfit.items[0], thumbnail_url: url }] }],
    }).success).toBe(false);
  });
});
