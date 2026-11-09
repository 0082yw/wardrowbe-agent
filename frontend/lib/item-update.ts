import type { Item, ItemTags } from '@/lib/types';

export const USER_TAG_FIELDS = [
  { key: 'size', maxLength: 32 },
  { key: 'care_instructions', maxLength: 255 },
  { key: 'source_url', maxLength: 500 },
] as const;
export type UserTagKey = typeof USER_TAG_FIELDS[number]['key'];
export type UserTagPatch = { [K in UserTagKey]?: string | null };
export type ItemUpdateData = Omit<Partial<Item>, 'tags'> & {
  tags?: Partial<Omit<ItemTags, UserTagKey>> & UserTagPatch;
};

export function mergeItemUpdate(item: Item, data: ItemUpdateData): Item {
  const tags: Record<string, unknown> = { ...item.tags };
  for (const [key, value] of Object.entries(data.tags ?? {})) {
    if (value === null) delete tags[key];
    else if (value !== undefined) tags[key] = value;
  }
  return { ...item, ...data, tags: tags as unknown as ItemTags };
}
