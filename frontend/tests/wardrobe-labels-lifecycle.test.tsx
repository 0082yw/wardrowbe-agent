import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { NextIntlClientProvider } from 'next-intl';
import { beforeEach, expect, it, vi } from 'vitest';
import type { ReactNode } from 'react';
import { ItemUserTags } from '@/components/item-user-tags';
import { ItemLifecycleActions } from '@/components/item-lifecycle-actions';
import { ApplicationAccessContext } from '@/lib/application-access-context';
import { mergeItemUpdate } from '@/lib/item-update';
import { queryKeys } from '@/lib/hooks/query-keys';
import type { Item } from '@/lib/types';
import { api } from '@/lib/api';
import wardrobe from '../messages/en/wardrobe.json';
import common from '../messages/en/common.json';

vi.unmock('next-intl');
vi.mock('@/lib/api', async original => ({
  ...await original<typeof import('@/lib/api')>(),
  api: { get: vi.fn(), patch: vi.fn(), post: vi.fn() }, setAccessToken: vi.fn(),
}));
vi.mock('sonner', () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }));

const item = {
  id: 'test-item', is_archived: false,
  tags: { colors: ['white'], style: [], season: [], pattern: 'striped' },
} as unknown as Item;

function mount(node: ReactNode, writable = true) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={client}>
    <NextIntlClientProvider locale="en" messages={{ wardrobe, common }}>
      <ApplicationAccessContext.Provider value={{ businessWrite: writable, reason: 'Not enabled',
        writeOperations: writable ? ['update', 'archive', 'restore'] : [] }}>{node}</ApplicationAccessContext.Provider>
    </NextIntlClientProvider>
  </QueryClientProvider>);
  return client;
}
beforeEach(() => vi.resetAllMocks());

it('saves only supported personal labels and preserves AI labels in cache', async () => {
  vi.mocked(api.patch).mockResolvedValue(item);
  mount(<ItemUserTags item={item} />);
  fireEvent.click(screen.getByRole('button', { name: wardrobe.userTags.add }));
  fireEvent.change(screen.getByLabelText(wardrobe.userTags.size), { target: { value: ' M ' } });
  fireEvent.change(screen.getByLabelText(wardrobe.userTags.care_instructions), { target: { value: 'Hand wash' } });
  fireEvent.click(screen.getByRole('button', { name: common.save }));
  await waitFor(() => expect(api.patch).toHaveBeenCalledWith('/items/test-item', {
    tags: { size: 'M', care_instructions: 'Hand wash', source_url: null },
  }));
  const original = { ...item, tags: { ...item.tags, size: 'M' } };
  expect(mergeItemUpdate(original, { tags: { size: null } }).tags).toEqual(item.tags);
  expect(original.tags.size).toBe('M');
});

it('deletes a single label with explicit null', async () => {
  vi.mocked(api.patch).mockResolvedValue(item);
  mount(<ItemUserTags item={{ ...item, tags: { ...item.tags, size: 'M' } }} />);
  fireEvent.click(screen.getByRole('button', { name: 'Remove Size' }));
  await waitFor(() => expect(api.patch).toHaveBeenCalledWith('/items/test-item', { tags: { size: null } }));
});

it('preserves edited labels when the API rejects the change', async () => {
  vi.mocked(api.patch).mockRejectedValue(new Error('test failure'));
  mount(<ItemUserTags item={{ ...item, tags: { ...item.tags, size: 'M' } }} />);
  fireEvent.click(screen.getByRole('button', { name: wardrobe.userTags.edit }));
  expect(screen.getByLabelText(wardrobe.userTags.size)).toHaveValue('M');
  fireEvent.change(screen.getByLabelText(wardrobe.userTags.size), { target: { value: 'L' } });
  fireEvent.click(screen.getByRole('button', { name: common.save }));
  expect(await screen.findByRole('alert')).toHaveTextContent(wardrobe.userTags.failed);
  expect(screen.getByLabelText(wardrobe.userTags.size)).toHaveValue('L');
  expect(api.patch).toHaveBeenCalledTimes(1);
});

it('archives only after confirmation and updates detail cache', async () => {
  const onComplete = vi.fn();
  const archived = { ...item, is_archived: true };
  vi.mocked(api.post).mockResolvedValue(archived);
  const client = mount(<ItemLifecycleActions item={item} onComplete={onComplete} />);
  fireEvent.click(screen.getByRole('button', { name: wardrobe.lifecycle.archive }));
  expect(api.post).not.toHaveBeenCalled();
  fireEvent.click(within(screen.getByRole('alertdialog')).getByRole('button', { name: wardrobe.lifecycle.archive }));
  await waitFor(() => expect(onComplete).toHaveBeenCalledTimes(1));
  expect(api.post).toHaveBeenCalledWith('/items/test-item/archive', { reason: undefined });
  expect(client.getQueryData(queryKeys.item(item.id))).toEqual(archived);
});

it('restores with the existing endpoint and keeps edits unavailable on archived items', async () => {
  const onComplete = vi.fn();
  const archived = { ...item, is_archived: true };
  vi.mocked(api.post).mockResolvedValue(item);
  const client = mount(<><ItemUserTags item={archived} /><ItemLifecycleActions item={archived} onComplete={onComplete} /></>);
  expect(screen.getByRole('button', { name: wardrobe.userTags.add })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: wardrobe.lifecycle.restore }));
  await waitFor(() => expect(onComplete).toHaveBeenCalledTimes(1));
  expect(api.post).toHaveBeenCalledWith('/items/test-item/restore', {});
  expect(client.getQueryData(queryKeys.item(item.id))).toEqual(item);
});

it('disables label and lifecycle writes in read-only mode', () => {
  mount(<><ItemUserTags item={item} /><ItemLifecycleActions item={item} onComplete={vi.fn()} /></>, false);
  expect(screen.getByRole('button', { name: wardrobe.userTags.add })).toBeDisabled();
  expect(screen.getByRole('button', { name: wardrobe.lifecycle.archive })).toBeDisabled();
  expect(api.patch).not.toHaveBeenCalled();
  expect(api.post).not.toHaveBeenCalled();
});
