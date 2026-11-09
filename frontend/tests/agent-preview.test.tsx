import type { Session } from 'next-auth';
import { QueryClient, QueryClientProvider, onlineManager } from '@tanstack/react-query';
import { cloneElement, isValidElement, type ReactNode } from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { NextIntlClientProvider } from 'next-intl';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AgentPage from '@/app/dashboard/agent/page';
import { Sidebar } from '@/components/sidebar';
import { MobileSidebar } from '@/components/mobile-sidebar';
import type { AgentPreviewResponse, AgentStatusResponse } from '@/lib/agent';
import { setAccessToken } from '@/lib/api';
import agent from '@/messages/en/agent.json';
import constants from '@/messages/en/constants.json';
import nav from '@/messages/en/nav.json';

vi.unmock('next-intl');
const auth = vi.hoisted(() => ({
  data: null as Session | null,
  status: 'authenticated' as 'authenticated' | 'unauthenticated' | 'loading',
}));
vi.mock('next-auth/react', () => ({ useSession: () => auth }));

const runtime = vi.hoisted(() => ({ data: undefined as AgentStatusResponse | undefined, isError: false }));
vi.mock('@/lib/hooks/use-agent-preview', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/lib/hooks/use-agent-preview')>(),
  useAgentStatus: () => runtime,
}));

const clients: QueryClient[] = [];
const preview: AgentPreviewResponse = {
  status: 'success',
  answer: 'A simulated preview from your queried wardrobe.',
  outfits: [{
    item_ids: ['00000000-0000-4000-8000-000000000001'],
    items: [{ id: '00000000-0000-4000-8000-000000000001', name: 'White shirt', type: 'shirt', thumbnail_url: null }],
    reasoning: 'This item was confirmed by the backend.',
    style_notes: 'Try this preview; it has not been saved.',
  }],
};

function response(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function renderPage(content: ReactNode = <AgentPage />) {
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  clients.push(client);
  const view = () => (
    <QueryClientProvider client={client}>
      <NextIntlClientProvider locale="en" messages={{ agent, constants, nav }} timeZone="UTC">
        {/* Explicit rerenders must reach hooks even when the session mock changes outside React. */}
        {isValidElement(content) ? cloneElement(content) : content}
      </NextIntlClientProvider>
    </QueryClientProvider>
  );
  return { ...render(view()), view };
}

function input(value: string) {
  fireEvent.change(screen.getByRole('textbox'), { target: { value } });
}

function submit() {
  fireEvent.submit(screen.getByRole('textbox').closest('form')!);
}

beforeEach(() => {
  vi.mocked(global.fetch).mockReset();
  runtime.data = undefined;
  runtime.isError = false;
  auth.data = { user: { id: 'test-user' }, accessToken: 'test-only-token', expires: '2099-01-01' };
  auth.status = 'authenticated';
  setAccessToken(null);
});

afterEach(() => {
  cleanup();
  onlineManager.setOnline(true);
  clients.forEach((client) => client.clear());
  clients.length = 0;
  setAccessToken(null);
});

describe('Agent assistant page', () => {
  it('labels remote mode and the tunnel while retaining an unverified GPU state', () => {
    runtime.data = { mode: 'remote_ollama', model: 'qwen3:4b', available: true, cpu_only: false,
      execution_mode: 'remote_gpu', tunnel_status: 'connected', gpu_verified: 'unknown' };
    renderPage();
    expect(screen.getByText(agent.runtimeBadge.remoteConfigured)).toBeInTheDocument();
    expect(screen.getByText(agent.runtimeDetails.remoteGpu)).toBeInTheDocument();
    expect(screen.getByText(agent.runtimeDetails.gpuReadinessOnly)).toBeInTheDocument();
    expect(screen.getByText(agent.tunnelState.connected)).toBeInTheDocument();
    expect(screen.queryByText(agent.runtimeBadge.configured)).not.toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });
  it('shows a disconnected remote tunnel clearly without switching to CPU', () => {
    runtime.data = { mode: 'remote_ollama', model: 'qwen3:4b', available: false, cpu_only: false,
      execution_mode: 'remote_gpu', tunnel_status: 'disconnected', gpu_verified: 'unknown' };
    renderPage();
    expect(screen.getByText(agent.runtimeBadge.remoteUnavailable)).toBeInTheDocument();
    expect(screen.getByText(agent.runtimeNotice.remoteUnavailable)).toBeInTheDocument();
    expect(screen.getByText(agent.tunnelState.disconnected)).toBeInTheDocument();
    expect(screen.queryByText(agent.runtimeDetails.localCpu)).not.toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });
  it('does not reuse a previous remote status after a metadata error', () => {
    runtime.data = { mode: 'remote_ollama', model: 'qwen3:4b', available: true, cpu_only: false,
      execution_mode: 'remote_gpu', tunnel_status: 'connected', gpu_verified: 'unknown' };
    runtime.isError = true;
    renderPage();
    expect(screen.getByText(agent.runtimeBadge.unknown)).toBeInTheDocument();
    expect(screen.queryByText(agent.runtimeBadge.remoteConfigured)).not.toBeInTheDocument();
    expect(screen.queryByText(agent.tunnelState.connected)).not.toBeInTheDocument();
  });
  it('switches between distinct outfit previews without another API request', async () => {
    const jeans = { id: '00000000-0000-4000-8000-000000000002', name: 'Jeans', type: 'jeans', thumbnail_url: null };
    const tops = [preview.outfits[0].items[0], { id: '00000000-0000-4000-8000-000000000003', name: 'Second top', type: 'shirt', thumbnail_url: null }];
    vi.mocked(global.fetch).mockResolvedValue(response({ kind: 'outfit', scene: 'daily', status: 'success',
      answer: 'Two outfits', outfits: tops.map((top, index) => ({ ...preview.outfits[0], title: 'Plan ' + (index + 1),
        item_ids: [top.id, jeans.id], items: [top, jeans] })) }));
    renderPage(); input('帮我搭配两套日常穿搭'); submit();
    expect(await screen.findByText('Plan 1')).toBeInTheDocument();
    expect(screen.getByText('Plan 2')).toBeInTheDocument();
    const selector = screen.getByRole('group', { name: agent.previewSelector });
    fireEvent.click(within(selector).getByRole('button', { name: agent.selectPreview.replace('{number}', '2') }));
    expect(screen.queryByText('Plan 1')).not.toBeInTheDocument();
    expect(screen.getByText('Plan 2')).toBeInTheDocument();
    expect(screen.getByText(agent.unsaved)).toBeInTheDocument();
    fireEvent.click(within(selector).getByRole('button', { name: agent.allPreviews }));
    expect(screen.getByText('Plan 1')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
    input('查询我的衬衫');
    expect(screen.queryByRole('group', { name: agent.previewSelector })).not.toBeInTheDocument();
  });
  it('shows one complete outfit with two signed photos and role labels', async () => {
    const owner = '00000000-0000-4000-8000-000000000099';
    const image = (file: string) => '/api/v1/images/' + owner + '/' + file
      + '.jpg?expires=2000000000&sig=' + 'a'.repeat(32);
    const shirt = { ...preview.outfits[0].items[0], thumbnail_url: image('shirt') };
    const jeans = { id: '00000000-0000-4000-8000-000000000002', name: 'Light blue jeans',
      type: 'jeans', thumbnail_url: image('jeans') };
    vi.mocked(global.fetch).mockResolvedValue(response({
      kind: 'outfit', scene: 'daily', status: 'success', answer: 'One verified complete outfit.',
      outfits: [{ ...preview.outfits[0], title: 'Daily outfit 1',
        item_ids: [shirt.id, jeans.id], items: [shirt, jeans] }],
    }));
    renderPage();
    input('帮我搭配一套日常穿搭');
    submit();
    expect(await screen.findByText('Daily outfit 1')).toBeInTheDocument();
    expect(screen.getByText('White shirt')).toBeInTheDocument();
    expect(screen.getByText('Light blue jeans')).toBeInTheDocument();
    expect(screen.getByText(agent.roles.base_top)).toBeInTheDocument();
    expect(screen.getByText(agent.roles.bottom)).toBeInTheDocument();
    expect(screen.getByText(agent.outfitPreviewOnly)).toBeInTheDocument();
    const topPhoto = screen.getByRole('img', { name: 'White shirt' });
    const bottomPhoto = screen.getByRole('img', { name: 'Light blue jeans' });
    expect(topPhoto).toHaveAttribute('src', new URL(shirt.thumbnail_url, window.location.origin).href);
    expect(bottomPhoto).toHaveAttribute('src', new URL(jeans.thumbnail_url, window.location.origin).href);
    fireEvent.error(bottomPhoto);
    expect(screen.getByRole('img', { name: 'Light blue jeans: ' + agent.imageUnavailable })).toBeInTheDocument();
    expect(topPhoto).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('shows missing outfit pieces without displaying a fake complete outfit', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response({ kind: 'outfit', scene: 'daily',
      status: 'no_items', answer: 'Missing a wearable bottom.', outfits: [] }));
    renderPage();
    input('帮我搭配一套日常穿搭');
    submit();
    expect(await screen.findByText('Missing a wearable bottom.')).toBeInTheDocument();
    expect(screen.getByText(agent.status.no_items)).toBeInTheDocument();
    expect(screen.queryByText('White shirt')).not.toBeInTheDocument();
  });
  it('shows an unconfirmed mode and does not submit a preview on mount', () => {
    renderPage();
    expect(screen.getByText(agent.runtimeBadge.unknown)).toBeInTheDocument();
    expect(screen.getByText(agent.runtimeNotice.unknown)).toBeInTheDocument();
    expect(screen.getByText(agent.emptyState)).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('identifies confirmed local configuration without claiming successful inference', () => {
    runtime.data = { mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: true, cpu_only: true };
    renderPage();
    expect(screen.getByText(agent.runtimeBadge.configured)).toBeInTheDocument();
    expect(screen.getByText(agent.runtimeNotice.configured)).toBeInTheDocument();
    expect(screen.queryByText(agent.runtimeBadge.unknown)).not.toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('shows backend unavailability and keeps the preview read-only', () => {
    runtime.data = { mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: false, cpu_only: false };
    renderPage();
    expect(screen.getByText(agent.runtimeBadge.unavailable)).toBeInTheDocument();
    expect(screen.getByText(agent.runtimeNotice.unavailable)).toBeInTheDocument();
    expect(screen.getByText(agent.emptyState)).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('does not label an unknown backend as a local or simulated model', () => {
    runtime.data = { mode: 'unknown', model: null, available: false, cpu_only: false };
    renderPage();
    expect(screen.getByText(agent.runtimeBadge.unknown)).toBeInTheDocument();
    expect(screen.queryByText(agent.runtimeBadge.configured)).not.toBeInTheDocument();
    expect(screen.queryByText(agent.runtimeBadge.unavailable)).not.toBeInTheDocument();
  });

  it('leaves the mode unconfirmed when a status error retains older readiness data', () => {
    runtime.data = { mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: true, cpu_only: true };
    runtime.isError = true;
    renderPage();
    expect(screen.getByText(agent.runtimeBadge.unknown)).toBeInTheDocument();
    expect(screen.getByText(agent.runtimeNotice.unknown)).toBeInTheDocument();
    expect(screen.queryByText(agent.runtimeBadge.configured)).not.toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('uses the authenticated API proxy and displays only the returned preview fields', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response(preview));
    renderPage();
    input('  帮我查询衬衫  ');
    submit();

    expect(await screen.findByText(preview.answer)).toBeInTheDocument();
    expect(screen.getByText(agent.status.success)).toBeInTheDocument();
    expect(screen.getByText('White shirt')).toBeInTheDocument();
    expect(screen.getByText('Type: Shirt')).toBeInTheDocument();
    expect(screen.getByText(preview.outfits[0].reasoning)).toBeInTheDocument();
    expect(screen.getByText(preview.outfits[0].style_notes)).toBeInTheDocument();
    expect(screen.getByText(agent.previewOnly)).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'White shirt: ' + agent.imageUnavailable })).toBeInTheDocument();
    expect(screen.getAllByRole('button')).toHaveLength(1);
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch).toHaveBeenCalledWith('/api/v1/agent/preview', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ message: '帮我查询衬衫' }),
      credentials: 'include',
      headers: expect.objectContaining({ Authorization: 'Bearer test-only-token' }),
      signal: expect.any(AbortSignal),
    }));
  });

  it('displays three combined query results as separate individual item previews', async () => {
    const outfits = [
      { name: 'White shirt', type: 'shirt' },
      { name: 'Black pants', type: 'pants' },
      { name: 'Walking shoes', type: 'shoes' },
    ].map((item, index) => {
      const id = `00000000-0000-4000-8000-00000000000${index + 1}`;
      return {
        ...preview.outfits[0], item_ids: [id],
        items: [{ ...item, id, thumbnail_url: null }],
      };
    });
    vi.mocked(global.fetch).mockResolvedValue(response({ ...preview, outfits }));
    renderPage();
    input('查询我的衬衫、裤子和鞋子');
    submit();
    await screen.findByText(preview.answer);
    outfits.forEach((outfit, index) => {
      expect(screen.getByText(agent.outfit.replace('{number}', String(index + 1)))).toBeInTheDocument();
      expect(screen.getByText(outfit.items[0].name)).toBeInTheDocument();
    });
    const lists = screen.getAllByRole('list');
    expect(lists).toHaveLength(3);
    lists.forEach((list) => expect(within(list).getAllByRole('listitem')).toHaveLength(1));
    expect(screen.getByText(agent.previewOnly)).toBeInTheDocument();
    expect(screen.getAllByRole('button')).toHaveLength(1);
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it.each(['', ' \n\t ', '\u200B'])('blocks empty input %# without a request', async (value) => {
    renderPage();
    input(value);
    submit();
    expect(await screen.findByText(agent.errors.blank)).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('rejects an oversized request before calling the API', async () => {
    renderPage();
    input('a'.repeat(10001));
    submit();
    expect(await screen.findByText(agent.errors.tooLong)).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('blocks duplicate submission while showing loading state', async () => {
    const pending = deferred<Response>();
    vi.mocked(global.fetch).mockReturnValue(pending.promise);
    renderPage();
    input('shirt');
    submit();
    submit();
    expect(await screen.findByRole('status')).toHaveTextContent(agent.loading);
    expect(screen.getByRole('button', { name: agent.generating })).toBeDisabled();
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    await act(async () => pending.resolve(response(preview)));
    expect(await screen.findByText(preview.answer)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: agent.generate })).toBeEnabled();
  });

  it.each(['needs_clarification', 'no_items', 'invalid_result'] as const)(
    'displays %s with the server answer and no outfit controls', async (status) => {
      vi.mocked(global.fetch).mockResolvedValue(response({ status, answer: 'An explanation.', outfits: [] }));
      renderPage();
      input('shirt');
      submit();
      expect(await screen.findByText('An explanation.')).toBeInTheDocument();
      expect(screen.getByText(agent.status[status])).toBeInTheDocument();
      expect(screen.getByText(agent.guidance[status])).toBeInTheDocument();
      expect(screen.queryByText('White shirt')).not.toBeInTheDocument();
      expect(screen.getAllByRole('button')).toHaveLength(1);
    },
  );

  it('submits an open-ended need and shows the backend clarification without inventing item previews', async () => {
    const answer = '你想先查询哪些衣物类型？';
    vi.mocked(global.fetch).mockResolvedValue(response({ status: 'needs_clarification', answer, outfits: [] }));
    renderPage();
    input('明天怎么穿？');
    submit();
    expect(await screen.findByText(answer)).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveValue('明天怎么穿？');
    expect(screen.getByText(agent.guidance.needs_clarification)).toBeInTheDocument();
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledWith('/api/v1/agent/preview', expect.objectContaining({
      body: JSON.stringify({ message: '明天怎么穿？' }),
    }));
  });

  it('shows a fallback only when the returned item name is missing', async () => {
    const unnamed = { ...preview, outfits: [{ ...preview.outfits[0], items: [{ ...preview.outfits[0].items[0], name: null }] }] };
    vi.mocked(global.fetch).mockResolvedValue(response(unnamed));
    renderPage();
    input('shirt');
    submit();
    expect(await screen.findByText(agent.unnamedItem)).toBeInTheDocument();
    expect(screen.getByText('Type: Shirt')).toBeInTheDocument();
  });

  it.each([
    [401, 'auth'], [422, 'invalidInput'], [429, 'rateLimited'], [503, 'failed'],
  ] as const)('handles HTTP %s without showing raw server details', async (status, key) => {
    vi.mocked(global.fetch).mockResolvedValue(response({ detail: 'internal server detail' }, status));
    renderPage();
    input('shirt');
    submit();
    expect(await screen.findByText(agent.errors[key])).toBeInTheDocument();
    expect(screen.queryByText('internal server detail')).not.toBeInTheDocument();
    expect(screen.queryByText(preview.answer)).not.toBeInTheDocument();
  });

  it('allows retry after a network failure without automatic retries', async () => {
    vi.mocked(global.fetch).mockRejectedValueOnce(new TypeError('offline')).mockResolvedValueOnce(response(preview));
    renderPage();
    input('shirt');
    submit();
    expect(await screen.findByText(agent.errors.failed)).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
    submit();
    expect(await screen.findByText(preview.answer)).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });

  it('fails promptly offline instead of queuing a preview for later', async () => {
    onlineManager.setOnline(false);
    vi.mocked(global.fetch).mockRejectedValue(new TypeError('offline'));
    renderPage();
    input('shirt');
    submit();
    expect(await screen.findByText(agent.errors.failed)).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('rejects malformed response data instead of rendering it as a valid outfit', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response({ ...preview, outfits: [] }));
    renderPage();
    input('shirt');
    submit();
    expect(await screen.findByText(agent.errors.invalidResponse)).toBeInTheDocument();
    expect(screen.queryByText(preview.answer)).not.toBeInTheDocument();
  });

  it.each(['success', 'failure'] as const)('ignores an old %s after the input and latest result change', async (oldOutcome) => {
    const old = deferred<Response>();
    vi.mocked(global.fetch).mockReturnValueOnce(old.promise).mockResolvedValueOnce(response(preview));
    renderPage();
    input('old request');
    submit();
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    const signal = vi.mocked(global.fetch).mock.calls[0][1]?.signal;

    input('shirt');
    expect(signal?.aborted).toBe(true);
    submit();
    expect(await screen.findByText(preview.answer)).toBeInTheDocument();

    await act(async () => {
      if (oldOutcome === 'success') old.resolve(response({ status: 'no_items', answer: 'Stale response', outfits: [] }));
      else old.reject(new TypeError('Stale failure'));
    });
    expect(screen.getByText(preview.answer)).toBeInTheDocument();
    expect(screen.queryByText('Stale response')).not.toBeInTheDocument();
    expect(screen.queryByText(agent.errors.failed)).not.toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });

  it('an old completion cannot clear the newer request loading state', async () => {
    const old = deferred<Response>();
    const latest = deferred<Response>();
    vi.mocked(global.fetch).mockReturnValueOnce(old.promise).mockReturnValueOnce(latest.promise);
    renderPage();
    input('old request');
    submit();
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    input('shirt');
    submit();
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
    await act(async () => old.resolve(response({ status: 'no_items', answer: 'Stale', outfits: [] })));
    expect(screen.getByRole('button', { name: agent.generating })).toBeDisabled();
    expect(screen.queryByText('Stale')).not.toBeInTheDocument();
    await act(async () => latest.resolve(response(preview)));
    expect(await screen.findByText(preview.answer)).toBeInTheDocument();
  });

  it('clears a previous preview when the request is edited', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response(preview));
    renderPage();
    input('shirt');
    submit();
    await screen.findByText(preview.answer);
    input('pants');
    expect(screen.queryByText(preview.answer)).not.toBeInTheDocument();
    expect(screen.getByText(agent.emptyState)).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('cancels on unmount without submitting another request', async () => {
    const pending = deferred<Response>();
    vi.mocked(global.fetch).mockReturnValue(pending.promise);
    const { unmount } = renderPage();
    input('shirt');
    submit();
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    const signal = vi.mocked(global.fetch).mock.calls[0][1]?.signal;
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => pending.resolve(response(preview)));
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('clears the previous account result and blocks unauthenticated submission', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response(preview));
    const page = renderPage();
    input('shirt');
    submit();
    await screen.findByText(preview.answer);
    auth.data = null;
    auth.status = 'unauthenticated';
    page.rerender(page.view());
    expect(screen.queryByText(preview.answer)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: agent.generate })).toBeDisabled();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('requires a valid session token even if a session user exists', () => {
    auth.data = { user: { id: 'test-user' }, expires: '2099-01-01' };
    renderPage();
    input('shirt');
    submit();
    expect(screen.getByRole('button', { name: agent.generate })).toBeDisabled();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('hides an existing preview and blocks new requests while authentication is loading', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response(preview));
    const page = renderPage();
    input('shirt');
    submit();
    await screen.findByText(preview.answer);
    auth.status = 'loading';
    page.rerender(page.view());
    expect(screen.queryByText(preview.answer)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: agent.generate })).toBeDisabled();
    submit();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('clears a completed preview on direct account switch and uses the new account token', async () => {
    const nextPreview = { ...preview, answer: 'The next account’s preview.' };
    vi.mocked(global.fetch).mockResolvedValueOnce(response(preview)).mockResolvedValueOnce(response(nextPreview));
    const page = renderPage();
    input('shirt');
    submit();
    await screen.findByText(preview.answer);
    auth.data = { user: { id: 'different-user' }, accessToken: 'different-test-token', expires: '2099-01-01' };
    page.rerender(page.view());
    expect(screen.queryByText(preview.answer)).not.toBeInTheDocument();
    expect(screen.getByText(agent.emptyState)).toBeInTheDocument();
    input('pants');
    submit();
    expect(await screen.findByText(nextPreview.answer)).toBeInTheDocument();
    expect(screen.queryByText(preview.answer)).not.toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(2);
    expect(global.fetch).toHaveBeenLastCalledWith('/api/v1/agent/preview', expect.objectContaining({
      body: JSON.stringify({ message: 'pants' }),
      headers: expect.objectContaining({ Authorization: 'Bearer different-test-token' }),
    }));
  });

  it('ignores a pending result after a different account signs in', async () => {
    const pending = deferred<Response>();
    vi.mocked(global.fetch).mockReturnValue(pending.promise);
    const page = renderPage();
    input('shirt');
    submit();
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    const signal = vi.mocked(global.fetch).mock.calls[0][1]?.signal;
    auth.data = { ...auth.data!, user: { id: 'different-user' } };
    page.rerender(page.view());
    expect(signal?.aborted).toBe(true);
    await act(async () => pending.resolve(response(preview)));
    expect(screen.queryByText(preview.answer)).not.toBeInTheDocument();
    expect(screen.getByText(agent.emptyState)).toBeInTheDocument();
  });
});


describe('Agent navigation', () => {
  it('is accessible from the desktop sidebar', () => {
    renderPage(<Sidebar />);
    expect(screen.getByRole('link', { name: nav.agentAssistant })).toHaveAttribute('href', '/dashboard/agent');
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('is accessible from the mobile sidebar', () => {
    renderPage(<MobileSidebar open onClose={vi.fn()} />);
    expect(screen.getByRole('link', { name: nav.agentAssistant })).toHaveAttribute('href', '/dashboard/agent');
    expect(global.fetch).not.toHaveBeenCalled();
  });
});



describe('Agent runtime metadata hook', () => {
  async function runtimeProbe() {
    const { useAgentStatus: useActualAgentStatus } = await vi.importActual<typeof import('@/lib/hooks/use-agent-preview')>('@/lib/hooks/use-agent-preview');
    return function RuntimeProbe() {
      const status = useActualAgentStatus();
      return <p data-testid="runtime-state">{status.isError ? 'unconfirmed' : status.data?.mode ?? 'waiting'}</p>;
    };
  }

  it('checks metadata through the authenticated proxy without generating a preview', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response({
      mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: true, cpu_only: true,
    }));
    const RuntimeProbe = await runtimeProbe();
    renderPage(<RuntimeProbe />);
    expect(await screen.findByText('local_ollama')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch).toHaveBeenCalledWith('/api/v1/agent/status', expect.objectContaining({
      method: 'GET', credentials: 'include',
      headers: expect.objectContaining({ Authorization: 'Bearer test-only-token' }),
      signal: expect.any(AbortSignal),
    }));
  });

  it.each([404, 503])('leaves model mode unconfirmed for HTTP %s without retry', async (status) => {
    vi.mocked(global.fetch).mockResolvedValue(response({ detail: 'private backend detail' }, status));
    const RuntimeProbe = await runtimeProbe();
    renderPage(<RuntimeProbe />);
    expect(await screen.findByText('unconfirmed')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(screen.queryByText('private backend detail')).not.toBeInTheDocument();
  });

  it('does not check metadata without an authenticated session token', async () => {
    auth.data = null;
    auth.status = 'unauthenticated';
    const RuntimeProbe = await runtimeProbe();
    renderPage(<RuntimeProbe />);
    expect(screen.getByTestId('runtime-state')).toHaveTextContent('waiting');
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('rejects malformed readiness metadata without treating it as a confirmed model', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response({
      mode: 'local_ollama', model: 'remote-cloud-model', available: true, cpu_only: true,
    }));
    const RuntimeProbe = await runtimeProbe();
    renderPage(<RuntimeProbe />);
    expect(await screen.findByText('unconfirmed')).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('does not display the previous account readiness during a session transition', async () => {
    const old = deferred<Response>();
    vi.mocked(global.fetch).mockReturnValueOnce(old.promise);
    const RuntimeProbe = await runtimeProbe();
    const page = renderPage(<RuntimeProbe />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    auth.data = null;
    auth.status = 'unauthenticated';
    page.rerender(page.view());
    await act(async () => old.resolve(response({
      mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: true, cpu_only: true,
    })));
    expect(screen.getByTestId('runtime-state')).toHaveTextContent('waiting');
    expect(screen.queryByText('local_ollama')).not.toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('cancels old metadata on direct account switch and checks readiness with the new account token', async () => {
    const old = deferred<Response>();
    const latest = deferred<Response>();
    vi.mocked(global.fetch).mockReturnValueOnce(old.promise).mockReturnValueOnce(latest.promise);
    const RuntimeProbe = await runtimeProbe();
    const page = renderPage(<RuntimeProbe />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    const oldSignal = vi.mocked(global.fetch).mock.calls[0][1]?.signal;
    auth.data = { user: { id: 'different-user' }, accessToken: 'different-test-token', expires: '2099-01-01' };
    page.rerender(page.view());
    await waitFor(() => expect(oldSignal?.aborted).toBe(true));
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
    expect(screen.getByTestId('runtime-state')).toHaveTextContent('waiting');
    expect(global.fetch).toHaveBeenLastCalledWith('/api/v1/agent/status', expect.objectContaining({
      headers: expect.objectContaining({ Authorization: 'Bearer different-test-token' }),
    }));
    await act(async () => old.resolve(response({
      mode: 'local_ollama', model: 'qwen3:0.6b-cpu', available: true, cpu_only: true,
    })));
    expect(screen.getByTestId('runtime-state')).toHaveTextContent('waiting');
    await act(async () => latest.resolve(response({
      mode: 'unknown', model: null, available: false, cpu_only: false,
    })));
    expect(await screen.findByText('unknown')).toBeInTheDocument();
    expect(screen.queryByText('local_ollama')).not.toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
});


describe('Agent preview images', () => {
  const signedImage = '/api/v1/images/00000000-0000-4000-8000-000000000099/shirt_thumb.jpg?expires=9999999999&sig=' + 'a'.repeat(32);

  function withImage(src: string | null) {
    return {
      ...preview,
      outfits: [{
        ...preview.outfits[0],
        items: [{ ...preview.outfits[0].items[0], thumbnail_url: src }],
      }],
    };
  }

  it('renders the backend signed image through the existing local image route', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response(withImage(signedImage)));
    renderPage();
    input('shirt');
    submit();
    const image = await screen.findByRole('img', { name: 'White shirt' });
    const displayed = new URL(image.getAttribute('src')!, window.location.origin);
    expect(displayed.origin).toBe(window.location.origin);
    expect(displayed.pathname + displayed.search).toBe(signedImage);
    expect(image).toHaveAttribute('loading', 'lazy');
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('keeps the preview usable when no image is available', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response(withImage(null)));
    renderPage();
    input('shirt');
    submit();
    expect(await screen.findByRole('img', { name: 'White shirt: ' + agent.imageUnavailable })).toBeInTheDocument();
    expect(screen.getByText(preview.answer)).toBeInTheDocument();
    expect(screen.getByText(preview.outfits[0].reasoning)).toBeInTheDocument();
  });

  it('replaces a failed image with a placeholder without losing the preview', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response(withImage(signedImage)));
    renderPage();
    input('shirt');
    submit();
    fireEvent.error(await screen.findByRole('img', { name: 'White shirt' }));
    expect(screen.queryByRole('img', { name: 'White shirt' })).not.toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'White shirt: ' + agent.imageUnavailable })).toBeInTheDocument();
    expect(screen.getByText(preview.answer)).toBeInTheDocument();
    expect(screen.getByText(preview.outfits[0].style_notes)).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });

  it('does not retry a denied signed image with an unsigned or public URL', async () => {
    vi.mocked(global.fetch).mockResolvedValue(response(withImage(signedImage)));
    renderPage();
    input('shirt');
    submit();
    // The browser reports an image HTTP denial through the same error event.
    fireEvent.error(await screen.findByRole('img', { name: 'White shirt' }));
    expect(screen.getByRole('img', { name: 'White shirt: ' + agent.imageUnavailable })).toBeInTheDocument();
    expect(document.querySelector('img')).not.toBeInTheDocument();
    expect(screen.getByText(preview.answer)).toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
    expect(global.fetch).toHaveBeenCalledWith('/api/v1/agent/preview', expect.any(Object));
  });

  it.each(['https://example.com/private.jpg', '//example.com/private.jpg', '/home/private.jpg'])(
    'does not render an untrusted image location %s',
    async (url) => {
      vi.mocked(global.fetch).mockResolvedValue(response(withImage(url)));
      renderPage();
      input('shirt');
      submit();
      expect(await screen.findByText(agent.errors.invalidResponse)).toBeInTheDocument();
      expect(screen.queryByRole('img')).not.toBeInTheDocument();
    },
  );
});

describe('Request execution evidence display', () => {
  const model = 'qwen3:4b-instruct-2507-q4_K_M';
  function makeProof() {
    const observed = Date.now() / 1000 - 1;
    return { model, model_digest: 'a'.repeat(64), process_id: 123, process_started_at: observed - 95,
    gpu_uuid: 'GPU-11111111-1111-1111-1111-111111111111', observed_at: observed,
    layers_loaded: 37, layers_total: 37, model_size_bytes: 3000, gpu_size_bytes: 3000,
    process_gpu_memory_bytes: 4096, memory_samples: [{observed_at: observed, process_gpu_memory_bytes: 4096}],
    request_id: 'b'.repeat(32), request_started_at: observed - 1, request_finished_at: observed + 1,
    evidence_id: 'c'.repeat(64) };
  }
  it('shows verified GPU only with this result and clears it on input change', async () => {
    const proof = makeProof();
    vi.mocked(global.fetch).mockResolvedValue(response({ ...preview, execution: {
      request_id: proof.request_id, execution_mode: 'remote_gpu', model,
      gpu_verified: true, gpu_evidence_status: 'verified', gpu_evidence: proof,
      fallback_fields: ['reasoning'],
    } }));
    renderPage();
    expect(screen.queryByText(agent.requestExecution.gpuVerified)).not.toBeInTheDocument();
    input('帮我搭配一套日常穿搭'); submit();
    expect(await screen.findByText(agent.requestExecution.gpuVerified)).toBeInTheDocument();
    expect(screen.getByText(agent.requestExecution.copyFallback)).toBeInTheDocument();
    input('查询我的衬衫');
    expect(screen.queryByText(agent.requestExecution.gpuVerified)).not.toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });
  it('shows unavailable evidence without implying CPU fallback or GPU success', async () => {
    const proof = makeProof();
    vi.mocked(global.fetch).mockResolvedValue(response({ ...preview, execution: {
      request_id: proof.request_id, execution_mode: 'remote_gpu', model, gpu_verified: 'unknown',
    } }));
    renderPage(); input('帮我搭配一套日常穿搭'); submit();
    expect(await screen.findByText(agent.requestExecution.gpuUnverified)).toBeInTheDocument();
    expect(screen.queryByText(agent.requestExecution.gpuVerified)).not.toBeInTheDocument();
  });
  it('rejects a bare GPU success flag without proof', async () => {
    const proof = makeProof();
    vi.mocked(global.fetch).mockResolvedValue(response({ ...preview, execution: {
      request_id: proof.request_id, execution_mode: 'remote_gpu', model, gpu_verified: true,
    } }));
    renderPage(); input('帮我搭配一套日常穿搭'); submit();
    expect(await screen.findByText(agent.errors.invalidResponse)).toBeInTheDocument();
    expect(screen.queryByText(agent.requestExecution.gpuVerified)).not.toBeInTheDocument();
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });
});
