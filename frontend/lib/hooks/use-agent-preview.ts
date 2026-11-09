'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useSession } from 'next-auth/react';
import { ZodError } from 'zod';
import { api, ApiError } from '@/lib/api';
import {
  agentPreviewSchema,
  agentStatusSchema,
  getAgentInputError,
  type AgentInputError,
  type AgentPreviewResponse,
  type AgentRequest,
} from '@/lib/agent';
import { applySessionToken } from '@/lib/hooks/use-session-token';
import { queryKeys } from '@/lib/hooks/query-keys';

type PreviewError = AgentInputError | 'auth' | 'invalidInput' | 'rateLimited' | 'failed' | 'invalidResponse';
interface PreviewState {
  owner?: string;
  pending: boolean;
  data?: AgentPreviewResponse;
  error?: PreviewError;
}

export function useAgentPreview() {
  const { data: session, status } = useSession();
  const owner = status === 'authenticated' && session?.accessToken ? session?.user?.id ?? session?.user?.email ?? undefined : undefined;
  const currentOwner = useRef(owner);
  currentOwner.current = owner;
  const sequence = useRef(0);
  const active = useRef<AbortController | null>(null);
  const busy = useRef(false);
  const [state, setState] = useState<PreviewState>({ pending: false });

  const { mutateAsync, reset } = useMutation({
    mutationFn: async ({ message, signal }: AgentRequest & { signal: AbortSignal }) => {
      applySessionToken(session);
      const response = await api.post<unknown>('/agent/preview', { message }, { signal });
      return agentPreviewSchema.parse(response);
    },
    retry: false,
    networkMode: 'always',
    gcTime: 0,
    meta: { toastsOwnErrors: true },
  });

  const cancel = useCallback(() => {
    sequence.current += 1;
    active.current?.abort();
    active.current = null;
    busy.current = false;
    reset();
    setState({ pending: false });
  }, [reset]);

  useEffect(() => {
    cancel();
    return () => {
      sequence.current += 1;
      active.current?.abort();
      active.current = null;
      busy.current = false;
    };
  }, [owner, cancel]);

  async function submit(message: string) {
    // The ref also blocks two submits before React has rendered the disabled button.
    if (busy.current) return;
    const error = getAgentInputError(message);
    if (error || !owner) {
      setState({ owner, pending: false, error: error ?? 'auth' });
      return;
    }
    const requestId = ++sequence.current;
    const requestOwner = owner;
    const controller = new AbortController();
    active.current = controller;
    busy.current = true;
    setState({ owner, pending: true });
    try {
      const data = await mutateAsync({ message: message.trim(), signal: controller.signal });
      if (requestId === sequence.current && currentOwner.current === requestOwner) {
        setState({ owner: requestOwner, pending: false, data });
      }
    } catch (cause) {
      if (requestId !== sequence.current || currentOwner.current !== requestOwner) return;
      let failure: PreviewError = 'failed';
      if (cause instanceof ZodError) failure = 'invalidResponse';
      else if (cause instanceof ApiError) {
        if (cause.status === 401) failure = 'auth';
        else if (cause.status === 422) failure = 'invalidInput';
        else if (cause.status === 429) failure = 'rateLimited';
      }
      setState({ owner: requestOwner, pending: false, error: failure });
    } finally {
      if (requestId === sequence.current) {
        active.current = null;
        busy.current = false;
      }
    }
  }

  // Never display another account's preview during a session transition.
  const visibleState: PreviewState = state.owner === owner ? state : { pending: false };
  return { ...visibleState, canSubmit: Boolean(owner), submit, cancel };
}

/** A separate authenticated metadata request; it never submits a preview. */
export function useAgentStatus() {
  const { data: session, status } = useSession();
  const owner = status === 'authenticated' && session?.accessToken ? session?.user?.id ?? session?.user?.email ?? undefined : undefined;

  return useQuery({
    queryKey: queryKeys.agentStatus(owner),
    enabled: Boolean(owner),
    queryFn: async ({ signal }) => {
      applySessionToken(session);
      return agentStatusSchema.parse(await api.get<unknown>('/agent/status', { signal }));
    },
    retry: false,
    networkMode: 'always',
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    gcTime: 0,
    meta: { toastsOwnErrors: true },
  });
}
