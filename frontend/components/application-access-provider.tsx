'use client';
import { useQuery } from '@tanstack/react-query';
import { useTranslations } from 'next-intl';
import { z } from 'zod';
import { api } from '@/lib/api';
import { queryKeys } from '@/lib/hooks/query-keys';
import { ApplicationAccessContext, writeOperations } from '@/lib/application-access-context';

export const applicationAccessSchema = z.object({
  mode: z.enum(['read_only_demo', 'full_app']),
  business_write: z.boolean(),
  write_operations: z.array(z.enum(writeOperations)).default([]),
  agent_preview: z.literal(true),
  reason_code: z.enum(['APP_READ_ONLY']).nullable(),
}).superRefine((value, ctx) => {
  if (new Set(value.write_operations).size !== value.write_operations.length ||
      (value.mode === 'read_only_demo' && value.write_operations.length > 0) ||
      value.business_write !== (value.mode === 'full_app') ||
      value.reason_code !== (value.mode === 'read_only_demo' ? 'APP_READ_ONLY' : null)) {
    ctx.addIssue({ code: z.ZodIssueCode.custom, message: 'Inconsistent application access' });
  }
});

export function ApplicationAccessProvider({ ownerId, children }: {
  ownerId: string | undefined; children: React.ReactNode;
}) {
  const t = useTranslations('common.runtime');
  const { data, isError } = useQuery({
    queryKey: queryKeys.applicationAccess(ownerId),
    queryFn: async () => {
      const result = await api.get<{ application?: unknown }>('/capabilities');
      return applicationAccessSchema.parse(result.application);
    },
    enabled: !!ownerId,
    retry: false,
    staleTime: 0,
    refetchOnWindowFocus: true,
  });
  // Do not retain write permission after a failed refresh or an owner change.
  const known = !!data && !isError;
  const businessWrite = known && data.business_write;
  const reason = businessWrite ? t('operationReason') : t(known ? 'readOnlyReason' : 'unknownReason');
  return (
    <ApplicationAccessContext.Provider value={{ businessWrite, reason, writeOperations: known ? data.write_operations : [] }}>
      <div role="status" className="mb-4 rounded-md border bg-muted/30 p-3 text-sm">
        <p className="font-medium">{t(!known ? 'unknownMode' : businessWrite ? 'fullMode' : 'readOnlyMode')}</p>
        {!businessWrite && reason && <p className="mt-1 text-muted-foreground">{reason}</p>}
      </div>
      {children}
    </ApplicationAccessContext.Provider>
  );
}
