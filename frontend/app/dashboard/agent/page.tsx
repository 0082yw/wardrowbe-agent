'use client';

import { useEffect, useState, type FormEvent } from 'react';
import Image from 'next/image';
import { Loader2, MessageSquare, Shirt, Sparkles } from 'lucide-react';
import { useTranslations } from 'next-intl';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { MAX_AGENT_MESSAGE_LENGTH } from '@/lib/agent';
import { RequestGpuEvidence } from '@/components/request-gpu-evidence';
import { useAgentPreview, useAgentStatus } from '@/lib/hooks/use-agent-preview';
import { useTypeLabel } from '@/lib/hooks/use-translated-constants';
import { ITEM_ROLE } from '@/lib/generated/garment-vocabulary';

function PreviewItemImage({ src, alt, unavailable }: {
  src: string | null;
  alt: string;
  unavailable: string;
}) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);

  return (
    <div className="relative h-20 w-20 shrink-0 overflow-hidden rounded-md bg-muted">
      {src && failedSrc !== src ? (
        <Image
          src={src}
          alt={alt}
          fill
          sizes="80px"
          unoptimized
          className="object-contain"
          onError={() => setFailedSrc(src)}
        />
      ) : (
        <div role="img" aria-label={alt + ': ' + unavailable} className="flex h-full items-center justify-center">
          <Shirt className="h-8 w-8 text-muted-foreground" aria-hidden="true" />
        </div>
      )}
    </div>
  );
}

export default function AgentPage() {
  const t = useTranslations('agent');
  const typeLabel = useTypeLabel();
  const [message, setMessage] = useState('');
  const { data, pending, error, canSubmit, submit, cancel } = useAgentPreview();
  const [selectedPreview, setSelectedPreview] = useState<number | null>(null);
  useEffect(() => setSelectedPreview(null), [data]);
  const { data: modelStatus, isError: statusError } = useAgentStatus();
  const modelMode = statusError || !modelStatus || modelStatus.mode === 'unknown'
    ? 'unknown'
    : modelStatus.mode === 'remote_ollama'
      ? modelStatus.available ? 'remoteConfigured' : 'remoteUnavailable'
      : modelStatus.available && modelStatus.cpu_only ? 'configured' : 'unavailable';

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void submit(message);
  }

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <header className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="rounded-xl bg-primary/10 p-3 text-primary">
            <MessageSquare className="h-6 w-6" aria-hidden="true" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">{t('title')}</h1>
          <Badge variant="secondary">{t('runtimeBadge.' + modelMode)}</Badge>
        </div>
        <p className="text-muted-foreground">{t('description')}</p>
      </header>

      <Alert>
        <AlertDescription>{t('runtimeNotice.' + modelMode)}</AlertDescription>
      </Alert>
      {modelStatus && modelStatus.mode !== 'unknown' && !statusError && (
        <div className="space-y-1 text-sm text-muted-foreground" aria-label={t('runtimeDetails.label')}>
          <p>{t(modelStatus.mode === 'remote_ollama' ? 'runtimeDetails.remoteGpu' : 'runtimeDetails.localCpu')}</p>
          {modelStatus.mode === 'remote_ollama' && <>
            <p>{t('tunnelState.' + modelStatus.tunnel_status)}</p>
            <p>{t('runtimeDetails.gpuReadinessOnly')}</p>
          </>}
        </div>
      )}

      <Card>
        <CardContent className="pt-6">
          <form onSubmit={handleSubmit} noValidate className="space-y-4" aria-busy={pending}>
            <div className="space-y-2">
              <label htmlFor="agent-message" className="text-sm font-medium">{t('inputLabel')}</label>
              <Textarea
                id="agent-message"
                value={message}
                rows={5}
                placeholder={t('placeholder')}
                aria-describedby={error ? 'agent-input-help agent-error' : 'agent-input-help'}
                aria-invalid={error === 'blank' || error === 'tooLong' || error === 'invalidInput'}
                onChange={(event) => {
                  cancel();
                  setMessage(event.target.value);
                }}
              />
              <div id="agent-input-help" className="flex flex-wrap justify-between gap-2 text-xs text-muted-foreground">
                <span>{t('inputHelp')}</span>
                <span>{t('characterCount', { count: Array.from(message).length, max: MAX_AGENT_MESSAGE_LENGTH })}</span>
              </div>
            </div>
            <Button type="submit" disabled={pending || !canSubmit} className="w-full sm:w-auto">
              {pending
                ? <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden="true" />
                : <Sparkles className="mr-2 h-4 w-4" aria-hidden="true" />}
              {pending ? t('generating') : t('generate')}
            </Button>
            {!canSubmit && <p className="text-sm text-muted-foreground">{t('errors.auth')}</p>}
          </form>
        </CardContent>
      </Card>

      <div aria-live="polite" aria-atomic="false" className="space-y-4">
        {error && (
          <Alert id="agent-error" variant="destructive">
            <AlertDescription>{t('errors.' + error)}</AlertDescription>
          </Alert>
        )}
        {pending && <p role="status" className="text-sm text-muted-foreground">{t('loading')}</p>}
        {data && (
          <section aria-label={t('results')} className="space-y-4">
            <Card>
              <CardContent className="space-y-3 pt-6">
                <Badge variant={data.status === 'invalid_result' ? 'destructive' : 'outline'}>
                  {t('status.' + data.status)}
                </Badge>
                <h2 className="font-semibold">{t('reply')}</h2>
                <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{data.answer}</p>
                {data.execution && <div className="space-y-1 text-xs text-muted-foreground" aria-label={t('requestExecution.label')}>
                  <p>{t('requestExecution.request', { id: data.execution.request_id.slice(0, 8) })}</p>
                  {data.execution.execution_mode === 'remote_gpu' && <RequestGpuEvidence execution={data.execution} />}
                  {data.execution.fallback_fields.length > 0 && <p>{t('requestExecution.copyFallback')}</p>}
                </div>}
                {data.status !== 'success' && (
                  <p className="text-sm text-muted-foreground">{t('guidance.' + data.status)}</p>
                )}
              </CardContent>
            </Card>
            {data.kind === 'outfit' && data.outfits.length > 1 && (
              <div role="group" aria-label={t('previewSelector')} className="flex flex-wrap gap-2">
                <Button type="button" size="sm" variant={selectedPreview === null ? 'default' : 'outline'}
                  aria-pressed={selectedPreview === null} onClick={() => setSelectedPreview(null)}>
                  {t('allPreviews')}
                </Button>
                {data.outfits.map((_, index) => (
                  <Button key={index} type="button" size="sm"
                    variant={selectedPreview === index ? 'default' : 'outline'}
                    aria-pressed={selectedPreview === index} onClick={() => setSelectedPreview(index)}>
                    {t('selectPreview', { number: index + 1 })}
                  </Button>
                ))}
              </div>
            )}
            <div className="grid gap-4 md:grid-cols-2">
              {data.outfits.map((outfit, index) => ({ outfit, index }))
                .filter(({ index }) => data.kind !== 'outfit' || selectedPreview === null || selectedPreview === index)
                .map(({ outfit, index }) => (
                <Card key={index}>
                  <CardHeader>
                    <CardTitle className="text-lg">{data.kind === 'outfit'
                      ? outfit.title : t('outfit', { number: index + 1 })}</CardTitle>
                    {data.kind === 'outfit' && <Badge variant="secondary">{t('scene.' + data.scene)}</Badge>}
                    {data.kind === 'outfit' && <span className="text-xs text-muted-foreground">{t('unsaved')}</span>}
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <ul className="space-y-3">
                      {outfit.items.map((item) => (
                        <li key={item.id} className="flex items-start gap-3 rounded-md bg-muted/50 p-3">
                          <PreviewItemImage
                            src={item.thumbnail_url}
                            alt={item.name?.trim() || t('unnamedItem')}
                            unavailable={t('imageUnavailable')}
                          />
                          <div className="min-w-0 space-y-1">
                            <p className="break-words font-medium">{item.name?.trim() || t('unnamedItem')}</p>
                            {data.kind === 'outfit' && <p className="text-xs text-muted-foreground">
                              {t('roles.' + ITEM_ROLE[item.type])}
                            </p>}
                            <p className="break-words text-sm text-muted-foreground">
                              {t('itemType', { type: typeLabel(item.type) })}
                            </p>
                          </div>
                        </li>
                      ))}
                    </ul>
                    <dl className="space-y-4 text-sm">
                      <div>
                        <dt className="font-medium">{t(data.kind === 'outfit' ? 'outfitReasoning' : 'reasoning')}</dt>
                        <dd className="mt-1 whitespace-pre-wrap break-words text-muted-foreground">{outfit.reasoning}</dd>
                      </div>
                      <div>
                        <dt className="font-medium">{t('styleNotes')}</dt>
                        <dd className="mt-1 whitespace-pre-wrap break-words text-muted-foreground">{outfit.style_notes}</dd>
                      </div>
                    </dl>
                  </CardContent>
                </Card>
              ))}
            </div>
            <p className="text-xs text-muted-foreground">{t(data.kind === 'outfit' ? 'outfitPreviewOnly' : 'previewOnly')}</p>
          </section>
        )}
        {!data && !pending && !error && (
          <Card className="border-dashed bg-muted/20">
            <CardContent className="py-10 text-center text-sm text-muted-foreground">{t('emptyState')}</CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
