'use client';
import { useEffect, useState } from 'react';
import { useTranslations } from 'next-intl';
import { currentRequestGpuState, GPU_EVIDENCE_MAX_AGE_MS, type RequestExecution } from '@/lib/gpu-evidence-state';
import { agentExecutionSchema } from '@/lib/agent';

export function RequestGpuEvidence({ execution }: { execution: RequestExecution }) {
  const t = useTranslations('agent.requestExecution');
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const current = Date.now();
    setNow(current);
    const observed = execution.gpu_evidence?.observed_at;
    if (execution.gpu_verified !== true || observed === undefined) return;
    const remaining = observed * 1000 + GPU_EVIDENCE_MAX_AGE_MS - current;
    if (remaining < 0 || remaining > GPU_EVIDENCE_MAX_AGE_MS) return;
    // One local expiry timer; no polling, network calls or persistent GPU sampling.
    const timer = setTimeout(() => setNow(Date.now()), remaining + 1);
    return () => clearTimeout(timer);
  }, [execution]);
  const parsed = agentExecutionSchema.safeParse(execution);
  const proof = parsed.success ? parsed.data.gpu_evidence : null;
  const verified = currentRequestGpuState(execution, now) === 'verified';

  return (
    <section className="space-y-1" aria-label={t('proofLabel')}>
      <p role="status">{t(verified ? 'gpuVerified' : 'gpuUnverified')}</p>
      {proof && (
        <>
          {!verified && <p>{t('proofExpired')}</p>}
          <dl className="grid grid-cols-1 gap-1 break-all sm:grid-cols-2">
            <div><dt>{t('proofRequest')}</dt><dd>{proof.request_id}</dd></div>
            <div><dt>{t('proofPid')}</dt><dd>{proof.process_id}</dd></div>
            <div><dt>{t('proofDevice')}</dt><dd>{proof.gpu_uuid}</dd></div>
            <div><dt>{t('proofLayers')}</dt><dd>{proof.layers_loaded}/{proof.layers_total}</dd></div>
            <div><dt>{t('proofMemory')}</dt><dd>{t('proofMemoryValue', { value: Math.round(Math.max(...proof.memory_samples.map(row =>
              row.process_gpu_memory_bytes)) / 1024 / 1024 * 10) / 10 })}</dd></div>
            <div><dt>{t('proofSamples')}</dt><dd>{proof.memory_samples.length}</dd></div>
          </dl>
          <p>{t('sampleScope')}</p>
        </>
      )}
    </section>
  );
}
