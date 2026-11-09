'use client';
import { useContext, useState } from 'react';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle } from '@/components/ui/alert-dialog';
import { useArchiveItem, useRestoreArchivedItem } from '@/lib/hooks/use-items';
import { ApplicationAccessContext, permitsOperation } from '@/lib/application-access-context';
import type { Item } from '@/lib/types';

export function ItemLifecycleActions({ item, onComplete }: { item: Item; onComplete: () => void }) {
  const t = useTranslations('wardrobe.lifecycle');
  const tc = useTranslations('common');
  const access = useContext(ApplicationAccessContext);
  const archive = useArchiveItem();
  const restore = useRestoreArchivedItem();
  const [confirm, setConfirm] = useState(false);
  const [error, setError] = useState(false);
  const pending = archive.isPending || restore.isPending;
  const operation = item.is_archived ? 'restore' : 'archive';

  async function apply() {
    if (pending || !permitsOperation(access, operation)) return;
    setError(false);
    try {
      if (item.is_archived) await restore.mutateAsync(item.id);
      else await archive.mutateAsync({ id: item.id });
      setConfirm(false);
      onComplete();
    } catch {
      setError(true);
    }
  }

  return (
    <section className="space-y-2 border-t pt-3" aria-label={t('title')}>
      <h3 className="text-sm font-medium">{t('title')}</h3>
      <p className="text-xs text-muted-foreground">{t(item.is_archived ? 'archivedHint' : 'activeHint')}</p>
      <Button type="button" variant="outline" size="sm" requiresWrite={operation} disabled={pending}
        onClick={() => item.is_archived ? void apply() : setConfirm(true)}>{t(operation)}</Button>
      {error && <p role="alert" className="text-sm text-destructive">{t('failed')}</p>}
      <AlertDialog open={confirm} onOpenChange={setConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('confirmTitle')}</AlertDialogTitle>
            <AlertDialogDescription>{t('confirmDescription')}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={pending}>{tc('cancel')}</AlertDialogCancel>
            <AlertDialogAction disabled={pending || !permitsOperation(access, 'archive')}
              onClick={event => { event.preventDefault(); void apply(); }}>{t('archive')}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}
