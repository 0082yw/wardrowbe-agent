'use client';
import { useContext, useState, type FormEvent } from 'react';
import { useTranslations } from 'next-intl';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useUpdateItem } from '@/lib/hooks/use-items';
import { ApplicationAccessContext, permitsOperation } from '@/lib/application-access-context';
import { USER_TAG_FIELDS, type UserTagKey, type UserTagPatch } from '@/lib/item-update';
import type { Item } from '@/lib/types';

export function ItemUserTags({ item }: { item: Item }) {
  const t = useTranslations('wardrobe.userTags');
  const tc = useTranslations('common');
  const access = useContext(ApplicationAccessContext);
  const mutation = useUpdateItem();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<Record<UserTagKey, string>>({
    size: '', care_instructions: '', source_url: '',
  });
  const [error, setError] = useState(false);
  const blocked = item.is_archived || mutation.isPending;
  const hasTags = USER_TAG_FIELDS.some(({ key }) => !!item.tags?.[key]);

  async function change(tags: UserTagPatch) {
    if (!permitsOperation(access, 'update') || item.is_archived || mutation.isPending) return;
    setError(false);
    try {
      await mutation.mutateAsync({ id: item.id, data: { tags } });
      setEditing(false);
    } catch {
      setError(true);
    }
  }

  function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const tags = Object.fromEntries(USER_TAG_FIELDS.map(({ key }) => [key, draft[key].trim() || null]));
    void change(tags);
  }

  return (
    <section className="space-y-2 border-t pt-3" aria-label={t('title')}>
      <h3 className="text-sm font-medium">{t('title')}</h3>
      <p className="text-xs text-muted-foreground">{t('hint')}</p>
      {editing ? (
        <form onSubmit={save} className="space-y-3">
          {USER_TAG_FIELDS.map(({ key, maxLength }) => (
            <div key={key} className="space-y-1">
              <Label htmlFor={'user-tag-' + item.id + '-' + key}>{t(key)}</Label>
              <Input id={'user-tag-' + item.id + '-' + key} value={draft[key]} maxLength={maxLength}
                disabled={blocked} onChange={event => setDraft({ ...draft, [key]: event.target.value })} />
            </div>
          ))}
          <div className="flex gap-2">
            <Button type="submit" requiresWrite="update" disabled={blocked}>{tc('save')}</Button>
            <Button type="button" variant="outline" disabled={mutation.isPending}
              onClick={() => { setEditing(false); setError(false); }}>{tc('cancel')}</Button>
          </div>
        </form>
      ) : (
        <>
          {hasTags ? <dl className="space-y-2">
            {USER_TAG_FIELDS.map(({ key }) => item.tags?.[key] ? (
              <div key={key} className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <dt className="text-xs text-muted-foreground">{t(key)}</dt>
                  <dd className="break-all text-sm">{item.tags[key]}</dd>
                </div>
                <Button type="button" size="sm" variant="ghost" requiresWrite="update" disabled={blocked}
                  aria-label={t('remove', { name: t(key) })}
                  onClick={() => void change({ [key]: null })}>{tc('delete')}</Button>
              </div>
            ) : null)}
          </dl> : <p className="text-sm text-muted-foreground">{t('empty')}</p>}
          <Button type="button" size="sm" variant="outline" requiresWrite="update" disabled={blocked}
            onClick={() => {
              setDraft({ size: item.tags?.size ?? '', care_instructions: item.tags?.care_instructions ?? '',
                source_url: item.tags?.source_url ?? '' });
              setError(false); setEditing(true);
            }}>{t(hasTags ? 'edit' : 'add')}</Button>
        </>
      )}
      {item.is_archived && <p className="text-xs text-muted-foreground">{t('restoreFirst')}</p>}
      {error && <p role="alert" className="text-sm text-destructive">{t('failed')}</p>}
    </section>
  );
}
