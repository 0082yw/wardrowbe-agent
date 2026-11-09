import { it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BulkActionToolbar } from '@/components/bulk-action-toolbar';
import { ApplicationAccessContext, writeOperations } from '@/lib/application-access-context';
vi.mock('next-intl', () => ({useTranslations: () => (key: string) => key}));
it('single-item full_app grants cannot authorize bulk mutations', () => {
 const noop=vi.fn();
 render(<ApplicationAccessContext.Provider value={{businessWrite:true,reason:'Not authorized',writeOperations}}>
 <BulkActionToolbar selection={{mode:'some',selectedIds:new Set(['test-id']),excludedIds:new Set()}}
 totalItems={2} pageItems={2} onSelectAll={noop} onSelectAllMatching={noop} onClear={noop}
 onDelete={noop} onReanalyze={noop} onRotate={noop} onRemoveBackground={noop} />
 </ApplicationAccessContext.Provider>);
 for(const name of ['delete','bulkActions.reanalyze','bulkActions.rotateCcw','bulkActions.rotateCw','bulkActions.removeBackground']){
   expect(screen.getByRole('button',{name})).toBeDisabled();
 }
 expect(noop).not.toHaveBeenCalled();
});
