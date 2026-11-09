import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { NextIntlClientProvider } from 'next-intl';
import common from '../messages/en/common.json';
import { ApplicationAccessProvider, applicationAccessSchema } from '@/components/application-access-provider';
import { ApplicationAccessContext, writeOperations, permitsOperation } from '@/lib/application-access-context';
import { Button } from '@/components/ui/button';
import { api, apiErrorMessage } from '@/lib/api';
vi.unmock('next-intl');

afterEach(() => { cleanup(); vi.restoreAllMocks(); });
function renderAccess(value: unknown, owner='user-a') {
  vi.spyOn(api,'get').mockResolvedValue(value);
  const client=new QueryClient({defaultOptions:{queries:{retry:false}}});
  const change=vi.fn();
  render(<QueryClientProvider client={client}>
    <NextIntlClientProvider locale="en" messages={{common}}>
      <ApplicationAccessProvider ownerId={owner}>
        <Button requiresWrite="update" onClick={change}>Save item</Button>
        <Button>Search</Button>
      </ApplicationAccessProvider>
    </NextIntlClientProvider>
  </QueryClientProvider>);
  return change;
}

describe('application access',()=>{
  it('uses actual readonly capability without disabling reads',async()=>{
    const change=renderAccess({application:{mode:'read_only_demo',business_write:false,
      agent_preview:true,reason_code:'APP_READ_ONLY'}});
    await screen.findByText('Read-only demo');
    expect(screen.getByRole('button',{name:'Save item'})).toBeDisabled();
    expect(screen.getByRole('button',{name:'Search'})).not.toBeDisabled();
    fireEvent.click(screen.getByRole('button',{name:'Save item'}));expect(change).not.toHaveBeenCalled();
    expect(api.get).toHaveBeenCalledWith('/capabilities');
  });
  it('enables business entry only after full mode is confirmed',async()=>{
    const change=renderAccess({application:{mode:'full_app',business_write:true,
      agent_preview:true,reason_code:null,write_operations:['update']}});
    const save=screen.getByRole('button',{name:'Save item'});
    await waitFor(()=>expect(save).not.toBeDisabled());
    fireEvent.click(save);expect(change).toHaveBeenCalledOnce();
  });
  it.each([{}, {application:{mode:'read_only_demo',business_write:true,agent_preview:true,reason_code:null}}])(
    'unknown or inconsistent capabilities never grant writes',async(value)=>{
      renderAccess(value);
      await waitFor(()=>expect(api.get).toHaveBeenCalledOnce());
      expect(screen.getByRole('button',{name:'Save item'})).toBeDisabled();
      expect(screen.getByText('Permissions unavailable')).toBeInTheDocument();
    });
  it('a missing owner cannot inherit another account write permission',()=>{
    renderAccess({},'');
    expect(api.get).not.toHaveBeenCalled();
    expect(screen.getByRole('button',{name:'Save item'})).toBeDisabled();
  });
  it('a malformed full mode cannot become trusted',()=>{
    expect(()=>applicationAccessSchema.parse({mode:'full_app',business_write:false,
      agent_preview:true,reason_code:null})).toThrow();
  });
  it('multipart and JSON errors use the same explicit reason',()=>{
    expect(apiErrorMessage({detail:{error_code:'APP_READ_ONLY',message:'Read only'}},'fallback')).toBe('Read only');
    expect(apiErrorMessage({detail:'Bad upload'},'fallback')).toBe('Bad upload');
  });
});


describe('operation-level permission mapping', () => {
  it.each(['create','update','delete','archive','restore','rotate'] as const)(
    'only enables the explicitly granted %s operation', (operation) => {
      render(<ApplicationAccessContext.Provider value={{businessWrite:true,reason:'Not allowed',writeOperations:[operation]}}>
        {writeOperations.map(value=><Button key={value} requiresWrite={value}>{value}</Button>)}
        <Button requiresWrite>Unscoped</Button>
      </ApplicationAccessContext.Provider>);
      for (const value of writeOperations) {
        expect(screen.getByRole('button',{name:value}).hasAttribute('disabled')).toBe(value!==operation);
      }
      expect(screen.getByRole('button',{name:'Unscoped'})).toBeDisabled();
    });
  it('does not grant legacy boolean-only or unknown bulk operations', () => {
    const access={businessWrite:true,reason:null};
    expect(permitsOperation(access,'create')).toBe(false);
    expect(permitsOperation({...access,writeOperations:[...writeOperations]},'bulk_create')).toBe(false);
  });
  it.each([
    {mode:'full_app',business_write:true,reason_code:null,write_operations:['bulk_create']},
    {mode:'full_app',business_write:true,reason_code:null,write_operations:['create','create']},
    {mode:'read_only_demo',business_write:false,reason_code:'APP_READ_ONLY',write_operations:['create']},
  ])('rejects malformed or excessive capability lists', value => {
    expect(()=>applicationAccessSchema.parse({...value,agent_preview:true})).toThrow();
  });
  it('missing operation list grants no operations', () => {
    expect(applicationAccessSchema.parse({mode:'full_app',business_write:true,reason_code:null,agent_preview:true}).write_operations).toEqual([]);
  });
});
