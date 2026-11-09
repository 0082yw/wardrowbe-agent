// @vitest-environment node
import { describe,it,expect,vi,afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import { GET,POST } from '@/app/api/v1/[...path]/route';
afterEach(()=>vi.restoreAllMocks());

describe('proxy policy and privacy',()=>{
 it('reports legacy runtime policy denial as 403 rather than backend unreachable',async()=>{
   vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error('Read-only Agent validation prohibits other write requests')));
   const response=await POST(new NextRequest('http://localhost/api/v1/items',{method:'POST'}));
   expect(response.status).toBe(403);
   const body=await response.json();
   expect(body.detail.error_code).toBe('APP_READ_ONLY');
   expect(JSON.stringify(body)).not.toContain('Unable to reach');
   vi.unstubAllGlobals();
 });
 it('never logs or returns a signed URL, user path or raw exception',async()=>{
   const secret='PRIVATE_SIGNATURE';
   const url='http://localhost/api/v1/images/private-user/private-file.jpg?expires=123&sig='+secret;
   const log=vi.spyOn(console,'error').mockImplementation(()=>{});
   vi.stubGlobal('fetch',vi.fn().mockRejectedValue(new Error(url)));
   const response=await GET(new NextRequest(url));
   expect(response.status).toBe(502);
   const output=JSON.stringify(log.mock.calls)+JSON.stringify(await response.json());
   for(const value of [secret,'private-user','private-file','expires=','sig='])expect(output).not.toContain(value);
   vi.unstubAllGlobals();
 });
});


describe('trusted policy error mapping', () => {
 it.each([
   ['APP_READ_ONLY', 403],
   ['APP_OPERATION_FORBIDDEN', 403],
   ['APP_AUTH_REQUIRED', 401],
 ])('preserves %s without claiming backend failure', async (code, status) => {
   const error = Object.assign(new Error('private diagnostic path'), { code });
   const log = vi.spyOn(console, 'error').mockImplementation(() => {});
   vi.stubGlobal('fetch', vi.fn().mockRejectedValue(error));
   try {
     const response = await POST(new NextRequest('http://localhost/api/v1/items', { method: 'POST' }));
     expect(response.status).toBe(status);
     const body = await response.json();
     expect(body.detail.error_code).toBe(code);
     expect(JSON.stringify(body) + JSON.stringify(log.mock.calls)).not.toContain('private diagnostic');
   } finally { vi.unstubAllGlobals(); }
 });
});
