import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Resolved per request, not at module scope: next.config.js rewrites are serialized into
// routes-manifest.json at build time, so a rewrite cannot honor a runtime BACKEND_URL in the
// prebuilt image. Route handlers are the only proxy layer that reads env at request time.
function backendUrl(): string {
  const url =
    process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://backend:8000';
  return url.replace(/\/+$/, '');
}

const STRIPPED_REQUEST_HEADERS = new Set([
  'connection',
  'content-length',
  'host',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
]);

// content-encoding and content-length must go: fetch has already decoded the body (the backend
// runs GZipMiddleware), so forwarding them would describe bytes the client never receives.
const STRIPPED_RESPONSE_HEADERS = new Set([
  'connection',
  'content-encoding',
  'content-length',
  'keep-alive',
  'trailer',
  'transfer-encoding',
  'upgrade',
]);

const METHODS_WITHOUT_BODY = new Set(['GET', 'HEAD']);

function buildRequestHeaders(request: NextRequest): Headers {
  const headers = new Headers();
  request.headers.forEach((value, key) => {
    if (!STRIPPED_REQUEST_HEADERS.has(key.toLowerCase())) {
      headers.set(key, value);
    }
  });
  return headers;
}

function buildResponseHeaders(upstream: Response): Headers {
  const headers = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!STRIPPED_RESPONSE_HEADERS.has(key.toLowerCase())) {
      headers.set(key, value);
    }
  });

  headers.delete('set-cookie');
  for (const cookie of upstream.headers.getSetCookie?.() ?? []) {
    headers.append('set-cookie', cookie);
  }

  return headers;
}

async function proxy(request: NextRequest): Promise<NextResponse> {
  const target = new URL(request.url);
  const backend = backendUrl();
  // pathname/search rather than the decoded `params.path`, so signed image URLs survive intact.
  const url = `${backend}${target.pathname}${target.search}`;

  const hasBody = !METHODS_WITHOUT_BODY.has(request.method);
  const init: RequestInit & { duplex?: 'half' } = {
    method: request.method,
    headers: buildRequestHeaders(request),
    // Manual because the default 'follow' throws on auth/mobile-callback, which redirects to
    // a wardrobe:// app scheme that fetch cannot follow.
    redirect: 'manual',
  };

  if (hasBody && request.body) {
    init.body = request.body;
    init.duplex = 'half';
  }

  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch (error) {
    const code = error instanceof Error ? (error as Error & { code?: string }).code : undefined;
    const readOnly = code === 'APP_READ_ONLY' || (error instanceof Error &&
      error.message === 'Read-only Agent validation prohibits other write requests');
    const policyDenied = code === 'APP_OPERATION_FORBIDDEN';
    const authRequired = code === 'APP_AUTH_REQUIRED';
    // Neither the URL nor exception message is safe: either may contain a signature,
    // user path, credentials, or other query parameters.
    console.error('API proxy request failed', {
      method: request.method,
      pathKind: target.pathname.includes('/images/') ? 'signed_image' : 'api',
      reason: readOnly ? 'application_read_only' : policyDenied ? 'application_operation_forbidden' :
        authRequired ? 'authentication_required' : 'upstream_unavailable',
    });
    return NextResponse.json(
      { detail: {
        error_code: readOnly ? 'APP_READ_ONLY' : policyDenied ? 'APP_OPERATION_FORBIDDEN' :
          authRequired ? 'APP_AUTH_REQUIRED' : 'BACKEND_UNAVAILABLE',
        message: readOnly
          ? 'This application is in read-only demo mode. Business changes are disabled.'
          : policyDenied ? 'This operation is not enabled by the trusted application policy.'
          : authRequired ? 'Sign in before changing wardrobe items.'
          : 'The backend is temporarily unavailable.',
      } },
      { status: authRequired ? 401 : readOnly || policyDenied ? 403 : 502 }
    );
  }

  return new NextResponse(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: buildResponseHeaders(upstream),
  });
}

export {
  proxy as GET,
  proxy as POST,
  proxy as PUT,
  proxy as PATCH,
  proxy as DELETE,
  proxy as HEAD,
  proxy as OPTIONS,
};
