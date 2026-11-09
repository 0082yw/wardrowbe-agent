import fs from 'node:fs';
import crypto from 'node:crypto';

/** Manifest is a pinned, server-owned file. No browser field is a mode source. */
export function loadPolicy(path, digest, mode = 'read_only_demo') {
  if (!['read_only_demo', 'full_app'].includes(mode)) throw new Error('Invalid application mode');
  const stat = fs.lstatSync(path);
  if (!stat.isFile() || stat.isSymbolicLink() || (stat.mode & 0o022) ||
      (typeof process.getuid === 'function' && stat.uid !== process.getuid())) {
    throw new Error('Unsafe permission manifest');
  }
  const bytes = fs.readFileSync(path);
  if (crypto.createHash('sha256').update(bytes).digest('hex') !== digest) throw new Error('Permission manifest changed');
  const manifest = JSON.parse(bytes);
  if (manifest.version !== 1 || manifest.default_mode !== 'read_only_demo' ||
      !Array.isArray(manifest.write_routes)) throw new Error('Unsupported permission manifest');
  return Object.freeze({ mode, manifest });
}

export function allowsHttp(policy, method, path) {
  method = method.toUpperCase();
  const m = policy.manifest;
  if (m.read_methods.includes(method)) return true;
  if (method === 'POST' && m.non_business_posts.includes(path)) return true;
  return policy.mode === 'full_app' && m.write_routes.some(
    route => route.method === method && new RegExp(route.pattern).test(path),
  );
}

function deny(code, message) {
  const error = new Error(message);
  error.code = code;
  throw error;
}

export function assertHttp(policy, url, method, headers) {
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || url.port !== '8001') {
    throw new Error('Application policy only permits the fixed backend');
  }
  if (!allowsHttp(policy, method, url.pathname)) {
    deny(policy.mode === 'read_only_demo' ? 'APP_READ_ONLY' : 'APP_OPERATION_FORBIDDEN', 'Application operation forbidden');
  }
  const business = !policy.manifest.read_methods.includes(method.toUpperCase()) &&
    !(method.toUpperCase() === 'POST' && policy.manifest.non_business_posts.includes(url.pathname));
  if (business && !new Headers(headers).get('authorization')?.startsWith('Bearer ')) {
    deny('APP_AUTH_REQUIRED', 'Authenticated backend token required');
  }
}
