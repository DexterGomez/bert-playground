// api.js — every network call goes through request(); errors become ApiError with a human message.

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request(path, { method = 'GET', body = null } = {}) {
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    throw new ApiError(`Network error — backend not running? (${err.message})`);
  }

  let data = null;
  try {
    data = await res.json();
  } catch (err) {
    /* non-JSON body; fall through */
  }

  if (!res.ok) {
    const detail = data && typeof data.detail === 'string' && data.detail
      ? data.detail
      : `HTTP ${res.status} ${res.statusText || ''}`.trim();
    throw new ApiError(detail, res.status);
  }
  return { data, clientMs: Math.max(1, Math.round(performance.now() - t0)) };
}

export const api = {
  meta: () => request('/api/meta'),
  tokenize: (model, text) => request('/api/tokenize', { method: 'POST', body: { model, text } }),
  forward: (model, text) => request('/api/forward', { method: 'POST', body: { model, text } }),
  layer: (model, text, layer) => request('/api/layer', { method: 'POST', body: { model, text, layer } }),
  similarity: (model, sentences) => request('/api/similarity', { method: 'POST', body: { model, sentences } }),
  probe: (probe, text) => request('/api/probe', { method: 'POST', body: { probe, text } }),
};
