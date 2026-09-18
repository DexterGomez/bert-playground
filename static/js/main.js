// main.js — state, event wiring, per-tab render orchestration.

import { api } from './api.js';
import {
  viridis, diverging, tokenColor, clamp, esc,
  drawHeatmap, drawLineChart, drawScatter, drawPlaceholder,
  bindMatrixHover, bindChartHover, bindScatterHover, showTip, hideTip,
} from './viz.js';
import {
  renderPipeline, renderTokenTable, renderCompareView, renderLegend,
  makeArcView, renderHeadGrid, renderProbePipeline,
} from './views.js';

/* ---------------- state ---------------- */

const state = {
  meta: null,
  model: null,
  text: '',
  tab: 'tokenize',
  visited: { embeddings: false, probe: false },

  tokenize: null,          // /api/tokenize for (model, text)
  compare: null,           // {model, text, data} for the "other" model
  forward: null,           // /api/forward for (model, text)
  stripesMax: 1e-6,
  layerIdx: 0,
  fullMode: false,
  embStretch: false,
  full768: null,           // {model, text, layer, values} from /api/layer
  normHl: null,            // highlighted token index in the norms chart

  attLayer: 0,
  attHead: 0,
  attStat: 'to_sep',

  embPool: 'cls',
  simAB: null,             // {a, b, data}
  simBank: null,           // {sentences, data}

  probeId: null,
  probeDirty: false,       // user edited probe input -> stop tracking top bar
  probeData: null,

  errorSource: null,
};

const PRESETS = [
  'I hate transformer encoders',
  'Los transformadores son increíbles',
  'The bank raised interest rates by the river bank',
];

const STAT_LABELS = {
  to_sep: 'mass on [SEP]',
  to_cls: 'mass on [CLS]',
  to_prev: 'mass on prev token',
  to_next: 'mass on next token',
};

/* ---------------- element cache ---------------- */

const $ = (id) => document.getElementById(id);
const els = {
  modelSelect: $('model-select'), mainText: $('main-text'),
  presetChips: $('preset-chips'), statusChip: $('status-chip'),
  errorBanner: $('error-banner'), errorMsg: $('error-msg'), errorDismiss: $('error-dismiss'),

  tokBody: $('tok-body'), tokEmpty: $('tok-empty'),
  tokCompare: $('tok-compare'), tokCompareView: $('tok-compare-view'),
  tokRaw: $('tok-raw'), tokNormalized: $('tok-normalized'), tokNormalizer: $('tok-normalizer'),
  tokPre: $('tok-pre'), tokWordpieces: $('tok-wordpieces'),
  tokTableBody: $('tok-table-body'), tokCount: $('tok-count'), tokDecoded: $('tok-decoded'),

  encBody: $('enc-body'), encEmpty: $('enc-empty'),
  encLayer: $('enc-layer'), encLayerLabel: $('enc-layer-label'),
  encPerLayer: $('enc-perlayer'), encPerToken: $('enc-pertoken'),
  encFull768: $('enc-full768'), encFullNote: $('enc-full-note'),
  encHeatmap: $('enc-heatmap'), encHeatmapScroll: $('enc-heatmap-scroll'), encHeatmapSub: $('enc-heatmap-sub'),
  encNorms: $('enc-norms'), encNormsLegend: $('enc-norms-legend'),
  encCosine: $('enc-cosine'), encCosineScroll: $('enc-cosine-scroll'), encCosSub: $('enc-cos-sub'),
  encMeanCos: $('enc-meancos'),

  attBody: $('att-body'), attEmpty: $('att-empty'),
  attLayer: $('att-layer'), attLayerLabel: $('att-layer-label'),
  attHead: $('att-head'), attHeadLabel: $('att-head-label'),
  attHeatmap: $('att-heatmap'), attHeatmapScroll: $('att-heatmap-scroll'), attHeatmapSub: $('att-heatmap-sub'),
  attArcs: $('att-arcs'),
  attStat: $('att-stat'), attHeadgrid: $('att-headgrid'),

  embPool: $('emb-pool'), embA: $('emb-a'), embB: $('emb-b'),
  embCosReadout: $('emb-cos-readout'), embCosSub: $('emb-cos-sub'),
  embBank: $('emb-bank'),
  embHeatmap: $('emb-heatmap'), embHeatmapScroll: $('emb-heatmap-scroll'), embHeatSub: $('emb-heat-sub'),
  embStretch: $('emb-stretch'),
  embPca: $('emb-pca'), embPcaSub: $('emb-pca-sub'), embEmpty: $('emb-empty'),

  probeSelect: $('probe-select'), probeText: $('probe-text'),
  probeBody: $('probe-body'), probeEmpty: $('probe-empty'),
  probeHsCanvas: $('probe-hs-canvas'), probeHsShape: $('probe-hs-shape'),
  probeClsCanvas: $('probe-cls-canvas'), probeClsShape: $('probe-cls-shape'),
  probeWCanvas: $('probe-w-canvas'), probeWShape: $('probe-w-shape'),
  probeLogitShape: $('probe-logit-shape'), probeLogits: $('probe-logits'),
  probeSoftmax: $('probe-softmax'), probeVerdict: $('probe-verdict'),
};

const probeMounts = {
  hsCanvas: els.probeHsCanvas, hsShape: els.probeHsShape,
  clsCanvas: els.probeClsCanvas, clsShape: els.probeClsShape,
  wCanvas: els.probeWCanvas, wShape: els.probeWShape,
  logitShape: els.probeLogitShape,
  logits: els.probeLogits, softmax: els.probeSoftmax, verdict: els.probeVerdict,
};

const arcView = makeArcView(els.attArcs);

/* ---------------- tiny utils ---------------- */

function debounce(fn, ms) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function trunc(s, n = 48) {
  s = String(s);
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

function squareCells(wrap, n, maxSide = 34) {
  const availW = Math.max(140, wrap.clientWidth - 66 - 6);
  return clamp(Math.floor(Math.min(availW / n, 470 / n)), 12, maxSide);
}

function maxAbsMatrix(m) {
  let mx = 1e-6;
  for (const row of m) for (const v of row) { const a = Math.abs(v); if (a > mx) mx = a; }
  return mx;
}

/* ---------------- status chip / errors ---------------- */

let inflight = 0;
let lastStatus = '—';
function setChip(text) {
  lastStatus = text;
  if (inflight === 0) els.statusChip.textContent = text;
}
function busy(on) {
  inflight = Math.max(0, inflight + (on ? 1 : -1));
  els.statusChip.classList.toggle('loading', inflight > 0);
  els.statusChip.textContent = inflight > 0 ? '…' : lastStatus;
}

const seq = {};
// Runs task(); ignores stale completions; never throws.
async function run(key, task) {
  const my = (seq[key] = (seq[key] || 0) + 1);
  busy(true);
  try {
    const out = await task();
    if (my === seq[key]) return { ok: true, ...out };
  } catch (err) {
    if (my === seq[key]) return { ok: false, err };
  } finally {
    busy(false);
  }
  return { ok: false, stale: true };
}

function showError(source, msg) {
  state.errorSource = source;
  els.errorBanner.hidden = false;
  els.errorMsg.textContent = msg;
}
function clearError(source) {
  if (state.errorSource === source) {
    els.errorBanner.hidden = true;
    state.errorSource = null;
  }
}

/* ---------------- meta ---------------- */

function modelLabel(id) {
  const m = ((state.meta && state.meta.models) || []).find((x) => x.id === id);
  return m ? (m.label || m.id) : id;
}
function otherModelId() {
  const models = (state.meta && state.meta.models) || [];
  const other = models.find((m) => m.id !== state.model);
  return other ? other.id : (state.model === 'distilbert-base-uncased' ? 'bert-base-cased' : 'distilbert-base-uncased');
}

async function loadMeta() {
  const r = await run('meta', () => api.meta());
  if (r.stale) return false;
  if (!r.ok) {
    showError('meta', `${r.err.message} — start the backend (tools/mock_server.py on :8304 or uvicorn app:app on :8303) and reload.`);
    return false;
  }
  state.meta = r.data;
  clearError('meta');

  els.modelSelect.innerHTML = '';
  for (const m of r.data.models || []) {
    const o = document.createElement('option');
    o.value = m.id;
    o.textContent = m.label || m.id;
    els.modelSelect.appendChild(o);
  }
  const models = r.data.models || [];
  state.model = models.some((m) => m.id === r.data.default_model)
    ? r.data.default_model : (models[0] && models[0].id);
  if (state.model) els.modelSelect.value = state.model;

  els.probeSelect.innerHTML = '';
  for (const p of r.data.probes || []) {
    const o = document.createElement('option');
    o.value = p.id;
    o.textContent = p.label || p.id;
    if (!p.available) { o.disabled = true; o.textContent += ' — unavailable'; }
    els.probeSelect.appendChild(o);
  }
  const firstAvail = (r.data.probes || []).find((p) => p.available);
  if (firstAvail) { state.probeId = firstAvail.id; els.probeSelect.value = state.probeId; }
  return true;
}

/* ---------------- fetchers ---------------- */

async function refreshTokenize() {
  const r = await run('tokenize', () => api.tokenize(state.model, state.text));
  if (r.stale) return;
  if (!r.ok) {
    state.tokenize = null;
    showError('tokenize', r.err.message);
    renderIfActive(['tokenize']);
    return;
  }
  state.tokenize = r.data;
  clearError('tokenize');
  setChip(`${(r.data.tokens || []).length} tokens · ${r.clientMs} ms`);
  renderIfActive(['tokenize']);
  if (els.tokCompare.checked) refreshCompare();
}

async function refreshCompare() {
  const otherId = otherModelId();
  if (!otherId) return;
  const r = await run('compare', () => api.tokenize(otherId, state.text));
  if (r.stale) return;
  if (!r.ok) { showError('compare', r.err.message); return; }
  state.compare = { model: otherId, text: state.text, data: r.data };
  clearError('compare');
  if (state.tab === 'tokenize' && els.tokCompare.checked) renderCompareNow();
}

function configureEncSliders(f) {
  const maxL = Math.max(0, (f.num_hidden_states || 1) - 1);
  els.encLayer.max = maxL;
  if (state.layerIdx > maxL) state.layerIdx = maxL;
  els.encLayer.value = state.layerIdx;
  updateEncLabels();
}

function configureAttSliders(f) {
  const a = f.attention || {};
  const ml = Math.max(0, (a.num_layers || 1) - 1);
  const mh = Math.max(0, (a.num_heads || 1) - 1);
  els.attLayer.max = ml;
  els.attHead.max = mh;
  if (state.attLayer > ml) state.attLayer = ml;
  if (state.attHead > mh) state.attHead = mh;
  els.attLayer.value = state.attLayer;
  els.attHead.value = state.attHead;
  updateAttLabels();
}

async function doForward() {
  const r = await run('forward', () => api.forward(state.model, state.text));
  if (r.stale) return;
  if (!r.ok) {
    state.forward = null;
    state.full768 = null;
    showError('forward', r.err.message);
    renderIfActive(['encoder', 'attention']);
    return;
  }
  const f = r.data;
  state.forward = f;
  state.full768 = null;
  let sm = 1e-6;
  if (Array.isArray(f.stripes)) {
    for (const layer of f.stripes) for (const row of layer) for (const v of row) {
      const a = Math.abs(v);
      if (a > sm) sm = a;
    }
  }
  state.stripesMax = sm;
  configureEncSliders(f);
  configureAttSliders(f);
  clearError('forward');
  setChip(`${(f.tokens || []).length} tokens · ${f.elapsed_ms} ms`);
  renderIfActive(['encoder', 'attention']);
  if (state.fullMode) fetchLayer();
}

function fullLayerValues() {
  const fl = state.full768;
  if (!fl || !state.fullMode) return null;
  if (fl.model !== state.model || fl.text !== state.text || fl.layer !== state.layerIdx) return null;
  return fl.values;
}

async function fetchLayer() {
  if (!state.fullMode || !state.forward) return;
  const layer = state.layerIdx;
  const r = await run('layer', () => api.layer(state.model, state.text, layer));
  if (r.stale) return;
  if (!r.ok) { showError('layer', r.err.message); updateFullUi(); return; }
  state.full768 = { model: state.model, text: state.text, layer, values: r.data.values };
  clearError('layer');
  if (state.tab === 'encoder') drawEncoderHeatmap();
  updateFullUi();
}
const fetchLayerDebounced = debounce(fetchLayer, 250);

function updateFullUi() {
  els.encFull768.classList.toggle('active', state.fullMode);
  els.encFull768.textContent = state.fullMode ? 'Back to 192 stripes' : 'Full 768 dims';
  const active = fullLayerValues();
  els.encFullNote.hidden = !active;
  if (active && state.full768) {
    const dims = state.full768.values[0] ? state.full768.values[0].length : 768;
    els.encFullNote.textContent = `/api/layer · full ${dims} dims @ layer ${state.layerIdx}`;
  }
}

function bankLines() {
  return els.embBank.value.split('\n').map((s) => s.trim()).filter(Boolean);
}

async function fetchAB() {
  const A = els.embA.value.trim();
  const B = els.embB.value.trim();
  if (!A || !B) { state.simAB = null; renderEmbResults(); return; }
  const r = await run('simab', () => api.similarity(state.model, [A, B]));
  if (r.stale) return;
  if (!r.ok) { state.simAB = null; showError('simab', r.err.message); renderEmbResults(); return; }
  state.simAB = { a: A, b: B, data: r.data };
  clearError('simab');
  setChip(`2 sentences · ${r.data.elapsed_ms} ms`);
  renderEmbResults();
}

async function fetchBank() {
  const lines = bankLines();
  if (!lines.length) { state.simBank = null; renderEmbResults(); return; }
  const r = await run('simbank', () => api.similarity(state.model, lines));
  if (r.stale) return;
  if (!r.ok) { state.simBank = null; showError('simbank', r.err.message); renderEmbResults(); return; }
  state.simBank = { sentences: lines, data: r.data };
  clearError('simbank');
  setChip(`${lines.length} sentences · ${r.data.elapsed_ms} ms`);
  renderEmbResults();
}

async function refreshProbe() {
  if (!state.probeId) return;
  const probeId = state.probeId;
  const text = els.probeText.value;
  const r = await run('probe', () => api.probe(probeId, text));
  if (r.stale) return;
  if (!r.ok) { state.probeData = null; showError('probe', r.err.message); renderProbeTab(); return; }
  state.probeData = r.data;
  clearError('probe');
  setChip(`${(r.data.tokens || []).length} tokens · ${r.data.elapsed_ms} ms`);
  renderProbeTab();
}
const refreshProbeDebounced = debounce(refreshProbe, 400);

/* ---------------- tab dispatch ---------------- */

function renderIfActive(names) {
  if (names.includes(state.tab)) renderTab(state.tab);
}

function renderTab(name) {
  if (name === 'tokenize') renderTokenizeTab();
  else if (name === 'encoder') renderEncoderTab();
  else if (name === 'attention') renderAttentionTab();
  else if (name === 'embeddings') { syncPoolButtons(); renderEmbResults(); }
  else if (name === 'probe') renderProbeTab();
}

function activateTab(name) {
  state.tab = name;
  document.querySelectorAll('.tab-button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('active', p.id === `tab-${name}`));
  hideTip();
  if (name === 'embeddings') {
    if (!state.visited.embeddings) state.visited.embeddings = true;
    if (!state.simAB) fetchAB();
    if (!state.simBank) fetchBank();
  }
  if (name === 'probe') {
    if (!state.visited.probe) state.visited.probe = true;
    if (!state.probeData) refreshProbe();
  }
  renderTab(name);
}

/* ---------------- tab 1: tokenize ---------------- */

function renderTokenizeTab() {
  const d = state.tokenize;
  els.tokEmpty.hidden = !!d;
  els.tokBody.hidden = !d;
  if (!d) {
    els.tokEmpty.textContent = 'No tokenize data yet — type a sentence above.';
    return;
  }
  renderPipeline({
    raw: els.tokRaw, normalized: els.tokNormalized, normalizer: els.tokNormalizer,
    pre: els.tokPre, wordpieces: els.tokWordpieces,
  }, d);
  renderTokenTable(els.tokTableBody, d);
  els.tokCount.textContent = `${(d.tokens || []).length} tokens`;
  els.tokDecoded.textContent = d.decoded == null ? '' : `"${d.decoded}"`;
  if (els.tokCompare.checked) renderCompareNow();
}

function renderCompareNow() {
  const cur = state.tokenize;
  if (!cur) return;
  const otherId = otherModelId();
  els.tokCompareView.hidden = false;
  const cmp = state.compare;
  const otherData = (cmp && cmp.model === otherId && cmp.text === state.text) ? cmp.data : null;
  renderCompareView(els.tokCompareView, [
    { label: modelLabel(state.model), data: cur },
    { label: modelLabel(otherId), data: otherData },
  ]);
}

/* ---------------- tab 2: encoder ---------------- */

function updateEncLabels() {
  els.encLayerLabel.textContent = `layer ${state.layerIdx} / ${els.encLayer.max}`;
}

function encoderMatrix() {
  const f = state.forward;
  if (!f) return null;
  const fl = fullLayerValues();
  if (fl && fl.length) return { values: fl, rows: fl[0].length, full: true };
  const s = (f.stripes || [])[state.layerIdx];
  return (s && s.length) ? { values: s, rows: s[0].length, full: false } : null;
}

function encoderColor(matrix) {
  const perToken = els.encPerToken.checked;
  const perLayer = els.encPerLayer.checked;
  if (perToken) {
    // rows are tokens: normalize each token to its own max activation
    const rowMax = matrix.map((row) => {
      let mx = 1e-6;
      for (const v of row) { const a = Math.abs(v || 0); if (a > mx) mx = a; }
      return mx;
    });
    return (v, i) => diverging(v, rowMax[i]);
  }
  const vmax = state.fullMode ? maxAbsMatrix(matrix)
    : (perLayer ? maxAbsMatrix(matrix) : (state.stripesMax || maxAbsMatrix(matrix)));
  return (v) => diverging(v, vmax);
}

function drawEncoderHeatmap() {
  const f = state.forward;
  const m = encoderMatrix();
  if (!f || !m) return;
  // rows = tokens, columns = dims. Show every k-th dim so cells stay readable
  // on a projector — the hover tooltip always reports the true dim index.
  const k = Math.max(1, Math.ceil(m.values[0].length / 48));
  const shown = k === 1 ? m.values : m.values.map((row) => row.filter((_, j) => j % k === 0));
  drawHeatmap(els.encHeatmap, els.encHeatmapScroll, {
    matrix: shown,
    color: encoderColor(m.values),
    rowLabels: f.tokens,
    rowSpecial: f.special_mask,
    colTickEvery: 8,
    colTickLabel: (j) => j * k,
    cellH: clamp(Math.floor(360 / Math.max(1, shown.length)), 8, 26),
  });
  els.encHeatmap._hit.stride = k;
  els.encHeatmapSub.textContent =
    `layer ${state.layerIdx}/${(f.num_hidden_states || 1) - 1} · ${m.values[0].length} dims${k > 1 ? `, every ${k}th shown` : ''}${m.full ? ' (full, /api/layer)' : ''}`;
}

function drawNorms() {
  const f = state.forward;
  if (!f || !f.norms || !f.norms.length) return;
  const n = (f.tokens || []).length;
  const series = (f.tokens || []).map((t, i) => ({
    values: f.norms.map((row) => row[i]),
    color: tokenColor(i, n),
    dash: !!(f.special_mask && f.special_mask[i]),
    label: t,
  }));
  let maxN = 1;
  for (const row of f.norms) for (const v of row) if (v > maxN) maxN = v;
  drawLineChart(els.encNorms, els.encNorms.parentElement, {
    series,
    yMin: 0,
    yMax: maxN * 1.12,
    xLabels: f.norms.map((_, k) => k),
    xTitle: 'layer',
    cursorIndex: state.layerIdx,
    height: 250,
    highlight: state.normHl,
    yFmt: (v) => v.toFixed(1),
  });
}

function drawNormsLegend() {
  const f = state.forward;
  if (!f) return;
  renderLegend(els.encNormsLegend, f.tokens, f.special_mask || [], tokenColor, (i) => {
    state.normHl = i;
    drawNorms();
  });
}

function drawCosine() {
  const f = state.forward;
  if (!f) return;
  const m = (f.cosine || [])[state.layerIdx];
  if (!m) return;
  const side = squareCells(els.encCosineScroll, m.length);
  drawHeatmap(els.encCosine, els.encCosineScroll, {
    matrix: m,
    color: (v) => viridis(v),
    colLabels: f.tokens,
    rowLabels: f.tokens,
    colSpecial: f.special_mask,
    rowSpecial: f.special_mask,
    cellW: side,
    cellH: side,
    targetH: 470,
  });
  // off-diagonal range: late layers squeeze into a narrow band (anisotropy)
  let lo = 1, hi = -1;
  for (let i = 0; i < m.length; i++) for (let j = 0; j < m.length; j++) {
    if (i === j) continue;
    if (m[i][j] < lo) lo = m[i][j];
    if (m[i][j] > hi) hi = m[i][j];
  }
  els.encCosSub.textContent =
    `layer ${state.layerIdx}/${(f.num_hidden_states || 1) - 1} · cos ${lo.toFixed(2)}–${hi.toFixed(2)}`;
}

function drawMeanCos() {
  const f = state.forward;
  if (!f || !f.mean_cosine || !f.mean_cosine.length) return;
  let mx = 0.1;
  for (const v of f.mean_cosine) if (v > mx) mx = v;
  drawLineChart(els.encMeanCos, els.encMeanCos.parentElement, {
    series: [{ values: f.mean_cosine, color: '#ffb454', width: 2 }],
    yMin: 0,
    yMax: mx * 1.3,
    xLabels: f.mean_cosine.map((_, k) => k),
    xTitle: 'layer',
    cursorIndex: state.layerIdx,
    cursorDot: true,
    cursorLabel: (v) => v.toFixed(3),
    height: 230,
    yFmt: (v) => v.toFixed(2),
  });
}

function renderEncoderTab() {
  const f = state.forward;
  els.encEmpty.hidden = !!f;
  els.encBody.hidden = !f;
  updateFullUi();
  if (!f) return;
  updateEncLabels();
  drawEncoderHeatmap();
  drawNorms();
  drawNormsLegend();
  drawCosine();
  drawMeanCos();
}

/* ---------------- tab 3: attention ---------------- */

function updateAttLabels() {
  els.attLayerLabel.textContent = `layer ${state.attLayer} / ${els.attLayer.max}`;
  els.attHeadLabel.textContent = `head ${state.attHead} / ${els.attHead.max}`;
  els.attHeatmapSub.textContent = `L${state.attLayer} · H${state.attHead}`;
}

function attWeights() {
  const f = state.forward;
  if (!f || !f.attention || !f.attention.weights) return null;
  const { num_layers, weights } = f.attention;
  const l = clamp(state.attLayer, 0, num_layers - 1);
  const h = clamp(state.attHead, 0, f.attention.num_heads - 1);
  return (weights[l] && weights[l][h]) ? weights[l][h] : null;
}

function drawAttHeatmap() {
  const f = state.forward;
  const w = attWeights();
  if (!f || !w) return;
  const wmax = maxAbsMatrix(w);
  const side = squareCells(els.attHeatmapScroll, w.length);
  drawHeatmap(els.attHeatmap, els.attHeatmapScroll, {
    matrix: w,
    color: (v) => viridis(v / wmax),
    colLabels: f.tokens,
    rowLabels: f.tokens,
    colSpecial: f.special_mask,
    rowSpecial: f.special_mask,
    cellW: side,
    cellH: side,
    targetH: 470,
  });
  els.attHeatmapSub.textContent = `L${state.attLayer} · H${state.attHead}`;
}

function renderHeadGridNow() {
  const f = state.forward;
  if (!f || !f.head_stats || !f.attention) return;
  renderHeadGrid(els.attHeadgrid, {
    stats: f.head_stats,
    numLayers: f.attention.num_layers,
    numHeads: f.attention.num_heads,
    stat: state.attStat,
    current: { layer: state.attLayer, head: state.attHead },
    onPick: (l, h) => {
      state.attLayer = l;
      state.attHead = h;
      els.attLayer.value = l;
      els.attHead.value = h;
      updateAttLabels();
      drawAttHeatmap();
      arcView.setRow(attWeights());
      renderHeadGridNow();
    },
  });
}

function renderAttentionTab() {
  const f = state.forward;
  els.attEmpty.hidden = !!f;
  els.attBody.hidden = !f;
  if (!f) return;
  updateAttLabels();
  drawAttHeatmap();
  arcView.setData({ tokens: f.tokens || [], specialMask: f.special_mask || [], subwordMask: f.subword_mask || [] });
  arcView.setRow(attWeights());
  renderHeadGridNow();
}

/* ---------------- tab 4: embeddings ---------------- */

function syncPoolButtons() {
  els.embPool.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.pool === state.embPool));
}

function renderEmbResults() {
  const pool = state.embPool;
  const poolName = pool === 'cls' ? '[CLS]' : 'mean';
  const cosKey = pool === 'cls' ? 'cos_cls' : 'cos_mean';
  const pcaKey = pool === 'cls' ? 'pca_cls' : 'pca_mean';

  if (state.simAB && state.simAB.data[cosKey]) {
    const m = state.simAB.data[cosKey];
    const v = (m && m[0] && typeof m[0][1] === 'number') ? m[0][1] : null;
    els.embCosReadout.textContent = v == null ? '—' : v.toFixed(3);
    els.embCosSub.textContent = `cosine(A, B) · ${poolName} pooling`;
  } else {
    els.embCosReadout.textContent = '—';
    els.embCosSub.textContent = state.errorSource === 'simab'
      ? 'similarity failed — see error above'
      : 'cosine(A, B) — type both sentences';
  }

  if (state.tab !== 'embeddings') return;

  const sb = state.simBank;
  if (!sb || !sb.data[cosKey]) {
    els.embEmpty.hidden = false;
    els.embEmpty.textContent = state.errorSource === 'simbank'
      ? 'similarity failed — see error above'
      : 'no sentences in the bank — add one per line';
    drawPlaceholder(els.embHeatmap, els.embHeatmapScroll, 'no sentences');
    drawPlaceholder(els.embPca, els.embPca.parentElement, 'no sentences');
    els.embHeatSub.textContent = '';
    els.embPcaSub.textContent = '';
    return;
  }
  els.embEmpty.hidden = true;

  const m = sb.data[cosKey];
  const side = squareCells(els.embHeatmapScroll, m.length);
  let lo = 1, hi = -1;
  for (let i = 0; i < m.length; i++) for (let j = 0; j < m.length; j++) {
    if (i === j) continue;
    if (m[i][j] < lo) lo = m[i][j];
    if (m[i][j] > hi) hi = m[i][j];
  }
  const stretch = state.embStretch && hi > lo;
  drawHeatmap(els.embHeatmap, els.embHeatmapScroll, {
    matrix: m,
    color: stretch ? (v) => viridis((v - lo) / (hi - lo)) : (v) => viridis(v),
    colLabels: m.map((_, i) => i),
    rowLabels: m.map((_, i) => i),
    cellW: side,
    cellH: side,
    leftW: 40,
    topH: 46,
    targetH: 470,
  });
  els.embHeatSub.textContent =
    `${m.length}×${m.length} · ${poolName} pooling · cos ${lo.toFixed(2)}–${hi.toFixed(2)}` +
    (stretch ? ' · stretched' : '');

  const pca = sb.data[pcaKey] || { x: [], y: [] };
  const pts = (pca.x || []).map((x, i) => ({
    x,
    y: (pca.y || [])[i] || 0,
    color: viridis(i / Math.max(1, (pca.x || []).length - 1)),
  }));
  drawScatter(els.embPca, els.embPca.parentElement, pts);
  els.embPcaSub.textContent = `${pts.length} sentences · ${poolName} pooling`;
}

/* ---------------- tab 5: probe ---------------- */

function renderProbeTab() {
  const d = state.probeData;
  els.probeEmpty.hidden = !!d;
  els.probeBody.hidden = !d;
  if (!d || state.tab !== 'probe') return;
  renderProbePipeline(probeMounts, d);
}

/* ---------------- events ---------------- */

function refreshForText() {
  state.text = els.mainText.value;
  refreshTokenize();
  doForward();
  if (!state.probeDirty) {
    els.probeText.value = state.text;
    if (state.visited.probe) refreshProbe();
  }
}
const scheduleTextRefresh = debounce(refreshForText, 400);

function wireEvents() {
  // top bar
  els.mainText.addEventListener('input', scheduleTextRefresh);
  els.modelSelect.addEventListener('change', () => {
    state.model = els.modelSelect.value;
    state.full768 = null;
    state.compare = null;
    refreshTokenize();
    doForward();
    if (state.visited.embeddings) { fetchAB(); fetchBank(); }
  });
  els.errorDismiss.addEventListener('click', () => {
    els.errorBanner.hidden = true;
    state.errorSource = null;
  });

  // tabs
  document.querySelectorAll('.tab-button').forEach((btn) => {
    btn.addEventListener('click', () => activateTab(btn.dataset.tab));
  });

  // tab 1
  els.tokCompare.addEventListener('change', () => {
    els.tokCompareView.hidden = !els.tokCompare.checked;
    if (els.tokCompare.checked) {
      refreshCompare();
      renderCompareNow();
    }
  });
  els.tokPre.addEventListener('mouseover', (e) => {
    const c = e.target.closest('.prechip');
    if (!c) return;
    showTip(e.clientX, e.clientY,
      `<span class="tt-mono">${esc(c.dataset.text)}</span><br>offsets [${c.dataset.start}, ${c.dataset.end}) into normalized text`);
  });
  els.tokPre.addEventListener('mouseout', (e) => { if (e.target.closest('.prechip')) hideTip(); });

  // tab 2
  els.encLayer.addEventListener('input', () => {
    state.layerIdx = +els.encLayer.value;
    updateEncLabels();
    if (state.tab === 'encoder') { drawEncoderHeatmap(); drawNorms(); drawCosine(); drawMeanCos(); }
    if (state.fullMode) fetchLayerDebounced();
  });
  els.encPerLayer.addEventListener('change', () => { if (state.tab === 'encoder') drawEncoderHeatmap(); });
  els.encPerToken.addEventListener('change', () => { if (state.tab === 'encoder') drawEncoderHeatmap(); });
  els.encFull768.addEventListener('click', () => {
    state.fullMode = !state.fullMode;
    state.full768 = null;
    if (state.fullMode) fetchLayer();
    else if (state.tab === 'encoder') drawEncoderHeatmap();
    updateFullUi();
  });

  // tab 3
  const attSliderInput = () => {
    state.attLayer = +els.attLayer.value;
    state.attHead = +els.attHead.value;
    updateAttLabels();
    if (state.tab !== 'attention') return;
    drawAttHeatmap();
    arcView.setRow(attWeights());
    renderHeadGridNow();
  };
  els.attLayer.addEventListener('input', attSliderInput);
  els.attHead.addEventListener('input', attSliderInput);
  els.attStat.addEventListener('change', () => {
    state.attStat = els.attStat.value;
    renderHeadGridNow();
  });
  els.attHeadgrid.addEventListener('mouseover', (e) => {
    const cell = e.target.closest('.hcell');
    if (!cell) return;
    const l = cell.dataset.layer, h = cell.dataset.head, v = +cell.dataset.value;
    showTip(e.clientX, e.clientY,
      `<span class="tt-mono">L${l} · H${h}</span> ${STAT_LABELS[state.attStat] || state.attStat} = <span class="tt-mono">${v.toFixed(4)}</span>`);
  });
  els.attHeadgrid.addEventListener('mouseout', (e) => { if (e.target.closest('.hcell')) hideTip(); });

  // canvas hover tooltips
  bindMatrixHover(els.encHeatmap, (i, j, v, hit) => {
    const tok = (hit.rowLabels && hit.rowLabels[i]) != null ? hit.rowLabels[i] : '?';
    const dim = j * (hit.stride || 1);
    return `<span class="tt-mono">${esc(tok)}</span> · dim ${dim} · <span class="tt-mono">${v >= 0 ? '+' : ''}${v.toFixed(3)}</span>`;
  });
  bindMatrixHover(els.encCosine, (i, j, v, hit) => {
    const a = hit.rowLabels[i], b = hit.colLabels[j];
    if (i === j) return `<span class="tt-mono">${esc(a)}</span> · self · cos = ${v.toFixed(4)}`;
    return `cos(<span class="tt-mono">${esc(a)}</span>, <span class="tt-mono">${esc(b)}</span>) = <span class="tt-mono">${v.toFixed(4)}</span>`;
  });
  bindChartHover(els.encNorms, (i, s, v, ser) =>
    `\u201C<span class="tt-mono">${esc(ser.label)}</span>\u201D · layer ${i} · ‖h‖ = <span class="tt-mono">${v.toFixed(2)}</span>`);
  bindChartHover(els.encMeanCos, (i, s, v) =>
    `layer ${i} · mean cosine = <span class="tt-mono">${v.toFixed(4)}</span>`);
  bindMatrixHover(els.attHeatmap, (i, j, v, hit) => {
    const a = hit.rowLabels[i], b = hit.colLabels[j];
    if (i === j) return `<span class="tt-mono">${esc(a)}</span> → itself · w = <span class="tt-mono">${v.toFixed(4)}</span>`;
    return `<span class="tt-mono">${esc(a)}</span> → <span class="tt-mono">${esc(b)}</span> · w = <span class="tt-mono">${v.toFixed(4)}</span><br><span class="dim">row sums to 1</span>`;
  });
  bindMatrixHover(els.embHeatmap, (i, j, v) => {
    const sb = state.simBank;
    if (!sb) return null;
    if (i === j) return `<span class="tt-mono">${i}</span> · cos = ${v.toFixed(4)} (self)<br>${esc(trunc(sb.sentences[i], 70))}`;
    return `<span class="tt-mono">${i} ↔ ${j}</span> · cos = <span class="tt-mono">${v.toFixed(4)}</span><br>${esc(trunc(sb.sentences[i], 70))}<br>${esc(trunc(sb.sentences[j], 70))}`;
  });
  bindScatterHover(els.embPca, (i) => {
    const sb = state.simBank;
    return sb ? `<span class="tt-mono">${i}</span> · ${esc(trunc(sb.sentences[i], 80))}` : null;
  });

  // tab 4
  els.embPool.querySelectorAll('button').forEach((b) => {
    b.addEventListener('click', () => {
      state.embPool = b.dataset.pool;
      syncPoolButtons();
      renderEmbResults();
    });
  });
  const scheduleAB = debounce(fetchAB, 400);
  els.embA.addEventListener('input', scheduleAB);
  els.embB.addEventListener('input', scheduleAB);
  els.embStretch.addEventListener('change', () => {
    state.embStretch = els.embStretch.checked;
    if (state.tab === 'embeddings') renderTab('embeddings');
  });
  const scheduleBank = debounce(fetchBank, 500);
  els.embBank.addEventListener('input', scheduleBank);

  // tab 5
  els.probeSelect.addEventListener('change', () => {
    state.probeId = els.probeSelect.value;
    refreshProbe();
  });
  els.probeText.addEventListener('input', () => {
    state.probeDirty = true;
    refreshProbeDebounced();
  });

  // resize -> redraw everything visible
  window.addEventListener('resize', debounce(() => renderTab(state.tab), 150));
}

/* ---------------- init ---------------- */

function buildPresets() {
  for (const p of PRESETS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = p;
    b.title = p;
    b.addEventListener('click', () => {
      els.mainText.value = p;
      refreshForText();
    });
    els.presetChips.appendChild(b);
  }
}

async function init() {
  buildPresets();
  wireEvents();
  els.statusChip.textContent = '…';
  const ok = await loadMeta();
  if (!ok) { els.statusChip.textContent = 'offline'; return; }
  state.text = els.mainText.value;
  els.probeText.value = state.text;
  refreshTokenize();
  doForward();
}

init();
