// views.js — DOM/SVG view builders (chips, tables, arcs, head grid, probe pipeline).

import { viridis, clamp, esc } from './viz.js';
import { drawMiniStripes, drawDivergingBars, drawMiniHeat } from './viz.js';

export function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

export function svgEl(tag, attrs = {}) {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

/* ---------------- tab 1: tokenize ---------------- */

export function tokenChip(tok, special, subword) {
  return el('span', `chip${special ? ' special' : ''}${subword ? ' subword' : ''}`, tok);
}

export function renderPipeline(m, data) {
  m.raw.textContent = data.text === '' ? '(empty string)' : `\u201C${data.text}\u201D`;
  m.normalized.textContent = data.normalized === '' ? '(empty)' : `\u201C${data.normalized}\u201D`;
  m.normalizer.textContent = data.normalizer || '';

  m.pre.innerHTML = '';
  if (!data.pre_tokens.length) m.pre.appendChild(el('span', 'dim', '(none)'));
  for (const p of data.pre_tokens) {
    const c = el('span', 'chip prechip', p.text);
    c.dataset.start = p.start;
    c.dataset.end = p.end;
    c.dataset.text = p.text;
    m.pre.appendChild(c);
  }

  m.wordpieces.innerHTML = '';
  data.tokens.forEach((t, i) => {
    m.wordpieces.appendChild(tokenChip(t, data.special_mask[i], data.subword_mask[i]));
  });
}

export function renderTokenTable(tbody, data) {
  tbody.innerHTML = '';
  data.tokens.forEach((t, i) => {
    const tr = document.createElement('tr');
    tr.appendChild(el('td', 'dim', String(i)));
    const tokTd = document.createElement('td');
    tokTd.appendChild(el('span', `tok-cell${data.special_mask[i] ? ' special' : ''}${data.subword_mask[i] ? ' subword' : ''}`, t));
    tr.appendChild(tokTd);
    tr.appendChild(el('td', null, String(data.input_ids[i])));
    tr.appendChild(el('td', 'dim', String(data.attention_mask[i])));
    const o = data.offsets[i] || [0, 0];
    tr.appendChild(el('td', 'dim', `${o[0]}:${o[1]}`));
    tbody.appendChild(tr);
  });
}

// cols: [{label, data} | {label, pending}]
export function renderCompareView(container, cols) {
  container.innerHTML = '';
  for (const col of cols) {
    const div = el('div', 'compare-col');
    div.appendChild(el('h4', null, col.label));
    const chips = el('div', 'chips');
    if (!col.data) {
      chips.appendChild(el('span', 'dim', 'loading…'));
    } else {
      col.data.tokens.forEach((t, i) => {
        chips.appendChild(tokenChip(t, col.data.special_mask[i], col.data.subword_mask[i]));
      });
      div.appendChild(el('span', 'dim mono', `${col.data.tokens.length} tokens — ${col.data.normalizer || ''}`));
    }
    div.appendChild(chips);
    container.appendChild(div);
  }
}

/* ---------------- legend (tab 2 norms) ---------------- */

export function renderLegend(container, tokens, specialMask, colorFn, onHover) {
  container.innerHTML = '';
  tokens.forEach((t, i) => {
    const c = el('span', `legend-chip${specialMask[i] ? ' special' : ''}`, t);
    c.style.borderLeftColor = colorFn(i, tokens.length);
    if (onHover) {
      c.addEventListener('mouseenter', () => onHover(i));
      c.addEventListener('mouseleave', () => onHover(null));
    }
    container.appendChild(c);
  });
}

/* ---------------- tab 3: attention arcs ----------------
   Chips are built once per token set / width; arcs re-render on hover/click
   without rebuilding chips, so mouseleave keeps working. */

export function makeArcView(root) {
  const SVG_H = 312;
  const CHIP_H = 30;
  const GAP = 8;
  const ARC_MAX = 230;

  let key = null;
  let tokens = [], special = [], subword = [];
  let row = null;
  let pinned = null, hovered = null;
  let svg = null, arcsG = null, hint = null, queryLabel = null;
  let chipData = [];
  let chipsY = 0;
  let lastWrapW = -1;

  function setData(d) {
    const k = d.tokens.join('\u0001');
    const wrapW = (root.parentElement ? root.parentElement.clientWidth : 0) || 900;
    if (k === key && wrapW === lastWrapW && svg) return;
    key = k;
    lastWrapW = wrapW;
    tokens = d.tokens;
    special = d.specialMask || [];
    subword = d.subwordMask || [];
    const prevPinned = pinned;
    build(wrapW, prevPinned);
  }

  function build(wrapW, prevPinned) {
    root.innerHTML = '';
    const widths = tokens.map((t) => clamp(t.length * 7.2 + 18, 38, 110));
    const total = widths.reduce((a, b) => a + b, 0) + GAP * Math.max(0, tokens.length - 1);
    const W = Math.max(wrapW, total + 20);
    svg = svgEl('svg', { width: W, height: SVG_H, class: 'arcs-svg' });
    arcsG = svgEl('g');
    svg.appendChild(arcsG);

    hint = svgEl('text', { x: W / 2, y: 24, 'text-anchor': 'middle', class: 'arc-hint' });
    hint.textContent = 'hover or click a token chip — arcs show where it looks';
    queryLabel = svgEl('text', { x: 10, y: 24, class: 'arc-query' });
    svg.appendChild(hint);
    svg.appendChild(queryLabel);

    chipsY = SVG_H - CHIP_H - 12;
    let x = (W - total) / 2;
    chipData = tokens.map((t, i) => {
      const w = widths[i];
      const g = svgEl('g', {
        class: `arc-chip${special[i] ? ' special' : ''}${subword[i] ? ' subword' : ''}`,
      });
      g.style.cursor = 'pointer';
      g.appendChild(svgEl('rect', { x, y: chipsY, width: w, height: CHIP_H, rx: 7 }));
      const maxChars = Math.max(3, Math.floor((w - 14) / 7));
      const label = t.length > maxChars ? `${t.slice(0, maxChars - 1)}…` : t;
      const txt = svgEl('text', { x: x + w / 2, y: chipsY + CHIP_H / 2 + 4, 'text-anchor': 'middle' });
      txt.textContent = label;
      g.appendChild(txt);
      g.addEventListener('mouseenter', () => { hovered = i; update(); });
      g.addEventListener('mouseleave', () => { hovered = null; update(); });
      g.addEventListener('click', () => { pinned = pinned === i ? null : i; update(); });
      svg.appendChild(g);
      const cd = { x, w, el: g };
      x += w + GAP;
      return cd;
    });

    root.appendChild(svg);
    pinned = prevPinned != null && prevPinned < tokens.length ? prevPinned
      : (tokens.length > 2 ? 1 : 0);
    hovered = null;
    update();
  }

  function update() {
    if (!svg) return;
    const s = hovered != null ? hovered : pinned;
    hint.style.display = s == null ? '' : 'none';
    queryLabel.textContent = s == null ? '' : `query: "${tokens[s]}"`;
    arcsG.innerHTML = '';
    chipData.forEach((cd) => cd.el.classList.remove('selected'));

    if (s == null || !row || !row[s]) {
      chipData.forEach((cd) => cd.el.setAttribute('opacity', 1));
      return;
    }
    const entries = row[s]
      .map((w, j) => ({ w, j }))
      .filter((e) => e.w >= 0.02)
      .sort((a, b) => b.w - a.w)
      .slice(0, 8);
    const targets = new Set(entries.map((e) => e.j));
    targets.add(s);
    chipData.forEach((cd, i) => cd.el.setAttribute('opacity', targets.has(i) ? 1 : 0.25));
    chipData[s].el.classList.add('selected');

    entries.forEach((e, idx) => {
      const a = chipData[s];
      const b = chipData[e.j];
      const x1 = a.x + a.w / 2;
      const x2 = b.x + b.w / 2;
      const y = chipsY - 2;
      let d, lx, ly;
      if (e.j === s) {
        const h = 36;
        d = `M ${x1 - 9} ${y} C ${x1 - 11} ${y - h - 14}, ${x1 + 11} ${y - h - 14}, ${x1 + 9} ${y}`;
        lx = x1;
        ly = y - h - 8;
      } else {
        const h = clamp(26 + Math.abs(x2 - x1) * 0.45, 30, ARC_MAX);
        d = `M ${x1} ${y} Q ${(x1 + x2) / 2} ${y - h}, ${x2} ${y}`;
        lx = (x1 + x2) / 2;
        ly = y - h + 16;
      }
      arcsG.appendChild(svgEl('path', {
        d,
        fill: 'none',
        // clamp the color domain too: plain viridis(w) is nearly black-on-dark for weak arcs
        stroke: viridis(0.3 + 0.7 * e.w),
        'stroke-width': (2 + 9 * e.w).toFixed(2),
        'stroke-opacity': (0.55 + 0.45 * e.w).toFixed(3),
        'stroke-linecap': 'round',
      }));
      if (idx < 3) {
        const t = svgEl('text', { x: lx, y: ly, 'text-anchor': 'middle', class: 'arc-weight' });
        t.textContent = e.w.toFixed(2);
        arcsG.appendChild(t);
      }
    });
  }

  function setRow(w) { row = w; update(); }

  return { setData, setRow };
}

/* ---------------- tab 3: head grid ---------------- */

export function renderHeadGrid(container, opts) {
  const { stats, numLayers, numHeads, stat, current, onPick } = opts;
  container.innerHTML = '';
  const statMax = Math.max(1e-6, ...stats.map((s) => (s && s[stat]) || 0));
  const grid = el('div', 'headgrid');
  grid.style.gridTemplateColumns = `44px repeat(${numHeads}, minmax(20px, 1fr))`;
  grid.appendChild(el('div', 'hlabel'));
  for (let h = 0; h < numHeads; h++) grid.appendChild(el('div', 'hlabel', `H${h}`));
  for (let l = 0; l < numLayers; l++) {
    grid.appendChild(el('div', 'hlabel', `L${l}`));
    for (let h = 0; h < numHeads; h++) {
      const s = stats[l * numHeads + h];
      const v = s ? (s[stat] ?? 0) : 0;
      const cell = el('div', 'hcell');
      cell.style.background = viridis(v / statMax);
      cell.dataset.layer = l;
      cell.dataset.head = h;
      cell.dataset.value = v;
      if (current && current.layer === l && current.head === h) cell.classList.add('current');
      cell.addEventListener('click', () => onPick(l, h));
      grid.appendChild(cell);
    }
  }
  container.appendChild(grid);
}

/* ---------------- tab 5: probe pipeline ---------------- */

export function renderProbePipeline(m, data) {
  m.hsShape.textContent = `[${(data.hidden_shape || []).join(', ')}]`;
  m.clsShape.textContent = '(768)';
  m.wShape.textContent = `[${(data.head_shape || []).join(' \u00d7 ')}]`;
  m.logitShape.textContent = `(${(data.labels || []).length})`;

  drawMiniStripes(m.hsCanvas, (data.tokens || []).length, data.cls_preview);
  drawDivergingBars(m.clsCanvas, data.cls_preview || []);
  drawMiniHeat(m.wCanvas, data.w_preview || [], (data.labels || []).map((s) => String(s).slice(0, 7)));

  // logits: signed bars around a center line
  m.logits.innerHTML = '';
  const labels = data.labels || [];
  const logits = data.logits || [];
  const maxAbs = Math.max(0.001, ...logits.map((v) => Math.abs(v)));
  labels.forEach((lab, i) => {
    const v = logits[i] || 0;
    const row = el('div', 'brow');
    row.appendChild(el('span', 'blabel', lab));
    const track = el('div', 'btrack');
    const bar = el('span', 'bbar');
    const w = (Math.abs(v) / maxAbs) * 48;
    bar.style.width = `${Math.max(w, 0.8).toFixed(2)}%`;
    if (v >= 0) { bar.style.left = '50%'; bar.style.background = '#ff6e6e'; }
    else { bar.style.right = '50%'; bar.style.background = '#6eaaff'; }
    track.appendChild(bar);
    row.appendChild(track);
    row.appendChild(el('span', 'bvalue', (v || 0).toFixed(3)));
    m.logits.appendChild(row);
  });

  // softmax: 0..1 bars, top class highlighted
  m.softmax.innerHTML = '';
  const probs = data.probs || [];
  labels.forEach((lab, i) => {
    const p = probs[i] || 0;
    const top = lab === data.top_label;
    const row = el('div', `brow${top ? ' top' : ''}`);
    row.appendChild(el('span', 'blabel', lab));
    const track = el('div', 'btrack');
    const bar = el('span', 'bbar');
    bar.style.left = '0';
    bar.style.width = `${(Math.max(p, 0.004) * 100).toFixed(2)}%`;
    bar.style.background = top ? '#ffb454' : '#3d7bd6';
    track.appendChild(bar);
    row.appendChild(track);
    row.appendChild(el('span', 'bvalue', `${(p * 100).toFixed(1)}%`));
    m.softmax.appendChild(row);
  });

  // verdict
  m.verdict.innerHTML = '';
  m.verdict.appendChild(el('span', 'dim', '\u2192'));
  m.verdict.appendChild(el('span', 'verdict', data.top_label || '—'));
  m.verdict.appendChild(el('span', 'prob mono', `${((data.top_prob || 0) * 100).toFixed(1)}%`));
}
