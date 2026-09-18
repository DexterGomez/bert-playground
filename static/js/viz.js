// viz.js — color scales, tooltip, canvas helpers (devicePixelRatio-crisp), generic chart drawers.

export const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';
export const SANS = 'system-ui, -apple-system, "Segoe UI", sans-serif';

export function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

/* ---------------- color scales ---------------- */

const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
];

// viridis-like sequential scale for t in [0, 1]
export function viridis(t) {
  const x = clamp(t, 0, 1) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x));
  const f = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}

// diverging blue (negative) <-> dark (0) <-> red (positive), v scaled by vmax
const DIV_NEG = [110, 170, 255];
const DIV_POS = [255, 110, 110];
const DIV_ZERO = [24, 28, 38];
export function diverging(v, vmax) {
  if (!(vmax > 0)) vmax = 1;
  const t = clamp(v / vmax, -1, 1);
  if (Math.abs(t) < 0.004) return 'rgb(24,28,38)';
  const c = t > 0 ? DIV_POS : DIV_NEG;
  const f = Math.sqrt(Math.abs(t)); // perceptual lift for small magnitudes
  const r = Math.round(DIV_ZERO[0] + (c[0] - DIV_ZERO[0]) * f);
  const g = Math.round(DIV_ZERO[1] + (c[1] - DIV_ZERO[1]) * f);
  const b = Math.round(DIV_ZERO[2] + (c[2] - DIV_ZERO[2]) * f);
  return `rgb(${r},${g},${b})`;
}

export function tokenColor(i, n) {
  const hue = Math.round((i * (360 / Math.max(n, 1)) + 20) % 360);
  return `hsl(${hue} 72% 64%)`;
}

/* ---------------- tooltip ---------------- */

let tipEl = null;
export function showTip(clientX, clientY, html) {
  if (!tipEl) tipEl = document.getElementById('tooltip');
  if (!tipEl) return;
  tipEl.innerHTML = html;
  tipEl.hidden = false;
  const w = tipEl.offsetWidth, h = tipEl.offsetHeight;
  let x = clientX + 14, y = clientY + 14;
  if (x + w > window.innerWidth - 8) x = clientX - w - 10;
  if (y + h > window.innerHeight - 8) y = clientY - h - 10;
  tipEl.style.left = `${Math.max(6, x)}px`;
  tipEl.style.top = `${Math.max(6, y)}px`;
}
export function hideTip() {
  if (tipEl) tipEl.hidden = true;
}

/* ---------------- canvas helpers ---------------- */

export function prepCanvas(canvas, cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(cssW * dpr));
  const h = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  canvas.style.width = `${Math.max(1, Math.round(cssW))}px`;
  canvas.style.height = `${Math.max(1, Math.round(cssH))}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  return ctx;
}

export function drawPlaceholder(canvas, wrap, message = 'no data') {
  const W = Math.max(120, wrap ? wrap.clientWidth : canvas.parentElement.clientWidth);
  const H = Math.max(60, Math.min(200, canvas.clientHeight || 80));
  const ctx = prepCanvas(canvas, W, H);
  ctx.fillStyle = '#5d6780';
  ctx.font = `12px ${SANS}`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(message, W / 2, H / 2);
}

/* ---------------- heatmap (matrix) ----------------
   Draws matrix[i][j]: i = row (y), j = column (x).
   Sets canvas._hit for hover binding. */
export function drawHeatmap(canvas, wrap, opts) {
  const { matrix, color } = opts; // color(v, i, j) -> css
  const rows = matrix.length, cols = matrix[0].length;
  const colLabels = opts.colLabels || null;
  const rowLabels = opts.rowLabels || null;
  const colSpecial = opts.colSpecial || null;
  const rowSpecial = opts.rowSpecial || null;
  const rowTickEvery = opts.rowTickEvery || 0;
  const rowTickLabel = opts.rowTickLabel || ((i) => i);
  const colTickEvery = opts.colTickEvery || 0;
  const colTickLabel = opts.colTickLabel || ((j) => j);

  const leftW = rowLabels ? (opts.leftW ?? 66) : (rowTickEvery ? 36 : 4);
  const topH = colLabels ? (opts.topH ?? 86) : (colTickEvery ? 22 : 6);
  const availW = Math.max(140, (wrap ? wrap.clientWidth : 900) - leftW - 6);

  const cellW = opts.cellW ?? clamp(Math.floor(availW / cols), opts.minCell ?? 9, opts.maxCell ?? 42);
  const cellH = opts.cellH ?? clamp(Math.floor((opts.targetH ?? 470) / rows), 0.6, opts.maxCellH ?? 24);

  const cssW = leftW + cols * cellW + 4;
  const cssH = topH + rows * cellH + 4;
  const ctx = prepCanvas(canvas, cssW, cssH);

  // column (top) labels, rotated to read bottom -> top
  if (colLabels) {
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const every = cellW < 7 ? Math.ceil(7 / cellW) : 1;
    for (let j = 0; j < cols; j += every) {
      const sp = colSpecial ? colSpecial[j] : false;
      ctx.fillStyle = sp ? '#ffb454' : '#98a2b8';
      let lab = String(colLabels[j]);
      if (lab.length > 14) lab = `${lab.slice(0, 13)}…`;
      ctx.save();
      ctx.translate(leftW + j * cellW + cellW / 2, topH - 4);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(lab, 3, 0);
      ctx.restore();
    }
  } else if (colTickEvery) {
    // horizontal numeric ticks (e.g. dim indices), no rotation needed
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillStyle = '#79839a';
    for (let j = 0; j < cols; j += colTickEvery) {
      ctx.fillText(String(colTickLabel(j)), leftW + j * cellW + cellW / 2, topH - 3);
    }
  }

  // row (left) labels
  ctx.font = `10px ${MONO}`;
  ctx.textBaseline = 'middle';
  if (rowLabels) {
    ctx.textAlign = 'right';
    for (let i = 0; i < rows; i++) {
      const sp = rowSpecial ? rowSpecial[i] : false;
      ctx.fillStyle = sp ? '#ffb454' : '#98a2b8';
      let lab = String(rowLabels[i]);
      const maxCh = Math.floor((leftW - 8) / 6.1);
      if (lab.length > maxCh) lab = `${lab.slice(0, maxCh - 1)}…`;
      ctx.fillText(lab, leftW - 6, topH + i * cellH + cellH / 2);
    }
  } else if (rowTickEvery) {
    ctx.textAlign = 'right';
    ctx.fillStyle = '#79839a';
    for (let i = 0; i < rows; i += rowTickEvery) {
      ctx.fillText(String(rowTickLabel(i)), leftW - 5, topH + i * cellH + Math.max(cellH / 2, 4));
    }
  }

  // cells
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      ctx.fillStyle = color(matrix[i][j], i, j);
      ctx.fillRect(leftW + j * cellW, topH + i * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  // frame
  ctx.strokeStyle = '#2a3245';
  ctx.lineWidth = 1;
  ctx.strokeRect(leftW - 0.5, topH - 0.5, cols * cellW + 1, rows * cellH + 1);

  canvas._hit = {
    matrix, leftW, topH, cellW, cellH, nRows: rows, nCols: cols,
    colLabels, rowLabels, colSpecial, rowSpecial,
  };
}

export function bindMatrixHover(canvas, describe) {
  canvas.addEventListener('mousemove', (e) => {
    const hit = canvas._hit;
    if (!hit) return;
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const j = Math.floor((x - hit.leftW) / hit.cellW);
    const i = Math.floor((y - hit.topH) / hit.cellH);
    if (i < 0 || j < 0 || i >= hit.nRows || j >= hit.nCols) { hideTip(); return; }
    const html = describe(i, j, hit.matrix[i][j], hit);
    if (html) showTip(e.clientX, e.clientY, html);
    else hideTip();
  });
  canvas.addEventListener('mouseleave', hideTip);
}

/* ---------------- line chart ----------------
   opts: { series: [{values, color, dash, width, label}], yMin, yMax, xLabels,
           cursorIndex, cursorColor, height, pad, highlight } */
export function drawLineChart(canvas, wrap, opts) {
  const series = opts.series;
  const n = series[0].values.length;
  const W = Math.max(160, (wrap ? wrap.clientWidth : canvas.parentElement.clientWidth));
  const H = opts.height ?? 250;
  const pad = { l: 40, r: 16, t: 14, b: 26, ...(opts.pad || {}) };
  const yMin = opts.yMin ?? 0;
  const yMax = opts.yMax ?? 1;
  const ctx = prepCanvas(canvas, W, H);

  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  const xOf = (i) => pad.l + (n <= 1 ? plotW / 2 : (plotW * i) / (n - 1));
  const yOf = (v) => pad.t + plotH * (1 - (v - yMin) / (yMax - yMin || 1));

  // y grid + labels
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  const yTicks = opts.yTicks ?? 4;
  for (let k = 0; k <= yTicks; k++) {
    const v = yMin + ((yMax - yMin) * k) / yTicks;
    const y = yOf(v);
    ctx.strokeStyle = '#1c2230';
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.stroke();
    ctx.fillStyle = '#79839a';
    ctx.fillText((opts.yFmt ?? ((x) => x.toFixed(1)))(v), pad.l - 6, y);
  }

  // x labels
  if (opts.xLabels) {
    ctx.textAlign = 'center';
    const every = Math.max(1, Math.ceil(n / 12));
    for (let i = 0; i < n; i += every) {
      ctx.fillStyle = '#79839a';
      ctx.fillText(String(opts.xLabels[i]), xOf(i), H - pad.b + 13);
    }
    ctx.textAlign = 'left';
    ctx.fillStyle = '#5d6780';
    ctx.font = `10px ${SANS}`;
    ctx.fillText(opts.xTitle || 'layer', pad.l, H - 4);
  }

  // series
  for (let s = 0; s < series.length; s++) {
    const ser = series[s];
    const hl = opts.highlight == null || opts.highlight === s;
    ctx.strokeStyle = ser.color;
    ctx.lineWidth = hl ? (ser.width ?? 1.5) : Math.max(1, (ser.width ?? 1.5) - 0.5);
    ctx.globalAlpha = hl ? 1 : 0.22;
    if (ser.dash) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = xOf(i), y = yOf(ser.values[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // cursor line at the selected layer
  if (opts.cursorIndex != null && opts.cursorIndex >= 0 && opts.cursorIndex < n) {
    const x = xOf(opts.cursorIndex);
    ctx.strokeStyle = opts.cursorColor || '#e8ebf2';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, pad.t);
    ctx.lineTo(x, H - pad.b);
    ctx.stroke();
    ctx.setLineDash([]);
    if (opts.cursorDot && series.length >= 1) {
      const v = series[0].values[opts.cursorIndex];
      ctx.fillStyle = opts.cursorColor || '#e8ebf2';
      ctx.beginPath();
      ctx.arc(x, yOf(v), 4, 0, Math.PI * 2);
      ctx.fill();
      if (opts.cursorLabel) {
        ctx.font = `11px ${MONO}`;
        ctx.textAlign = x > W - 70 ? 'right' : 'left';
        ctx.fillText(opts.cursorLabel(v), x + (x > W - 70 ? -8 : 8), yOf(v) - 8);
      }
    }
  }

  canvas._chart = { pad, n, xOf, yOf, yMin, yMax, series, plotH };
}

export function bindChartHover(canvas, describe) {
  canvas.addEventListener('mousemove', (e) => {
    const ch = canvas._chart;
    if (!ch) return;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const t = (mx - ch.pad.l) / ((canvas.clientWidth - ch.pad.l - ch.pad.r) || 1);
    const i = clamp(Math.round(t * (ch.n - 1)), 0, ch.n - 1);
    // nearest series vertically at that x
    let best = -1, bestD = Infinity;
    for (let s = 0; s < ch.series.length; s++) {
      const d = Math.abs(ch.yOf(ch.series[s].values[i]) - my);
      if (d < bestD) { bestD = d; best = s; }
    }
    if (best < 0) { hideTip(); return; }
    const html = describe(i, best, ch.series[best].values[i], ch.series[best]);
    if (html) showTip(e.clientX, e.clientY, html);
    else hideTip();
  });
  canvas.addEventListener('mouseleave', hideTip);
}

/* ---------------- scatter ---------------- */

export function drawScatter(canvas, wrap, points, opts = {}) {
  const W = Math.max(160, (wrap ? wrap.clientWidth : canvas.parentElement.clientWidth));
  const H = opts.height ?? Math.min(360, Math.max(240, W));
  const ctx = prepCanvas(canvas, W, H);
  const pad = 34;

  let x0 = Math.min(0, ...points.map((p) => p.x)), x1 = Math.max(0, ...points.map((p) => p.x));
  let y0 = Math.min(0, ...points.map((p) => p.y)), y1 = Math.max(0, ...points.map((p) => p.y));
  if (x1 - x0 < 1e-6) { x0 -= 0.5; x1 += 0.5; }
  if (y1 - y0 < 1e-6) { y0 -= 0.5; y1 += 0.5; }
  const xOf = (v) => pad + (W - 2 * pad) * ((v - x0) / (x1 - x0));
  const yOf = (v) => H - pad - (H - 2 * pad) * ((v - y0) / (y1 - y0));

  // zero axes
  ctx.strokeStyle = '#232a3a';
  if (x0 < 0 && x1 > 0) { ctx.beginPath(); ctx.moveTo(xOf(0), pad); ctx.lineTo(xOf(0), H - pad); ctx.stroke(); }
  if (y0 < 0 && y1 > 0) { ctx.beginPath(); ctx.moveTo(pad, yOf(0)); ctx.lineTo(W - pad, yOf(0)); ctx.stroke(); }
  ctx.strokeStyle = '#1c2230';
  ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad);

  const pix = points.map((p, i) => ({ i, x: xOf(p.x), y: yOf(p.y), p }));
  for (const q of pix) {
    ctx.fillStyle = points[q.i].color || '#ffb454';
    ctx.beginPath();
    ctx.arc(q.x, q.y, 5.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#cdd6e8';
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    // alternate labels above / below the dot so clustered points stay readable
    ctx.fillText(String(points[q.i].label ?? q.i), q.x + 7, q.y - 4 + (q.i % 2 ? 14 : 0));
  }
  canvas._scatter = pix;
}

export function bindScatterHover(canvas, describe) {
  canvas.addEventListener('mousemove', (e) => {
    const pts = canvas._scatter;
    if (!pts) return;
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bestD = 15 * 15;
    for (const q of pts) {
      const d = (q.x - mx) * (q.x - mx) + (q.y - my) * (q.y - my);
      if (d < bestD) { bestD = d; best = q.i; }
    }
    if (best < 0) { hideTip(); return; }
    const html = describe(best);
    if (html) showTip(e.clientX, e.clientY, html);
    else hideTip();
  });
  canvas.addEventListener('mouseleave', hideTip);
}

/* ---------------- probe mini-canvases ---------------- */

// schematic token stripes derived from the real cls_preview vector
export function drawMiniStripes(canvas, nTokens, seed, cssW) {
  const W = Math.max(120, cssW || canvas.parentElement.clientWidth || 240);
  const cols = 96;
  const rowH = clamp(Math.floor(170 / Math.max(nTokens, 1)), 2, 9);
  const H = nTokens * rowH + 4;
  const ctx = prepCanvas(canvas, W, H);
  const seedArr = (seed && seed.length) ? seed : [0.1, -0.1];
  const L = seedArr.length;
  const cellW = W / cols;
  for (let i = 0; i < nTokens; i++) {
    for (let d = 0; d < cols; d++) {
      const pos = (d / cols) * L + i * 1.31;
      const i0 = Math.floor(pos) % L;
      const i1 = (i0 + 1) % L;
      const f = pos - Math.floor(pos);
      const v = seedArr[i0] + (seedArr[i1] - seedArr[i0]) * f
        + 0.12 * Math.sin(i * 2.7 + d * 0.29);
      ctx.fillStyle = diverging(v, 0.7);
      ctx.fillRect(d * cellW, 2 + i * rowH, cellW + 0.5, rowH + 0.5);
    }
  }
}

// vertical diverging bars (e.g. cls_preview)
export function drawDivergingBars(canvas, values, cssW) {
  const W = Math.max(120, cssW || canvas.parentElement.clientWidth || 220);
  const H = 92;
  const ctx = prepCanvas(canvas, W, H);
  const vmax = Math.max(0.001, ...values.map((v) => Math.abs(v)));
  const bw = W / values.length;
  const mid = H / 2;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const h = (Math.abs(v) / vmax) * (mid - 8);
    ctx.fillStyle = v >= 0 ? '#ff6e6e' : '#6eaaff';
    const y = v >= 0 ? mid - h : mid;
    ctx.fillRect(i * bw + 2, y, bw - 4, Math.max(1.5, h));
  }
  ctx.strokeStyle = '#3a4257';
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(W, mid);
  ctx.stroke();
}

// small heat strip with row labels (e.g. w_preview)
export function drawMiniHeat(canvas, matrix, rowLabels, cssW) {
  const leftW = 64; // wide enough for the longest label ("surprise") at 10px mono
  const W = Math.max(140, (cssW || canvas.parentElement.clientWidth || 240));
  const cell = 12;
  const H = matrix.length * cell + 4;
  const ctx = prepCanvas(canvas, W, H);
  const vmax = Math.max(0.001, ...matrix.flat().map((v) => Math.abs(v)));
  const cols = matrix[0].length;
  const cw = (W - leftW - 4) / cols;
  ctx.font = `10px ${MONO}`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < matrix.length; i++) {
    ctx.fillStyle = '#98a2b8';
    ctx.fillText(String(rowLabels[i] ?? i), leftW - 5, 2 + i * cell + cell / 2);
    for (let j = 0; j < cols; j++) {
      ctx.fillStyle = diverging(matrix[i][j], vmax);
      ctx.fillRect(leftW + j * cw, 2 + i * cell, cw + 0.5, cell + 0.5);
    }
  }
}
