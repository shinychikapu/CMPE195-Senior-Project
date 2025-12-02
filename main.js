// ---------------- HEATMAP (from your matrix_to_scatter, integrated) ----------------
const hm = { svg: null, g: null, zoom: null, data: [], order: [], selection: null, zTransform: null, map: null, info: document.getElementById('hmInfo') };
// === Annotations (BEDPE-like) ===
const ann = {
  rows: [],          // parsed rows filtered to current chromosome
  binSize: 1_000_000,
  chr: 'chr1',
  useColorFromFile: true,
  defaultRGB: '0,255,0',
  show: true
};
// === TAD/Loop overlay state (derived from ann.rows) ===
const tadLoop = {
  tads: [],   // {chr,x1,x2,y1,y2,color,features,id}
  loops: [],  // {chr,a,b,color,score}
};

const useTADClusters = document.getElementById('useTADClusters');
const showLoopEdges = document.getElementById('showLoopEdges');
const loopsAffectLayout = document.getElementById('loopsAffectLayout');

useTADClusters.addEventListener('change', () => { rerunLayoutDebounced(); });
showLoopEdges.addEventListener('change', () => { draw(); });
loopsAffectLayout.addEventListener('change', () => { rerunLayoutDebounced(); });

function normalizeChr(s) {
  if (!s) return '';
  s = String(s).trim();
  if (!s) return '';
  if (s.startsWith('chr') || s.startsWith('CHR')) return s.replace(/^CHR/, 'chr');
  return 'chr' + s;
}

function rgbOrFallback(colorStr, fallback) {
  if (!colorStr) return `rgb(${fallback})`;
  const m = String(colorStr).match(/^\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*$/);
  if (!m) return `rgb(${fallback})`;
  const [r,g,b] = m.slice(1).map(Number).map(v => Math.max(0, Math.min(255, v)));
  return `rgb(${r},${g},${b})`;
}

function bpToBin(bp, binSize) {
  // Arrowhead/HiCCUPS usually define TAD/loop anchors in bp; your matrix bins are uniform.
  // Round to nearest bin start to align visually with heatmap pixels.
  return Math.round(Number(bp) / Number(binSize)) * Number(binSize);
}
function parseBEDPE(text, targetChr, binSize) {
  const out = [];
  const lines = String(text || '').split(/\r?\n/);
  // try to detect header (first line has non-numeric tokens or "chr1")
  const startIdx = lines.length && /chr/i.test(lines[0]) && /\bchr1?\b/i.test(lines[0]) ? 1 : 0;

  for (let li = startIdx; li < lines.length; li++) {
    const line = lines[li].trim();
    if (!line || line.startsWith('#')) continue;
    const p = line.split(/\s+/);
    if (p.length < 6) continue;

    const c1 = normalizeChr(p[0]);
    const x1 = +p[1], x2 = +p[2];
    const c2 = normalizeChr(p[3]);
    const y1 = +p[4], y2 = +p[5];

    // Keep only intra-chromosomal for overlay (squares on the diagonal)
    if (c1 !== c2) continue;
    if (targetChr && c1 !== targetChr) continue;
    if (!Number.isFinite(x1) || !Number.isFinite(x2) || !Number.isFinite(y1) || !Number.isFinite(y2)) continue;

    const color = p[6] && /^\d/.test(p[6]) ? p[6] : null; // optional RGB
    const features = p.slice(7).map(Number).filter(v => Number.isFinite(v));

    // Convert bp to bin-aligned coordinates (so they line up with matrix bands)
    const X1 = bpToBin(Math.min(x1, x2), binSize);
    const X2 = bpToBin(Math.max(x1, x2), binSize);
    const Y1 = bpToBin(Math.min(y1, y2), binSize);
    const Y2 = bpToBin(Math.max(y1, y2), binSize);

    out.push({ chr: c1, x1: X1, x2: X2, y1: Y1, y2: Y2, color, features });
  }
  return out;
}

function parseRAW(text) {
  const out = []; const lines = String(text || '').trim().split(/\r?\n/);
  for (const line of lines) {
    const p = line.trim().split(/\s+/); if (p.length < 2) continue;
    const i = +p[0], j = +p[1], v = p[2] ? +p[2] : 1;
    if (Number.isFinite(i) && Number.isFinite(j) && Number.isFinite(v)) out.push({ i, j, value: v });
  }
  return out;
}

function drawHeatmap(rawText) {
  const triples = parseRAW(rawText);
  if (!triples.length) { hm.info.textContent = 'No valid i j value lines.'; return; }
  // symmetric map + indices
  const set = new Set(); const map = new Map(); let vmax = 0;
  for (const { i, j, value } of triples) {
    set.add(i); set.add(j);
    map.set(i + ',' + j, value); map.set(j + ',' + i, value);
    if (value > vmax) vmax = value;
  }
  hm.map = map;
  hm.order = Array.from(set).sort((a, b) => a - b);

  const W = 780, H = 780, padL = 18, padT = 18;
  const cell = Math.max(2, Math.floor(Math.min(W, H) / Math.sqrt(hm.order.length + 1)));
  const width = Math.min(W, hm.order.length * cell);
  const height = Math.min(H, hm.order.length * cell);

  hm.svg = d3.select('#heatmap').attr('viewBox', `0 0 ${width + padL * 2} ${height + padT * 2}`);
  hm.svg.selectAll('*').remove();
  hm.g = hm.svg.append('g').attr('transform', `translate(${padL},${padT})`);
  const x = d3.scaleBand().domain(hm.order).range([0, width]).padding(0);
  const y = d3.scaleBand().domain(hm.order).range([0, height]).padding(0);
  const color = d3.scaleSequential(d3.interpolateOrRd).domain([0, Math.log1p(vmax)]);

  // dense grid
  const grid = [];
  for (const i of hm.order) {
    for (const j of hm.order) {
      const v = map.get(i + ',' + j) || 0;
      grid.push({ i, j, value: v });
    }
  }

  hm.g.selectAll('rect').data(grid).join('rect')
    .attr('x', d => x(d.i)).attr('y', d => y(d.j))
    .attr('width', x.bandwidth()).attr('height', y.bandwidth())
    .attr('fill', d => color(Math.log1p(d.value)));

  hm.zoom = d3.zoom().scaleExtent([0.5, 12]).on('zoom', ev => { hm.g.attr('transform', ev.transform); hm.zTransform = ev.transform; });
  d3.select('#hmPan').on('click', () => { disableBoxSelect(); hm.svg.call(hm.zoom.transform, hm.zTransform || d3.zoomIdentity); });
  d3.select('#hmSelect').on('click', () => enableBoxSelect(x, y));
  d3.select('#hmClear').on('click', () => { if (hm.selection) { hm.selection.remove(); hm.selection = null; } });
  hm.svg.call(hm.zoom).on('dblclick.zoom', null);

  // after the heatmap cells are created
  splitTADsAndLoops();       
  drawHeatmapOverlay();
  hm.info.textContent = `Matrix: ${hm.order.length}×${hm.order.length} • cells drawn: ${grid.length}`;
}

function refreshAnnotations() {
  // If user changes chr or bin size after loading, re-filter/re-bin if original text is unavailable.
  // Minimal approach: keep only rows matching ann.chr (already stored); user can reload file if needed.
  drawHeatmapOverlay();
}
function splitTADsAndLoops() {
  const T = [], L = [];
  for (const r of ann.rows) {
    // consider TAD if x-range ~ y-range (allow small bin-size tolerance)
    const isTAD = Math.abs((r.x2 - r.x1) - (r.y2 - r.y1)) <= ann.binSize * 0.5
               && Math.abs(r.x1 - r.y1) <= ann.binSize * 0.5
               && Math.abs(r.x2 - r.y2) <= ann.binSize * 0.5;
    if (isTAD) {
      T.push({ ...r, id: `${r.chr}:${r.x1}-${r.x2}` });
    } else {
      // Loop anchor = start bins of each side (use lower bound)
      const a = Math.min(r.x1, r.x2);
      const b = Math.min(r.y1, r.y2);
      const score = (r.features && r.features.length) ? r.features[0] : 1;
      L.push({ chr: r.chr, a, b, color: r.color, score });
    }
  }
  tadLoop.tads = T;
  tadLoop.loops = L;
}

function drawHeatmapOverlay() {
  if (!hm.g) return;
  hm.g.selectAll('.annbox').remove();
  if (!ann.show || !ann.rows.length) return;

  // We need band scales from current heatmap draw
  // Recreate them from hm.order and the heatmap's width/height in the 'g' bbox
  const gbox = hm.g.node().getBBox();
  const width = gbox.width, height = gbox.height;
  const x = d3.scaleBand().domain(hm.order).range([0, width]).padding(0);
  const y = d3.scaleBand().domain(hm.order).range([0, height]).padding(0);

  // Helper: convert a bp coordinate to the bin "index" your matrix uses.
  // Your RAW uses integer IDs that typically equal bin starts (for 1Mb bins: 0, 1e6, 2e6, ...).
  function coordToKey(bp) {
    // If your matrix nodes are the bin start positions, use the rounded bin start we computed:
    return bp; // bpToBin already done in parseBEDPE
  }

  const bw = x.bandwidth();
  const bh = y.bandwidth();

  const data = ann.rows
    .map(r => {
      const xi = coordToKey(r.x1), xj = coordToKey(r.x2);
      const yi = coordToKey(r.y1), yj = coordToKey(r.y2);
      // find band positions for start and end (inclusive range)
      const xiPix = x(xi), yiPix = y(yi), xjPix = x(xj), yjPix = y(yj);
      if (xiPix == null || yiPix == null || xjPix == null || yjPix == null) return null;

      // Compute pixel rect covering the bin range (x1..x2, y1..y2)
      const x0 = Math.min(xiPix, xjPix);
      const y0 = Math.min(yiPix, yjPix);
      const w = (Math.abs(xjPix - xiPix) + bw);
      const h = (Math.abs(yjPix - yiPix) + bh);

      const color = ann.useColorFromFile ? rgbOrFallback(r.color, ann.defaultRGB) : `rgb(${ann.defaultRGB})`;
      return { x: x0, y: y0, w, h, color, r };
    })
    .filter(Boolean);

  hm.g.selectAll('.annbox')
  .data(data)
  .enter()
  .append('rect')
  .attr('class', 'annbox')
  .attr('x', d => d.x)
  .attr('y', d => d.y)
  .attr('width', d => d.w)
  .attr('height', d => d.h)
  .attr('fill', 'none')
  .attr('stroke', d => d.color)
  .attr('stroke-width', 1.5)
  .style('cursor', 'pointer')
  .on('click', (ev, d) => {
    // Build RAW lines within the rectangle just like box-select does
    const picked = [];
    const bw = x.bandwidth(), bh = y.bandwidth();
    const box = { x: d.x, y: d.y, w: d.w, h: d.h };
    for (const i of hm.order) {
      const xi = x(i);
      for (const j of hm.order) {
        const yj = y(j);
        const hit = !(xi > box.x + box.w || xi + bw < box.x || yj > box.y + box.h || yj + bh < box.y);
        if (hit) {
          const v = hm.map.get(i + ',' + j);
          if (v != null) picked.push(`${i}\t${j}\t${v}`);
        }
      }
    }
    if (picked.length) { build(picked.join('\n')); }
  });

}

function enableBoxSelect(x, y) {
  hm.svg.on('.zoom', null);
  if (hm.selection) { hm.selection.remove(); hm.selection = null; }
  let start = null;

  hm.svg.on('mousedown.box', (ev) => {
    const p = d3.pointer(ev, hm.g.node());
    start = { x: p[0], y: p[1] };
    hm.selection = hm.g.append('rect')
      .attr('x', start.x).attr('y', start.y).attr('width', 0).attr('height', 0)
      .attr('class', 'selection');
  });

  hm.svg.on('mousemove.box', (ev) => {
    if (!hm.selection || !start) return;
    const p = d3.pointer(ev, hm.g.node());
    const x0 = Math.min(start.x, p[0]), y0 = Math.min(start.y, p[1]);
    const w = Math.abs(p[0] - start.x), h = Math.abs(p[1] - start.y);
    hm.selection.attr('x', x0).attr('y', y0).attr('width', w).attr('height', h);
  });

  hm.svg.on('mouseup.box', () => {
    if (!hm.selection) return;
    const box = hm.selection.node().getBBox();
    // pick cells whose rect overlaps box; then convert to RAW lines
    const picked = [];
    const bw = x.bandwidth(), bh = y.bandwidth();
    for (const i of hm.order) {
      const xi = x(i);
      for (const j of hm.order) {
        const yj = y(j);
        const hit = !(xi > box.x + box.width || xi + bw < box.x || yj > box.y + box.height || yj + bh < box.y);
        if (hit) {
          const v = hm.map.get(i + ',' + j);
          if (v != null) picked.push(`${i}\t${j}\t${v}`);
        }
      }
    }
    if (picked.length) { build(picked.join('\n')); } // ← send to graph renderer
    disableBoxSelect();
  });
}
function disableBoxSelect() {
  hm.svg.on('mousedown.box', null).on('mousemove.box', null).on('mouseup.box', null);
  if (hm.zoom) hm.svg.call(hm.zoom);
}

// ---------------- GRAPH (your graph3.html code, unchanged in behavior) ----------------
(function () {
  // ---------- DOM ----------
  const $ = id => document.getElementById(id);
  const stage = $('stage'), canvas = $('canvas'), ctx = canvas.getContext('2d');
  const fileInput = $('file'), statusEl = $('status'), errEl = $('err');
  const search = $('search'), legend = $('legend');
  const edgeAlpha = $('edgeAlpha'), edgeAlphaVal = $('edgeAlphaVal');
  const alphaGamma = $('alphaGamma'), alphaGammaVal = $('alphaGammaVal');
  const labelPct = $('labelPct'), labelPctVal = $('labelPctVal');
  const toggleCross = $('toggleCross'), toggleSelf = $('toggleSelf');
  const toggleClusterLayout = $('toggleClusterLayout'), toggleSpectral = $('toggleSpectral');
  const deterministic = $('deterministic'), seedInput = $('seed');
  const iters = $('iters'), itersVal = $('itersVal');
  const kSpring = $('kSpring'), kSpringVal = $('kSpringVal');
  const kRepel = $('kRepel'), kRepelVal = $('kRepelVal');
  const kGravity = $('kGravity'), kGravityVal = $('kGravityVal');
  const wExp = $('wExp'), wExpVal = $('wExpVal');
  const lenScale = $('lenScale'), lenScaleVal = $('lenScaleVal');
  const wPercentile = $('wPercentile'), wPercentileVal = $('wPercentileVal');
  const wInfluence = $('wInfluence'), wInfluenceVal = $('wInfluenceVal');
  const sepMult = $('sepMult'), sepMultVal = $('sepMultVal');
  const interK = $('interK'), interKVal = $('interKVal');
  const itersIntra = $('itersIntra'), itersIntraVal = $('itersIntraVal');
  const nodeCountEl = $('nodeCount'), edgeCountEl = $('edgeCount'), comCountEl = $('comCount');
  // Annotation DOM controls
  const annFile = document.getElementById('annFile');
  const annChrInput = document.getElementById('annChr');
  const binSizeInput = document.getElementById('binSize');
  const annShowInput = document.getElementById('annShow');
  const annUseColorInput = document.getElementById('annUseColor');
  const annColorInput = document.getElementById('annColor');


  // ---------- Seeded RNG ----------
  let _rng = mulberry32(12345);
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = (t + Math.imul(t ^ t >>> 7, 61 | t)) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function reseed(seed) { _rng = mulberry32((seed | 0) || 1); }
  function rnd() { return _rng(); }
  function rand(a, b) { return a + rnd() * (b - a); }
  function jitter(s = 1) { return (rnd() - 0.5) * s; }

  // ---------- State ----------
  let nodes = [], indexOf = new Map(), clusters = {}, clusterOf = {}, clusterColors = {}, degrees = {}, edges = [], adjacency = new Map();
  let edgesW = []; // {a,b,w,n}
  let weightsNorm = []; let wMin = 1, wMax = 1;
  let N = 0, pos = [], vel = [], cam = { x: 0, y: 0, scale: 1 }, hoverIdx = -1;
  let viewW = 0, viewH = 0, dpr = 1;

  // cluster-level
  let clusterIds = [], clusterCenter = {}, clusterSize = {}, interW = {}, strongIntraByCluster = {};

  // ---------- Utils ----------
  const TAU = Math.PI * 2;
  const clamp = (v, min, max) => v < min ? min : (v > max ? max : v);
  const colorFor = k => `hsl(${(137.508 * k) % 360},70%,55%)`;
  const hsla = (h, a) => h.replace('hsl', 'hsla').replace(')', `, ${a})`);
  const setStatus = msg => statusEl.innerHTML = msg;
  const showError = msg => { errEl.style.display = 'block'; errEl.textContent = msg; };
  const clearError = () => { errEl.style.display = 'none'; errEl.textContent = ''; };

  // ---------- Louvain (weighted, compact) ----------
  function louvainWeighted(nodeList, edges) {
    const nodes = nodeList.slice(), idx = new Map(nodes.map((id, i) => [id, i])), n = nodes.length;
    const neigh = Array.from({ length: n }, () => new Map()); let m2 = 0;
    for (const e of edges) {
      const i = idx.get(e.a), j = idx.get(e.b), w = e.w; if (i == null || j == null) continue;
      neigh[i].set(j, (neigh[i].get(j) || 0) + w); neigh[j].set(i, (neigh[j].get(i) || 0) + w); m2 += 2 * w;
    }
    if (m2 <= 0) m2 = 1;
    let com = new Array(n).fill(0).map((_, i) => i), totalW = new Array(n).fill(0), nodeW = new Array(n).fill(0);
    for (let i = 0; i < n; i++) { let s = 0; neigh[i].forEach(w => s += w); nodeW[i] = s; totalW[com[i]] += s; }
    let improved = true, passes = 0;
    while (improved && passes < 20) {
      improved = false; passes++;
      let order = [...Array(n).keys()]; for (let i = order.length - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0;[order[i], order[j]] = [order[j], order[i]]; }
      for (const i of order) {
        const ci = com[i]; totalW[ci] -= nodeW[i];
        const k_i = nodeW[i], neighComW = new Map();
        neigh[i].forEach((w, j) => { const cj = com[j]; neighComW.set(cj, (neighComW.get(cj) || 0) + w); });
        let bestC = ci, bestGain = 0, k_i_over_m2 = k_i / m2;
        neighComW.forEach((sumIn, c) => {
          const gain = (sumIn * 2) / m2 - 2 * k_i_over_m2 * (totalW[c] / m2);
          if (gain > bestGain + 1e-12) { bestGain = gain; bestC = c; }
        });
        if (bestC !== ci) { com[i] = bestC; improved = true; }
        totalW[com[i]] += nodeW[i];
      }
    }
    const map = new Map(); let cidx = 0; for (const c of com) { if (!map.has(c)) map.set(c, cidx++); }
    const out = new Array(n); for (let i = 0; i < n; i++) out[i] = map.get(com[i]); return out;
  }

  // ---------- Spectral init ----------
  function spectralInitStrongEdges(strongEdges) {
    const deg = new Float64Array(N);
    const W = Array.from({ length: N }, () => new Map());
    for (const e of strongEdges) {
      if (e.a === e.b) continue;
      const i = indexOf.get(e.a), j = indexOf.get(e.b);
      const w = e.w; if (i == null || j == null) continue;
      W[i].set(j, (W[i].get(j) || 0) + w);
      W[j].set(i, (W[j].get(i) || 0) + w);
      deg[i] += w; deg[j] += w;
    }
    function mulS(x) {
      const y = new Float64Array(N);
      for (let i = 0; i < N; i++) {
        const di = deg[i] > 0 ? 1 / Math.sqrt(deg[i]) : 0;
        let sum = 0;
        W[i].forEach((wij, j) => { const dj = deg[j] > 0 ? 1 / Math.sqrt(deg[j]) : 0; sum += (wij * di * dj) * x[j]; });
        y[i] = sum;
      }
      return y;
    }
    function normalize(v) { let n = 0; for (let i = 0; i < N; i++) n += v[i] * v[i]; n = Math.sqrt(n) || 1; for (let i = 0; i < N; i++) v[i] /= n; }
    function dot(a, b) { let s = 0; for (let i = 0; i < N; i++) s += a[i] * b[i]; return s; }

    let v1 = new Float64Array(N); for (let i = 0; i < N; i++) v1[i] = rnd() - 0.5; normalize(v1);
    for (let t = 0; t < 60; t++) { v1 = mulS(v1); normalize(v1); }
    let v2 = new Float64Array(N); for (let i = 0; i < N; i++) v2[i] = rnd() - 0.5;
    let proj = dot(v2, v1); for (let i = 0; i < N; i++) v2[i] -= proj * v1[i]; normalize(v2);
    for (let t = 0; t < 60; t++) { v2 = mulS(v2); proj = dot(v2, v1); for (let i = 0; i < N; i++) v2[i] -= proj * v1[i]; normalize(v2); }

    let sx = 0, sy = 0; for (let i = 0; i < N; i++) { sx = Math.max(sx, Math.abs(v1[i])); sy = Math.max(sy, Math.abs(v2[i])); }
    const scale = (Math.min(viewW, viewH) || 1200) * 0.35;
    for (let i = 0; i < N; i++) {
      pos[i].x = (v1[i] / (sx || 1)) * scale + jitter(8);
      pos[i].y = (v2[i] / (sy || 1)) * scale + jitter(8);
      vel[i].x = vel[i].y = 0;
    }
  }

  // ---------- File / Build ----------
  fileInput.addEventListener('change', (e) => {
    try {
      clearError();
      const f = e.target.files && e.target.files[0];
      if (!f) { setStatus('No file selected.'); return; }
      setStatus('Reading…');
      const r = new FileReader();
      r.onerror = () => showError('Failed to read file.');
      r.onload = () => {
        try {
          const raw = String(r.result || ''); if (!raw.trim()) { showError('File is empty.'); return; }
          drawHeatmap(raw);   // <-- update heatmap from same file
          build(raw);         // <-- and render graph
        } catch (err) { showError('Error while parsing: ' + (err?.stack || err)); }
      };
      r.readAsText(f);
    } catch (err) { showError('Unexpected error: ' + (err?.stack || err)); }
  });

  // Expose build so heatmap box-select can call it
  window.build = build;

  function build(text) {
    if (deterministic.checked) reseed(Number(seedInput.value) || 1);

    setStatus('Parsing…');
    const lines = text.split(/\r?\n/);
    const nodeSet = new Set(), edgeSet = new Set();
    adjacency = new Map(); edges = []; edgesW = []; wMin = Infinity; wMax = -Infinity;

    for (const raw of lines) {
      if (!raw) continue;
      const p = raw.split('\t'); if (p.length < 2) continue;
      const a = (p[0] || '').trim(), b = (p[1] || '').trim(); if (!a || !b) continue;
      const wVal = (p[2] !== undefined && p[2] !== '' ? Number(p[2]) : 1);
      const w = isFinite(wVal) ? wVal : 1;

      nodeSet.add(a); nodeSet.add(b);
      const key = (a <= b) ? a + '|' + b : b + '|' + a; if (edgeSet.has(key)) continue;
      edgeSet.add(key); edges.push([a, b]); edgesW.push({ a, b, w });
      wMin = Math.min(wMin, w); wMax = Math.max(wMax, w);

      if (!adjacency.has(a)) adjacency.set(a, new Set());
      if (!adjacency.has(b)) adjacency.set(b, new Set());
      adjacency.get(a).add(b); adjacency.get(b).add(a);
    }

    nodes = Array.from(nodeSet); if (!nodes.length) throw new Error('No nodes parsed. Expected: node1\\tnode2\\tweight');
    N = nodes.length;

    const span = (wMax - wMin) || 1;
    weightsNorm = []; for (const e of edgesW) { e.n = (e.w - wMin) / span; weightsNorm.push(e.n); }
    weightsNorm.sort((a, b) => a - b);

    const wMap = new Map(); for (const n of nodes) wMap.set(n, new Map());
    degrees = {};
    for (const e of edgesW) { wMap.get(e.a).set(e.b, e.n); wMap.get(e.b).set(e.a, e.n); }
    for (const n of nodes) { let d = 0; for (const m of (adjacency.get(n) || [])) d += (wMap.get(n).get(m) ?? 1); degrees[n] = d; }

    const louvainCom = louvainWeighted(nodes, edgesW);
    // shared across both branches
    // assign to module-scope (no 'let' here)
    clusters = {};
    clusterOf = {};
    clusterColors = {};
    order = [];
    // ---- Optional override: color nodes by TADs instead of Louvain
    if (useTADClusters.checked) {
      // Build a palette on the fly
      console.log('Using TADs for clustering');
      const colorForTad = (k) => `hsl(${(137.508 * k) % 360},70%,55%)`;
      // index TADs by numeric id
      const tadIndex = new Map(); let tcount = 0;

      // assign each node (bin-start string) to a TAD by genomic span
      for (const id of nodes) {
        const pos = Number(id); // your node ids are bin starts
        let tid = -1;
        for (let i = 0; i < tadLoop.tads.length; i++) {
          const t = tadLoop.tads[i];
          if (pos >= t.x1 && pos < t.x2) { tid = i; break; }
        }
        const cid = (tid >= 0) ? (tadIndex.has(tid) ? tadIndex.get(tid) : tadIndex.set(tid, tcount).get(tid)) : -1;
        const clusterId = (cid >= 0) ? cid : 999999; // unassigned goes into one bucket
        if (clusterId === 999999) {
        console.warn(`Node ${id} (pos: ${pos}) not assigned to any TAD.`);
    }
        (clusters[clusterId] || (clusters[clusterId] = [])).push(id);
        clusterOf[id] = clusterId;
        if (clusterColors[clusterId] == null) clusterColors[clusterId] = (cid >= 0) ? colorForTad(clusterId) : '#888888';
      }

      // remake order, nodes, indexOf with this cluster grouping (keeps your legend logic intact)
      order = Object.keys(clusters).map(c => ({ c: +c, size: clusters[c].length }))
                                        .sort((a, b) => b.size - a.size).map(o => o.c);
      order.forEach(c => clusters[c].sort((a, b) => degrees[b] - degrees[a] || (a < b ? -1 : 1)));
      nodes = []; order.forEach(c => nodes.push(...clusters[c])); indexOf = new Map(nodes.map((id, i) => [id, i]));
      pos = new Array(N); vel = new Array(N);
      const spread = Math.min(viewW, viewH) || 1000;
      for (let i = 0; i < N; i++) {
        pos[i] = { x: rand(-spread * 0.25, spread * 0.25), y: rand(-spread * 0.25, spread * 0.25) };
        vel[i] = { x: 0, y: 0 };
      }
      cam.x = 0; cam.y = 0; cam.scale = 1;

      // Recompute cluster-level meta for layout (so clustered layout still works)
      buildClusterMeta();

      // Rebuild legend for TAD groups
      legend.innerHTML = '';
      for (const c of order) {
        const chip = document.createElement('div'); chip.className = 'chip';
        const dot = document.createElement('div'); dot.className = 'dot'; dot.style.background = clusterColors[c];
        const span = document.createElement('span'); span.textContent = (c === 999999) ? `Unassigned • ${clusters[c].length}` : `TAD ${c} • ${clusters[c].length}`;
        chip.appendChild(dot); chip.appendChild(span);
        chip.onclick = () => zoomToCluster(c);
        legend.appendChild(chip);
      }
    } else 
    {
    for (let i = 0; i < N; i++) {
      const c = louvainCom[i]; const id = nodes[i];
      (clusters[c] || (clusters[c] = [])).push(id); clusterOf[id] = c; if (clusterColors[c] == null) clusterColors[c] = colorFor(c);
    }

    const order = Object.keys(clusters).map(c => ({ c: +c, size: clusters[c].length })).sort((a, b) => b.size - a.size).map(o => o.c);
    order.forEach(c => clusters[c].sort((a, b) => degrees[b] - degrees[a] || (a < b ? -1 : 1)));
    nodes = []; order.forEach(c => nodes.push(...clusters[c])); indexOf = new Map(nodes.map((id, i) => [id, i]));

    pos = new Array(N); vel = new Array(N);
    const spread = Math.min(viewW, viewH) || 1000;
    for (let i = 0; i < N; i++) { pos[i] = { x: rand(-spread * 0.25, spread * 0.25), y: rand(-spread * 0.25, spread * 0.25) }; vel[i] = { x: 0, y: 0 }; }
    cam.x = 0; cam.y = 0; cam.scale = 1;

    buildClusterMeta();

    legend.innerHTML = '';
    for (const c of order) {
      const chip = document.createElement('div'); chip.className = 'chip';
      const dot = document.createElement('div'); dot.className = 'dot'; dot.style.background = clusterColors[c];
      const span = document.createElement('span'); span.textContent = `Cluster ${c} • ${clusters[c].length}`;
      chip.appendChild(dot); chip.appendChild(span);
      chip.onclick = () => zoomToCluster(c);
      legend.appendChild(chip);
    }
    // end of the TAD branch, right after you build the legend
    nodeCountEl.textContent = N;
    edgeCountEl.textContent = edges.length;
    comCountEl.textContent = order.length;
    setStatus('Laying out…');
    runLayout();
    fitView();
    setStatus('Rendered ✓');
    resize();
  }
    nodeCountEl.textContent = N; edgeCountEl.textContent = edges.length; comCountEl.textContent = order.length;
    setStatus('Laying out…');
    runLayout();
    fitView();  // auto-fit after upload/selection
    setStatus('Rendered ✓');
    resize();
  }

  function buildClusterMeta() {
    clusterIds = Object.keys(clusters).map(x => +x);
    const R = (Math.min(viewW, viewH) || 1000) * 0.35 + 60;
    clusterCenter = {}; clusterSize = {}; interW = {};
    const C = clusterIds.length;
    for (let k = 0; k < C; k++) {
      const cid = clusterIds[k]; clusterSize[cid] = clusters[cid].length;
      const ang = (k / C) * TAU; clusterCenter[cid] = { x: R * Math.cos(ang), y: R * Math.sin(ang) };
    }
    for (const e of edgesW) {
      const ca = clusterOf[e.a], cb = clusterOf[e.b]; if (ca === cb) continue;
      const key = (ca <= cb) ? ca + '|' + cb : cb + '|' + ca; interW[key] = (interW[key] || 0) + e.w;
    }
    strongIntraByCluster = {};
    const p = Number(wPercentile.value); const idxCut = Math.floor(weightsNorm.length * (p / 100));
    const cut = weightsNorm[Math.max(0, Math.min(weightsNorm.length - 1, idxCut))] ?? 0;
    for (const cid of clusterIds) strongIntraByCluster[cid] = [];
    for (const e of edgesW) {
      if (e.a === e.b && !toggleSelf.checked) continue;
      if (clusterOf[e.a] !== clusterOf[e.b]) continue;
      if (e.n >= cut) strongIntraByCluster[clusterOf[e.a]].push(e);
    }
    wPercentileVal.textContent = `${p}th`;
  }

  // ---------- Layout orchestration ----------
  function runLayout() {
    if (deterministic.checked) reseed(Number(seedInput.value) || 1);

    if (toggleSpectral.checked) {
      const p = Number(wPercentile.value);
      const idxCut = Math.floor(weightsNorm.length * (p / 100));
      const cut = weightsNorm[Math.max(0, Math.min(weightsNorm.length - 1, idxCut))] ?? 0;
      const strongGlobal = edgesW.filter(e => e.n >= cut && (toggleSelf.checked || e.a !== e.b));
      spectralInitStrongEdges(strongGlobal);
    }

    if (toggleClusterLayout.checked) {
      relaxClusterCentersAttractive();
      relaxIntraClusters();
      draw();
    } else {
      runGlobalForces();
    }
    fitView();
  }

  function relaxClusterCentersAttractive() {
    const steps = 380;
    const sep = Number(sepMult.value) / 100;
    const kBaseAttr = Number(interK.value) / 100; interKVal.textContent = kBaseAttr.toFixed(2);

    const kBaseRep = 0.9 * sep;
    const dt = 0.04;
    const velC = {}; for (const cid of clusterIds) velC[cid] = { x: 0, y: 0 };

    let maxInter = 0; for (const k in interW) if (interW[k] > maxInter) maxInter = interW[k];
    const wNorm = w => (maxInter > 0 ? w / maxInter : 0);

    function desiredDist(ca, cb, w) {
      const sa = Math.sqrt(clusterSize[ca] || 1), sb = Math.sqrt(clusterSize[cb] || 1);
      const base = (Math.min(viewW, viewH) || 1000) * 0.25 * (1 + 0.12 * (sa + sb) / (Math.sqrt(nodes.length) + 1));
      return base / (0.25 + 0.75 * wNorm(w) + 1e-6);
    }

    for (let t = 0; t < steps; t++) {
      for (const cid of clusterIds) { velC[cid].x *= 0.86; velC[cid].y *= 0.86; }

      for (let a = 0; a < clusterIds.length; a++) {
        for (let b = a + 1; b < clusterIds.length; b++) {
          const ca = clusterIds[a], cb = clusterIds[b];
          const key = (ca <= cb) ? ca + '|' + cb : cb + '|' + ca;
          const w = (interW[key] || 0);
          const pa = clusterCenter[ca], pb = clusterCenter[cb];
          let dx = pb.x - pa.x, dy = pb.y - pa.y;
          let dist = Math.hypot(dx, dy) || 1e-6;

          if (w > 0) {
            const ideal = desiredDist(ca, cb, w);
            const kAttr = kBaseAttr * (0.2 + 0.8 * wNorm(w));
            const fAttr = kAttr * (dist - ideal);
            const fxA = fAttr * dx / dist, fyA = fAttr * dy / dist;
            velC[ca].x += fxA * dt; velC[ca].y += fyA * dt;
            velC[cb].x -= fxA * dt; velC[cb].y -= fyA * dt;
          }
          const fRep = kBaseRep / (dist * dist);
          const fxR = fRep * (-dx / dist), fyR = fRep * (-dy / dist);
          velC[ca].x += fxR * dt; velC[ca].y += fyR * dt;
          velC[cb].x -= fxR * dt; velC[cb].y -= fyR * dt;
        }
      }
      for (const cid of clusterIds) {
        velC[cid].x += (-clusterCenter[cid].x * 0.02) * dt;
        velC[cid].y += (-clusterCenter[cid].y * 0.02) * dt;
      }
      for (const cid of clusterIds) {
        clusterCenter[cid].x += velC[cid].x * dt;
        clusterCenter[cid].y += velC[cid].y * dt;
      }
    }
  }

  function relaxIntraClusters() {
    const ITER = Number(itersIntra.value);
    const kRepBase = Number(kRepel.value);
    const kEdgeBase = Number(kSpring.value) / 100;
    const beta = Number(wExp.value) / 100;
    const lenS = Number(lenScale.value) / 100;
    const kCenter = 0.08;
    const dt = 0.02;

    for (let it = 0; it < ITER; it++) {
      for (let i = 0; i < N; i++) { vel[i].x *= 0.86; vel[i].y *= 0.86; }

      for (const cid of clusterIds) {
        const members = clusters[cid] || [];
        for (let a = 0; a < members.length; a++) {
          const ia = indexOf.get(members[a]);
          for (let b = a + 1; b < members.length; b++) {
            const ib = indexOf.get(members[b]);
            let dx = pos[ia].x - pos[ib].x, dy = pos[ia].y - pos[ib].y;
            let d2 = dx * dx + dy * dy; if (d2 < 1e-4) { pos[ia].x += rnd(); pos[ia].y += rnd(); d2 = 1; }
            const inv = 1 / Math.sqrt(d2), f = kRepBase * inv * inv;
            const fx = f * dx * inv, fy = f * dy * inv;
            vel[ia].x += fx * dt; vel[ia].y += fy * dt; vel[ib].x -= fx * dt; vel[ib].y -= fy * dt;
          }
        }
      }

      for (const cid of clusterIds) {
        for (const e of (strongIntraByCluster[cid] || [])) {
          const i = indexOf.get(e.a), j = indexOf.get(e.b);
          const dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y;
          const dist = Math.max(0.001, Math.hypot(dx, dy));
          const wEff = Math.pow(Math.max(0, e.n), beta);
          const ideal = (28 * lenS) / (0.12 + wEff);
          const kEdge = kEdgeBase * (0.25 + 0.75 * wEff);
          const f = kEdge * (dist - ideal);
          const fx = f * dx / dist, fy = f * dy / dist;
          vel[i].x += fx * dt; vel[i].y += fy * dt; vel[j].x -= fx * dt; vel[j].y -= fy * dt;
        }
      }

      for (const cid of clusterIds) {
        const c = clusterCenter[cid];
        for (const id of (clusters[cid] || [])) {
          const i = indexOf.get(id);
          vel[i].x += (c.x - pos[i].x) * kCenter * dt;
          vel[i].y += (c.y - pos[i].y) * kCenter * dt;
        }
      }

      for (let i = 0; i < N; i++) { pos[i].x += vel[i].x * dt; pos[i].y += vel[i].y * dt; }
    }
  }

  function runGlobalForces() {
    const ITER = Number(iters.value);
    const kEdgeBase = Number(kSpring.value) / 100;
    const kRep = Number(kRepel.value);
    const kG = Number(kGravity.value) / 100;
    const beta = Number(wExp.value) / 100;
    const lenS = Number(lenScale.value) / 100;
    const dt = 0.02;

    const p = Number(wPercentile.value), idxCut = Math.floor(weightsNorm.length * (p / 100));
    const cut = weightsNorm[Math.max(0, Math.min(weightsNorm.length - 1, idxCut))] ?? 0;
    const strongEdges = edgesW.filter(e => e.n >= cut && (toggleSelf.checked || e.a !== e.b));

    for (let it = 0; it < ITER; it++) {
      for (let i = 0; i < N; i++) { vel[i].x *= 0.86; vel[i].y *= 0.86; }

      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          let dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
          let d2 = dx * dx + dy * dy; if (d2 < 1e-4) { pos[i].x += rnd(); pos[i].y += rnd(); d2 = 1; }
          const inv = 1 / Math.sqrt(d2), f = kRep * inv * inv;
          const fx = f * dx * inv, fy = f * dy * inv;
          vel[i].x += fx * dt; vel[i].y += fy * dt; vel[j].x -= fx * dt; vel[j].y -= fy * dt;
        }
      }

      for (const e of strongEdges) {
        const i = indexOf.get(e.a), j = indexOf.get(e.b);
        const dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y;
        const dist = Math.max(0.001, Math.hypot(dx, dy));
        const wEff = Math.pow(Math.max(0, e.n), beta);
        const ideal = (28 * lenS) / (0.12 + wEff);
        const kEdge = kEdgeBase * (0.25 + 0.75 * wEff);
        const f = kEdge * (dist - ideal);
        const fx = f * dx / dist, fy = f * dy / dist;
        vel[i].x += fx * dt; vel[i].y += fy * dt; vel[j].x -= fx * dt; vel[j].y -= fy * dt;
      }

      if (kG > 0) { for (let i = 0; i < N; i++) { vel[i].x += (-pos[i].x * kG) * dt; vel[i].y += (-pos[i].y * kG) * dt; } }
      for (let i = 0; i < N; i++) { pos[i].x += vel[i].x * dt; pos[i].y += vel[i].y * dt; }
    }
    draw();
  }

  // ---------- View / Render ----------
  function resize() {
    const rect = stage.getBoundingClientRect();
    viewW = Math.max(1, rect.width | 0); viewH = Math.max(1, rect.height | 0);
    dpr = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = viewW * dpr; canvas.height = viewH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  window.addEventListener('resize', resize);
  new ResizeObserver(resize).observe(stage);

  function worldToScreen(x, y) { const cx = viewW / 2, cy = viewH / 2; return { x: Math.round(cx + (x + cam.x) * cam.scale), y: Math.round(cy + (y + cam.y) * cam.scale) }; }
  function screenToWorld(sx, sy) { const cx = viewW / 2, cy = viewH / 2; return { x: (sx - cx) / cam.scale - cam.x, y: (sy - cy) / cam.scale - cam.y }; }
  function clearCanvas() { ctx.clearRect(0, 0, viewW, viewH); }

  function fitView(pad = 36) {
    if (!nodes.length) return;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (let i = 0; i < N; i++) {
      const p = pos[i]; if (!isFinite(p.x) || !isFinite(p.y)) continue;
      if (p.x < xmin) xmin = p.x; if (p.x > xmax) xmax = p.x;
      if (p.y < ymin) ymin = p.y; if (p.y > ymax) ymax = p.y;
    }
    if (!isFinite(xmin) || !isFinite(xmax) || !isFinite(ymin) || !isFinite(ymax)) return;
    const w = Math.max(1e-3, xmax - xmin), h = Math.max(1e-3, ymax - ymin);
    const sx = (viewW - pad * 2) / w, sy = (viewH - pad * 2) / h;
    cam.scale = clamp(Math.min(sx, sy), 0.2, 6);
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2; cam.x = -cx; cam.y = -cy;
    draw();
  }

  function draw() {
    clearCanvas(); if (!nodes.length) return;

    const showCross = toggleCross.checked;
    const alphaBase = Number(edgeAlpha.value) / 100;
    const gamma = Number(alphaGamma.value) / 100;
    const beta = Number(wExp.value) / 100;

    // edges
    for (const e of edgesW) {
      if (!toggleSelf.checked && e.a === e.b) continue;
      const i = indexOf.get(e.a), j = indexOf.get(e.b);
      const ca = clusterOf[e.a], cb = clusterOf[e.b];
      const intra = (ca === cb); if (!intra && !showCross) continue;

      if (e.a === e.b) {
        const p = worldToScreen(pos[i].x, pos[i].y);
        const rNode = Math.max(2, 2.0 * cam.scale);
        const rLoop = rNode + Math.max(0.7 * cam.scale, 0.9);
        const off = 0.3 * rLoop;
        ctx.save(); ctx.lineWidth = Math.max(1, 1.1 * cam.scale);
        ctx.strokeStyle = hsla(clusterColors[ca], 0.8);
        ctx.beginPath(); ctx.arc(p.x + off, p.y - off, rLoop, 0, TAU); ctx.stroke(); ctx.restore();
        continue;
      }

      const wEff = Math.pow(Math.max(0, e.n), beta);
      const wPow = Math.pow(Math.max(1e-4, wEff), Math.max(0.2, gamma * 1.6));
      const minA = intra ? 0.01 : 0.004;
      const a = clamp(minA + (alphaBase - minA) * wPow, minA, 1);
      const lw = 0.7 + 1.8 * wPow * cam.scale;
      ctx.lineWidth = Math.max(0.6, lw);

      ctx.strokeStyle = hsla(clusterColors[ca], a);
      const p1 = worldToScreen(pos[i].x, pos[i].y), p2 = worldToScreen(pos[j].x, pos[j].y);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
    }

    // nodes
    const r = Math.max(2, 3 * cam.scale);
    for (let i = 0; i < N; i++) {
      const p = worldToScreen(pos[i].x, pos[i].y);
      ctx.fillStyle = clusterColors[clusterOf[nodes[i]]];
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, TAU); ctx.fill();
    }

    // labels
    const pct = Number(labelPct.value); labelPctVal.textContent = pct + '%';
    const showCount = Math.max(3, Math.round(N * pct / 100));
    const top = nodes.slice().sort((a, b) => degrees[b] - degrees[a] || (a < b ? -1 : 1)).slice(0, showCount);
    if (cam.scale > 0.6 || pct >= 10) {
      ctx.font = `${Math.max(10, Math.floor(11 * cam.scale))}px system-ui, sans-serif`;
      ctx.textBaseline = 'middle'; ctx.fillStyle = '#e6edf8';
      for (const id of top) { const i = indexOf.get(id), p = worldToScreen(pos[i].x, pos[i].y); ctx.fillText(id, p.x + 6, p.y); }
    }

    // hover
    if (hoverIdx >= 0) {
      const nid = nodes[hoverIdx], neigh = adjacency.get(nid) || new Set();
      ctx.lineWidth = Math.max(1.3, 1.7 * cam.scale); ctx.strokeStyle = '#ffffffbb';
      for (const m of neigh) {
        const ia = hoverIdx, ib = indexOf.get(m);
        if (m === nid) {
          const p = worldToScreen(pos[ia].x, pos[ia].y);
          const rNode = Math.max(2, 2.0 * cam.scale);
          const rLoop = rNode + Math.max(0.7 * cam.scale, 0.9);
          const off = 0.3 * rLoop;
          ctx.beginPath(); ctx.arc(p.x + off, p.y - off, rLoop, 0, TAU); ctx.stroke();
          continue;
        }
        const p1 = worldToScreen(pos[ia].x, pos[ia].y), p2 = worldToScreen(pos[ib].x, pos[ib].y);
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      }
      const p = worldToScreen(pos[hoverIdx].x, pos[hoverIdx].y);
      ctx.strokeStyle = '#ffffffaa'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.x, p.y, r + 2, 0, TAU); ctx.stroke();
    }
  }

  function debounce(fn, ms = 120) { let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); }; }
  const rerunLayout = () => { try { setStatus('Laying out…'); } catch { } try { runLayout(); } catch (e) { console.error(e); } try { fitView(); } catch { } try { setStatus('Rendered ✓'); } catch { } };
  const rerunLayoutDebounced =
  (window.rerunLayoutDebounced ||= debounce(rerunLayout, 120));

  const side = document.getElementById('side');
  side.addEventListener('input', (e) => {
    const t = e.target;
    if (!(t instanceof HTMLInputElement || t instanceof HTMLSelectElement)) return;
    rerunLayoutDebounced();
  });
  side.addEventListener('change', (e) => {
    const t = e.target; if (!(t instanceof HTMLInputElement || t instanceof HTMLSelectElement)) return;
    rerunLayoutDebounced();
  });

  // Interaction
  let panning = false, lastX = 0, lastY = 0;
  canvas.addEventListener('mousedown', e => { panning = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('mouseup', () => panning = false);
  window.addEventListener('mousemove', e => {
    if (!nodes.length) return;
    if (panning) { cam.x += (e.clientX - lastX) / cam.scale; cam.y += (e.clientY - lastY) / cam.scale; lastX = e.clientX; lastY = e.clientY; draw(); return; }
    let best = -1, bestD = 18;
    for (let i = 0; i < N; i++) { const p = worldToScreen(pos[i].x, pos[i].y); const d = Math.hypot(p.x - e.clientX, p.y - e.clientY); if (d < bestD) { bestD = d; best = i; } }
    hoverIdx = best; draw();
  });
  canvas.addEventListener('wheel', e => {
    if (!nodes.length) return; e.preventDefault();
    const before = screenToWorld(e.clientX, e.clientY), zoom = Math.exp(-e.deltaY * 0.0012);
    cam.scale = clamp(cam.scale * zoom, 0.2, 6);
    const after = screenToWorld(e.clientX, e.clientY);
    cam.x += before.x - after.x; cam.y += before.y - after.y; draw();
  }, { passive: false });

  // Controls
  edgeAlpha.addEventListener('input', () => { edgeAlphaVal.textContent = (Number(edgeAlpha.value) / 100).toFixed(2); draw(); });
  alphaGamma.addEventListener('input', () => { alphaGammaVal.textContent = (Number(alphaGamma.value) / 100).toFixed(2); draw(); });
  labelPct.addEventListener('input', () => { labelPctVal.textContent = labelPct.value + '%'; draw(); });
  toggleCross.addEventListener('change', draw);
  toggleSelf.addEventListener('change', () => { buildClusterMeta(); runLayout(); });
  toggleClusterLayout.addEventListener('change', () => { runLayout(); });
  toggleSpectral.addEventListener('change', () => { runLayout(); });
  deterministic.addEventListener('change', () => { });
  seedInput.addEventListener('change', () => { });
  sepMult.addEventListener('input', () => { sepMultVal.textContent = (Number(sepMult.value) / 100).toFixed(2) + '×'; });
  interK.addEventListener('input', () => { interKVal.textContent = (Number(interK.value) / 100).toFixed(2); });
  iters.addEventListener('input', () => itersVal.textContent = iters.value);
  itersIntra.addEventListener('input', () => itersIntraVal.textContent = itersIntra.value);
  kSpring.addEventListener('input', () => kSpringVal.textContent = (Number(kSpring.value) / 100).toFixed(2));
  kRepel.addEventListener('input', () => kRepelVal.textContent = kRepel.value);
  kGravity.addEventListener('input', () => kGravityVal.textContent = (Number(kGravity.value) / 100).toFixed(2));
  wExp.addEventListener('input', () => wExpVal.textContent = (Number(wExp.value) / 100).toFixed(2));
  lenScale.addEventListener('input', () => lenScaleVal.textContent = (Number(lenScale.value) / 100).toFixed(2));
  wInfluence.addEventListener('input', () => wInfluenceVal.textContent = wInfluence.value + '%');
  wPercentile.addEventListener('input', () => { wPercentileVal.textContent = wPercentile.value + 'th'; buildClusterMeta(); runLayout(); });
  annChrInput.addEventListener('change', () => { ann.chr = normalizeChr(annChrInput.value); refreshAnnotations(); });
  binSizeInput.addEventListener('change', () => { ann.binSize = Math.max(1, Number(binSizeInput.value)||1); refreshAnnotations(); });
  annShowInput.addEventListener('change', () => { ann.show = annShowInput.checked; drawHeatmapOverlay(); });
  annUseColorInput.addEventListener('change', () => { ann.useColorFromFile = annUseColorInput.checked; drawHeatmapOverlay(); });
  annColorInput.addEventListener('change', () => { ann.defaultRGB = annColorInput.value || '0,255,0'; drawHeatmapOverlay(); });

  annFile.addEventListener('change', (e) => {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => {
      ann.rows = parseBEDPE(rd.result, ann.chr, ann.binSize);
      drawHeatmapOverlay(); // draw on current heatmap if present
    };
    rd.readAsText(f);
  });

  search.addEventListener('keydown', e => {
    if (e.key !== 'Enter' || !nodes.length) return;
    const q = search.value.trim(); if (!indexOf.has(q)) { setStatus(`Node "${q}" not found.`); return; }
    const i = indexOf.get(q); hoverIdx = i; const p = pos[i]; cam.x = -p.x; cam.y = -p.y; cam.scale = 1.8; draw();
  });

  function zoomToCluster(cid) {
    const ids = (clusters[cid] || []); if (!ids.length) return;
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const n of ids) { const i = indexOf.get(n); xmin = Math.min(xmin, pos[i].x); xmax = Math.max(xmax, pos[i].x); ymin = Math.min(ymin, pos[i].y); ymax = Math.max(ymax, pos[i].y); }
    const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2; cam.x = -cx; cam.y = -cy; cam.scale = 1.6; draw();
  }

  function init() {
    edgeAlphaVal.textContent = (Number(edgeAlpha.value) / 100).toFixed(2);
    alphaGammaVal.textContent = (Number(alphaGamma.value) / 100).toFixed(2);
    labelPctVal.textContent = labelPct.value + '%';
    itersVal.textContent = iters.value; itersIntraVal.textContent = itersIntra.value;
    kSpringVal.textContent = (Number(kSpring.value) / 100).toFixed(2);
    kRepelVal.textContent = kRepel.value; kGravityVal.textContent = (Number(kGravity.value) / 100).toFixed(2);
    wExpVal.textContent = (Number(wExp.value) / 100).toFixed(2);
    lenScaleVal.textContent = (Number(lenScale.value) / 100).toFixed(2);
    wInfluenceVal.textContent = wInfluence.value + '%';
    wPercentileVal.textContent = wPercentile.value + 'th';
    sepMultVal.textContent = (Number(sepMult.value) / 100).toFixed(2) + '×';
    interKVal.textContent = (Number(interK.value) / 100).toFixed(2);
    new ResizeObserver(() => resize()).observe(stage);
    resize();
  }
  window.addEventListener('load', init);
})();

window.addEventListener('error', (e) => {
  const box = document.getElementById('err');
  if (box) { box.style.display = 'block'; box.textContent = 'Uncaught error: ' + (e?.error?.stack || e?.message || e); }
});
