// heatmap.js
import { hm, ann, tadLoop } from './state.js';

// ---- Helpers shared with graph.js ----
export function normalizeChr(s) {
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
  const [r, g, b] = m.slice(1).map(Number).map(v => Math.max(0, Math.min(255, v)));
  return `rgb(${r},${g},${b})`;
}

export function bpToBin(bp, binSize) {
  const x = Number(bp);
  const s = Number(binSize) || 1;
  return Math.floor(x / s) * s;
}

// Parse BEDPE-like annotation file -> rows used for TADs/loops
export function parseBEDPE(text, targetChr, binSize) {
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

// ---- RAW → heatmap ----
function parseRAW(text) {
  const out = [];
  const lines = String(text || '').trim().split(/\r?\n/);
  for (const line of lines) {
    const p = line.trim().split(/\s+/);
    if (p.length < 2) continue;
    const i = +p[0], j = +p[1], v = p[2] ? +p[2] : 1;
    if (Number.isFinite(i) && Number.isFinite(j) && Number.isFinite(v)) {
      out.push({ i, j, value: v });
    }
  }
  return out;
}

export function drawHeatmap(rawText) {
  const triples = parseRAW(rawText);
  if (!triples.length) {
    if (hm.info) hm.info.textContent = 'No valid i j value lines.';
    return;
  }

  // symmetric map + indices
  const set = new Set();
  const map = new Map();
  let vmax = 0;
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
    .attr('x', d => x(d.i))
    .attr('y', d => y(d.j))
    .attr('width', x.bandwidth())
    .attr('height', y.bandwidth())
    .attr('fill', d => color(Math.log1p(d.value)));

  hm.zoom = d3.zoom()
    .scaleExtent([0.5, 12])
    .on('zoom', ev => {
      hm.g.attr('transform', ev.transform);
      hm.zTransform = ev.transform;
    });

  d3.select('#hmPan')
    .on('click', () => {
      disableBoxSelect();
      hm.svg.call(hm.zoom.transform, hm.zTransform || d3.zoomIdentity);
    });

  d3.select('#hmSelect')
    .on('click', () => enableBoxSelect(x, y));

  d3.select('#hmClear')
    .on('click', () => {
      if (hm.selection) {
        hm.selection.remove();
        hm.selection = null;
      }
    });

  hm.svg.call(hm.zoom).on('dblclick.zoom', null);

  // after the heatmap cells are created
  splitTADsAndLoops();
  drawHeatmapOverlay();

  if (hm.info) {
    hm.info.textContent = `Matrix: ${hm.order.length}×${hm.order.length} • cells drawn: ${grid.length}`;
  }
}

// Called when chr/binSize/color options change
export function refreshAnnotations() {
  // Minimal approach: re-draw overlay with current ann.rows + options
  drawHeatmapOverlay();
}

export function splitTADsAndLoops() {
  const T = [];
  const L = [];

  for (const r of ann.rows) {
    // consider TAD if x-range ~ y-range (near diagonal)
    const isTAD =
      Math.abs((r.x2 - r.x1) - (r.y2 - r.y1)) <= ann.binSize * 0.5 &&
      Math.abs(r.x1 - r.y1) <= ann.binSize * 0.5 &&
      Math.abs(r.x2 - r.y2) <= ann.binSize * 0.5;

    const cssColor = ann.useColorFromFile
      ? rgbOrFallback(r.color, ann.defaultRGB)  // -> "rgb(255,255,0)"
      : `rgb(${ann.defaultRGB})`;

    if (isTAD) {
      T.push({
        ...r,
        id: `${r.chr}:${r.x1}-${r.x2}`,
        cssColor
      });
    } else {
      const a = Math.min(r.x1, r.x2);
      const b = Math.min(r.y1, r.y2);
      const score = (r.features && r.features.length) ? r.features[0] : 1;
      L.push({ chr: r.chr, a, b, color: cssColor, score });
    }
  }

  tadLoop.tads = T;
  tadLoop.loops = L;
}


export function drawHeatmapOverlay() {
  if (!hm.g) return;
  hm.g.selectAll('.annbox').remove();
  if (!ann.show || !ann.rows.length) return;

  // Figure out the pixel extent of the heatmap
  const gbox = hm.g.node().getBBox();
  const width = gbox.width;
  const height = gbox.height;

  const x = d3.scaleBand().domain(hm.order).range([0, width]).padding(0);
  const y = d3.scaleBand().domain(hm.order).range([0, height]).padding(0);

  const bw = x.bandwidth();
  const bh = y.bandwidth();

  // Map annotation rows → rectangles in pixel space
  const data = ann.rows
    .map(r => {
      const xi = r.x1;
      const yi = r.y1;
      const xj = r.x2;
      const yj = r.y2;

      const xiPix = x(xi);
      const yiPix = y(yi);
      const xjPix = x(xj);
      const yjPix = y(yj);

      if (xiPix == null || yiPix == null || xjPix == null || yjPix == null) {
        return null;
      }

      const x0 = Math.min(xiPix, xjPix);
      const y0 = Math.min(yiPix, yjPix);
      const w = Math.abs(xjPix - xiPix) + bw;
      const h = Math.abs(yjPix - yiPix) + bh;

      // Use precomputed TAD color if present, otherwise derive it
      const color = r.cssColor || (
        ann.useColorFromFile
          ? rgbOrFallback(r.color, ann.defaultRGB)
          : `rgb(${ann.defaultRGB})`
      );

      return { x: x0, y: y0, w, h, color, r };
    })
    .filter(Boolean);

  // Draw rectangles
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
      // Same click → build-graph logic you had before
      const box = { x: d.x, y: d.y, w: d.w, h: d.h };
      const picked = [];
      for (const i of hm.order) {
        const xi = x(i);
        for (const j of hm.order) {
          const yj = y(j);
          const hit = !(
            xi > box.x + box.w ||
            xi + bw < box.x ||
            yj > box.y + box.h ||
            yj + bh < box.y
          );
          if (hit) {
            const v = hm.map.get(i + ',' + j);
            if (v != null) picked.push(`${i}\t${j}\t${v}`);
          }
        }
      }
      if (picked.length && window.build) {
        window.build(picked.join('\n'));
      }
    });
}


// Box-select to send RAW subset to graph
function enableBoxSelect(x, y) {
  hm.svg.on('.zoom', null);
  if (hm.selection) {
    hm.selection.remove();
    hm.selection = null;
  }
  let start = null;

  hm.svg.on('mousedown.box', (ev) => {
    const p = d3.pointer(ev, hm.g.node());
    start = { x: p[0], y: p[1] };
    hm.selection = hm.g.append('rect')
      .attr('x', start.x)
      .attr('y', start.y)
      .attr('width', 0)
      .attr('height', 0)
      .attr('class', 'selection');
  });

  hm.svg.on('mousemove.box', (ev) => {
    if (!hm.selection || !start) return;
    const p = d3.pointer(ev, hm.g.node());
    const x0 = Math.min(start.x, p[0]);
    const y0 = Math.min(start.y, p[1]);
    const w = Math.abs(p[0] - start.x);
    const h = Math.abs(p[1] - start.y);
    hm.selection
      .attr('x', x0)
      .attr('y', y0)
      .attr('width', w)
      .attr('height', h);
  });

  hm.svg.on('mouseup.box', () => {
    if (!hm.selection) return;
    const box = hm.selection.node().getBBox();
    const picked = [];
    const bw = x.bandwidth(), bh = y.bandwidth();

    for (const i of hm.order) {
      const xi = x(i);
      for (const j of hm.order) {
        const yj = y(j);
        const hit = !(
          xi > box.x + box.width ||
          xi + bw < box.x ||
          yj > box.y + box.height ||
          yj + bh < box.y
        );
        if (hit) {
          const v = hm.map.get(i + ',' + j);
          if (v != null) picked.push(`${i}\t${j}\t${v}`);
        }
      }
    }
    if (picked.length && window.build) {
      window.build(picked.join('\n'));
    }
    disableBoxSelect();
  });
}

function disableBoxSelect() {
  hm.svg
    .on('mousedown.box', null)
    .on('mousemove.box', null)
    .on('mouseup.box', null);

  if (hm.zoom) hm.svg.call(hm.zoom);
}
