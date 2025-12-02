// state.js
export const hm = {
  svg: null,
  g: null,
  zoom: null,
  data: [],
  order: [],
  selection: null,
  zTransform: null,
  map: null,
  info: document.getElementById('hmInfo')
};

// Annotations (BEDPE-like)
export const ann = {
  rows: [],          // parsed rows filtered to current chromosome
  binSize: 1_000_000,
  chr: 'chr1',
  useColorFromFile: true,
  defaultRGB: '0,255,0',
  show: true
};

// TAD/Loop overlay state (derived from ann.rows)
export const tadLoop = {
  tads: [],   // {chr,x1,x2,y1,y2,color,features,id}
  loops: []   // {chr,a,b,color,score}
};
