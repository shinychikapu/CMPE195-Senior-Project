
/* === mgv Selection + Undo/Redo Patch ===
   Drop this <script src="selection_patch.js"></script> at the END of mgv.html
   right before </body>. Requires heatmapCanvas in DOM.
*/

(function(){

console.log("Selection Patch Loaded");

// ---------------------- STATE ----------------------------------
const state = {
  selection: { x:0, y:0, w:0, h:0 },
  history: [],
  future: []
};

function saveState(){
  state.history.push(JSON.parse(JSON.stringify(state.selection)));
  state.future = [];
}

function undo(){
  if(state.history.length === 0) return;
  state.future.push(JSON.parse(JSON.stringify(state.selection)));
  state.selection = state.history.pop();
  renderBox();
}
function redo(){
  if(state.future.length === 0) return;
  state.history.push(JSON.parse(JSON.stringify(state.selection)));
  state.selection = state.future.pop();
  renderBox();
}

// ---------------------- KEYBOARD ----------------------------------
document.addEventListener("keydown", e=>{
  if(e.ctrlKey && !e.shiftKey && e.key === "z") undo();
  if(e.ctrlKey && e.shiftKey && e.key.toLowerCase()==="z") redo();
});

// ---------------------- BUILD SELECTION BOX ----------------------------------
const heatmap = document.getElementById("heatmapCanvas");
if(!heatmap){ console.warn("heatmapCanvas not found"); return; }

const parent = heatmap.parentElement;

const box = document.createElement("div");
box.id = "mgvSelectionBox";
box.style.position="absolute";
box.style.border="2px solid #0af";
box.style.pointerEvents="auto";
box.style.display="none";
parent.appendChild(box);

// Four resize handles
const corners = ["nw","ne","sw","se"];
const handles = {};
corners.forEach(c=>{
  const h = document.createElement("div");
  h.className = "mgvHandle mgvHandle-"+c;
  h.style.position="absolute";
  h.style.width="12px";
  h.style.height="12px";
  h.style.background="#0af";
  h.style.cursor=c+"-resize";

  if(c.includes("n")) h.style.top="-6px";
  if(c.includes("s")) h.style.bottom="-6px";
  if(c.includes("w")) h.style.left="-6px";
  if(c.includes("e")) h.style.right="-6px";

  box.appendChild(h);
  handles[c]=h;
});

// ---------------------- RENDER ----------------------------------
function renderBox(){
  box.style.display="block";
  box.style.left = state.selection.x + "px";
  box.style.top  = state.selection.y + "px";
  box.style.width  = state.selection.w + "px";
  box.style.height = state.selection.h + "px";
}

// ---------------------- DRAGGING ----------------------------------
let dragging=false, offsetX=0, offsetY=0;

box.addEventListener("mousedown", e=>{
  if(e.target.classList.contains("mgvHandle")) return;
  dragging=true;
  offsetX=e.offsetX;
  offsetY=e.offsetY;
  saveState();
});

window.addEventListener("mousemove", e=>{
  if(!dragging) return;
  const r = heatmap.getBoundingClientRect();
  state.selection.x = e.clientX - r.left - offsetX;
  state.selection.y = e.clientY - r.top - offsetY;
  renderBox();
});
window.addEventListener("mouseup", ()=> dragging=false);

// ---------------------- RESIZING ----------------------------------
let resizing=false, resizeCorner=null;

Object.entries(handles).forEach(([corner, el])=>{
  el.addEventListener("mousedown", e=>{
    resizing=true;
    resizeCorner=corner;
    e.stopPropagation();
    saveState();
  });
});

window.addEventListener("mousemove", e=>{
  if(!resizing) return;
  const r = heatmap.getBoundingClientRect();
  const mx = e.clientX - r.left;
  const my = e.clientY - r.top;
  const sel = state.selection;

  if(resizeCorner.includes("n")){
    const diff = sel.y - my;
    sel.y = my;
    sel.h += diff;
  }
  if(resizeCorner.includes("s")){
    sel.h = my - sel.y;
  }
  if(resizeCorner.includes("w")){
    const diff = sel.x - mx;
    sel.x = mx;
    sel.w += diff;
  }
  if(resizeCorner.includes("e")){
    sel.w = mx - sel.x;
  }

  renderBox();
});

window.addEventListener("mouseup", ()=>{ resizing=false; resizeCorner=null; });

// ---------------------- PUBLIC API ----------------------------------
window.mgvSelectionPatch = {
  setSelection(x,y,w,h){
    state.selection={x,y,w,h};
    renderBox();
  },
  getSelection(){
    return JSON.parse(JSON.stringify(state.selection));
  },
  undo, redo
};

})();
