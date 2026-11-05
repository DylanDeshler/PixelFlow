"""
Interactive comparison of runs, one self-contained HTML file (no libraries, no network): FID (or IS / precision / recall)
vs images seen, vs training compute and vs parameters, for every run under --runs_root.

  run list    every run with in-loop evals, searchable; checked by default if it has evals at >= --min_epochs epochs
  legend      the checked runs; click one to hide / show it on every chart, double-click to show only it (again: all)
  charts      FID vs images seen (sample efficiency), vs training PFLOPs (images x 3 x forward FLOPs: compute efficiency),
              vs trainable parameters at a chosen epoch; hover a point for its numbers
  table       the visible runs' numbers (the same data as the charts)
A run keeps its colour + marker whatever else is selected (6 colours x 6 markers, by its position in the full list).

Parameters and forward GFLOPs per image come from config.json (train_jit.py records them since 2026-11-05) or, for older
runs, from rebuilding the model from its saved architecture (jit_common.model_cost, one image on CPU, a few seconds per
run), cached in {out}.cost_cache.json so reruns are fast. Runs whose model can't be measured still appear on the
images chart. Intermediate evals are valid shorter runs here: JiT's lr is constant after warm-up and its EMAs are fixed.

$ python plot_runs_html.py                                    # runs/in1k_128 -> runs/plots/runs.html
$ python plot_runs_html.py --runs_root=runs/in100_128 --out=runs/plots/in100.html
"""

import os
import glob
import json
import time

import plot_style as ps
from util.progress import bar

# -----------------------------------------------------------------------------
runs_root = 'runs/in1k_128' # scanned recursively for run directories (config.json + diagnostics/diag_*.json)
min_epochs = 5.0 # runs with evals at >= this many epochs start checked
img_size = 128 # for runs whose config.json predates img_size / num_classes being recorded
num_classes = 1000
out = 'runs/plots/runs.html'
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read())
# -----------------------------------------------------------------------------

METRICS = {'val/fid': 'FID-10k', 'val/is': 'Inception score', 'val/precision': 'precision', 'val/recall': 'recall'}

def find_runs(root):
    found = []
    for cfg in glob.glob(os.path.join(root, '**', 'config.json'), recursive=True):
        d = os.path.dirname(cfg)
        if os.path.basename(d) == 'previous' or '/previous/' in d: # archived configs of resumed runs
            continue
        if glob.glob(os.path.join(d, 'diagnostics', 'diag_*.json')):
            found.append(d)
    return sorted(found)

def measure(d, res, cache):
    # (params, fwd_gflops) or (None, None): config.json, then the cache, then a rebuild
    if 'fwd_gflops' in res:
        return res.get('params'), res['fwd_gflops']
    key = f"{d}:{os.path.getmtime(os.path.join(d, 'config.json'))}"
    if key not in cache:
        try:
            import torch
            from jit_common import build_denoiser, model_cost
            with torch.device('cpu'):
                net = build_denoiser(res['model_name'], res.get('img_size', img_size), res.get('num_classes', num_classes),
                                     noise_scale=res.get('noise_scale', 1.0), arch=res.get('arch')).net
            c = model_cost(net, res.get('img_size', img_size), res.get('num_classes', num_classes))
            cache[key] = [c['params'], c['fwd_gflops']]
        except Exception as e: # other families' forwards may not fit the probe
            print(f'  {d}: cost not measured ({type(e).__name__}: {str(e)[:120]})')
            cache[key] = [None, None]
    return tuple(cache[key])

def label_tokens(d, root, arch):
    # the run's name as tokens: model dir (in-context length made explicit: a preset's 32 tokens leave no trace in the
    # name) + run name. The page drops the tokens every visible run shares, so labels show only what differs
    rel = os.path.relpath(d, root).split(os.sep)
    model, leaf = (rel[0], rel[-1]) if len(rel) > 1 else ('', rel[0])
    toks = [t for t in model.split('_') if t and not (t.startswith('icl') and t[3:].isdigit())]
    if arch and 'in_context_len' in arch:
        toks.append(f"icl{arch['in_context_len']}")
    return toks + [t for t in leaf.split('_') if t]

dirs = find_runs(runs_root)
assert dirs, f'no runs with evals under {runs_root}'
cache_path = os.path.splitext(out)[0] + '.cost_cache.json'
cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
runs = []
for d in bar(dirs, f'reading {len(dirs)} runs'):
    cfg = json.load(open(os.path.join(d, 'config.json')))
    res = cfg.get('resolved', {})
    ipe = res.get('iters_per_epoch')
    batch = res.get('effective_batch_size') or cfg.get('batch_size', 0) * cfg.get('gradient_accumulation_steps', 1)
    if not ipe or not batch:
        continue
    params, gfl = measure(d, res, cache)
    evals = []
    for f in glob.glob(os.path.join(d, 'diagnostics', 'diag_*.json')):
        it = int(f.rsplit('_', 1)[1][:-5])
        try:
            sc = json.load(open(f)).get('scalars', {})
        except json.JSONDecodeError:
            continue
        if 'val/fid' not in sc:
            continue
        e = {'epoch': round(it / ipe, 3), 'images': it * batch / 1e6, 'pflops': it * batch * 3 * gfl / 1e6 if gfl else None}
        e.update({k.split('/')[1]: sc[k] for k in METRICS if k in sc})
        evals.append(e)
    if not evals:
        continue
    evals.sort(key=lambda e: e['epoch'])
    ips = None
    log = os.path.join(d, 'train_log.jsonl')
    if os.path.exists(log):
        try:
            v = sorted(r['imgs_per_sec'] for r in map(json.loads, open(log)) if 'imgs_per_sec' in r and r.get('epoch', 0) > 0.5)
            ips = round(v[len(v) // 2]) if v else None
        except json.JSONDecodeError:
            pass
    runs.append({'path': os.path.relpath(d, runs_root), 'tokens': label_tokens(d, runs_root, res.get('arch')), 'params': params, 'gflops': gfl,
                 'ips': ips, 'evals': evals, 'max_epoch': evals[-1]['epoch'], 'default': evals[-1]['epoch'] >= min_epochs - 0.25})
os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
with open(cache_path, 'w') as f:
    json.dump(cache, f)

data = {'root': runs_root, 'generated': time.strftime('%Y-%m-%d %H:%M'), 'runs': runs, 'metrics': {k.split('/')[1]: v for k, v in METRICS.items()},
        'colors': ps.CATEGORICAL, 'surface': ps.SURFACE, 'ink': ps.INK, 'ink2': ps.INK_2, 'grid': ps.GRID}

HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Runs: __ROOT__</title>
<style>
:root { --surface: __SURFACE__; --ink: __INK__; --ink2: __INK2__; --grid: __GRID__; --line: #d6d5cf; }
* { box-sizing: border-box; }
body { margin: 0; font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--ink); background: var(--surface); }
header { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line); }
header h1 { font-size: 15px; margin: 0 8px 0 0; font-weight: 600; }
header .muted, .muted { color: var(--ink2); }
select, input[type=search], button { font: inherit; color: var(--ink); background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 4px 8px; }
button { cursor: pointer; } button:hover { border-color: var(--ink2); }
main { display: grid; grid-template-columns: 330px 1fr; min-height: calc(100vh - 50px); }
#side { border-right: 1px solid var(--line); padding: 10px; display: flex; flex-direction: column; gap: 8px; }
#side .row { display: flex; gap: 6px; }
#runlist { overflow: auto; flex: 1; max-height: calc(100vh - 150px); }
#runlist label { display: flex; gap: 6px; align-items: flex-start; padding: 3px 2px; border-radius: 4px; cursor: pointer; word-break: break-all; }
#runlist label:hover { background: #f1f0ec; }
#runlist svg, #legend svg { flex-shrink: 0; margin-top: 1px; }
#runlist .ep { margin-left: auto; color: var(--ink2); white-space: nowrap; padding-left: 6px; }
#content { padding: 10px 16px; min-width: 0; }
#legend { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-bottom: 8px; user-select: none; }
#legend .item { display: flex; gap: 5px; align-items: center; cursor: pointer; padding: 2px 4px; border-radius: 4px; }
#legend .item:hover { background: #f1f0ec; }
#legend .item.off { opacity: 0.35; }
#charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
.chart h2 { font-size: 13px; font-weight: 600; margin: 4px 0 2px; }
svg text { fill: var(--ink2); font-size: 11px; }
#tip { position: fixed; pointer-events: none; background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px;
       box-shadow: 0 2px 8px rgba(0,0,0,.08); display: none; max-width: 420px; word-break: break-all; z-index: 10; }
table { border-collapse: collapse; margin-top: 14px; width: 100%; }
th, td { text-align: right; padding: 3px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; white-space: normal; word-break: break-all; }
th { color: var(--ink2); font-weight: 600; }
details summary { cursor: pointer; color: var(--ink2); margin-top: 12px; }
</style></head><body>
<header>
  <h1>Runs</h1><span class="muted">__ROOT__ &middot; generated __GENERATED__</span>
  <label>metric <select id="metric"></select></label>
  <label>parameters chart at epoch <select id="atep"></select></label>
  <span class="muted">legend: click hides / shows a run, double-click shows only it</span>
</header>
<main>
<div id="side">
  <input type="search" id="search" placeholder="filter runs (space-separated terms)">
  <div class="row"><button id="all">all shown</button><button id="none">none</button><button id="def">default (&ge; __MINEP__ epochs)</button></div>
  <div id="runlist"></div>
</div>
<div id="content">
  <div id="legend"></div>
  <div id="charts">
    <div class="chart"><h2>vs images seen</h2><svg id="c_images"></svg></div>
    <div class="chart"><h2>vs training compute</h2><svg id="c_flops"></svg></div>
    <div class="chart"><h2>vs parameters</h2><svg id="c_params"></svg></div>
  </div>
  <details><summary>table of the visible runs</summary><div id="table"></div></details>
</div>
</main>
<div id="tip"></div>
<script>
const D = __DATA__;
const MARKERS = ['circle', 'square', 'triangle', 'diamond', 'down', 'plus'];
D.runs.forEach((r, i) => { r.color = D.colors[i % D.colors.length]; r.marker = MARKERS[Math.floor(i / D.colors.length) % MARKERS.length]; r.i = i; });
const state = { selected: new Set(D.runs.filter(r => r.default).map(r => r.i)), hidden: new Set(), metric: 'fid', atep: null };
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const fmt = (v, d = 2) => v == null ? '–' : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : (+v).toFixed(d));

function markerPath(kind, x, y, s) { // s = half size
  switch (kind) {
    case 'square': return `M${x-s},${y-s}h${2*s}v${2*s}h${-2*s}z`;
    case 'triangle': return `M${x},${y-s*1.15}L${x+s*1.1},${y+s*0.8}H${x-s*1.1}z`;
    case 'down': return `M${x},${y+s*1.15}L${x+s*1.1},${y-s*0.8}H${x-s*1.1}z`;
    case 'diamond': return `M${x},${y-s*1.3}L${x+s*1.1},${y}L${x},${y+s*1.3}L${x-s*1.1},${y}z`;
    case 'plus': { const t = s * 0.45; return `M${x-t},${y-s}h${2*t}v${s-t}h${s-t}v${2*t}h${t-s}v${s-t}h${-2*t}v${t-s}h${t-s}v${-2*t}h${s-t}z`; }
    default: return `M${x-s},${y}a${s},${s} 0 1,0 ${2*s},0a${s},${s} 0 1,0 ${-2*s},0`;
  }
}
function swatch(r) { const s = el('svg', { width: 16, height: 16 }); el('path', { d: markerPath(r.marker, 8, 8, 5), fill: r.color }, s); return s; }
function ticks(lo, hi, n = 5) {
  const span = hi - lo || Math.abs(hi) || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10)); return out;
}
function relabel() { // label = the tokens not shared by every checked run ('base' if none): stable while the legend hides runs
  const vis = D.runs.filter(r => state.selected.has(r.i)); const common = vis.length > 1 ? vis.map(r => new Set(r.tokens)).reduce((a, b) => new Set([...a].filter(t => b.has(t)))) : new Set();
  for (const r of D.runs) r.label = r.tokens.filter(t => !common.has(t)).join(' ') || 'base';
}
function visibleRuns() { return D.runs.filter(r => state.selected.has(r.i) && !state.hidden.has(r.i)); }

// one chart: series = [{run, pts: [{x, y, ev}]}], line = connect the points
function chart(svg, series, xlabel, ylabel, line, labels) {
  svg.innerHTML = '';
  const W = svg.parentNode.clientWidth || 480, H = 330, m = { l: 52, r: 16, t: 10, b: 40 };
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const pts = series.flatMap(s => s.pts);
  if (!pts.length) { el('text', { x: W / 2, y: H / 2, 'text-anchor': 'middle' }, svg).textContent = 'no data for the visible runs'; return; }
  let [x0, x1] = [Math.min(...pts.map(p => p.x)), Math.max(...pts.map(p => p.x))];
  let [y0, y1] = [Math.min(...pts.map(p => p.y)), Math.max(...pts.map(p => p.y))];
  const minSpan = Math.abs(x1) * 0.1 || 1; if (x1 - x0 < minSpan) { const c = (x0 + x1) / 2; x0 = c - minSpan / 2; x1 = c + minSpan / 2; }
  const px = (x1 - x0) * 0.04, py = (y1 - y0) * 0.06 || 1; x0 -= px; x1 += px; y0 -= py; y1 += py;
  const sx = x => m.l + (x - x0) / (x1 - x0) * (W - m.l - m.r), sy = y => H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
  for (const v of ticks(y0, y1)) { el('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: D.grid }, svg); el('text', { x: m.l - 6, y: sy(v) + 4, 'text-anchor': 'end' }, svg).textContent = v; }
  for (const v of ticks(x0, x1, 6)) { el('line', { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, stroke: D.grid }, svg); el('text', { x: sx(v), y: H - m.b + 15, 'text-anchor': 'middle' }, svg).textContent = v; }
  el('text', { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle' }, svg).textContent = xlabel;
  el('text', { x: 12, y: (m.t + H - m.b) / 2, 'text-anchor': 'middle', transform: `rotate(-90 12 ${(m.t + H - m.b) / 2})` }, svg).textContent = ylabel;
  for (const s of series) {
    if (line && s.pts.length > 1) el('path', { d: s.pts.map((p, k) => `${k ? 'L' : 'M'}${sx(p.x)},${sy(p.y)}`).join(''), fill: 'none', stroke: s.run.color, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
  }
  for (const s of series) for (const p of s.pts) {
    el('path', { d: markerPath(s.run.marker, sx(p.x), sy(p.y), 4.5), fill: s.run.color, stroke: D.surface, 'stroke-width': 2 }, svg);
    const hit = el('circle', { cx: sx(p.x), cy: sy(p.y), r: 10, fill: 'transparent', style: 'cursor:pointer' }, svg);
    hit.addEventListener('mousemove', ev => showTip(ev, s.run, p.ev));
    hit.addEventListener('mouseleave', () => { $('tip').style.display = 'none'; });
    if (labels) { const right = sx(p.x) > W * 0.6; // near the right edge: label to the left of the point
      el('text', { x: sx(p.x) + (right ? -8 : 8), y: sy(p.y) - 6, 'text-anchor': right ? 'end' : 'start' }, svg).textContent = s.run.label.length > 40 ? s.run.label.slice(0, 38) + '…' : s.run.label; }
  }
}
function showTip(ev, r, e) {
  const t = $('tip');
  t.innerHTML = `<b>${r.label}</b><br>epoch ${fmt(e.epoch, 1)} &middot; ${fmt(e.images, 1)}M images` + (e.pflops != null ? ` &middot; ${fmt(e.pflops, 0)} PFLOPs` : '') +
    `<br>FID ${fmt(e.fid)}` + (e.is != null ? ` &middot; IS ${fmt(e.is)}` : '') + (e.precision != null ? ` &middot; precision ${fmt(e.precision, 3)} &middot; recall ${fmt(e.recall, 3)}` : '') +
    `<br><span class="muted">${r.params ? (r.params / 1e6).toFixed(2) + 'M params' : 'params ?'} &middot; ${r.gflops ? r.gflops.toFixed(2) + ' GFLOP/img fwd' : 'FLOPs ?'}${r.ips ? ' &middot; ' + r.ips + ' img/s' : ''}</span>`;
  t.style.display = 'block';
  const x = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 8), y = Math.min(ev.clientY + 14, window.innerHeight - t.offsetHeight - 8);
  t.style.left = x + 'px'; t.style.top = y + 'px';
}
function nearest(r, ep) { let best = null; for (const e of r.evals) if (Math.abs(e.epoch - ep) <= 0.5 && (!best || Math.abs(e.epoch - ep) < Math.abs(best.epoch - ep))) best = e; return best; }

function render() {
  const mk = state.metric, ylab = D.metrics[mk];
  const vis = visibleRuns();
  relabel();
  const series = (fx) => vis.map(r => ({ run: r, pts: r.evals.filter(e => e[mk] != null && fx(e) != null).map(e => ({ x: fx(e), y: e[mk], ev: e })) })).filter(s => s.pts.length);
  chart($('c_images'), series(e => e.images), 'images seen (M)', ylab, true);
  chart($('c_flops'), series(e => e.pflops), 'training compute (PFLOPs: images × 3 × forward FLOPs)', ylab, true);
  const ps = vis.map(r => { const e = nearest(r, state.atep); return e && r.params && e[mk] != null ? { run: r, pts: [{ x: r.params / 1e6, y: e[mk], ev: e }] } : null; }).filter(Boolean);
  chart($('c_params'), ps, 'trainable parameters (M)', `${ylab} at epoch ${state.atep}`, false, ps.length <= 4);
  // legend: the selected runs
  const lg = $('legend'); lg.innerHTML = '';
  for (const r of D.runs.filter(r => state.selected.has(r.i))) {
    const it = document.createElement('div'); it.className = 'item' + (state.hidden.has(r.i) ? ' off' : '');
    it.appendChild(swatch(r)); it.appendChild(document.createTextNode(r.label)); it.title = r.path;
    let timer = null;
    it.addEventListener('click', () => { clearTimeout(timer); timer = setTimeout(() => { state.hidden.has(r.i) ? state.hidden.delete(r.i) : state.hidden.add(r.i); render(); }, 220); });
    it.addEventListener('dblclick', () => { clearTimeout(timer); const sel = [...state.selected];
      const only = sel.every(i => i === r.i || state.hidden.has(i)) && !state.hidden.has(r.i);
      state.hidden = new Set(only ? [] : sel.filter(i => i !== r.i)); render(); });
    lg.appendChild(it);
  }
  // table
  const eps = [...new Set(vis.flatMap(r => r.evals.map(e => Math.round(e.epoch))))].sort((a, b) => a - b);
  let h = `<table><tr><th>run</th><th>params</th><th>GFLOP/img</th><th>img/s</th>${eps.map(e => `<th>${ylab} @${e}</th>`).join('')}</tr>`;
  for (const r of vis) h += `<tr><td title="${r.path}">${r.label}</td><td>${r.params ? (r.params / 1e6).toFixed(2) + 'M' : '–'}</td><td>${fmt(r.gflops)}</td><td>${r.ips || '–'}</td>` +
    eps.map(ep => { const e = nearest(r, ep); return `<td>${e ? fmt(e[mk], mk === 'fid' || mk === 'is' ? 2 : 3) : ''}</td>`; }).join('') + '</tr>';
  $('table').innerHTML = h + '</table>';
}
function renderList() {
  const terms = $('search').value.toLowerCase().split(/\s+/).filter(Boolean);
  const box = $('runlist'); box.innerHTML = '';
  for (const r of D.runs) {
    if (terms.length && !terms.every(t => r.path.toLowerCase().includes(t))) continue;
    const lab = document.createElement('label'); lab.title = r.path;
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = state.selected.has(r.i);
    cb.addEventListener('change', () => { cb.checked ? state.selected.add(r.i) : (state.selected.delete(r.i), state.hidden.delete(r.i)); render(); });
    lab.appendChild(cb); lab.appendChild(swatch(r)); lab.appendChild(document.createTextNode(r.path.replace(/(^|\/)muon_/, '$1')));
    const ep = document.createElement('span'); ep.className = 'ep'; ep.textContent = fmt(r.max_epoch, 0) + ' ep'; lab.appendChild(ep);
    box.appendChild(lab);
  }
}
for (const [k, v] of Object.entries(D.metrics)) { const o = document.createElement('option'); o.value = k; o.textContent = v; $('metric').appendChild(o); }
const allEps = [...new Set(D.runs.flatMap(r => r.evals.map(e => Math.round(e.epoch))))].filter(e => e > 0).sort((a, b) => a - b);
for (const e of allEps) { const o = document.createElement('option'); o.value = e; o.textContent = e; $('atep').appendChild(o); }
state.atep = allEps.includes(20) ? 20 : allEps[allEps.length - 1]; $('atep').value = state.atep;
$('metric').addEventListener('change', e => { state.metric = e.target.value; render(); });
$('atep').addEventListener('change', e => { state.atep = +e.target.value; render(); });
$('search').addEventListener('input', renderList);
const visibleInList = () => D.runs.filter(r => { const t = $('search').value.toLowerCase().split(/\s+/).filter(Boolean); return t.every(x => r.path.toLowerCase().includes(x)); });
$('all').addEventListener('click', () => { visibleInList().forEach(r => state.selected.add(r.i)); state.hidden.clear(); renderList(); render(); });
$('none').addEventListener('click', () => { visibleInList().forEach(r => { state.selected.delete(r.i); state.hidden.delete(r.i); }); renderList(); render(); });
$('def').addEventListener('click', () => { state.selected = new Set(D.runs.filter(r => r.default).map(r => r.i)); state.hidden.clear(); renderList(); render(); });
window.addEventListener('resize', render);
relabel(); renderList(); render();
</script></body></html>'''

page = HTML
for k, v in {'__ROOT__': runs_root, '__GENERATED__': data['generated'], '__MINEP__': f'{min_epochs:g}', '__SURFACE__': ps.SURFACE,
             '__INK__': ps.INK, '__INK2__': ps.INK_2, '__GRID__': ps.GRID, '__DATA__': json.dumps(data).replace('</', '<\\/')}.items():
    page = page.replace(k, v)
with open(out, 'w') as f:
    f.write(page)
n_def = sum(r['default'] for r in runs)
print(f"wrote {out}: {len(runs)} runs ({n_def} with >= {min_epochs:g} epochs shown by default), "
      f"{sum(r['gflops'] is None for r in runs)} without a FLOP / parameter count")
