"""Live local server version of inspect_checkpoints.py -- genuinely on-demand
generation instead of a precomputed grid: pick ANY param combo / trajectory /
start-frame / rollout length in the browser and it runs the rollout on the
GPU right then, no precompute step, no size ceiling from embedding data.

The static HTML report (inspect_checkpoints.py) is still the right tool for
sharing a specific finding as a link. This is the right tool for browsing --
"let me just look at a bunch of stuff in the validation set" -- since a
self-contained artifact fundamentally can't do that (no backend it's allowed
to call). This one *is* the backend, running on your own machine against
your own GPU, so it has no such restriction.

Usage:
    python3 src/niko/eval/inspect_server.py \\
        --model "winner_roll16|configs/context_cond_helmholtz_rotation_roll16.yaml|/path/to/winner_ep7.pt" \\
        --model "baseline_roll16|configs/context_cond_helmholtz_roll16.yaml|/path/to/baseline_ep7.pt" \\
        --data-dir /home/sl8rv/datasets/rayleigh_benard/data \\
        --context-frames 6 --device cuda:1 --port 8765

Then open http://localhost:8765 . Models load once at startup and stay
resident; each rollout request reuses them. --context-frames is fixed at
startup (baked into the encoder's architecture) -- everything else (param
combo, trajectory, start frame, rollout length) is a live control.

Single-threaded on purpose: one GPU, one model forward pass at a time, no
point pretending otherwise with a thread pool.
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dataloader as dl
from eval.inspect_checkpoints import (
    CHANNEL_NAMES, DIAG_KEYS, METRICS,
    _load_model, _parse_model_arg, _fetch_window, build_report_data,
)

STATE = {}  # populated in main(): models, args, param_combos, loader_cache


def _scan_param_combos(data_dir: str, valid_subdir: str = "valid"):
    """One-time directory scan: every file's (param tuple, n_traj, time_steps),
    read directly from HDF5 (cheap -- shapes and a couple of scalars, no field
    data loaded). This is what populates the param-combo dropdown."""
    valid_dir = os.path.join(data_dir, valid_subdir)
    combos = []
    for path in sorted(glob.glob(os.path.join(valid_dir, "*.hdf5"))):
        params = dl.read_params_from_h5(path)
        if params is None:
            continue
        with h5py.File(path, "r") as f:
            shape = f["t0_fields/buoyancy"].shape  # (n_traj, T, x, y)
            n_traj, time_steps = int(shape[0]), int(shape[1])
        combos.append({"params": list(params), "n_traj": n_traj, "time_steps": time_steps})
    return combos


def _get_val_loader(param_tuple, rollout_steps: int):
    """LRU-ish cache: rebuilding a loader just re-derives the valid-window
    list from an already-known T (cheap), but no reason to redo it on every
    request for the same (param, rollout_steps) pair."""
    key = (param_tuple, rollout_steps)
    cache = STATE["loader_cache"]
    if key in cache:
        return cache[key]
    args = STATE["args"]
    _, val_loader, _ = dl.create_param_dataloaders(
        args.data_dir, batch_size=1, context_frames=args.context_frames,
        predict_frames=rollout_steps, num_workers=0, param_choices=[param_tuple],
        shuffle_val=False,
    )
    if len(cache) >= 12:
        cache.pop(next(iter(cache)))
    cache[key] = val_loader
    return val_loader


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        print(f"[server] {self.address_string()} {fmt % a}")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._send_html(FRONTEND_HTML)
            elif parsed.path == "/api/meta":
                self._handle_meta()
            elif parsed.path == "/api/rollout":
                self._handle_rollout(parse_qs(parsed.query))
            else:
                self._send_json({"error": f"unknown path {parsed.path}"}, status=404)
        except Exception as e:
            traceback.print_exc()
            self._send_json({"error": str(e)}, status=500)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_meta(self):
        self._send_json({
            "labels": list(STATE["models"].keys()),
            "paramCombos": STATE["param_combos"],
            "contextFrames": STATE["args"].context_frames,
            "channels": CHANNEL_NAMES,
            "metrics": list(METRICS.keys()),
            "diagKeys": [{"key": k, "title": t} for k, t in DIAG_KEYS],
        })

    def _handle_rollout(self, q):
        t0 = time.time()
        param_tuple = tuple(float(x) for x in q["params"][0].split(","))
        traj_id = int(q["traj_id"][0])
        start_frame = int(q["start_frame"][0])
        rollout_steps = int(q["rollout_steps"][0])
        args = STATE["args"]

        val_loader = _get_val_loader(param_tuple, rollout_steps)
        xb, yb, bparams = _fetch_window(val_loader, traj_id, start_frame, args.context_frames, rollout_steps)
        xb, yb = xb.to(args.device), yb.to(args.device)
        frames, metrics, diagnostics = build_report_data(
            STATE["models"], xb, yb, args.device, args.render_scale,
            args.diag_kappa, args.diag_nu, args.diag_g,
        )
        end_frame = start_frame + args.context_frames + rollout_steps - 1
        self._send_json({
            "frames": frames, "metrics": metrics, "diagnostics": diagnostics,
            "frameRange": f"{start_frame}-{end_frame}",
            "genSeconds": round(time.time() - t0, 2),
        })


# ---------------------------------------------------------------------------
# Frontend -- same visual identity as inspect_checkpoints.py's static report,
# but every control is fetch()-backed instead of reading precomputed JSON.
# ---------------------------------------------------------------------------

FRONTEND_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Rollout Inspector (live)</title>
<style>
:root {
  --bg: #eef1f2; --panel: #ffffff; --border: #d9dfe1; --fg: #161a1c; --muted: #5c6b6e;
  --accent: #0d7d6f; --accent-warm: #c2622a; --shadow: 0 1px 2px rgba(15, 30, 30, 0.06);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #10151a; --panel: #171d22; --border: #262f35; --fg: #e7ecec; --muted: #8b9a9d;
    --accent: #3fc0af; --accent-warm: #e2934f; --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
  }
}
:root[data-theme="dark"] {
  --bg: #10151a; --panel: #171d22; --border: #262f35; --fg: #e7ecec; --muted: #8b9a9d;
  --accent: #3fc0af; --accent-warm: #e2934f; --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg); margin: 0; padding: 32px 28px 64px;
  font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 21px; font-weight: 650; letter-spacing: -0.01em; margin: 0 0 6px; }
.sub { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
.eyebrow {
  font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--accent); margin: 40px 0 12px; padding-left: 12px; border-left: 3px solid var(--accent);
}
.eyebrow:first-of-type { margin-top: 28px; }
code, .mono, td, th, #stepLabel, #genStatus { font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace; }
.controls {
  display: flex; align-items: center; gap: 18px; flex-wrap: wrap; margin-bottom: 14px;
  padding: 12px 16px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 7px; box-shadow: var(--shadow);
}
.controls label { display: flex; align-items: center; gap: 7px; }
select, input[type=number] {
  font: inherit; color: var(--fg); background: var(--bg); border: 1px solid var(--border);
  border-radius: 5px; padding: 3px 6px;
}
input[type=number] { width: 64px; }
input[type=range] { accent-color: var(--accent); width: 220px; }
button {
  font: inherit; font-weight: 600; color: #fff; background: var(--accent); border: none;
  border-radius: 5px; padding: 6px 16px; cursor: pointer;
}
button:disabled { opacity: 0.5; cursor: default; }
#stepLabel { min-width: 88px; font-size: 12px; color: var(--muted); }
#genStatus { font-size: 12px; color: var(--muted); }
#genStatus.err { color: #b91c1c; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 10px; }
.cell { border: 1px solid var(--border); border-top: 2px solid var(--accent); border-radius: 7px; padding: 9px; background: var(--panel); box-shadow: var(--shadow); }
.cell .cap { font-size: 11px; color: var(--muted); margin-bottom: 7px; }
.cell img { width: 100%; display: block; border-radius: 4px; image-rendering: pixelated; }
.chart-wrap { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
svg.chart { background: var(--panel); border: 1px solid var(--border); border-radius: 7px; box-shadow: var(--shadow); }
.legend { display: flex; flex-direction: column; gap: 7px; font-size: 12px; padding-top: 10px; min-width: 140px; }
.legend .sw { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; }
.overflow { overflow-x: auto; border-radius: 7px; }
table { border-collapse: collapse; font-size: 12px; width: 100%; background: var(--panel); }
th, td { border: 1px solid var(--border); padding: 6px 10px; text-align: right; font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align: left; color: var(--muted); font-family: ui-sans-serif, -apple-system, sans-serif; }
td.ratio { color: var(--muted); }
thead th { background: var(--bg); font-family: ui-sans-serif, -apple-system, sans-serif; font-weight: 600; }
.empty { color: var(--muted); padding: 24px; text-align: center; border: 1px dashed var(--border); border-radius: 7px; }
</style></head>
<body><div class="wrap">
<h1>Rollout Inspector <span style="color:var(--muted); font-weight:400;">— live</span></h1>
<div class="sub" id="metaLine">loading…</div>

<div class="eyebrow">Pick a window</div>
<div class="controls">
  <label>Param combo <select id="paramSel"></select></label>
  <label>Trajectory <select id="trajSel"></select></label>
  <label>Start frame <input type="number" id="startInput" min="0" value="0"></label>
  <label>Rollout steps <input type="number" id="stepsInput" min="1" value="20"></label>
  <button id="genBtn">Generate</button>
  <span id="genStatus"></span>
</div>

<div class="eyebrow">Rollout viewer</div>
<div class="controls">
  <label>Channel <select id="channelSel"></select></label>
  <label>Step <input type="range" id="stepSlider" min="0" value="0"><span id="stepLabel"></span></label>
</div>
<div class="grid" id="gtGrid"><div class="empty">Pick a window above and hit Generate.</div></div>
<div class="grid" id="modelGrid"></div>

<div class="eyebrow">Metrics over rollout</div>
<div class="controls">
  <label>Metric <select id="metricSel"></select></label>
  <label>Channel <select id="metricChannelSel"><option value="__avg__">average over channels</option></select></label>
</div>
<div class="chart-wrap">
  <svg id="chart" class="chart" width="640" height="320"></svg>
  <div class="legend" id="legend"></div>
</div>

<div class="eyebrow">Physics diagnostics — aggregated over the rollout shown above</div>
<div class="overflow">
<table>
  <thead><tr id="diagHeader"><th></th></tr></thead>
  <tbody id="diagBody"></tbody>
</table>
</div>
</div>

<script>
const COLORS = ["#0d7d6f", "#c2622a", "#6b5b9a", "#3b6ea5", "#a04868", "#5c8a3f"];
let META = null, RESULT = null;

async function loadMeta() {
  META = await (await fetch("/api/meta")).json();
  document.getElementById("metaLine").textContent =
    `models: ${META.labels.join(", ")}  ·  context frames: ${META.contextFrames}  ·  ` +
    `${META.paramCombos.length} param combo(s) in the validation set`;

  const paramSel = document.getElementById("paramSel");
  META.paramCombos.forEach((c, i) =>
    paramSel.add(new Option(`Rayleigh=${c.params[0].toExponential(0)}, Prandtl=${c.params[1]} (${c.n_traj} traj, T=${c.time_steps})`, i)));
  paramSel.addEventListener("input", onParamChange);
  onParamChange();

  const channelSel = document.getElementById("channelSel");
  META.channels.forEach((c, i) => channelSel.add(new Option(c, i)));

  const metricSel = document.getElementById("metricSel");
  META.metrics.forEach(m => metricSel.add(new Option(m, m)));

  const metricChannelSel = document.getElementById("metricChannelSel");
  META.channels.forEach(c => metricChannelSel.add(new Option(c, c)));

  const diagHeader = document.getElementById("diagHeader");
  diagHeader.innerHTML = "<th></th>" + META.labels.map(lb => `<th colspan="2">${lb}</th>`).join("") + "<th>ground truth</th>";

  const legend = document.getElementById("legend");
  legend.innerHTML = META.labels.map((lb, i) =>
    `<div><span class="sw" style="background:${COLORS[i % COLORS.length]}"></span>${lb}</div>`
  ).join("") + `<div><span class="sw" style="background:#9ca3af"></span>ground truth</div>`;

  [channelSel, document.getElementById("stepSlider")].forEach(el => el.addEventListener("input", renderFrames));
  [metricSel, metricChannelSel].forEach(el => el.addEventListener("input", renderChart));
  document.getElementById("genBtn").addEventListener("click", generate);
}

function onParamChange() {
  const combo = META.paramCombos[+document.getElementById("paramSel").value];
  const trajSel = document.getElementById("trajSel");
  trajSel.innerHTML = "";
  for (let i = 0; i < combo.n_traj; i++) trajSel.add(new Option(`#${i}`, i));
  const startInput = document.getElementById("startInput");
  startInput.max = combo.time_steps - META.contextFrames - 1;
  startInput.value = Math.floor(combo.time_steps / 2);
}
document.addEventListener("DOMContentLoaded", loadMeta);

async function generate() {
  const btn = document.getElementById("genBtn");
  const status = document.getElementById("genStatus");
  btn.disabled = true;
  status.className = "";
  status.textContent = "generating…";
  const combo = META.paramCombos[+document.getElementById("paramSel").value];
  const q = new URLSearchParams({
    params: combo.params.join(","),
    traj_id: document.getElementById("trajSel").value,
    start_frame: document.getElementById("startInput").value,
    rollout_steps: document.getElementById("stepsInput").value,
  });
  try {
    const res = await fetch("/api/rollout?" + q.toString());
    const json = await res.json();
    if (json.error) throw new Error(json.error);
    RESULT = json;
    status.textContent = `done in ${json.genSeconds}s — frames ${json.frameRange}`;
    document.getElementById("stepSlider").max = RESULT.frames[0].length - 1;
    document.getElementById("stepSlider").value = 0;
    renderFrames();
    renderChart();
    renderDiag();
  } catch (e) {
    status.className = "err";
    status.textContent = "failed: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderFrames() {
  if (!RESULT) return;
  const c = +document.getElementById("channelSel").value, k = +document.getElementById("stepSlider").value;
  document.getElementById("stepLabel").textContent = `step ${k + 1} / ${RESULT.frames[0].length}`;
  const entry = RESULT.frames[c][k];
  document.getElementById("gtGrid").innerHTML =
    `<div class="cell"><div class="cap">ground truth — ${META.channels[c]}</div><img src="${entry.gt}"></div>`;
  document.getElementById("modelGrid").innerHTML = META.labels.map(lb => `
    <div class="cell"><div class="cap">${lb} — prediction</div><img src="${entry.models[lb].pred}"></div>
    <div class="cell"><div class="cap">${lb} — |error|</div><img src="${entry.models[lb].err}"></div>
  `).join("");
}

function renderDiag() {
  if (!RESULT) return;
  document.getElementById("diagBody").innerHTML = META.diagKeys.map(({key, title}) => {
    const modelCells = META.labels.map(lb => {
      const d = RESULT.diagnostics[lb][key];
      return `<td>${d.pred.toPrecision(4)}</td><td class="ratio">${d.ratio.toFixed(3)}</td>`;
    }).join("");
    const target = RESULT.diagnostics[META.labels[0]][key].target;
    return `<tr><td>${title}</td>${modelCells}<td>${target.toPrecision(4)}</td></tr>`;
  }).join("");
}

function renderChart() {
  if (!RESULT) return;
  const metric = document.getElementById("metricSel").value;
  const chCh = document.getElementById("metricChannelSel").value;
  const svg = document.getElementById("chart");
  const W = 640, H = 320, padL = 46, padR = 14, padT = 14, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  function seriesFor(label) {
    if (chCh === "__avg__") {
      const perCh = META.channels.map(c => RESULT.metrics[metric][c][label]);
      return perCh[0].map((_, k) => perCh.reduce((s, arr) => s + arr[k], 0) / perCh.length);
    }
    return RESULT.metrics[metric][chCh][label];
  }

  const allSeries = META.labels.map(seriesFor);
  let vmax = Math.max(...allSeries.flat(), 1e-6);
  const K = allSeries[0].length;
  const x = i => padL + (i / (K - 1)) * plotW;
  const y = v => padT + plotH - (v / vmax) * plotH;

  let svgEls = "";
  for (let g = 0; g <= 4; g++) {
    const v = vmax * g / 4;
    const yy = y(v);
    svgEls += `<line x1="${padL}" y1="${yy}" x2="${W - padR}" y2="${yy}" style="stroke:var(--border)" stroke-width="1"/>`;
    svgEls += `<text x="${padL - 6}" y="${yy + 3}" font-size="10" style="fill:var(--muted)" text-anchor="end">${v.toFixed(3)}</text>`;
  }
  const xt = [0, Math.floor(K / 2), K - 1];
  xt.forEach(i => {
    svgEls += `<text x="${x(i)}" y="${H - 8}" font-size="10" style="fill:var(--muted)" text-anchor="middle">${i + 1}</text>`;
  });
  META.labels.forEach((lb, li) => {
    const s = allSeries[li];
    const d = s.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    svgEls += `<path d="${d}" fill="none" stroke="${COLORS[li % COLORS.length]}" stroke-width="2"/>`;
  });
  svg.innerHTML = svgEls;
}
</script>
</body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, dest="models",
                   help="'label|config_path|checkpoint_path[|old_key>new_key]', repeatable")
    p.add_argument("--data-dir", default="/home/sl8rv/datasets/rayleigh_benard/data")
    p.add_argument("--context-frames", type=int, default=6,
                   help="fixed for the whole server -- baked into the loaded models' encoders")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--render-scale", type=float, default=0.5,
                   help="single-window requests aren't size-constrained like the static report, "
                        "so this can stay higher for sharper images")
    p.add_argument("--diag-kappa", type=float, default=0.1)
    p.add_argument("--diag-nu", type=float, default=None)
    p.add_argument("--diag-g", type=float, default=1.0)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    models = {}
    for spec in args.models:
        label, config_path, ckpt_path, key_remap = _parse_model_arg(spec)
        print(f"loading {label} <- {ckpt_path}")
        model, _epoch = _load_model(config_path, ckpt_path, args.device, args.context_frames, key_remap)
        models[label] = (model, _epoch)

    print("scanning validation set for param combos ...")
    param_combos = _scan_param_combos(args.data_dir)
    print(f"found {len(param_combos)} param combo(s)")

    STATE["models"] = models
    STATE["args"] = args
    STATE["param_combos"] = param_combos
    STATE["loader_cache"] = {}

    server = HTTPServer((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
