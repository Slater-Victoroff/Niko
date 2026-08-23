"""Generate an interactive HTML report for eyeballing checkpoint rollouts.

Loads one or more checkpoints against the same held-out trajectory, runs each
one's autoregressive rollout, and writes a single self-contained HTML file
with:
  - a scrubbable rollout viewer (channel selector + step slider) showing
    ground truth, each model's prediction, and its abs-error map, all on a
    shared per-channel color scale so panels are visually comparable
  - per-step metric curves (vrmse and fixed-reference NRMSE, per channel or
    averaged) for every model on one chart
  - a physics-diagnostics summary table (incompressibility, buoyancy/momentum
    residuals, vorticity, spectral hf-frac) per model, reusing
    training.physics_diagnostics.compute_physics_diagnostics

Metrics are intentionally factored into small `_metric_*` functions -- adding
a new one (spectral error, a custom physics score, whatever) is a matter of
writing one more function and adding it to METRICS, not touching the
rendering code. That's the "esp if we start evaluating on more than vrmse"
part.

Usage:
    python3 src/niko/eval/inspect_checkpoints.py \\
        --model "winner_roll16|configs/context_cond_helmholtz_rotation_roll16.yaml|/path/to/winner_ep7.pt" \\
        --model "baseline_roll16|configs/context_cond_helmholtz_roll16.yaml|/path/to/baseline_ep7.pt" \\
        --data-dir /home/sl8rv/datasets/rayleigh_benard/data \\
        --param-combo 1e8,1.0 --context-frames 6 --rollout-steps 20 \\
        --traj-ids 0,1,2 --start-frames 0,60,120 --output report.html

Each --model is "label|config_path|checkpoint_path", optionally with a 4th
"|old_key>new_key" segment for renamed-module checkpoint remapping (see
EXPERIMENT_LOG.md §-- the operator rename that broke load_state_dict once).
Repeat --model for as many checkpoints as you want side by side.

--traj-ids/--start-frames define a grid: every (trajectory, start-frame)
combination in it gets its own full rollout, precomputed and embedded, with a
frontend dropdown for each axis to switch between them in the browser --
there's no live backend behind the report (it's a static, self-contained
HTML file), so "frontend control" here means picking a grid worth exploring
up front rather than an on-demand query. Keep the grid modest: each combo
costs real file size (channels x steps x models worth of images), so e.g. 3
trajectories x 4 start-frames = 12 combos is already a reasonable ceiling at
default --rollout-steps/--render-scale. If the written file creeps past
~14MB, shrink --rollout-steps, --render-scale, or the grid itself.
"""
import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.build import build_model
from training.losses import well_style_vrmse
from training.physics_diagnostics import compute_physics_diagnostics
import dataloader as dl

CHANNEL_NAMES = ["pressure", "buoyancy", "velocity_x", "velocity_y"]

# Curated subset of compute_physics_diagnostics's ~25 keys -- one
# representative "residual"/rms row per physical constraint group, each with
# _pred/_target/_ratio already computed by that function.
DIAG_KEYS = [
    ("incomp_div_rms", "|div V| rms (incompressibility)"),
    ("buoy_residual", "buoyancy transport residual"),
    ("mom_u_residual", "u-momentum residual"),
    ("mom_v_residual", "v-momentum residual"),
    ("vort_rms", "vorticity rms"),
    ("spec_b_hf_frac", "buoyancy high-freq energy frac"),
    ("spec_u_hf_frac", "velocity high-freq energy frac"),
]


# ---------------------------------------------------------------------------
# Metrics (per step, per channel) -- add a new one here to extend the report.
# ---------------------------------------------------------------------------

def _metric_vrmse(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """well_style_vrmse, per (step, channel). pred/target: [1, K, C, H, W]."""
    return well_style_vrmse(pred, target)[0].cpu().numpy()  # [K, C]


def _metric_nrmse_fixed_ref(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """RMSE / a single per-channel std computed once over this whole
    trajectory's target window (not recomputed per step -- see
    EXPERIMENT_LOG.md §14 for why that matters for cross-step comparison)."""
    mse = (pred - target).float().pow(2).mean(dim=(-2, -1))[0]  # [K, C]
    ref_std = target.float().std(dim=(0, 1, 3, 4)).clamp_min(1e-7)  # [C]
    return (mse.sqrt() / ref_std[None, :]).cpu().numpy()


METRICS = {
    "vrmse": _metric_vrmse,
    "nrmse (fixed ref)": _metric_nrmse_fixed_ref,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(config_path: str, ckpt_path: str, device: str, context_frames: int, key_remap=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["encoder"]["context_frames"] = context_frames
    cfg["context_cond_encoder"]["context_frames"] = context_frames
    model = build_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if key_remap:
        sd = {k.replace(*key_remap): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{ckpt_path}: checkpoint mismatch, missing={missing}, unexpected={unexpected}")
    model.eval()
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return model, epoch


def _parse_model_arg(spec: str):
    parts = spec.split("|")
    if len(parts) not in (3, 4):
        raise ValueError(f"--model must be 'label|config|checkpoint[|old_key>new_key]', got: {spec}")
    label, config_path, ckpt_path = parts[:3]
    key_remap = None
    if len(parts) == 4:
        old, new = parts[3].split(">")
        key_remap = (old, new)
    return label, config_path, ckpt_path, key_remap


# ---------------------------------------------------------------------------
# Colormaps (no matplotlib dependency -- plain numpy lerp)
# ---------------------------------------------------------------------------

def _diverging_rgb(x: np.ndarray) -> np.ndarray:
    """x in [-1, 1] -> RGB uint8, blue (low) - white (0) - red (high)."""
    x = np.clip(x, -1.0, 1.0)
    lo = np.array([33, 102, 172], dtype=np.float32)
    mid = np.array([247, 247, 247], dtype=np.float32)
    hi = np.array([178, 24, 43], dtype=np.float32)
    t_neg = np.clip(-x, 0, 1)[..., None]
    t_pos = np.clip(x, 0, 1)[..., None]
    rgb = mid * (1 - t_neg - t_pos) + lo * t_neg + hi * t_pos
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _sequential_rgb(x: np.ndarray) -> np.ndarray:
    """x in [0, 1] -> RGB uint8, white (0) - orange - dark red (1). Used for
    abs-error maps, where only magnitude (not sign) matters."""
    x = np.clip(x, 0.0, 1.0)
    lo = np.array([255, 255, 255], dtype=np.float32)
    mid = np.array([253, 174, 97], dtype=np.float32)
    hi = np.array([103, 0, 31], dtype=np.float32)
    t = x[..., None]
    rgb = np.where(t < 0.5, lo * (1 - 2 * t) + mid * (2 * t), mid * (2 - 2 * t) + hi * (2 * t - 1))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _png_data_uri(rgb: np.ndarray, scale: float = 1.0) -> str:
    img = Image.fromarray(rgb, mode="RGB")
    if scale != 1.0:
        # BILINEAR (not NEAREST) so downsampling actually smooths high-frequency
        # noise instead of just subsampling it -- matters a lot for PNG size, since
        # these are smooth colormap gradients and noise is what defeats compression.
        # CSS image-rendering:pixelated in the report re-blocks it on upscale, so this
        # doesn't cost the blocky look, just the file size.
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BILINEAR)
    # Quantize to a small adaptive palette -- these are synthetic colormap outputs
    # (a few hundred distinct colors at most even in RGB truecolor), so an indexed
    # PNG with a fitted palette is dramatically smaller than truecolor with
    # negligible visible loss, which is what actually keeps a many-step, many-model
    # report under the artifact size budget.
    img = img.convert("P", palette=Image.ADAPTIVE, colors=32)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Report data assembly
# ---------------------------------------------------------------------------

def _fetch_window(val_loader, traj_id: int, start_frame: int, context_frames: int, rollout_steps: int):
    """Look up the loader index whose PairIndex matches (traj_id, start_frame) --
    create_param_dataloaders/H5RayleighBenardFields enumerates every valid (file,
    trajectory, start-frame) window up front as dataset._pairs, in the exact fixed
    order the (shuffle_val=False) loader iterates, so this is just a lookup, not a
    re-implementation of the windowing logic."""
    pairs = val_loader.dataset._pairs
    match_idx = next(
        (idx for idx, pr in enumerate(pairs) if pr.traj_idx == traj_id and pr.t == start_frame),
        None,
    )
    if match_idx is None:
        traj_ids = sorted({pr.traj_idx for pr in pairs})
        starts_here = sorted(pr.t for pr in pairs if pr.traj_idx == traj_id)
        lo_hi = f"[{starts_here[0]}, {starts_here[-1]}]" if starts_here else "(none -- traj-id not in this file)"
        raise ValueError(
            f"traj-id {traj_id} start-frame {start_frame} not found. traj-id must be one of {traj_ids}; "
            f"for traj-id {traj_id}, start-frame must be in {lo_hi} given --context-frames {context_frames} "
            f"--rollout-steps {rollout_steps} (start-frame's ceiling shrinks as rollout-steps grows)."
        )
    for i, (xb, yb, bparams) in enumerate(val_loader):
        if i == match_idx:
            return xb, yb, bparams
    raise AssertionError("unreachable: match_idx was found in _pairs but not in the loader's iteration")


def build_report_data(models: dict, xb: torch.Tensor, yb: torch.Tensor, device: str, render_scale: float,
                       diag_kappa: float, diag_nu, diag_g: float):
    K = yb.shape[1]
    preds = {}
    with torch.no_grad():
        for label, (model, _epoch) in models.items():
            preds[label] = model(xb, steps=K, return_initial_encode=False)

    yb_np = yb[0].cpu().numpy()  # [K, C, H, W]

    # Per-channel symmetric color range from ground truth, shared by GT and
    # every model's prediction so panels are visually comparable.
    ch_vabs = [max(1e-6, float(np.abs(yb_np[:, c]).max())) for c in range(len(CHANNEL_NAMES))]
    # Per-channel error color range, shared across all models and steps.
    err_vmax = [1e-6] * len(CHANNEL_NAMES)
    for label in preds:
        err = (preds[label] - yb).abs()[0].cpu().numpy()  # [K, C, H, W]
        for c in range(len(CHANNEL_NAMES)):
            err_vmax[c] = max(err_vmax[c], float(err[:, c].max()))

    frames = []  # frames[channel][step] = {"gt": uri, "models": {label: {"pred": uri, "err": uri}}}
    for c, cname in enumerate(CHANNEL_NAMES):
        vabs = ch_vabs[c]
        chan_frames = []
        for k in range(K):
            gt_uri = _png_data_uri(_diverging_rgb(yb_np[k, c] / vabs), scale=render_scale)
            step_entry = {"gt": gt_uri, "models": {}}
            for label in preds:
                pred_np = preds[label][0, k, c].detach().cpu().numpy()
                err_np = np.abs(pred_np - yb_np[k, c])
                step_entry["models"][label] = {
                    "pred": _png_data_uri(_diverging_rgb(pred_np / vabs), scale=render_scale),
                    "err": _png_data_uri(_sequential_rgb(err_np / err_vmax[c]), scale=render_scale),
                }
            chan_frames.append(step_entry)
        frames.append(chan_frames)

    metrics = {}
    for metric_name, fn in METRICS.items():
        metrics[metric_name] = {}
        for c, cname in enumerate(CHANNEL_NAMES):
            metrics[metric_name][cname] = {}
        for label in preds:
            vals = fn(preds[label], yb)  # [K, C]
            for c, cname in enumerate(CHANNEL_NAMES):
                metrics[metric_name][cname][label] = [float(v) for v in vals[:, c]]

    diagnostics = {}
    for label, pred in preds.items():
        d = compute_physics_diagnostics(pred, yb, kappa=diag_kappa, nu=diag_nu, g=diag_g)
        row = {}
        for key, _title in DIAG_KEYS:
            row[key] = {
                "pred": d.get(f"{key}_pred", d.get(key)),
                "target": d.get(f"{key}_target"),
                "ratio": d.get(f"{key}_ratio"),
            }
        diagnostics[label] = row

    return frames, metrics, diagnostics


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def render_html(combos: dict, labels, traj_ids, start_frames, meta: dict) -> str:
    """combos: {(traj_id, start_frame): {"frames":..., "metrics":..., "diagnostics":...,
    "frameRange": "a-b"}} -- one full report's worth of data per grid point. Trajectory
    and start-frame selection happens client-side (switching which combo's data feeds
    the existing channel/step/metric controls), since a live backend to compute a new
    window on demand isn't an option for a self-contained artifact -- see the comment on
    --start-frames in main() for how the grid is chosen."""
    import json
    combo_key = lambda t, s: f"{t}_{s}"
    data = {
        "channels": CHANNEL_NAMES,
        "labels": labels,
        "trajIds": traj_ids,
        "startFrames": start_frames,
        "combos": {combo_key(t, s): v for (t, s), v in combos.items()},
        "diagKeys": [{"key": k, "title": t} for k, t in DIAG_KEYS],
        "meta": meta,
    }
    data_json = json.dumps(data)

    diag_header = "".join(f'<th colspan="2">{lb}</th>' for lb in labels)
    diag_subheader = "".join('<th>pred</th><th>pred/target</th>' for _ in labels)

    meta_rows = "".join(f"<div><span class='k'>{k}</span><span class='v'>{v}</span></div>" for k, v in meta.items())

    return f"""<title>Rollout Inspector</title>
<style>
:root {{
  --bg: #eef1f2; --panel: #ffffff; --border: #d9dfe1; --fg: #161a1c; --muted: #5c6b6e;
  --accent: #0d7d6f; --accent-warm: #c2622a; --shadow: 0 1px 2px rgba(15, 30, 30, 0.06);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #10151a; --panel: #171d22; --border: #262f35; --fg: #e7ecec; --muted: #8b9a9d;
    --accent: #3fc0af; --accent-warm: #e2934f; --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
  }}
}}
:root[data-theme="dark"] {{
  --bg: #10151a; --panel: #171d22; --border: #262f35; --fg: #e7ecec; --muted: #8b9a9d;
  --accent: #3fc0af; --accent-warm: #e2934f; --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--fg); margin: 0; padding: 32px 28px 64px;
  font: 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; }}
h1 {{
  font-size: 21px; font-weight: 650; letter-spacing: -0.01em; margin: 0 0 6px; text-wrap: balance;
}}
.eyebrow {{
  font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--accent); margin: 40px 0 12px; padding-left: 12px; border-left: 3px solid var(--accent);
}}
.eyebrow:first-of-type {{ margin-top: 28px; }}
code, .mono, td, th, #stepLabel, .meta .v {{
  font-family: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, monospace;
}}
.meta {{ display: flex; flex-wrap: wrap; gap: 5px 22px; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
.meta .k {{ margin-right: 5px; font-family: ui-sans-serif, -apple-system, sans-serif; }}
.meta .v {{ color: var(--fg); font-variant-numeric: tabular-nums; }}
.controls {{
  display: flex; align-items: center; gap: 22px; flex-wrap: wrap; margin-bottom: 14px;
  padding: 12px 16px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 7px; box-shadow: var(--shadow);
}}
.controls label {{ display: flex; align-items: center; gap: 7px; }}
select {{
  font: inherit; color: var(--fg); background: var(--bg); border: 1px solid var(--border);
  border-radius: 5px; padding: 3px 6px;
}}
input[type=range] {{ accent-color: var(--accent); width: 220px; }}
#stepLabel {{ min-width: 88px; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 10px; }}
.cell {{
  border: 1px solid var(--border); border-top: 2px solid var(--accent); border-radius: 7px;
  padding: 9px; background: var(--panel); box-shadow: var(--shadow);
}}
.cell .cap {{ font-size: 11px; color: var(--muted); margin-bottom: 7px; }}
.cell img {{ width: 100%; display: block; border-radius: 4px; image-rendering: pixelated; }}
.chart-wrap {{ display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }}
svg.chart {{ background: var(--panel); border: 1px solid var(--border); border-radius: 7px; box-shadow: var(--shadow); }}
.legend {{ display: flex; flex-direction: column; gap: 7px; font-size: 12px; padding-top: 10px; min-width: 140px; }}
.legend .sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 7px; }}
#frameRange {{ font-size: 12px; color: var(--muted); }}
.overflow {{ overflow-x: auto; border-radius: 7px; }}
table {{ border-collapse: collapse; font-size: 12px; width: 100%; background: var(--panel); }}
th, td {{ border: 1px solid var(--border); padding: 6px 10px; text-align: right; font-variant-numeric: tabular-nums; }}
th:first-child, td:first-child {{ text-align: left; color: var(--muted); font-family: ui-sans-serif, -apple-system, sans-serif; }}
td.ratio {{ color: var(--muted); }}
thead th {{ background: var(--bg); font-family: ui-sans-serif, -apple-system, sans-serif; font-weight: 600; }}
:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
</style>
<div class="wrap">
<h1>Rollout Inspector</h1>
<div class="meta">{meta_rows}</div>

<div class="eyebrow">Trajectory</div>
<div class="controls">
  <label>Trajectory <select id="trajSel"></select></label>
  <label>Start frame <select id="startSel"></select></label>
  <span id="frameRange"></span>
</div>

<div class="eyebrow">Rollout viewer</div>
<div class="controls">
  <label>Channel <select id="channelSel"></select></label>
  <label>Step <input type="range" id="stepSlider" min="0" value="0"><span id="stepLabel"></span></label>
</div>
<div class="grid" id="gtGrid"></div>
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
  <thead><tr><th></th>{diag_header}<th>ground truth</th></tr>
  <tr><th></th>{diag_subheader}<th></th></tr></thead>
  <tbody id="diagBody"></tbody>
</table>
</div>
</div>

<script>
const DATA = {data_json};
const COLORS = ["#0d7d6f", "#c2622a", "#6b5b9a", "#3b6ea5", "#a04868", "#5c8a3f"];
const DIAG_LABELS = {json.dumps(labels)};

const trajSel = document.getElementById("trajSel");
DATA.trajIds.forEach(t => trajSel.add(new Option(`#${{t}}`, t)));

const startSel = document.getElementById("startSel");
DATA.startFrames.forEach(s => startSel.add(new Option(`frame ${{s}}`, s)));
// Default to a middle start frame, not frame 0 -- frame 0 is always the
// trajectory's own undeveloped initial condition (see --start-frames help).
startSel.selectedIndex = Math.floor((DATA.startFrames.length - 1) / 2);

function currentCombo() {{
  return DATA.combos[`${{trajSel.value}}_${{startSel.value}}`];
}}

const channelSel = document.getElementById("channelSel");
DATA.channels.forEach((c, i) => channelSel.add(new Option(c, i)));

const metricSel = document.getElementById("metricSel");
Object.keys(currentCombo().metrics).forEach(m => metricSel.add(new Option(m, m)));

const metricChannelSel = document.getElementById("metricChannelSel");
DATA.channels.forEach(c => metricChannelSel.add(new Option(c, c)));

const stepSlider = document.getElementById("stepSlider");

const legend = document.getElementById("legend");
legend.innerHTML = DATA.labels.map((lb, i) =>
  `<div><span class="sw" style="background:${{COLORS[i % COLORS.length]}}"></span>${{lb}}</div>`
).join("") + `<div><span class="sw" style="background:#9ca3af"></span>ground truth (dashed)</div>`;

function onComboChange() {{
  const combo = currentCombo();
  document.getElementById("frameRange").textContent = `showing frames ${{combo.frameRange}}`;
  stepSlider.max = combo.frames[0].length - 1;
  if (+stepSlider.value > +stepSlider.max) stepSlider.value = 0;
  renderFrames();
  renderChart();
  renderDiag();
}}

function renderFrames() {{
  const combo = currentCombo();
  const c = +channelSel.value, k = +stepSlider.value;
  document.getElementById("stepLabel").textContent = `step ${{k + 1}} / ${{combo.frames[0].length}}`;
  const entry = combo.frames[c][k];
  document.getElementById("gtGrid").innerHTML =
    `<div class="cell"><div class="cap">ground truth — ${{DATA.channels[c]}}</div><img src="${{entry.gt}}"></div>`;
  document.getElementById("modelGrid").innerHTML = DATA.labels.map(lb => `
    <div class="cell"><div class="cap">${{lb}} — prediction</div><img src="${{entry.models[lb].pred}}"></div>
    <div class="cell"><div class="cap">${{lb}} — |error|</div><img src="${{entry.models[lb].err}}"></div>
  `).join("");
}}

function renderDiag() {{
  const combo = currentCombo();
  document.getElementById("diagBody").innerHTML = DATA.diagKeys.map(({{key, title}}) => {{
    const modelCells = DIAG_LABELS.map(lb => {{
      const d = combo.diagnostics[lb][key];
      return `<td>${{d.pred.toPrecision(4)}}</td><td class="ratio">${{d.ratio.toFixed(3)}}</td>`;
    }}).join("");
    const target = combo.diagnostics[DIAG_LABELS[0]][key].target;
    return `<tr><td>${{title}}</td>${{modelCells}}<td>${{target.toPrecision(4)}}</td></tr>`;
  }}).join("");
}}

function renderChart() {{
  const combo = currentCombo();
  const metric = metricSel.value;
  const chCh = metricChannelSel.value;
  const svg = document.getElementById("chart");
  const W = 640, H = 320, padL = 46, padR = 14, padT = 14, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  function seriesFor(label) {{
    if (chCh === "__avg__") {{
      const perCh = DATA.channels.map(c => combo.metrics[metric][c][label]);
      return perCh[0].map((_, k) => perCh.reduce((s, arr) => s + arr[k], 0) / perCh.length);
    }}
    return combo.metrics[metric][chCh][label];
  }}

  const allSeries = DATA.labels.map(seriesFor);
  let vmax = Math.max(...allSeries.flat(), 1e-6);
  const K = allSeries[0].length;

  const x = i => padL + (i / (K - 1)) * plotW;
  const y = v => padT + plotH - (v / vmax) * plotH;

  let svgEls = "";
  // gridlines + axis labels
  for (let g = 0; g <= 4; g++) {{
    const v = vmax * g / 4;
    const yy = y(v);
    svgEls += `<line x1="${{padL}}" y1="${{yy}}" x2="${{W - padR}}" y2="${{yy}}" style="stroke:var(--border)" stroke-width="1"/>`;
    svgEls += `<text x="${{padL - 6}}" y="${{yy + 3}}" font-size="10" style="fill:var(--muted)" text-anchor="end">${{v.toFixed(3)}}</text>`;
  }}
  const xt = [0, Math.floor(K / 2), K - 1];
  xt.forEach(i => {{
    svgEls += `<text x="${{x(i)}}" y="${{H - 8}}" font-size="10" style="fill:var(--muted)" text-anchor="middle">${{i + 1}}</text>`;
  }});

  DATA.labels.forEach((lb, li) => {{
    const s = allSeries[li];
    const d = s.map((v, i) => `${{i === 0 ? "M" : "L"}}${{x(i).toFixed(1)}},${{y(v).toFixed(1)}}`).join(" ");
    svgEls += `<path d="${{d}}" fill="none" stroke="${{COLORS[li % COLORS.length]}}" stroke-width="2"/>`;
  }});

  svg.innerHTML = svgEls;
}}

[trajSel, startSel].forEach(el => el.addEventListener("input", onComboChange));
[channelSel, stepSlider].forEach(el => el.addEventListener("input", renderFrames));
[metricSel, metricChannelSel].forEach(el => el.addEventListener("input", renderChart));
onComboChange();
</script>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, dest="models",
                   help="'label|config_path|checkpoint_path[|old_key>new_key]', repeatable")
    p.add_argument("--data-dir", default="/home/sl8rv/datasets/rayleigh_benard/data")
    p.add_argument("--param-combo", default=None, help="e.g. '1e8,1.0' -- picks which trajectory family to draw from")
    p.add_argument("--context-frames", type=int, default=6)
    p.add_argument("--rollout-steps", type=int, default=20)
    p.add_argument("--traj-ids", default="0",
                   help="comma-separated physical trajectory ids (0..n_traj-1 in the matched file) -- "
                        "each one gets a frontend selector, so pass a few (e.g. '0,1,2') for real "
                        "in-browser trajectory control instead of just one")
    p.add_argument("--start-frames", default=None,
                   help="comma-separated frames within the trajectory to start the context window at, "
                        "each gets a frontend selector -- e.g. '0,60,120'. Frame 0 is always the "
                        "trajectory's own initial condition (Rayleigh-Benard starts near-uniform; "
                        "convection rolls/plumes take a while to form), so include a few later values "
                        "to reach the developed/turbulent regime. Default: auto -- --num-start-frames "
                        "points evenly spaced from 0 to the latest frame --rollout-steps allows.")
    p.add_argument("--num-start-frames", type=int, default=4,
                   help="used only when --start-frames is not given: how many auto-spaced start points")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--render-scale", type=float, default=0.3,
                   help="downsize factor for embedded PNGs -- keep report size in check; shrink further "
                        "if generating a large trajectory x start-frame grid pushes past ~14MB")
    p.add_argument("--diag-kappa", type=float, default=0.1)
    p.add_argument("--diag-nu", type=float, default=None)
    p.add_argument("--diag-g", type=float, default=1.0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    param_choices = None
    if args.param_combo:
        vals = tuple(float(v) for v in args.param_combo.split(","))
        param_choices = [vals]

    models = {}
    for spec in args.models:
        label, config_path, ckpt_path, key_remap = _parse_model_arg(spec)
        print(f"loading {label} <- {ckpt_path}")
        model, epoch = _load_model(config_path, ckpt_path, args.device, args.context_frames, key_remap)
        models[label] = (model, epoch)

    _, val_loader, _ = dl.create_param_dataloaders(
        args.data_dir, batch_size=1, context_frames=args.context_frames,
        predict_frames=args.rollout_steps, num_workers=0, param_choices=param_choices,
        shuffle_val=False,
    )

    traj_ids = [int(x) for x in args.traj_ids.split(",")]

    if args.start_frames is not None:
        start_frames = sorted({int(x) for x in args.start_frames.split(",")})
    else:
        pairs = val_loader.dataset._pairs
        starts_here = sorted(pr.t for pr in pairs if pr.traj_idx == traj_ids[0])
        if not starts_here:
            raise ValueError(f"traj-id {traj_ids[0]} has no valid windows for "
                              f"--context-frames {args.context_frames} --rollout-steps {args.rollout_steps}")
        lo, hi, n = starts_here[0], starts_here[-1], args.num_start_frames
        start_frames = sorted({round(lo + i * (hi - lo) / max(1, n - 1)) for i in range(n)})
    print(f"grid: traj_ids={traj_ids} x start_frames={start_frames} "
          f"({len(traj_ids) * len(start_frames)} combos)")

    labels = list(models.keys())
    combos = {}
    bparams = None
    for traj_id in traj_ids:
        for start_frame in start_frames:
            xb, yb, bp = _fetch_window(val_loader, traj_id, start_frame, args.context_frames, args.rollout_steps)
            xb, yb = xb.to(args.device), yb.to(args.device)
            bparams = bp
            end_frame = start_frame + args.context_frames + args.rollout_steps - 1
            print(f"  traj {traj_id} frames {start_frame}-{end_frame} ...")
            frames, metrics, diagnostics = build_report_data(
                models, xb, yb, args.device, args.render_scale, args.diag_kappa, args.diag_nu, args.diag_g,
            )
            combos[(traj_id, start_frame)] = {
                "frames": frames, "metrics": metrics, "diagnostics": diagnostics,
                "frameRange": f"{start_frame}-{end_frame}",
            }

    meta = {
        "trajectory params": bparams.tolist(),
        "context frames": args.context_frames,
        "rollout steps": args.rollout_steps,
    }
    for label, (_model, epoch) in models.items():
        meta[f"{label} epoch"] = epoch

    html = render_html(combos, labels, traj_ids, start_frames, meta)
    Path(args.output).write_text(html)
    print(f"wrote {args.output} ({Path(args.output).stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
