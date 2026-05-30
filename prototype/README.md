# Tensegrity — Offline Prototype & Audition Harness (Phase 0)

The Python implementation of the variational synthesis engine. It is both the
**sound-design edit loop** (author patches, render, listen, read the numbers) and the
**gradient oracle** the C++ core is validated against.

## Setup (one time)

`numpy` is required; `scipy` and `autograd` usually need installing. Use a virtual
environment so your global Python stays clean:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

(On macOS/Linux: `.venv/bin/python` instead of `.venv\Scripts\python`.)

## The fast edit loop — author and audition patches

This is the loop to live in until the C++ VST is playable.

```powershell
.venv\Scripts\python audition.py                 # render every patch in patches.py
.venv\Scripts\python audition.py frustrated_bell # render one patch by name
```

For each patch it writes `out_<name>.wav` and prints the headline verifier metrics:

- **`centroid_r`** — correlation between the rendered spectral centroid and the
  patch's intended centroid curve. ~0.95 means the brightness gesture is tracking.
- **`periodicity_mean`** — mean normalized autocorrelation at the target pitch lag.
  ~0.8 confirms the pitch constraint actually holds.

A patch can converge to a good number while sounding dead, so **always listen too**,
and apply the **kill criterion**: if it sounds like a filtered oscillator, the
constraints aren't fighting — add tension or cut the patch.

### Editing patches

Open [`patches.py`](patches.py) and edit the `PATCHES` list. A patch is data:

```python
{
  "name": "my_patch", "seconds": 2.0,
  "reg": "smooth", "reg_w": 2e-3, "init": "noise",
  "terms": [
    {"c": "centroid", "w": 1.0, "target": ramp(400, 3400)},
    {"c": "autocorr", "w": 4.0, "hz": ramp(110, 220), "target": 0.85},
    {"c": "power",    "w": 2.0, "target": 0.25},
    {"c": "envelope", "w": 3.0, "target": ENV_PERC},
    {"c": "formants", "w": 0.0, "vowel": "a", "lifter": 30},  # w=0 -> off (gated)
  ],
}
```

- **Weight `w` is the gate.** A term at `w = 0` contributes nothing and is skipped —
  exactly as the real engine will skip it — so leave optional constraints in at 0 and
  dial them up to taste.
- **Targets can move over the take.** Use the [`curves.py`](curves.py) helpers
  (`const`, `ramp`, `breakpoints`, `sine`) or a bare number. `t` runs 0→1 across the
  render.
- Patch design is **picking constraints that fight** — a fixed pitch period *and* a
  percussive envelope *and* a rising centroid *and* a fixed loudness.

## Files

| File | Role |
|---|---|
| `variational_synth.py` | The verified solver + constraints (the **gradient oracle**). Run directly to render its built-in `demo`. Import-safe. |
| `curves.py` | Time-curve primitives for declarative targets (`const`, `ramp`, `breakpoints`, `sine`). |
| `patches.py` | The editable patch library — the sound-design surface. |
| `audition.py` | Compiles declarative patches → solver, renders to WAV, prints verifier metrics. The CLI above. |

## What's inside the oracle

- **Solver** — hand-rolled Adam over the autograd gradient, warm-started per block,
  50% overlap-add Hann windowing for continuity (`solve_block`, `render`).
- **Base constraints** — `c_centroid`, `c_tilt`, `c_power`, `c_autocorr` (pitch via
  normalized autocorrelation), `c_envelope` (short-time RMS / transient shape).
- **Optional formant module** — `c_formants` matches a cepstrally-smoothed spectral
  envelope to a sum of Gaussians; `VOWELS` holds recipes. Opt-in, never in the base set.
- **Regularizers** — `reg_l2`, `reg_smooth`, `reg_tv` (the "voice").
- **Init sources** — `noise` / `filtered_noise` / `sine` block-0 basins; same
  constraints, different init → different sound.
