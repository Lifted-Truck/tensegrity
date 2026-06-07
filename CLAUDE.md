# CLAUDE.md — Tensegrity working memory

Agent-facing context for this repo. Read this first, then the README for the full
overview/roadmap. This file captures the *working state* and *hard-won findings* that
the README doesn't — the things you'd otherwise have to rediscover.

## What this project is

**Tensegrity** is a novel sound-synthesis engine: each block of audio samples is the
**solution of a constraint optimization** (gradient descent), not generated forward by
an oscillator.

```
x* = argmin_x E(x),   E(x) = Σ_k w_k · C_k(x)  +  reg_w · R(x)
```

The novelty is **over-constrained, mutually-frustrated constraint sets** — the "voice"
is the *character of the compromise* the solver reaches between constraints that cannot
all be satisfied at once. **Kill criterion:** if a patch sounds like a filtered
oscillator, the constraints aren't fighting — add tension or cut the patch.

Full thesis, design invariants, repo structure, and the phased roadmap live in
[`README.md`](README.md). Design briefs in [`docs/`](docs/). **Don't duplicate the
roadmap here — read it there.**

## Who I'm working with

**Julian** (GitHub `Lifted-Truck`, julian.beall.smith@gmail.com). Designs original
synthesis paradigms as C++/JUCE VST3 plugins. **DSP- and math-fluent** — explanations
can assume that; engage on architecture and tradeoffs, don't over-explain basics.

Sibling project: **Morphos** (`../morphogenic`), a mature JUCE VST3 morphogenetic synth
(sound particles on a 2D vector field). Phase 8 goal: embed the Tensegrity core in
Morphos so the field drives constraint targets.

### Standing preferences (honor these)
- **Early testability above all.** Front-load whatever produces audible/playable results
  soonest. He wants to be *hands-on tuning the sound by ear*, not reviewing plans. The
  ear is the irreplaceable input — when a metric goes loose, defer to listening.
- **Self-contained, portable core.** The DSP engine (`core/`, planned C++17) must be
  **framework-free — no JUCE types cross its boundary** — so it ports into Morphos and
  elsewhere. The VST is a thin adapter. Do **not** mirror the Morphos interface.
- **Optional features are weight-gated, not bolted on.** Every optional constraint is a
  continuous term whose **weight is its gate**: weight 0 = contributes nothing to
  energy/gradient = identical to disabled = CPU-skipped; blends in smoothly above 0.
  There is no separate enable boolean — the slider *is* the toggle, 0 = off.
- Prefer **declarative/data edit surfaces** he can tweak without touching solver internals.

## Toolchain / environment

- C++17 / JUCE 8.x / CMake + FetchContent, VST3, **Windows** (VS 2026, x64), auditions in
  **Ableton**. Core/plugin split is enforced from Phase 1 onward.
- **Prototype runs in a venv**, NOT global Python (global has only numpy). Always use it:
  ```
  cd prototype
  .venv\Scripts\python.exe <script>        # Windows
  ```
  `.venv` has numpy 2.4.6, scipy 1.17.1, autograd (from `requirements.txt`). Global Python
  is 3.12.10 with numpy only — don't rely on it.
- **`.venv/` and `*.wav` are gitignored.** Consequence for machine-switching: the venv and
  all audio (including `prototype/test1_balloon.wav`, `test2_roar.wav`, and rendered
  outputs) do **not** sync via git. On a fresh clone: recreate the venv
  (`python -m venv .venv` then `pip install -r prototype/requirements.txt`) and re-copy any
  test audio files manually.

## Current status

- **Phase 0 complete**: offline Python solver + declarative audition harness, verified.
- **Phase 1 (portable C++ `core/`) is the next build target** — not started. `core/` and
  `plugin/` hold only README stubs so far.
- Recent work has been in the prototype: a "squeaking balloon" patch and an
  **audio-derived-constraints** experiment (derive constraints from sound files via
  cepstral envelopes). See findings below.

## Prototype map (`prototype/`)

The prototype is the **reference oracle** every future C++ constraint is validated against.

- **`variational_synth.py`** — the solver + constraint/regularizer library (the gradient
  oracle). Constants: `SR=44100`, `BLOCK=1024`, `HOP=512`, `ITERS=90` (offline; real-time
  target ~8–16 warm-started), `LR=0.05`. Adam over autograd gradients; warm-start (roll
  prev block by hop); 50% overlap-add Hann for continuity.
  - Constraints: `c_centroid`, `c_tilt` (spectral; need FFT-adjoint in C++),
    `c_power`, `c_autocorr` (pitch), `c_envelope` (short-time RMS contour) — these three
    are **time-domain, no-FFT gradients = easiest to port**; `c_formants` (cepstral
    envelope → Gaussian targets, optional), `c_envelope_match` (match a *given* measured
    cepstral envelope — generalizes `c_formants`).
  - Regularizers (the "voice"): `reg_l2`, `reg_smooth` (lowpass bias), `reg_tv`.
  - Init sources (block-0 basin, also "voice"): `noise`, `filtered_noise`, `sine`.
- **`audition.py` + `patches.py`** — the **declarative sound-design edit loop**. Author
  patches as data (weights + time-curve targets) in `patches.py`; `python audition.py`
  batch-renders each to WAV and prints metrics (centroid-tracking correlation `centroid_r`,
  `periodicity_mean`). No solver code to touch. Weight-gating: terms with `w<=0` are skipped.
  - Term types: `centroid`, `tilt`, `power`, `autocorr` (uses `"hz"` key), `envelope`,
    `formants`. Regs: `"l2" | "smooth" | "tv"`.
  - **Julian has hand-tuned weights in `patches.py` (autocorr/envelope at 5.0 in several
    patches) — these are intentional, do NOT revert.**
- **`curves.py`** — time-curve primitives, all return `f(t)` with t∈[0,1]: `const(v)`,
  `ramp(a,b)`, `breakpoints(points)`, `sine(center, depth, cycles, phase)`, `as_curve(x)`.
- **`from_audio.py`** — the audio-derived-constraints experiment (see below). Loads two
  recordings, extracts per-block cepstral envelopes, renders a "fight" (both envelopes at
  comparable weight) vs a "baseline" (one envelope = kill-criterion control), prints an
  env-distance table. `load_wav()` handles mono/resample(48k→44.1k)/normalize/trim.

### Short renders look bad — that's expected
Metrics on <0.5s smoke renders (e.g. negative `centroid_r`) are normal: too few blocks to
settle. Full 2s renders at `ITERS=90` approach the targets. Not a bug.

## Key findings — audio-derived constraints (cepstrum experiment)

The idea: "derive constraints from sound files" by extracting each source's **cepstral
spectral envelope** (low-quefrency = timbre/formants, decoupled from pitch) and imposing
two incompatible ones on a new sound so they fight.

**What's proven:** the machinery works. On *synthetic divergent tones* (dark+steep-rolloff
vs bright+flat-rolloff), the baseline collapses onto its target and the fight sits *between*
the two — a genuine negotiated compromise, not a crossfade.

**What real recordings exposed (balloon squeak vs roar):**
1. **The shape-only cepstral-envelope descriptor is too loose to transfer a real timbre.**
   `c_envelope_match` matches the *mean-subtracted, liftered, log-magnitude* envelope shape.
   A render aimed at the balloon overshoots to centroid 7447 Hz (balloon 3714, roar 1997)
   yet still "matches" — log compression hides a high-freq hash that dominates the *linear*
   spectrum. Many spectrally-dissimilar waveforms satisfy the objective equally.
2. **Reachability is a real wall.** The roar (broadband, near what `reg_smooth`+noise
   naturally produces) is reachable; the balloon squeak is **not** — every regularizer ×
   init still lands roar-ward even aimed point-blank at the balloon. *Constraint-matching
   can only reach timbres inside the manifold that (constraints + regularizer + init) span.*
3. Adding a source-derived `c_centroid` does **not** rescue it and can anti-correlate with
   the env-distance metric — the solver's objective and the judging metric disagree. This
   metric has gone as far as it usefully can; **judge by ear.**

**Lesson for the project:** "derive constraints from a file" is only as good as the
descriptor derived; a single liftered log-envelope underdetermines the spectrum. Tensegrity
was never about *reproducing* a sound (that's a vocoder) — the on-thesis artifact is the
*compromise* between incompatible real timbres.

### Open fork (where we left off) — Julian's call
- **A. Richer descriptor:** build a per-bin / mel-weighted `c_spectrum_match` that pins
  brightness + fine structure (most faithful, biggest build, drifts toward vocoder).
- **B. Lean into the thesis (recommended):** keep the loose match; judge `out_fight.wav` by
  ear as a distinct third object, formalize if it sings.
- **C. Reachable sources:** pick two real sounds that both live in the synth's manifold
  (e.g. two sustained broadband textures) so they fight cleanly without the reachability wall.

## Conventions

- Memory/preferences also live in the user's auto-memory (user profile + early-testability
  feedback). Keep this file and those consistent.
- When proposing a build order, call out the **fastest path to "you can hear/play this."**
- Verify patches two ways — instrumented metric **and** the ear. A patch can converge to a
  number while sounding dead.
