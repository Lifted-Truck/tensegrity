# Tensegrity — Offline Prototype (Phase 0)

The reference implementation of the variational synthesis engine, in Python. It
renders a patch to `variational_demo.wav` and serves two jobs:

1. **Sound design sandbox** — declare frustrated constraint sets and hear the
   compromise the solver reaches, before committing anything to C++.
2. **Gradient oracle** — every constraint here gets its gradient for free via
   `autograd`. The C++ core's hand-derived analytic gradients are validated against
   this implementation (see [`../docs/architecture.md`](../docs/architecture.md) §6).

## Run

```bash
pip install -r requirements.txt
python variational_synth.py        # writes variational_demo.wav next to the script
```

## What's inside

- **Solver** — hand-rolled Adam over the autograd gradient, warm-started per block,
  with 50% overlap-add Hann windowing for continuity (`solve_block`, `render`).
- **Base constraints** — `c_centroid`, `c_tilt`, `c_power`, `c_autocorr` (pitch via
  normalized autocorrelation), `c_envelope` (short-time RMS / transient shape).
- **Optional formant module** — `c_formants` matches a cepstrally-smoothed spectral
  envelope to a sum of Gaussians; `VOWELS` holds formant recipes. This stays **opt-in**
  in the final engine, never part of the base set.
- **Regularizers** — `reg_l2`, `reg_smooth`, `reg_tv` (the "voice").
- **Demo patches** — `demo` (over-constrained: pitch glide + rising brightness +
  percussive contour + fixed loudness) and `demo_vowel` (vowel morph where formant
  resonance fights a percussive envelope).

## Editing patches

A patch is a function `t -> [(weight, constraint), ...]` where `t` runs 0→1 across the
take. Swap the `constraint_fn` passed to `render()` in `__main__`, or change `reg` /
`reg_w`, to audition different instruments. Keep the kill criterion in mind: if it
sounds like a filtered oscillator, the constraints aren't fighting.
