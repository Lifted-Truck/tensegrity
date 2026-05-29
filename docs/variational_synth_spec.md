# Variational Synthesis — build brief

## The principle
The waveform is not generated forward by an oscillator/model. Each block of `N`
samples is the **solution of an optimization**: declare constraints (what must be
true of the sound) plus a regularizer (which of the many satisfying waveforms you
prefer), and let gradient descent find the block.

```
x* = argmin_x  E(x),   E(x) = Σ_k w_k · C_k(x)  +  reg_w · R(x)
```

You sculpt the constraint manifold; the sound is whatever lives at the bottom of
it. This inverts causality relative to every forward method — and it's the only
reason this escapes the "clever controller bolted onto an oscillator" trap.

## The one design insight that matters
**The novelty is entirely in over-constrained, mutually-frustrated constraint
sets.** A single magnitude-spectrum target with free phase, solved from noise, is
just a filter with extra steps — it collapses to subtractive/additive synthesis.
The new sound only appears when you impose constraints that *cannot all be
satisfied at once* (e.g. a fixed pitch period AND a percussive amplitude envelope
AND a rising centroid AND a fixed loudness). The solver negotiates a compromise no
forward synth would ever produce, and the *character of that compromise* — which
constraint wins where — is the instrument's voice. Design patches by picking
constraints that fight.

Corollary kill-criterion: if a patch sounds indistinguishable from a filtered
oscillator, your constraints aren't frustrating each other. Add tension or
abandon the patch.

## Architecture
- **Block solver.** Per block, run Adam on `grad(E)`. Offline: ~80–150 iters to
  near-convergence. Real-time: a *fixed small* iteration count (~8–16),
  warm-started — an "anytime" solver that *tracks* the moving targets rather than
  fully resolving them. The tracking lag is not a bug; it's a playable feel (the
  instrument "chases" your gestures, like a heavy resonator).
- **Continuity.** Independent per-block solves click at the seams. Fix with
  warm-start (init each block from the previous block rolled by the hop) + 50%
  overlap-add with a Hann window. Secondary option: an explicit boundary penalty
  tying `x[0]` and the first difference to the previous block's tail.
- **State.** The only state carried between blocks is the previous waveform (for
  warm-start). Everything else is the live constraint set.

## Constraint library
All must be differentiable (autodiff offline; hand-derived gradients in C++ for
real-time — each is just an FFT plus reductions, cheap). Implemented in the
prototype: spectral centroid, spectral tilt, RMS power, normalized
autocorrelation at a lag (→ pitch), short-time RMS envelope (→ transient shape).
Worth adding: full magnitude-envelope target (use sparingly — see insight above),
spectral flatness, inharmonicity (autocorrelation at non-integer-related lags),
and an amplitude-modulation-band energy term as a rough sensory-roughness proxy.

## Under-determination = the voice
With few constraints the solution space is large; the **regularizer and the
initialization pick the point**, so they are timbral controls, not housekeeping:
- L2 on `x` → low-energy, smoother. Smoothness (squared first diff) → low-pass-ish,
  darker. Total variation → flatter segments with hard edges (buzzier, more
  formant-like).
- Init from previous block → continuity; from filtered noise → breathier basins;
  from a sinusoid → more tonal basins. Same constraints, different init, different
  sound. Expose init-source as a discrete "voice" selector.

## Controller mapping
The playable parameters are: constraint **targets**, the **weights** `w_k` (how
hard each constraint pulls — a "focus" macro), `reg_w`, the regularizer choice,
and the init source.
- Continuous gestures (XY pad, mod wheel, envelopes) → constraint targets.
- **Synergy with your existing vector-field engine:** retarget it. Instead of the
  field driving oscillator/additive parameters, let the field's position drive the
  *constraint targets* (centroid, pitch lag, envelope shape). The attractor then
  traverses a manifold of *constraint configurations*, and the variational solver
  realizes each point as sound. That composes your two ideas into one instrument
  where the field decides what must be true and the solver decides how.

## Real-time porting plan
- Do **not** run Python/autograd in the audio thread. Phase 0 (offline, the
  prototype) only validates that patches sound worth chasing.
- Phase 1: port to the VST audio thread in C++ with **hand-written gradients** for
  the locked-in constraint set (FFT via your existing kit + a handful of vector
  reductions; the Adam update is trivial). Fixed iteration budget per block sized
  to your block period minus headroom.
- Phase 2: voices (regularizer/init presets), constraint-weight macros, and the
  vector-field retarget.

## Verifier protocol
The whole point of last conversation's discipline. Keep the loop tight:
1. Offline: render → listen → A/B regularizers and constraint weights.
2. Instrument the render to confirm constraints are actually met (the prototype's
   centroid-tracking correlation and periodicity readouts do this — a patch can
   "converge" to a number while sounding dead; check both the metric and the ear).
3. Kill criterion above: collapses-to-a-filter → cut it.

## Status
Offline prototype (`variational_synth.py`) runs and is verified: across a 1 s
render the solved waveform tracked a 400→3400 Hz centroid sweep at r≈0.95 while
holding ~0.78 normalized periodicity at the target pitch lag, under simultaneous
percussive-envelope and loudness constraints — i.e. the frustrated set resolves
into a coherent, controllable sound. `variational_demo.wav` is a 2 s example of
that patch (pitch glide A2→A3, rising brightness, percussive contour).
