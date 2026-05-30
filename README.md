# Tensegrity

**A variational sound synthesis engine.** Each block of samples is not generated
forward by an oscillator — it is the **solution of an optimization**. You declare
constraints (what must be true of the sound) plus a regularizer (which of the many
satisfying waveforms you prefer), and gradient descent finds the block.

```
x* = argmin_x  E(x),   E(x) = Σ_k w_k · C_k(x)  +  reg_w · R(x)
```

You sculpt a constraint manifold; the sound is whatever lives at the bottom of it.
This inverts causality relative to every forward synthesis method.

> **Why "Tensegrity"?** A tensegrity structure holds its shape through a balance of
> members in tension and compression — integrity *through* tension. This engine's
> voice is exactly that: the character of the compromise the solver reaches between
> constraints that **cannot all be satisfied at once**. The tension is the instrument.

Design background: [`docs/variational_synth_spec.md`](docs/variational_synth_spec.md)
· formant extension: [`docs/variational_synth_formants.md`](docs/variational_synth_formants.md)
· engineering plan: [`docs/architecture.md`](docs/architecture.md)

---

## The one idea that matters

The novelty is **entirely** in over-constrained, mutually-frustrated constraint
sets. A single magnitude target with free phase collapses to subtractive/additive
synthesis — a filter with extra steps. The new sound only appears when constraints
*fight* (a fixed pitch period **and** a percussive envelope **and** a rising
centroid **and** a fixed loudness). The solver negotiates a compromise no forward
synth would produce, and the *character of that compromise* is the voice.

**Kill criterion:** if a patch sounds indistinguishable from a filtered oscillator,
the constraints aren't frustrating each other. Add tension or cut the patch.

---

## Design invariants

These are the non-negotiables that shape every phase below. Violate one and the
project loses the property that makes it worth building.

1. **Portable, framework-free core.** The DSP engine (`core/`) is plain C++17 with
   **zero plugin-framework dependencies** — no JUCE types cross its boundary. The
   VST is a thin adapter; Morphos (and anything else) embeds the same core. The
   engine is the product; the plugin is one host.
2. **Frustration is the feature.** The architecture must keep arbitrary constraint
   combinations cheap to declare and weight, because designing patches *is* picking
   constraints that fight.
3. **Optional features are weight-gated, not bolted on.** Every optional constraint
   (formants first, others to follow) is a continuous term whose **weight is its
   gate**: at weight 0 it contributes nothing to the energy or gradient — identical to
   disabled, and skipped entirely for zero CPU — and blends in smoothly above 0. There
   is no separate "enable" boolean; the slider *is* the toggle, with 0 as off. This
   keeps the base engine general and makes adding a new optional constraint a uniform,
   non-invasive act. Default patches sit at weight 0 for these.
4. **Verifier discipline.** Every patch is checked two ways: instrumented metrics
   (centroid-tracking correlation, periodicity readout) **and** the ear. A patch can
   converge to a number while sounding dead — check both.
5. **Anytime real-time solving.** In the audio thread the solver runs a *fixed small*
   iteration budget, warm-started — it *tracks* moving targets rather than fully
   resolving them. The tracking lag is a playable feel, not a bug.

---

## Repository structure

```
tensegrity/
├── README.md            # this file — overview + roadmap
├── docs/
│   ├── variational_synth_spec.md      # original design brief (the principle)
│   ├── variational_synth_formants.md  # formant-constraint design (optional module)
│   └── architecture.md                # engineering plan: core API, streaming, gradients
├── prototype/           # Phase 0 — offline Python verifier (DONE)
│   ├── variational_synth.py
│   ├── requirements.txt
│   └── README.md
├── core/                # Phase 1+ — portable, JUCE-free C++ DSP engine (planned)
└── plugin/              # Phase 3+ — thin JUCE VST3 wrapper around core (planned)
```

---

## Quickstart — run the offline prototype

The prototype renders a patch to a `.wav` and is the reference implementation every
C++ constraint is validated against.

```bash
cd prototype
pip install -r requirements.txt
python variational_synth.py        # writes variational_demo.wav
```

It renders a deliberately over-constrained patch (pitch glide A2→A3, rising
brightness, percussive contour, fixed loudness) — the frustrated set resolving into
a coherent, controllable sound. A second `demo_vowel` patch exercises the optional
formant module.

---

## Roadmap

Phase 0 is complete (the prototype is in `prototype/`). Everything after it is the
build-out toward a shippable VST and a portable engine. Phases are sequential but
the **core/plugin split is enforced from Phase 1 onward** — never let plugin
concerns leak into the engine.

### Phase 0 — Offline prototype & verifier ✅
- [x] Python variational solver: Adam over autograd gradients, warm-start + 50%
      overlap-add Hann continuity.
- [x] Base constraints: spectral centroid, spectral tilt, RMS power, autocorrelation
      (pitch), short-time RMS envelope (transient shape).
- [x] Regularizers: L2, smoothness, total variation.
- [x] Optional formant constraint (cepstral-envelope match to Gaussian targets) with
      vowel recipes and a lifter macro.
- [x] Instrumented verification: centroid-tracking correlation + periodicity readout.

### Phase 1 — Portable core scaffold (`core/`, C++17)
The engine, JUCE-free, with a parity harness proving it matches the Python.
- [ ] CMake library target `tensegrity-core`, no plugin dependencies.
- [ ] Pluggable FFT abstraction (inject host FFT; vendor a small FFT for standalone
      tests and Python parity).
- [ ] Solver core: hand-rolled Adam, warm-start, 50% overlap-add `BlockRenderer`.
- [ ] Base constraints with **hand-derived analytic gradients** (the central lift —
      autograd is offline-only): centroid, tilt, power, autocorrelation, RMS envelope.
- [ ] Regularizers: L2 / smooth / TV.
- [ ] **Parity test harness:** render identical patches in C++ and Python, compare
      waveform + metrics (centroid r, periodicity) to confirm gradients match autograd.

### Phase 2 — Streaming + real-time solver
Turn the block engine into a real-time stream.
- [ ] Internal fixed analysis block (e.g. 1024) decoupled from host buffer size via a
      ring buffer; correct overlap-add across arbitrary host buffers.
- [ ] Latency reporting (block + hop) to the host.
- [ ] Fixed small iteration budget (~8–16), warm-started "anytime" tracking solver.
- [ ] CPU-budget instrumentation; iteration count sized to block period minus headroom.

### Phase 3 — Minimal VST3 (`plugin/`, JUCE) — first playable instrument
- [ ] Thin JUCE VST3 wrapper hosting `tensegrity-core`. Monophonic first.
- [ ] MIDI note → pitch (autocorrelation lag) constraint target.
- [ ] Expose the playable parameters: constraint **targets**, **weights** (a "focus"
      macro per constraint), `reg_w`, regularizer choice, and **init source** (the
      discrete "voice" selector — noise / sinusoid / previous block).
- [ ] APVTS parameters + state serialisation; audition in a DAW.

### Phase 4 — Optional formant module (first weight-gated constraint)
- [ ] Generic weight-gating in the solver: any term with weight ≤ ε is skipped (no
      FFT, no gradient) so a zeroed constraint costs nothing; above ε it blends in
      continuously. Formants are the first consumer of this; future opt-in constraints
      reuse it unchanged.
- [ ] Formant constraint ported with analytic gradient, driven entirely by its weight
      (0 = off, identical to absent).
- [ ] Vowel presets, lifter macro (15–60), time-varying / morphing formant targets.
- [ ] Verify cohabitation per the formant ear-protocol: envelope bumps at target Hz,
      periodicity preserved, RMS envelope still respected.

### Phase 5 — Expressive control & patch design
- [ ] Additional constraints: spectral flatness, inharmonicity (non-integer-related
      autocorrelation lags), AM-band energy (sensory-roughness proxy).
- [ ] Regularizer morphing and init-source voicing as first-class timbral controls.
- [ ] Patch/preset system; factory patches built under the kill-criterion discipline.

### Phase 6 — Polyphony / voice management
Variational synthesis is naturally expensive per voice (each voice is a full block
solve), so polyphony is scoped deliberately, not assumed.
- [ ] Measure per-voice solver cost; ship mono/paraphonic, then a small voice pool.
- [ ] Voice stealing; shared vs. per-voice solver-state strategy.

### Phase 7 — Performance
- [ ] SIMD on gradient reductions and the Adam update.
- [ ] FFT pooling / reuse across constraints sharing a spectrum.
- [ ] Auto-tuned iteration budget; quality settings.
- [ ] Offline-render correctness (buffer-clock advance under non-realtime render).

### Phase 8 — Morphos integration
The synergy both specs point to: let a vector field decide *what must be true*, and
the solver decide *how*.
- [ ] Embed `tensegrity-core` in Morphos as a Timbral Anchor engine (and/or a
      Transient source) — drop-in, since the core carries no JUCE dependency.
- [ ] Retarget the Manifold field to drive constraint **targets** (centroid, pitch
      lag, formant F1/F2) instead of oscillator/additive parameters. The attractor
      traverses a manifold of constraint configurations; the solver realizes each
      point as sound.

### Phase 9 — Cross-platform & product
- [ ] macOS port (AU + VST3 universal binary, code signing & notarisation).
- [ ] Preset browser, GUI polish, parameter automation coverage, documentation.

---

## Decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Synthesis paradigm | Variational (per-block optimization) | The project thesis — sound as the solution of a frustrated constraint set |
| Core language | C++17, framework-free | Portability is the point: one engine, many hosts |
| Core ↔ plugin split | Hard boundary; no JUCE types in `core/` | Lets the engine drop into the VST, Morphos, or a standalone with no rework |
| Plugin framework | JUCE 8.x, VST3 (Windows first) | Industry standard; matches the sibling Morphos toolchain |
| Gradients | Autodiff offline (Python), **hand-derived analytic** in C++ | Cannot run autograd in the audio thread; each constraint is an FFT + reductions with a standard adjoint |
| Continuity | Warm-start + 50% overlap-add Hann | Independent per-block solves click at the seams; this is the prototype's proven fix |
| Real-time solver | Fixed ~8–16 iters, warm-started ("anytime") | Tracks moving targets; the lag is a playable feel, not an error |
| Optional constraints | Continuous weight-gated; weight 0 = off (CPU-skipped) | One uniform mechanism for formants and any future opt-in feature; no boolean toggles |
| Verification | Instrumented metrics **and** ear, every patch | A patch can converge numerically while sounding dead |
| Build system | CMake + FetchContent | Reproducible, IDE-agnostic; mirrors Morphos |

---

## Relationship to Morphos

Tensegrity is a sibling to [Morphos](../morphogenic) (morphogenetic synthesis). They
are independent instruments, but Phase 8 unites them: because Tensegrity's engine is
framework-free, Morphos can host it as an Anchor engine where the vector field drives
the constraint targets. Tensegrity is built **standalone first** specifically so this
integration is a clean embed rather than a coupling.

## Status

Phase 0 complete and verified. Phase 1 (portable C++ core) is the next build target.

## License

TODO — choose before first external contribution (MIT and Apache-2.0 are the usual
candidates for a portable library; note JUCE's own licensing for the plugin layer).
