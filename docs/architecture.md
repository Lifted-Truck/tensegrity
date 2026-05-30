# Tensegrity — Architecture & Engineering Plan

This document scopes *how* the engine is built. The [README](../README.md) covers the
*what* and the roadmap; this covers the API surface, the threading/streaming model,
the gradient strategy, and the testing discipline. It is the reference for Phases 1–3.

The governing constraint is **portability**: the engine is a plain C++17 library with
no plugin-framework dependency, so the same code powers the Tensegrity VST, an embed
inside Morphos, and a standalone/offline renderer. Everything below serves that.

---

## 1. Layers

```
┌──────────────────────────────────────────────────────────────┐
│  Hosts (interchangeable)                                       │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ Tensegrity   │  │ Morphos Anchor   │  │ Offline render  │  │
│  │ VST3 (JUCE)  │  │ engine (embed)   │  │ / parity tests  │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬────────┘  │
└─────────┼───────────────────┼─────────────────────┼───────────┘
          │   only via the engine's C++ API (no JUCE types cross)
          ▼                   ▼                     ▼
┌──────────────────────────────────────────────────────────────┐
│  tensegrity-core   (C++17, framework-free)                     │
│                                                                │
│  StreamEngine ── ring buffer, overlap-add, latency             │
│      └─ BlockRenderer ── warm-start, Hann OLA, prev-block state │
│            └─ VariationalSolver ── Adam, fixed iteration budget │
│                  ├─ ConstraintSet ── weighted Σ w_k·C_k         │
│                  │     Constraint: value(x) + grad(x) [analytic]│
│                  └─ Regularizer ── L2 / smooth / TV             │
│            └─ Fft (injected interface)                          │
└──────────────────────────────────────────────────────────────┘
```

No layer above the line appears below it. The core never includes a JUCE header.

---

## 2. Core API sketch

Illustrative, not final — pins down the boundary so hosts and engine can be built
against a stable contract.

```cpp
namespace tns {

// --- injected FFT so the host can supply its own (JUCE dsp::FFT, pffft, etc.) ---
struct Fft {
    virtual ~Fft() = default;
    virtual void forward(const float* time, std::complex<float>* freq) = 0;  // size N
    virtual void inverse(const std::complex<float>* freq, float* time) = 0;
};

// --- a constraint computes a scalar penalty AND its gradient w.r.t. the block ---
// Analytic gradients (hand-derived). `work` holds shared FFT scratch so constraints
// that need the spectrum don't each recompute it.
struct Constraint {
    virtual ~Constraint() = default;
    virtual float value(const float* x, int N, SpectralWork& work) = 0;
    virtual void  accumulateGrad(const float* x, int N, SpectralWork& work,
                                 float weight, float* gradOut) = 0;  // += weight·∂C/∂x
};

enum class Regularizer { L2, Smooth, TotalVariation };
enum class InitSource  { PrevBlock, FilteredNoise, Sinusoid };  // the "voice" selector

// --- everything the solver needs for one block; this IS the live patch ---
struct PatchState {
    std::vector<std::pair<float, Constraint*>> terms;  // (weight, constraint)
    Regularizer reg     = Regularizer::Smooth;
    float       regW    = 2e-3f;
    InitSource  init     = InitSource::PrevBlock;
    int         iterations = 12;     // small + warm-started for real-time
    float       lr      = 0.05f;
};

class StreamEngine {
public:
    StreamEngine(int blockSize, int sampleRate, Fft& fft);
    void  setPatch(const PatchState&);          // swap targets/weights/reg/init live
    void  process(const float* in, float* out, int numSamples);  // host-buffer agnostic
    int   latencySamples() const;               // block + hop, report to host
};

} // namespace tns
```

Key points:
- **`PatchState` is the instrument.** Targets, weights, regularizer, init source,
  iteration budget. Moving these over time is "playing." A host maps MIDI/automation/a
  Morphos field onto this struct.
- **Weight is the gate — no boolean toggles.** Optional constraints (formants and any
  future opt-in feature) are ordinary terms whose weight controls their influence on a
  continuous scale. At weight 0 a term contributes nothing to the energy *and* nothing
  to the gradient, so it is exactly equivalent to being absent. The solver **skips any
  term with weight ≤ ε** before touching its FFT/reductions, so a zeroed constraint
  costs zero CPU — the slider blends a feature in from free-and-off to full influence
  with no discontinuity and no separate enable flag. This is one uniform mechanism for
  the whole optional-feature surface.
- **Constraints carry their own analytic gradient.** No autodiff at runtime.
- **`StreamEngine::process` is host-buffer agnostic** — it owns the fixed analysis
  block internally (see §4), so a host calling it with 64- or 512-sample buffers gets
  correct overlap-add either way.

---

## 3. Gradient strategy (the central engineering lift)

The Python prototype gets gradients for free via `autograd`. The audio thread cannot —
autodiff allocates and is far too slow. So every constraint that ships in C++ needs a
**hand-derived analytic gradient**. Each is tractable because each constraint is an
FFT plus vector reductions, and each FFT/reduction has a standard adjoint.

Per-constraint adjoint notes (gradients to derive and unit-test against autograd):

| Constraint | Forward | Gradient path |
|---|---|---|
| `c_power` | `(rms(x) − target)²` | Closed form in time domain; no FFT |
| `c_autocorr` | `(r(lag) − target)²` where `r` is normalized autocorr | Closed form in time domain; product/quotient rule on the normalized lag term |
| `c_centroid` | `(centroid − target)²` on `|FFT(x)|` | ∂centroid/∂|X|, then ∂|X|/∂X, then inverse-FFT adjoint back to `x` |
| `c_tilt` | `−slope · Σ ramp·\hat m` on normalized magnitude | Same magnitude→`x` adjoint chain as centroid |
| `c_envelope` | short-time RMS contour error | Per-frame chain rule into the owning time samples; no FFT |
| `c_formants` (opt) | cepstral-envelope match to Gaussians | Longest chain: FFT→log→IFFT→quefrency window→FFT→magnitude; each step's adjoint is standard. ~5× a single FFT |

Discipline: **derive, then verify against autograd** (§6) before a constraint is
considered done. The prototype is the oracle.

The Adam update itself is trivial and identical to the Python (`solve_block`).

---

## 4. Streaming model

The prototype renders offline: it knows the whole performance and emits a contiguous
buffer. A plugin gets arbitrary host buffers in real time. Bridging that:

- **Fixed internal analysis block** `N` (e.g. 1024) with hop `N/2`, independent of the
  host's `numSamples`. The host buffer size varies (64–2048); the solver block does not.
- **Input ring buffer** accumulates incoming samples until a full hop is available,
  then the engine solves the next block, warm-started from the previous one
  (`init = roll(prev, −hop)`), and **overlap-adds** with a Hann window into an output
  ring buffer that `process` drains back to the host.
- **Latency** = one block plus a hop of look-ahead inherent to OLA; reported to the
  host via `latencySamples()` so the DAW compensates.
- **Pitch / targets update at block boundaries** and are linearly smoothed within the
  block to avoid zipper artifacts, mirroring Morphos's buffer-rate smoothing.

### Real-time vs offline solving
- **Offline** (parity tests, future bounce/export): full iteration count (~90), as in
  the prototype.
- **Real-time:** a *fixed small* budget (~8–16 iterations), warm-started. The solver
  **tracks** the moving constraint targets rather than fully resolving each block. The
  tracking lag is intentional — it is the "heavy resonator chasing your gesture" feel.
  Iteration count is sized to (block period − headroom); expose a quality setting.

### Audio-thread rules (when wrapped by the VST)
The core must be allocation-free on the hot path so the host's `processBlock` stays
real-time-safe: no heap allocation, no locks, no blocking calls once `setPatch` has
sized the buffers. Patch swaps publish a new `PatchState` via the host's existing
parameter mechanism; the engine reads it at block boundaries.

---

## 5. Voice / pitch model

Pitch enters as a constraint target — the autocorrelation lag (`lag = SR/f0`). MIDI
note → `f0` → target lag, smoothed across blocks for glides.

Polyphony is deliberately deferred (Phase 6) because **each voice is a full block
solve** — variational synthesis is intrinsically expensive per voice, unlike an
oscillator. The plan:
1. **Monophonic** first (Phase 3): one solver, pitch from the held/last note.
2. **Paraphonic** option: multiple pitch-lag targets inside one solve (the frustrated
   set negotiates a chord-like spectrum) — cheap but timbrally coupled.
3. **True polyphony:** N independent `StreamEngine`s in a small voice pool with
   stealing (Phase 6), gated on measured per-voice CPU cost.

This is a genuine scoping risk worth stating plainly: do not assume oscillator-style
128-voice polyphony. The instrument's character is monophonic-leaning by construction.

---

## 6. Testing & verification discipline

Two independent checks, both required (README invariant #4):

1. **Numeric parity (automated).** A harness renders identical patches through the C++
   core and the Python prototype and asserts the waveforms and summary metrics match
   within tolerance: centroid-tracking correlation (target ≈ 0.95 on the demo sweep),
   normalized periodicity at the target lag (≈ 0.78 on the demo), RMS-envelope
   adherence. This is how a hand-derived gradient is proven correct against autograd.
2. **Ear + instrumentation (manual).** Render → listen → A/B regularizers and weights.
   Confirm constraints are actually met (a patch can converge to a number while
   sounding dead). Apply the **kill criterion**: if it sounds like a filtered
   oscillator, the constraints aren't fighting — add tension or cut it.

A per-constraint gradient check (finite-difference vs analytic vs autograd) gates each
constraint into the core.

---

## 7. Open engineering questions

Flagged now, resolved as the relevant phase lands — not blockers for Phase 1.

- **FFT vendoring.** Which standalone FFT for the core's own builds and parity tests
  (pffft, KissFFT, or pocketfft)? The VST can inject `juce::dsp::FFT`, but the core
  needs a default so it builds without a host.
- **Block size as a parameter vs. fixed.** Larger blocks resolve low pitches and
  formants better but add latency and cost; smaller blocks track faster. Likely a
  patch/quality setting, but the OLA + warm-start machinery must handle a configurable
  `N` from the start.
- **Real-time iteration budget tuning.** How few iterations still track a fast gesture
  acceptably? Needs measurement on the real solver — informs the Phase 2 quality knob.
- **Init-source voicing in a stream.** `FilteredNoise` / `Sinusoid` inits give
  different basins per the spec, but in a continuous stream the warm-start from the
  previous block dominates after block 0. Decide whether init-source re-seeds on note
  onset, continuously blends, or only applies at voice start.
- **Paraphonic frustration.** Whether multiple simultaneous pitch-lag targets in one
  solve produce a usable chord or just mush — an experiment for Phase 6, possibly
  earlier as a prototype patch.
- **Morphos embed surface.** Whether Tensegrity enters Morphos as a Timbral Anchor
  engine, a Transient source, or both, and how the field maps onto `PatchState`
  targets. Designed in Phase 8 against the then-stable core API.
