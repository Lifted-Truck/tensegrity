# Variational Synthesis: Formant Constraints — Integration & Design

## Where formants live in the architecture

The **formant constraint operates at a different timescale than pitch**, so they coexist rather than compete. A source–filter model emerges naturally from the constraint set:

- **Pitch/periodicity** (autocorrelation): governs the **fine structure** (harmonics spaced at the pitch interval)
- **Formants**: govern the **spectral envelope** (which harmonics are boosted/cut)
- **Envelope** (amplitude transient): governs **temporal shape** (attack, sustain, decay)
- **Centroid, tilt, power**: govern **gross spectral balance** (overall brightness, loudness)

They operate on independent degrees of freedom, so a patch with all four constraints is overconstrained, but the over-constraint is *productive* — the solver negotiates among them to produce something no forward synth would generate.

## The constraint

```python
def c_formants(x, formants, lifter=30):
    # formants: list of (center_hz, gain_db, bandwidth_hz)
    env = _envelope_cepstral(x, lifter)
    env = env - anp.mean(env)                      # shape-only: ignore level
    target = np.zeros(BLOCK // 2)
    for f_hz, gain_db, bw_hz in formants:
        target = target + gain_db * anp.exp(-0.5 * ((freqs_pos - f_hz) / bw_hz) ** 2)
    target = target - np.mean(target)
    return anp.mean((env - target) ** 2)
```

**Design choices:**

1. **Cepstral envelope** — we extract the smooth spectral envelope via real-cepstrum lowpass (lifter) instead of raw bins. Why: it decouples the spectral shape from pitch fine-structure, so formants pull on the overall curve, not individual harmonics. The pitch constraint is free to fill in harmonics.

2. **Mean-subtraction** — both envelope and target are mean-removed (shape-only). Why: if you didn't, the constraint would waste gradient correcting absolute level; with mean-subtraction it focuses entirely on placing peaks. This also makes it insensitive to your power constraint.

3. **Lifter as a design parameter** — controls how many cepstral quefrencies are kept:
   - `lifter=15`: very smooth, resolves only broad bandwidths (good for single-resonance instruments, sub-bass)
   - `lifter=30`: resolves ~2–3 closely-spaced formants (typical speech, vowels)
   - `lifter=50`: fine-grained, can separate 4+ formants but sensitive to noise in higher partials

   Lifter is a **playable parameter** — crossfading between lifter settings is a smooth morph between resolutions, usable as a macro.

4. **Gain in dB** — formant specifications are given as (center_hz, gain_db, bandwidth_hz). Gains are in dB not linear amplitude because that's how speech processing specifies them and it gives more intuitive control (6 dB = 2x energy, etc.).

## Integration with the solver

No changes to the main solver loop — formants plug into `constraint_fn` like any other term. Example:

```python
def demo_vowel(t):
    pitch_hz = 110 * (1 + 0.5 * anp.sin(2 * anp.pi * t))
    pitch_lag = int(SR / pitch_hz)
    
    # 'ah' vowel (IPA [ɑ]): F1~730, F2~1090
    formants = [
        (730,  8.0, 80),      # F1: strong, narrow
        (1090, 6.0, 100),     # F2: slightly weaker
    ]
    
    return [
        (1.0, lambda x: c_formants(x, formants, lifter=30)),
        (3.0, lambda x: c_autocorr(x, pitch_lag, 0.85)),
        (1.5, lambda x: c_power(x, 0.2)),
    ]
```

This yields a pitched vowel 'ah' with formant peaks at speech formant frequencies.

## The interesting frustrations

Formants are *resonances* — they want to ring and sustain. Your existing envelope constraints want them to decay or truncate. This is musically productive:

**1. Resonance vs. percussive transient.** A formant constraint fighting an amplitude-envelope constraint (e.g., fast attack, quick decay) produces a sound where the formant peaks persist slightly after the amplitude envelope collapses — like a plucked resonator. The peaks ring through the attack, then fade:

```python
ENV_PERC = np.array([1.0, 0.5, 0.2, 0.08, 0.02])
return [
    (2.0, lambda x: c_formants(x, vowels, lifter=30)),
    (3.0, lambda x: c_envelope(x, ENV_PERC)),       # fights the resonance
    (4.0, lambda x: c_autocorr(x, lag, 0.85)),
]
```

Result: formant ringing decaying slower than the amplitude envelope — percussive-but-resonant, like a muted string.

**2. Formant migration + centroid conflict.** A constraint that moves formants in frequency (via a time-varying formants list) while another constraint pulls the overall centroid elsewhere creates spectral tension — the envelope is stretched, and the solver trades off which formants win. Use for morphing between vowels that pull centroid in opposite directions.

**3. Pitch vs. formant bandwidth interaction.** A tightly-spaced formant set (narrow bandwidth) with a low pitch (wide harmonic spacing) forces the solver to choose: do harmonics land on formant peaks or between them? This tension is inaudible in the constraints but hearable in the resulting timbre — a kind of phase-space friction.

## Vowel recipes

Standard speech formant values (in Hz) for /i/, /ɛ/, /ɑ/, /ɔ/, /u/ across a range of speakers:

```python
VOWELS = {
    'i':  [(280, 8, 60),   (2600, 5, 120)],           # IPA [i]
    'e':  [(530, 7, 80),   (2000, 5, 100)],           # IPA [e]
    'a':  [(730, 8, 90),   (1090, 6, 110)],           # IPA [ɑ]
    'o':  [(570, 7, 90),   (1050, 4, 120)],           # IPA [ɔ]
    'u':  [(440, 7, 70),   (1340, 4, 100)],           # IPA [u]
}
```

Lifter 30 resolves the two main formants (F1, F2) cleanly; a third (F3~3–4 kHz) appears but is weak. For closer-to-speech synthesis add a third formant target, but increase lifter to 50.

## Real-time implementation

**Cepstral envelope extraction** is cheap: FFT → log-magnitude → real-IFFT → window in quefrency → real-FFT → take magnitude. The gradients are as well — all ops are standard signal-processing primitives. Precompute the lifter window and formant Gaussian targets.

**Hand-derived gradient** (pseudo):

```cpp
// Forward: E = mean((env - target)²)
// where env = envelope_cepstral(x, lifter), target is const
// Backward:
dE/dx = (2/N) * (env - target) ⊗ grad_envelope_cepstral

// grad_envelope_cepstral chains:
// x -> FFT -> log -> IFFT -> window -> FFT -> extract -> dE/dy_env -> ...
// Each step has a standard adjoint.
```

The per-block cost is about 5x a single FFT + vector operations. With 8–16 iterations per block at your buffer size, this is ~5–10% of your CPU budget on modern hardware.

## Playable parameters for the instrument

- **Formant targets** (frequency, gain, bandwidth): expose as continuous controllers (XY pad for F1/F2, sliders for bandwidth)
- **Lifter** (15–60): morph macro between fine and coarse envelope resolution
- **Formant weight** `w_formant`: blend between full-formant and free-spectrum (when 0, the constraint vanishes)
- **Time-varying formants**: parameterize formants as functions of a gesture parameter (t) so vowels morph smoothly (vowel-formant-morphing is a standard speech synth technique; here it's just another timed constraint set)

## Composition with the vector-field engine

Retarget your field to drive formant targets over time — the attractor position → (F1, F2, gain, lifter). This gives you a morphing vowel space traversed by the strange attractor, with the periodicity constraint (pitch lag) holding the voice intact. The field decides *what vowel*, the periodicity + envelope constraints negotiate the *timbre around* it.

## Verification / ear protocol

When you add formants to a patch, verify:

1. **Spectral envelope has bumps at target frequencies.** Plot the cepstrally-smoothed envelope and check for Gaussian-like peaks near your formant centers. If not, increase lifter or increase formant weight.

2. **Periodicity still holds.** Measure autocorrelation at the target pitch lag — formants shouldn't suppress it.

3. **Envelope decay is still respected.** Plot the block-wise RMS and confirm it follows your envelope constraint, not the formant ringing.

If all three hold, the constraints are cohabiting cleanly. If any fails, you've hit a frustration zone — which is fine, but now it's intentional, not accidental.
