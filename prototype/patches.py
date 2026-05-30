"""
Declarative patch library — the sound-design edit surface.

Edit weights and targets here, then run `python audition.py` to render every patch to
a WAV and print verifier metrics. No solver code to touch.

A patch is a dict:

    {
      "name":    "frustrated_bell",     # output file -> out_frustrated_bell.wav
      "seconds": 2.0,
      "reg":     "smooth",              # "l2" | "smooth" | "tv"  (the regularizer / "voice")
      "reg_w":   2e-3,
      "init":    "noise",               # "noise" | "filtered_noise" | "sine"  (block-0 basin)
      "terms":   [ ... ],               # the frustrated constraint set
    }

Each term is a dict with a constraint name `c`, a weight `w`, and per-constraint
targets. **Weight is the gate:** a term with w = 0 contributes nothing and is skipped
entirely (exactly as the real engine will skip it), so you can leave optional
constraints in a patch at w = 0 and dial them in.

    {"c": "centroid", "w": 1.0, "target": ramp(400, 3400)}   # target Hz over the take
    {"c": "tilt",     "w": 1.0, "target": 0.5}               # +up / -down the band
    {"c": "power",    "w": 2.0, "target": 0.25}              # target RMS
    {"c": "autocorr", "w": 4.0, "hz": ramp(110, 220), "target": 0.85}  # pitch + strength
    {"c": "envelope", "w": 3.0, "target": ENV_PERC}          # short-time RMS contour
    {"c": "formants", "w": 0.0, "vowel": "a", "lifter": 30}  # optional; w=0 -> off
"""
from curves import const, ramp, breakpoints, sine

# Short-time RMS contours (per-frame, normalized). Used by the envelope constraint.
ENV_PERC = [1.0, 0.6, 0.38, 0.24, 0.15, 0.09, 0.05, 0.03]   # fast percussive decay
ENV_SWELL = [0.15, 0.4, 0.7, 0.95, 1.0, 0.95, 0.85, 0.7]    # pad-like swell

PATCHES = [
    # The headline frustrated patch from the spec: a fixed pitch period, a percussive
    # contour, a rising centroid, and a fixed loudness — none fully satisfiable at once.
    {
        "name": "frustrated_bell",
        "seconds": 2.0,
        "reg": "smooth", "reg_w": 2e-3, "init": "noise",
        "terms": [
            {"c": "centroid", "w": 1.0, "target": ramp(400, 3400)},
            {"c": "autocorr", "w": 4.0, "hz": ramp(110, 220), "target": 0.85},
            {"c": "power",    "w": 2.0, "target": 0.25},
            {"c": "envelope", "w": 3.0, "target": ENV_PERC},
            # Optional formant colour, left off. Raise w to hear it engage.
            {"c": "formants", "w": 0.0, "vowel": "a", "lifter": 30},
        ],
    },

    # The early-playable subset: only time-domain constraints (no FFT in the gradient).
    # This is the frustration the first rough VST can ship with — pitch vs. transient
    # vs. loudness — before any spectral-gradient work lands. Brightness is absent here.
    {
        "name": "time_domain_only",
        "seconds": 2.0,
        "reg": "smooth", "reg_w": 2e-3, "init": "noise",
        "terms": [
            {"c": "autocorr", "w": 4.0, "hz": ramp(110, 165), "target": 0.85},
            {"c": "power",    "w": 2.0, "target": 0.25},
            {"c": "envelope", "w": 3.0, "target": ENV_PERC},
        ],
    },

    # Optional formant module exercised: a vowel with pitch wobble, the formant
    # resonance fighting a percussive envelope.
    {
        "name": "vowel_pluck",
        "seconds": 2.0,
        "reg": "smooth", "reg_w": 2e-3, "init": "sine",
        "terms": [
            {"c": "formants", "w": 2.0, "vowel": "a", "lifter": 30},
            {"c": "autocorr", "w": 4.0, "hz": sine(130, 8, cycles=2), "target": 0.85},
            {"c": "power",    "w": 1.5, "target": 0.22},
            {"c": "envelope", "w": 2.0, "target": ENV_PERC},
        ],
    },
]
