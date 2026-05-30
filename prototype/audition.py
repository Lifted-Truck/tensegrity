"""
Audition harness — render declarative patches and check them against the ear AND the
numbers (the verifier discipline from the spec).

Usage:
    python audition.py                 # render every patch in patches.py
    python audition.py vowel_pluck     # render one patch by name

For each patch it writes out_<name>.wav and prints the headline verifier metrics:
  - centroid_r        : correlation between the rendered spectral centroid and the
                        patch's intended centroid curve (target ~0.95 when tracking well)
  - periodicity_mean  : mean normalized autocorrelation at the target pitch lag
                        (target ~0.8 — confirms the pitch constraint actually holds)

A patch can converge to a number while sounding dead, so always listen too. And apply
the kill criterion: if it sounds like a filtered oscillator, the constraints aren't
fighting — add tension or cut the patch.

This driver reuses variational_synth.py (the verified prototype / gradient oracle)
for all solving; it only adds the declarative layer and the measurement readouts.
"""
import sys
import numpy as np
from scipy.io import wavfile

import variational_synth as vs
from curves import as_curve
import patches as patchlib

REGS = {"l2": vs.reg_l2, "smooth": vs.reg_smooth, "tv": vs.reg_tv}


def _formant_list(term, t):
    """Resolve a formants term into [(center_hz, gain_db, bandwidth_hz), ...] at time t."""
    if "vowel" in term:
        return [(float(f), float(g), float(bw)) for (f, g, bw) in vs.VOWELS[term["vowel"]]]
    return [(as_curve(f)(t), float(g), float(bw)) for (f, g, bw) in term["formants"]]


def build_constraint_fn(patch):
    """Compile a declarative patch into the constraint_fn(t) -> [(w, c), ...] the solver wants."""
    terms = patch["terms"]

    def constraint_fn(t):
        out = []
        for term in terms:
            w = float(term["w"])
            if w <= 0.0:                       # weight-gating: skip zeroed terms entirely
                continue
            c = term["c"]
            if c == "centroid":
                tgt = as_curve(term["target"])(t)
                out.append((w, lambda x, tgt=tgt: vs.c_centroid(x, tgt)))
            elif c == "tilt":
                slope = as_curve(term["target"])(t)
                out.append((w, lambda x, s=slope: vs.c_tilt(x, s)))
            elif c == "power":
                tgt = as_curve(term["target"])(t)
                out.append((w, lambda x, tgt=tgt: vs.c_power(x, tgt)))
            elif c == "autocorr":
                hz = as_curve(term["hz"])(t)
                lag = max(1, int(vs.SR / hz))
                tgt = as_curve(term.get("target", 0.85))(t)
                out.append((w, lambda x, lag=lag, tgt=tgt: vs.c_autocorr(x, lag, tgt)))
            elif c == "envelope":
                env = np.asarray(term["target"], dtype=float)
                out.append((w, lambda x, env=env: vs.c_envelope(x, env)))
            elif c == "formants":
                fmts = _formant_list(term, t)
                lifter = int(term.get("lifter", 30))
                out.append((w, lambda x, f=fmts, l=lifter: vs.c_formants(x, f, lifter=l)))
            else:
                raise ValueError(f"unknown constraint {c!r} in patch {patch['name']!r}")
        return out

    return constraint_fn


def render_patch(patch):
    np.random.seed(0)                          # reproducible renders
    cf = build_constraint_fn(patch)
    reg = REGS[patch.get("reg", "smooth")]
    sig = vs.render(patch["seconds"], cf, reg, patch.get("reg_w", 2e-3),
                    init_source=patch.get("init", "noise"))
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.9
    return sig


def analyze(sig, patch):
    """Re-measure the rendered audio on the synthesis grid and compare to intent."""
    N, HOP = vs.BLOCK, vs.HOP
    n = max((len(sig) - N) // HOP, 1)
    active = {term["c"]: term for term in patch["terms"] if float(term["w"]) > 0.0}

    measured_cen, intended_cen, periodicity = [], [], []
    for i in range(n):
        t = i / max(n - 1, 1)
        x = sig[i * HOP: i * HOP + N]
        if len(x) < N:
            break
        if "centroid" in active:
            m = np.abs(np.fft.fft(x))[: N // 2]
            measured_cen.append(np.sum(vs.freqs_pos * m) / (np.sum(m) + 1e-9))
            intended_cen.append(as_curve(active["centroid"]["target"])(t))
        if "autocorr" in active:
            hz = as_curve(active["autocorr"]["hz"])(t)
            lag = max(1, int(vs.SR / hz))
            xc = x - np.mean(x)
            periodicity.append(np.mean(xc[:-lag] * xc[lag:]) / (np.mean(xc ** 2) + 1e-9))

    report = {}
    if len(intended_cen) > 1:
        report["centroid_r"] = float(np.corrcoef(measured_cen, intended_cen)[0, 1])
    if periodicity:
        report["periodicity_mean"] = float(np.mean(periodicity))
    return report


def run(patch):
    print(f"[{patch['name']}] rendering {patch['seconds']}s "
          f"(reg={patch.get('reg', 'smooth')}, init={patch.get('init', 'noise')}) ...")
    sig = render_patch(patch)
    fname = f"out_{patch['name']}.wav"
    wavfile.write(fname, vs.SR, (sig * 32767).astype(np.int16))
    metrics = analyze(sig, patch)
    summary = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()) or "(no metric-bearing terms)"
    print(f"  -> {fname}   {summary}\n")


if __name__ == "__main__":
    selected = sys.argv[1:]
    chosen = [p for p in patchlib.PATCHES if not selected or p["name"] in selected]
    if not chosen:
        print(f"no patch matched {selected!r}. Available: "
              f"{', '.join(p['name'] for p in patchlib.PATCHES)}")
        sys.exit(1)
    for patch in chosen:
        run(patch)
