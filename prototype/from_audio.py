"""
Audio-derived constraints -- proof of pipeline with test tones.

The on-thesis version of "derive constraints from sound files": extract the cepstral
spectral envelope (the 'timbre filter') from TWO sources and impose BOTH on a new
sound at comparable weight, so they fight on the same spectral degrees of freedom.
The output is the solver's negotiated compromise between two incompatible timbres --
NOT a crossfade.

For contrast it also renders the well-determined version (one envelope + pitch), which
should collapse to a clean filtered tone -- the kill-criterion baseline. If the "fight"
output is audibly distinct from both sources AND from that baseline, the idea holds.

Run:
    .venv\\Scripts\\python from_audio.py
Writes (next to this script):
    src_A.wav, src_B.wav   -- the two analyzed sources (dark vs bright timbre)
    out_single.wav         -- A's envelope + pitch only (well-determined baseline)
    out_fight.wav          -- A's AND B's envelopes fighting (the novel compromise)
and prints, per output, the mean cepstral-envelope distance to A and to B. A genuine
compromise sits BETWEEN the two; the baseline sits on top of A.

Drop in real recordings later by replacing make_tone(...) with wavfile.read(...).
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

import variational_synth as vs

SR, BLOCK, HOP = vs.SR, vs.BLOCK, vs.HOP
SECONDS = 2.0

# --- which knobs to turn (kept at the top so this stays an edit surface) -----------
PITCH_HZ = 120.0      # shared pitch so the conflict is purely about timbre/envelope
LIFTER   = 30         # cepstral resolution of the extracted envelope
W_A, W_B = 2.0, 2.0   # the two timbre pulls -- keep comparable to force a compromise
W_PITCH  = 4.0
W_POWER  = 1.5


def make_tone(f0, formants, seconds=SECONDS, rolloff=1.0):
    """A harmonic tone at f0 whose spectrum is shaped by a formant envelope.

    formants: list of (center_hz, gain_db, bandwidth_hz). This is just a convenient way
    to manufacture two sources with clearly different timbres to analyze.
    """
    n = int(seconds * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    k = 1
    while k * f0 < SR / 2 * 0.95:
        fk = k * f0
        gdb = 0.0
        for fc, g, bw in formants:
            gdb += g * np.exp(-0.5 * ((fk - fc) / bw) ** 2)
        amp = (10.0 ** (gdb / 20.0)) / (k ** rolloff)
        sig += amp * np.sin(2 * np.pi * fk * t)
        k += 1
    return sig / (np.max(np.abs(sig)) + 1e-9) * 0.9


def load_wav(path, seconds=SECONDS):
    """Read a real recording -> mono float in [-1,1] at the engine SR, trimmed/padded
    to `seconds`. This is the drop-in replacement for make_tone: same return contract
    (a 1-D float array the analysis treats identically), so nothing downstream changes.
    """
    sr, d = wavfile.read(path)
    d = d.astype(np.float64)
    if d.ndim > 1:                       # stereo -> mono (average channels)
        d = d.mean(axis=1)
    # normalize by the integer full-scale of the source dtype (int16/int32) or peak
    peak = np.max(np.abs(d)) + 1e-9
    d = d / peak
    if sr != SR:                         # resample to the engine's rate (e.g. 48k->44.1k)
        from math import gcd
        g = gcd(sr, SR)
        d = resample_poly(d, SR // g, sr // g)
    n = int(seconds * SR)
    if len(d) >= n:
        d = d[:n]
    else:
        d = np.pad(d, (0, n - len(d)))
    return d / (np.max(np.abs(d)) + 1e-9) * 0.9


def analyze_envelopes(signal, lifter=LIFTER):
    """Per-synthesis-block cepstral spectral envelope across the whole signal."""
    n_blocks = max((len(signal) - BLOCK) // HOP, 1)
    envs = []
    for i in range(n_blocks):
        x = signal[i * HOP: i * HOP + BLOCK]
        if len(x) < BLOCK:
            x = np.pad(x, (0, BLOCK - len(x)))
        envs.append(np.asarray(vs._envelope_cepstral(x, lifter)))
    return envs


def env_at(envs, t):
    return envs[min(int(t * (len(envs) - 1) + 0.5), len(envs) - 1)]


def build_constraints(envsA, envsB=None):
    """A's timbre, optionally B's timbre too, plus pitch + power. envsB=None -> baseline."""
    lag = max(1, int(SR / PITCH_HZ))

    def cf(t):
        terms = [(W_A, lambda x, e=env_at(envsA, t): vs.c_envelope_match(x, e, LIFTER))]
        if envsB is not None:
            terms.append((W_B, lambda x, e=env_at(envsB, t): vs.c_envelope_match(x, e, LIFTER)))
        terms.append((W_PITCH, lambda x: vs.c_autocorr(x, lag, 0.85)))
        terms.append((W_POWER, lambda x: vs.c_power(x, 0.2)))
        return terms

    return cf


def env_distance(signal, envs_ref):
    """Mean shape-only cepstral-envelope distance between a signal and a reference set."""
    eo = analyze_envelopes(signal)
    n = min(len(eo), len(envs_ref))
    d = 0.0
    for i in range(n):
        a = eo[i] - np.mean(eo[i])
        b = envs_ref[i] - np.mean(envs_ref[i])
        d += float(np.mean((a - b) ** 2))
    return d / max(n, 1)


def write(name, sig):
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.9
    wavfile.write(name, SR, (sig * 32767).astype(np.int16))


def render(cf):
    np.random.seed(0)
    sig = vs.render(SECONDS, cf, vs.reg_smooth, 2e-3, init_source="noise")
    return sig / (np.max(np.abs(sig)) + 1e-9) * 0.9


if __name__ == "__main__":
    # Two real recordings with very different timbres -- their cepstral envelopes are the
    # two incompatible pulls. (Swap these paths, or fall back to make_tone(...) for tones.)
    print("loading + analyzing sources ...")
    srcA = load_wav("test1_balloon.wav")   # balloon squeak  -> timbre A
    srcB = load_wav("test2_roar.wav")      # roar            -> timbre B
    write("src_A.wav", srcA)
    write("src_B.wav", srcB)
    envsA = analyze_envelopes(srcA)
    envsB = analyze_envelopes(srcB)

    print("rendering baseline (A's envelope + pitch only) ...")
    out_single = render(build_constraints(envsA, None))
    write("out_single.wav", out_single)

    print("rendering fight (A's AND B's envelopes, comparable weight) ...")
    out_fight = render(build_constraints(envsA, envsB))
    write("out_fight.wav", out_fight)

    print("\n%-12s  %10s  %10s" % ("output", "dist->A", "dist->B"))
    for name, sig in [("src_A", srcA), ("src_B", srcB),
                      ("out_single", out_single), ("out_fight", out_fight)]:
        print("%-12s  %10.4f  %10.4f" % (name, env_distance(sig, envsA), env_distance(sig, envsB)))
    print("\nRead: out_single should sit ON A (small dist->A, large dist->B).")
    print("out_fight should sit BETWEEN A and B -- the negotiated compromise.")
