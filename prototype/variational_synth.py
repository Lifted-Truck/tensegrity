"""
Variational synthesis — offline prototype / verifier.

Principle: the waveform for each block is not GENERATED forward, it is the
SOLUTION of an optimization. You declare constraints (what must be true of the
sound) and a regularizer (which of the many waveforms satisfying them you
prefer). Gradient descent finds a block of samples minimizing
    E(x) = Sum_k  w_k * C_k(x)   +   reg_w * R(x)
The constraint targets, weights, regularizer and initialization are the
"instrument": move them over time and you are playing.

Dependency: autograd (pip install autograd). Pure-numpy semantics, so every
constraint below ports directly to hand-written C++ gradients for the real-time
VST — there is no framework lock-in. Swap `autograd.numpy` for `torch` verbatim
if you'd rather use GPU autodiff.
"""
import autograd.numpy as anp
import autograd.numpy.fft as afft
from autograd import grad
import numpy as np
from scipy.io import wavfile

SR     = 44100
BLOCK  = 1024
HOP    = BLOCK // 2          # 50% overlap-add
ITERS  = 90                  # offline: track to near-convergence. real-time: ~8-16, warm-started.
LR     = 0.05

freqs_pos = np.fft.fftfreq(BLOCK, d=1.0 / SR)[: BLOCK // 2]

# ----------------------------------------------------------------------------
# Differentiable constraints. Each returns a scalar penalty given block x.
# ----------------------------------------------------------------------------
def _mag(x):
    X = afft.fft(x)
    return anp.abs(X)[: BLOCK // 2]

def c_centroid(x, target_hz):
    m = _mag(x)
    centroid = anp.sum(freqs_pos * m) / (anp.sum(m) + 1e-9)
    return ((centroid - target_hz) / 1000.0) ** 2          # scaled to ~unit

def c_tilt(x, slope):
    # push spectral energy up (slope>0) or down (slope<0) the band
    m = _mag(x)
    m = m / (anp.sum(m) + 1e-9)
    ramp = np.linspace(-1, 1, BLOCK // 2)
    return -slope * anp.sum(ramp * m)                       # reward, not squared error

def c_power(x, target_rms):
    rms = anp.sqrt(anp.mean(x ** 2) + 1e-9)
    return (rms - target_rms) ** 2

def c_autocorr(x, lag, target):
    # normalized autocorrelation at `lag` in [-1,1]; >0 imposes periodicity -> pitch
    xc = x - anp.mean(x)
    denom = anp.mean(xc ** 2) + 1e-9
    r = anp.mean(xc[:-lag] * xc[lag:]) / denom
    return (r - target) ** 2

def _envelope_cepstral(x, lifter=30):
    # cepstrally-smoothed log-magnitude envelope (real cepstrum, lowpass in quefrency)
    logmag = anp.log(anp.abs(afft.fft(x)) + 1e-6)
    cep = anp.real(afft.ifft(logmag))
    w = np.zeros(BLOCK)
    w[:lifter] = 1.0
    w[-lifter + 1:] = 1.0
    env = anp.real(afft.fft(cep * w))[: BLOCK // 2]
    return env

def c_formants(x, formants, lifter=30):
    # formants: list of (center_hz, gain_db, bandwidth_hz)
    # match cepstrally-smoothed envelope (shape only, not level) to sum of Gaussians
    env = _envelope_cepstral(x, lifter)
    env = env - anp.mean(env)  # shape-only, insensitive to overall loudness
    target = np.zeros(BLOCK // 2)
    for f_hz, gain_db, bw_hz in formants:
        # Gaussian bump centered at f_hz with width bw_hz, height gain_db
        target = target + gain_db * anp.exp(-0.5 * ((freqs_pos - f_hz) / bw_hz) ** 2)
    target = target - np.mean(target)
    return anp.mean((env - target) ** 2)

def c_envelope_match(x, target_env, lifter=30):
    # generalization of c_formants: match the cepstral spectral envelope (shape only)
    # to a GIVEN target envelope array -- e.g. one extracted from a real sound file via
    # _envelope_cepstral. Same gradient path as c_formants; only the target differs.
    env = _envelope_cepstral(x, lifter)
    env = env - anp.mean(env)
    tgt = target_env - anp.mean(target_env)
    return anp.mean((env - tgt) ** 2)

def c_envelope(x, target_env):
    # match short-time RMS contour -> attack / transient shape
    nf = len(target_env)
    fl = BLOCK // nf
    e = anp.array([anp.sqrt(anp.mean(x[i*fl:(i+1)*fl] ** 2) + 1e-9) for i in range(nf)])
    e = e / (anp.max(e) + 1e-9)
    return anp.mean((e - target_env) ** 2)

# ----------------------------------------------------------------------------
# Regularizers — the "voice". Same constraints, different reg -> different timbre.
# ----------------------------------------------------------------------------
def reg_l2(x):     return anp.mean(x ** 2)
def reg_smooth(x): return anp.mean((x[1:] - x[:-1]) ** 2)    # favors low frequencies
def reg_tv(x):     return anp.mean(anp.abs(x[1:] - x[:-1]))  # favors flat segments + edges

# ----------------------------------------------------------------------------
# Per-block solver (hand-rolled Adam over autograd gradient)
# ----------------------------------------------------------------------------
def solve_block(energy_fn, init, iters=ITERS, lr=LR):
    g = grad(energy_fn)
    x = init.copy()
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        gr = g(x)
        m = b1 * m + (1 - b1) * gr
        v = b2 * v + (1 - b2) * gr ** 2
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        x = x - lr * mh / (np.sqrt(vh) + eps)
    return x

# ----------------------------------------------------------------------------
# Render: time-varying constraint set = the performance gesture
# ----------------------------------------------------------------------------
def _init_block(source):
    # block-0 basin selector (the "voice"). Same constraints, different init -> different sound.
    if source == "noise":
        return np.random.randn(BLOCK) * 0.01            # broad basin, breathier
    if source == "filtered_noise":
        k = np.hanning(9); k = k / k.sum()
        return np.convolve(np.random.randn(BLOCK), k, mode="same") * 0.01
    if source == "sine":
        return 0.01 * np.sin(2 * np.pi * 220 * np.arange(BLOCK) / SR)  # tonal basin
    raise ValueError(f"unknown init_source {source!r}")


def render(seconds, constraint_fn, reg, reg_w, init_source="noise"):
    n_blocks = int(seconds * SR / HOP)
    win = np.hanning(BLOCK)
    out = np.zeros(HOP * n_blocks + BLOCK)
    prev = _init_block(init_source)                 # block-0 seed (default: noise, as before)
    for i in range(n_blocks):
        t = i / max(n_blocks - 1, 1)                # 0..1 performance position
        terms = constraint_fn(t)                    # list of (weight, constraint)
        def energy(x, terms=terms):
            e = 0.0
            for w, c in terms:
                e = e + w * c(x)
            return e + reg_w * reg(x)
        init = np.roll(prev, -HOP)                  # warm-start from prev block -> continuity
        x = solve_block(energy, init)
        out[i*HOP : i*HOP + BLOCK] += x * win        # overlap-add
        prev = x
        if i % 20 == 0:
            print(f"  block {i}/{n_blocks}")
    return out

# ----------------------------------------------------------------------------
# A deliberately OVER-CONSTRAINED, mutually-frustrated demo.
# The novelty lives here: centroid, a fixed pitch period, and a percussive
# envelope cannot all be satisfied cleanly, so the solver negotiates a
# compromise no forward synth would produce.
# ----------------------------------------------------------------------------
ENV_PERC = np.array([1.0, 0.6, 0.38, 0.24, 0.15, 0.09, 0.05, 0.03])  # fast decay

# Vowel formant sets (Hz, gain_dB, bandwidth_Hz)
VOWELS = {
    'i':  [(280, 8.0, 60),   (2600, 5.0, 120)],
    'e':  [(530, 7.0, 80),   (2000, 5.0, 100)],
    'a':  [(730, 8.0, 90),   (1090, 6.0, 110)],
    'o':  [(570, 7.0, 90),   (1050, 4.0, 120)],
    'u':  [(440, 7.0, 70),   (1340, 4.0, 100)],
}

def demo(t):
    pitch_lag = int(SR / (110 * (1 + t)))            # glide A2 -> A3 over the take
    bright    = 400 + 3000 * t                        # rising centroid
    return [
        (1.0, lambda x: c_centroid(x, bright)),
        (4.0, lambda x: c_autocorr(x, pitch_lag, 0.85)),   # strong periodicity
        (2.0, lambda x: c_power(x, 0.25)),
        (3.0, lambda x: c_envelope(x, ENV_PERC)),          # percussive contour each block
    ]

def demo_vowel(t):
    """Morph through vowels (a -> e -> i) with pitch oscillation, formants + envelope fight."""
    pitch_hz = 110 * (1 + 0.3 * np.sin(2 * np.pi * t))
    pitch_lag = int(SR / pitch_hz)
    
    # morph vowels: a -> e -> i over a 1-second loop
    vowel_pos = (t % 1.0) * 3  # 0..3 spans three transitions
    if vowel_pos < 1.0:
        alpha = vowel_pos
        f1 = (1 - alpha) * VOWELS['a'][0][0] + alpha * VOWELS['e'][0][0]
        f2 = (1 - alpha) * VOWELS['a'][1][0] + alpha * VOWELS['e'][1][0]
        formants = [(f1, 7.5, 85), (f2, 5.5, 105)]
    elif vowel_pos < 2.0:
        alpha = vowel_pos - 1.0
        f1 = (1 - alpha) * VOWELS['e'][0][0] + alpha * VOWELS['i'][0][0]
        f2 = (1 - alpha) * VOWELS['e'][1][0] + alpha * VOWELS['i'][1][0]
        formants = [(f1, 7.5, 70), (f2, 5.5, 110)]
    else:
        alpha = vowel_pos - 2.0
        f1 = (1 - alpha) * VOWELS['i'][0][0] + alpha * VOWELS['a'][0][0]
        f2 = (1 - alpha) * VOWELS['i'][1][0] + alpha * VOWELS['a'][1][0]
        formants = [(f1, 8.0, 90), (f2, 5.5, 110)]
    
    return [
        (2.0, lambda x, f=formants: c_formants(x, f, lifter=30)),     # resonance pulls
        (4.0, lambda x: c_autocorr(x, pitch_lag, 0.85)),              # periodicity holds
        (1.5, lambda x: c_power(x, 0.22)),
        (2.0, lambda x: c_envelope(x, ENV_PERC)),                     # percussive kills resonance
    ]

if __name__ == "__main__":
    np.random.seed(0)
    sig = render(seconds=2.0, constraint_fn=demo, reg=reg_smooth, reg_w=2e-3)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.9
    wavfile.write("variational_demo.wav", SR, (sig * 32767).astype(np.int16))
    print("rendered", len(sig), "samples; peak", round(float(np.max(np.abs(sig))), 3))
