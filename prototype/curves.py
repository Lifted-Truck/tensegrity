"""
Time-curve primitives for declarative patches.

A patch parameter (a constraint target, a weight, a pitch) is either a constant or a
function of the performance position `t` in [0, 1] (0 = start of the take, 1 = end).
Each helper below returns such a function. `as_curve` accepts either a bare number or
an existing callable so patch authors can write `0.25` or `ramp(400, 3400)`
interchangeably.
"""
import numpy as np


def const(v):
    """Hold a fixed value for the whole take."""
    return lambda t: float(v)


def ramp(a, b):
    """Linear sweep from a (at t=0) to b (at t=1)."""
    return lambda t: float(a + (b - a) * t)


def breakpoints(points):
    """Piecewise-linear curve through (t, value) points; clamps outside the range.

    e.g. breakpoints([(0.0, 730), (0.5, 530), (1.0, 280)]) sweeps a formant.
    """
    ts = [float(p[0]) for p in points]
    vs = [float(p[1]) for p in points]
    return lambda t: float(np.interp(t, ts, vs))


def sine(center, depth, cycles=1.0, phase=0.0):
    """Sinusoidal modulation around `center` with amplitude `depth`."""
    return lambda t: float(center + depth * np.sin(2 * np.pi * (cycles * t + phase)))


def as_curve(x):
    """Coerce a number-or-callable into a callable f(t). Bare numbers become constants."""
    return x if callable(x) else const(x)
