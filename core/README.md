# tensegrity-core (planned — Phase 1+)

The portable variational synthesis engine: **plain C++17, zero plugin-framework
dependencies.** This is the product. The Tensegrity VST, a future Morphos Anchor
engine, and the offline parity harness all consume the same code here — nothing
JUCE-specific ever crosses this boundary.

Empty for now; this directory is scaffolded so the structure and intent are explicit.
Build-out begins at Phase 1 (see [`../README.md`](../README.md) roadmap).

## Planned contents

- `CMakeLists.txt` — static library target `tensegrity-core`, no plugin deps.
- `solver/` — `VariationalSolver` (Adam, fixed iteration budget), `BlockRenderer`
  (warm-start + 50% overlap-add Hann), `StreamEngine` (ring buffer, host-buffer-
  agnostic streaming, latency reporting).
- `constraints/` — base constraints with **hand-derived analytic gradients**:
  centroid, tilt, power, autocorrelation (pitch), short-time RMS envelope.
- `constraints/formants/` — the **optional** formant module (cepstral-envelope match),
  gated behind a feature toggle and its own weight.
- `regularizers/` — L2, smoothness, total variation.
- `fft/` — injectable `Fft` interface plus a vendored default for standalone builds.
- `tests/` — per-constraint gradient checks and the Python-parity harness.

API contract and design rationale: [`../docs/architecture.md`](../docs/architecture.md).
