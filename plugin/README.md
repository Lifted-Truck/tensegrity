# Tensegrity plugin (planned — Phase 3+)

A **thin** JUCE VST3 wrapper around [`../core`](../core). The plugin's only job is to
adapt the host world (MIDI, automation, audio buffers, GUI) onto the engine's
framework-free API — all DSP lives in `tensegrity-core`. If logic here grows beyond
adapting, it probably belongs in the core instead.

Empty for now; scaffolded to make the structure explicit. Build-out begins at Phase 3.

## Planned contents

- `CMakeLists.txt` — `juce_add_plugin` (VST3, IS_SYNTH), linking `tensegrity-core`,
  JUCE pulled via FetchContent (mirrors the Morphos toolchain: VS 2026 / x64).
- `source/PluginProcessor.*` — audio callback; maps MIDI note → pitch-lag target and
  APVTS parameters → `tns::PatchState`; reports engine latency to the host.
- `source/PluginEditor.*` — controls for the playable parameters: per-constraint
  targets and weights (the "focus" macros), `reg_w`, regularizer choice, and the
  init-source "voice" selector.
- `source/Parameters.h` — stable APVTS parameter ID strings (never rename post-ship).

## Build (once scaffolded)

```powershell
cmake --preset debug          # first time; downloads JUCE via FetchContent
cmake --build --preset debug
```

Prerequisites mirror Morphos: Visual Studio 2026 (Desktop development with C++),
CMake 3.22+. Output: `build/Tensegrity_artefacts/Debug/VST3/Tensegrity.vst3`.
