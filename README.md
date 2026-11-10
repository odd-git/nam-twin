# nam-twin

Turn a set of [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) captures of one
amplifier into a model of that amplifier with its real controls.

A NAM capture is a snapshot of an amp at one position of its knobs. Given several captures taken at
known settings, plus a circuit model of the amp, nam-twin finds the values the schematic does not give
(component tolerances, pot tapers, supply voltages, tube spread, levels) so that the circuit matches every
capture at that capture's settings. The knobs then behave as the circuit says they do, between and beyond
the captured positions.

This is system identification, not a conversion of the network: the captures stand in for the amplifier
(any test signal can be played through them), the schematic supplies the structure.

**Status: early.** Stage 1 (small-signal response) runs; the rest is planned.

## How many captures

With K continuous controls:

| Captures | What you get |
|---|---|
| 1 | The schematic does everything; the capture only sets levels and overall voicing |
| 2K+1 | Centre plus min and max of every knob: each pot's taper is identified (recommended minimum) |
| 2K+1, plus 3 along the gain and 1 per input | The nonlinear behaviour along the volume as well |

The fit reports how well the given captures pin each parameter down; what they do not determine stays at
its schematic value.

## Stages

1. **Linear**: third-octave response of each capture at a level where it is linear, against the circuit.
   Fits pot tapers and values, tone-stack and coupling capacitors, feedback, transformer corners.
2. **Nonlinear** (planned): compression and harmonics against level. Supplies, bias, tubes, sag.
3. **Joint refinement and leave-one-out validation** (planned): fit on all captures but one, predict it.
4. **Neural residual** (planned): a small knob-conditioned WaveNet on what the circuit misses, exported as
   a `.nam` and kept only if it improves the leave-one-out error.

## Use

    python3 namtwin.py amps/head200.json selftest   # fake captures with known parameters must be recovered
    python3 namtwin.py amps/head200.json linear     # stage 1; writes work/<amp>/params.json

Needs numpy and scipy.

## Adding an amp

nam-twin knows nothing about any amplifier. An amp is:

- `amps/<name>.json`: the parameters to fit, each with nominal value, bounds and scale;
- `amps/<name>.local.json` (untracked): paths to the renderer, the captures list and the `.nam` files;
- a captures list: `{"captures": [{"file", "channel", "knobs": {...}}, ...]}`;
- a **renderer**: an executable taking a job file,

      { "base": ..., "params": { name: value, ... },
        "renders": [ { "in": path, "out": path, "nam": path }
                   | { "in": path, "out": path, "knobs": {...}, "channel": ... } ] }

  that plays each `in` (raw float32 mono, 48 kHz) through the capture or through the circuit model with
  those parameters and knobs, and writes `out`.

The first amp is a 1968 200 W head; its circuit model and renderer live in a separate plugin project.
Captures are not included: bring your own.

## Credits

Built on [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) by Steven Atkinson: the
captures are NAM models, played back with
[NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore). This project is not affiliated
with or endorsed by NAM.
