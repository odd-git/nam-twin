"""nam-twin: fit a circuit amp model's parameters to NAM captures taken at known knob settings.

The amp is a renderer executable (see amps/*.json): given a job file it runs raw float32 mono 48 kHz audio
through a .nam capture or through the circuit model with a parameter set and the capture's knob settings.
"""
import json, os, subprocess, sys
import numpy as np

FS = 48000
BANDS = 1000.0 * 2.0 ** (np.arange(-12, 11) / 3.0)  # third octaves, 63 Hz .. 10 kHz


class Amp:
    def __init__(self, config_path):
        self.cfg = json.load(open(config_path))
        # Machine-specific paths (renderer, captures, library) live next to it in <name>.local.json, untracked.
        self.cfg.update(json.load(open(config_path[:-5] + ".local.json")))
        self.work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work",
                                 os.path.splitext(os.path.basename(config_path))[0])
        os.makedirs(self.work, exist_ok=True)
        caps = json.load(open(self.cfg["captures"]))["captures"]
        # Captures through a pedal, or with unknown settings, say nothing about the amp alone.
        self.captures = [c for c in caps if c.get("knobs") and not c.get("pedal")]

    def path(self, name):
        return os.path.join(self.work, name)

    def _run(self, renders, params=None):
        job = self.path("job-%d.json" % os.getpid())
        json.dump({"base": self.cfg.get("base", "B"), "params": params or {}, "renders": renders}, open(job, "w"))
        if not hasattr(self, "server"):  # kept alive: the renderer's start-up costs far more than a render
            self.server = subprocess.Popen([self.cfg["renderer"], "--serve"], stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, text=True)
        self.server.stdin.write(job + "\n")
        self.server.stdin.flush()
        if self.server.stdout.readline().strip() != "0":
            raise RuntimeError("renderer failed on " + job)

    def teacher(self, x, captures=None):
        """The captures' outputs for signal x: array [capture, sample]."""
        return self._render(x, captures, lambda c: {"nam": os.path.join(self.cfg["library"], c["file"])}, None)

    def circuit(self, x, params, captures=None):
        """The circuit model's outputs for signal x at each capture's settings."""
        return self._render(x, captures, lambda c: {"knobs": c["knobs"], "channel": c["channel"]},
                            {**self.cfg.get("fixed", {}), **params})

    def _render(self, x, captures, spec, params):
        captures = self.captures if captures is None else captures
        tag = "%d" % os.getpid()
        src = self.path("in-%s.f32" % tag)
        np.asarray(x, dtype="<f4").tofile(src)
        outs = [self.path("out-%s-%d.f32" % (tag, i)) for i in range(len(captures))]
        self._run([dict(spec(c), **{"in": src, "out": o}) for c, o in zip(captures, outs)], params)
        y = np.stack([np.fromfile(o, dtype="<f4") for o in outs])
        for f in [src] + outs:
            os.remove(f)
        return y


def sweep(seconds=3.0, f0=30.0, f1=18000.0):
    """Unit-amplitude logarithmic sine sweep with short fades."""
    t = np.arange(int(seconds * FS)) / FS
    k = np.log(f1 / f0)
    x = np.sin(2 * np.pi * f0 * seconds / k * (np.exp(t / seconds * k) - 1))
    fade = int(0.01 * FS)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return x


def band_db(x, y):
    """Gain from x to y in dB per third-octave band (energy ratio). y may be [n, samples]."""
    f = np.fft.rfftfreq(len(x), 1 / FS)
    px, py = np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(y)) ** 2
    out = []
    for c in BANDS:
        m = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        out.append(10 * np.log10(py[..., m].sum(-1) / px[m].sum() + 1e-30))
    return np.stack(out, -1)


# --- Parameters: the optimiser works on u, one number per parameter (log of the value for positive
# quantities that scale, the value itself for tapers).
class Space:
    def __init__(self, spec, stage, names=None):
        self.names = names or [n for n, p in spec.items() if p["stage"] == stage]
        self.spec = spec
        f = lambda n, k: np.log(spec[n][k]) if spec[n]["log"] else spec[n][k]
        self.x0 = np.array([f(n, "nominal") for n in self.names])
        self.lo = np.array([f(n, "lo") for n in self.names])
        self.hi = np.array([f(n, "hi") for n in self.names])

    def params(self, u):
        return {n: float(np.exp(v) if self.spec[n]["log"] else v) for n, v in zip(self.names, u)}


LIN_DB = -60.0  # the captures are level-independent (linear) here: measured, -60 and -50 dBFS agree


def fit_linear(amp, fixed=None, captures=None, verbose=True, names=None, truth=None, lam=0.0, target=None):
    """Stage 1: match each capture's small-signal third-octave response. Returns (params, report).
    With `truth` (a parameter set) the target is the circuit itself instead of the captures: the self-test.
    `lam` weights a prior that pulls each parameter toward its nominal value (in units of a quarter of its
    range); lam = 0 is the unpenalised fit. `target` lets the caller pass a precomputed teacher response."""
    from scipy.optimize import least_squares
    captures = amp.captures if captures is None else captures
    x = sweep(1.5) * 10 ** (LIN_DB / 20)
    if target is None:
        target = band_db(x, amp.teacher(x, captures) if truth is None else amp.circuit(x, truth, captures))
    space = Space(amp.cfg["params"], "linear", names)
    sd = np.maximum((space.hi - space.lo) / 4, 1e-9)

    def data(u):
        return (band_db(x, amp.circuit(x, {**(fixed or {}), **space.params(u)}, captures)) - target).ravel()

    def residual(u):
        return np.concatenate([data(u), lam * (u - space.x0) / sd])

    def jac(u):
        # Forward differences with an absolute step. scipy's own step is relative to the value, and for a
        # parameter sitting at 0 (a linear taper) it shrinks to 1e-8, where the model does not respond.
        r, h = residual(u), 1e-3
        cols = []
        for i in range(len(u)):
            v = u.copy()
            v[i] += h if u[i] + h <= space.hi[i] else -h
            cols.append((residual(v) - r) / (v[i] - u[i]))
        return np.stack(cols, 1)

    r0 = data(space.x0)
    fit = least_squares(residual, space.x0, jac=jac, bounds=(space.lo, space.hi), loss="soft_l1", f_scale=2.0,
                        x_scale=np.maximum(0.1, np.abs(space.hi - space.lo) / 4),
                        xtol=1e-10, ftol=1e-10, verbose=2 if verbose else 0, max_nfev=80)
    r1 = data(fit.x)
    # How well the captures pin each parameter down: its standard error, from the Jacobian at the optimum.
    jtj = fit.jac.T @ fit.jac
    sigma2 = (r1 ** 2).sum() / max(1, len(r1) - len(fit.x))
    err = np.sqrt(np.diag(np.linalg.pinv(jtj)) * sigma2)
    report = {"rms_db_before": rms_rows(r0, len(captures)), "rms_db_after": rms_rows(r1, len(captures)),
              "stderr": dict(zip(space.names, err.tolist())), "at_bound": [
                  n for n, v, a, b in zip(space.names, fit.x, space.lo, space.hi) if min(v - a, b - v) < 1e-3 * (b - a)]}
    return {**(fixed or {}), **space.params(fit.x)}, report


def rms_rows(r, n):
    return np.sqrt((r.reshape(n, -1) ** 2).mean(1)).tolist()


def heldout(amp, lams, folds=5, verbose=True):
    """Leave-k-out: fit on all captures but a fold, predict the fold, for each prior weight.
    Every finished fold is saved to work/<amp>/heldout.json, and a rerun skips it: an interrupted run resumes.
    Returns {lam: mean held-out rms dB}."""
    caps = amp.captures
    x = sweep(1.5) * 10 ** (LIN_DB / 20)
    teacher = band_db(x, amp.teacher(x))  # [capture, band]
    path = amp.path("heldout.json")
    done = json.load(open(path)) if os.path.exists(path) else {}
    out = {}
    for lam in lams:
        key = "%g" % lam
        per = done.setdefault(key, {"folds": {}, "per_capture": [None] * len(caps)})
        for k in range(folds):
            if str(k) in per["folds"]:
                continue
            test = [i for i in range(len(caps)) if i % folds == k]
            train = [i for i in range(len(caps)) if i % folds != k]
            params, _ = fit_linear(amp, captures=[caps[i] for i in train], verbose=False, lam=lam,
                                   target=teacher[train])
            pred = band_db(x, amp.circuit(x, params, [caps[i] for i in test]))
            for i, e in zip(test, np.sqrt(((pred - teacher[test]) ** 2).mean(1))):
                per["per_capture"][i] = float(e)
            per["folds"][str(k)] = params
            json.dump(done, open(path + ".tmp", "w"), indent=1)
            os.replace(path + ".tmp", path)
            if verbose:
                print("lam %-6g fold %d/%d done" % (lam, k + 1, folds), flush=True)
        out[lam] = float(np.mean(per["per_capture"]))
        if verbose:
            print("lam %-6g held-out mean rms %.2f dB" % (lam, out[lam]), flush=True)
    return out


if __name__ == "__main__":
    amp = Amp(sys.argv[1])
    if sys.argv[2] == "selftest":
        # Fake captures from the circuit with known, moved parameters: the fit must find them again.
        spec = amp.cfg["params"]
        names = list(spec)[:6]
        truth = {n: spec[n]["nominal"] * 1.25 if spec[n]["log"] else spec[n]["nominal"] + 0.6 for n in names}
        got, rep = fit_linear(amp, names=names, truth=truth, verbose=False)
        u = lambda n, v: np.log(v) if spec[n]["log"] else v
        for n in names:
            # A parameter the captures barely constrain may sit anywhere along its flat direction: it is
            # held to 3 standard errors, the well-determined ones to 3%.
            tol = max(0.03, 3 * rep["stderr"][n])
            print("%-14s truth %10.4g  found %10.4g  (allowed %.2f)" % (n, truth[n], got[n], tol))
            assert abs(u(n, got[n]) - u(n, truth[n])) < tol, n + " not recovered"
        worst = max(rep["rms_db_after"])
        print("worst capture: %.3f dB" % worst)
        assert worst < 0.15, "the fit does not reproduce its own fake captures"
        print("selftest ok")
    elif sys.argv[2] == "heldout":
        lams = [float(v) for v in sys.argv[3].split(",")]
        heldout(amp, lams)
    elif sys.argv[2] == "linear":
        lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        params, rep = fit_linear(amp, lam=lam)
        json.dump(params, open(amp.path("params.json"), "w"), indent=1)
        json.dump(rep, open(amp.path("linear-report.json"), "w"), indent=1)
        spec = amp.cfg["params"]
        print("\n%-28s before  after  (rms dB over third octaves, level included)" % "capture")
        for c, a, b in zip(amp.captures, rep["rms_db_before"], rep["rms_db_after"]):
            print("%-28s %6.1f %6.1f" % (c["file"][:-4][-28:], a, b))
        print("mean %.2f -> %.2f dB\n" % (np.mean(rep["rms_db_before"]), np.mean(rep["rms_db_after"])))
        for n, e in rep["stderr"].items():
            unit = "x/÷ %.2f" % np.exp(e) if spec[n]["log"] else "± %.2f" % e
            print("%-14s %10.4g -> %10.4g  (%s)%s" % (n, spec[n]["nominal"], params[n], unit, "  AT BOUND" if n in rep["at_bound"] else ""))
