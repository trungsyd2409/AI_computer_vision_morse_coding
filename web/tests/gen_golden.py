"""Generate tests/golden.json from the real Python detector and composer.

The browser port (public/js/blinks.js and morse.js) must behave exactly like
blink_morse/blinks.py and blink_morse/morse.py. This script runs the Python
modules on a set of deterministic traces and stores every output, and
logic.test.mjs replays the same traces through the JavaScript port.

Usage (from the web/ folder):
    python tests/gen_golden.py                 # uses ../blink_morse
    python tests/gen_golden.py path/to/blink_morse
"""

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str, folder: Path):
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def find_package() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    return (HERE.parent.parent / "blink_morse").resolve()


# -- trace generation ---------------------------------------------------------

def make_trace(seed: int, fps: float, seconds: float, time_unit: float,
               noise: float, drift: bool, winks: bool, face_loss: bool):
    """A list of frames: [t, left_raw, right_raw, reset_flag]."""
    rng = random.Random(seed)
    # Plan a list of (start, end, left_closed, right_closed) closures.
    closures = []
    t = rng.uniform(0.3, 0.8)
    while t < seconds - 1:
        kind = rng.random()
        if kind < 0.45:
            length = rng.uniform(0.4, 1.3) * time_unit        # dot
        elif kind < 0.8:
            length = rng.uniform(1.6, 4.0) * time_unit        # dash
        elif kind < 0.9:
            length = rng.uniform(5.0, 12.0) * time_unit       # long hold
        else:
            length = rng.uniform(0.2, 0.5) * time_unit        # very short
        left = right = True
        if winks and rng.random() < 0.2:
            if rng.random() < 0.5:
                right = False
            else:
                left = False
        closures.append((t, t + length, left, right))
        gap = rng.choice([1, 1, 1, 2, 3, 3, 4, 7, 9]) * time_unit
        t += length + gap * rng.uniform(0.8, 1.3)

    lost_at = []
    if face_loss:
        lost_at = sorted(rng.uniform(1, seconds - 1) for _ in range(2))

    frames = []
    now = 5.0 + rng.random()  # arbitrary clock origin
    start = now
    base_l = rng.uniform(0.02, 0.35)
    base_r = rng.uniform(0.02, 0.35)
    step = 1.0 / fps
    next_loss = 0
    while now - start < seconds:
        rel = now - start
        l = base_l
        r = base_r
        if drift:
            wob = 0.12 * (1 + math.sin(rel * 0.7)) / 2
            l += wob
            r += wob * 0.8
        for (a, b, cl, cr) in closures:
            if a <= rel < b:
                # ramp over ~2 frames like a real eyelid
                ramp = min(1.0, (rel - a) / (2 * step), (b - rel) / (2 * step) + 0.2)
                if cl:
                    l = base_l + (rng.uniform(0.7, 0.98) - base_l) * ramp
                if cr:
                    r = base_r + (rng.uniform(0.7, 0.98) - base_r) * ramp
        l = min(max(l + rng.gauss(0, noise), 0.0), 1.0)
        r = min(max(r + rng.gauss(0, noise), 0.0), 1.0)
        reset = False
        if next_loss < len(lost_at) and rel >= lost_at[next_loss]:
            reset = True
            next_loss += 1
        frames.append([now, l, r, reset])
        now += step * rng.uniform(0.85, 1.2)  # timestamp jitter
    return frames


def run_detector(blinks, morse, params, frames):
    det = blinks.BlinkDetector(params["close_threshold"], params["time_unit"])
    comp = morse.MorseComposer(params["letter_gap"], params["word_gap"], params["max_text"])
    out = []
    for t, l, r, reset in frames:
        if reset:
            det.reset(t)
        events = det.update(l, r, t)
        sym = []
        for e in events:
            ce = comp.add_symbol("." if e.kind == blinks.BlinkKind.DOT else "-")
            sym.append({
                "kind": "dot" if e.kind == blinks.BlinkKind.DOT else "dash",
                "started": e.started,
                "fired": e.fired,
                "composer": [ce.kind.name.lower(), ce.value],
            })
        pause = det.pause_time(t)
        cev = [[c.kind.name.lower(), c.value] for c in comp.update(pause)]
        pstate = comp.pause_state(pause)
        out.append({
            "events": sym,
            "pose": det.pose,
            "closed": det.closed,
            "left": det.left.value,
            "right": det.right.value,
            "baseL": det.left.baseline,
            "baseR": det.right.baseline,
            "closedTime": det.closed_time(t),
            "pause": pause,
            "composerEvents": cev,
            "pauseState": [pstate[0], pstate[1]],
            "code": comp.code,
            "text": comp.text,
        })
    return out


def composer_ops(morse, seed: int, count: int):
    rng = random.Random(seed)
    comp = morse.MorseComposer(0.3, 0.7, 12)
    ops, results = [], []
    pause = 0.0
    for _ in range(count):
        r = rng.random()
        if r < 0.45:
            op = ["add", rng.choice([".", "-"])]
            e = comp.add_symbol(op[1])
            res = [e.kind.name.lower(), e.value]
            pause = 0.0
        elif r < 0.7:
            pause += rng.choice([0.05, 0.1, 0.2, 0.3, 0.45, 0.75, 1.0])
            op = ["update", pause]
            res = [[c.kind.name.lower(), c.value] for c in comp.update(pause)]
        elif r < 0.8:
            op = ["end"]
            e = comp.end_letter()
            res = None if e is None else [e.kind.name.lower(), e.value]
        elif r < 0.93:
            op = ["delete"]
            e = comp.delete()
            res = None if e is None else [e.kind.name.lower(), e.value]
        elif r < 0.97:
            op = ["clear"]
            e = comp.clear()
            res = [e.kind.name.lower(), e.value]
        else:
            op = ["state", pause]
            res = list(comp.pause_state(pause))
        ops.append(op)
        results.append({"result": res, "code": comp.code, "text": comp.text})
    return {"ops": ops, "results": results}


def main():
    folder = find_package()
    blinks = load("blinks", folder)
    morse = load("morse", folder)

    scenarios = []
    configs = [
        # name, seed, fps, seconds, time_unit, close_threshold, noise, drift, winks, face_loss
        ("default_clean", 1, 30, 20, 0.10, 0.45, 0.01, False, False, False),
        ("default_noisy", 2, 30, 20, 0.10, 0.45, 0.05, True, True, False),
        ("slow_unit", 3, 30, 25, 0.25, 0.45, 0.03, True, True, True),
        ("fast_unit", 4, 60, 15, 0.06, 0.40, 0.02, False, True, False),
        ("low_fps", 5, 15, 25, 0.20, 0.50, 0.03, True, False, True),
        ("winks_heavy", 6, 30, 20, 0.12, 0.35, 0.04, True, True, True),
        ("high_noise", 7, 30, 20, 0.15, 0.55, 0.10, True, True, True),
        ("tiny_text", 8, 30, 30, 0.10, 0.45, 0.02, False, False, False, 6),
    ]
    for cfg in configs:
        name, seed, fps, seconds, unit, thr, noise, drift, winks, loss = cfg[:10]
        max_text = cfg[10] if len(cfg) > 10 else 200
        params = {
            "close_threshold": thr,
            "time_unit": unit,
            "letter_gap": unit * 3,
            "word_gap": unit * 7,
            "max_text": max_text,
        }
        frames = make_trace(seed, fps, seconds, unit, noise, drift, winks, loss)
        scenarios.append({
            "name": name,
            "params": params,
            "frames": frames,
            "expected": run_detector(blinks, morse, params, frames),
        })

    table = {
        "table": morse.MORSE_TABLE,
        "maxCodeLength": morse.MAX_CODE_LENGTH,
        "decode": {code: morse.decode(code) for code in
                   list(morse.CODE_TO_CHAR) + ["", ".......", "..--", "-.-.-.-", "...---"]},
        "candidates": {p: morse.candidates(p) for p in
                       ["", ".", "-", "..", ".-", "-.", "--", "...", ".-.", "-..-", ".----", "......."]},
    }

    composer = [composer_ops(morse, seed, 400) for seed in (11, 12, 13)]

    data = {"scenarios": scenarios, "morse": table, "composer": composer}
    target = HERE / "golden.json"
    target.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    events = sum(len(f["events"]) for s in scenarios for f in s["expected"])
    letters = sum(1 for s in scenarios for f in s["expected"]
                  for c in f["composerEvents"] if c[0] == "letter")
    print(f"wrote {target.name}: {len(scenarios)} scenarios, "
          f"{events} blink events, {letters} letters, "
          f"{target.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
