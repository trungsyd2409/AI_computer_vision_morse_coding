// Replays the traces from golden.json (made by gen_golden.py from the real
// Python modules) through the JavaScript port and compares every output.
// Run: node --test tests/

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { MORSE_TABLE, MAX_CODE_LENGTH, decode, candidates, MorseComposer } from "../public/js/morse.js";
import { BlinkDetector } from "../public/js/blinks.js";

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(here, "golden.json"), "utf8"));
const EPS = 1e-9;

function near(actual, expected, label) {
  assert.ok(
    Math.abs(actual - expected) <= EPS,
    `${label}: got ${actual}, expected ${expected}`
  );
}

test("morse table matches the Python table", () => {
  assert.deepEqual(MORSE_TABLE, golden.morse.table);
  assert.equal(MAX_CODE_LENGTH, golden.morse.maxCodeLength);
});

test("decode and candidates match", () => {
  for (const [code, expected] of Object.entries(golden.morse.decode)) {
    assert.equal(decode(code), expected, `decode(${code})`);
  }
  for (const [prefix, expected] of Object.entries(golden.morse.candidates)) {
    assert.deepEqual(candidates(prefix), expected, `candidates(${prefix})`);
  }
});

test("composer operation sequences match", () => {
  for (const [index, run] of golden.composer.entries()) {
    const comp = new MorseComposer(0.3, 0.7, 12);
    run.ops.forEach((op, i) => {
      let result;
      switch (op[0]) {
        case "add": {
          const e = comp.addSymbol(op[1]);
          result = [e.kind, e.value];
          break;
        }
        case "update":
          result = comp.update(op[1]).map((e) => [e.kind, e.value]);
          break;
        case "end": {
          const e = comp.endLetter();
          result = e ? [e.kind, e.value] : null;
          break;
        }
        case "delete": {
          const e = comp.delete();
          result = e ? [e.kind, e.value] : null;
          break;
        }
        case "clear": {
          const e = comp.clear();
          result = [e.kind, e.value];
          break;
        }
        case "state":
          result = comp.pauseState(op[1]);
          break;
        default:
          throw new Error(`unknown op ${op[0]}`);
      }
      const exp = run.results[i];
      const where = `run ${index} op ${i} ${JSON.stringify(op)}`;
      if (op[0] === "state") {
        assert.equal(result[0], exp.result[0], where);
        near(result[1], exp.result[1], where);
      } else {
        assert.deepEqual(result, exp.result, where);
      }
      assert.equal(comp.code, exp.code, `${where} code`);
      assert.equal(comp.text, exp.text, `${where} text`);
    });
  }
});

for (const scenario of golden.scenarios) {
  test(`detector trace: ${scenario.name}`, () => {
    const p = scenario.params;
    const det = new BlinkDetector(p.close_threshold, p.time_unit);
    const comp = new MorseComposer(p.letter_gap, p.word_gap, p.max_text);
    let events = 0;

    scenario.frames.forEach(([t, l, r, reset], i) => {
      const exp = scenario.expected[i];
      const where = `${scenario.name} frame ${i}`;
      if (reset) det.reset(t);
      const got = det.update(l, r, t);

      assert.equal(got.length, exp.events.length, `${where} event count`);
      got.forEach((e, k) => {
        const x = exp.events[k];
        assert.equal(e.kind, x.kind, `${where} kind`);
        near(e.started, x.started, `${where} started`);
        near(e.fired, x.fired, `${where} fired`);
        const ce = comp.addSymbol(e.kind === "dot" ? "." : "-");
        assert.deepEqual([ce.kind, ce.value], x.composer, `${where} composer symbol`);
        events += 1;
      });

      const pause = det.pauseTime(t);
      const cev = comp.update(pause).map((c) => [c.kind, c.value]);
      assert.deepEqual(cev, exp.composerEvents, `${where} composer events`);

      assert.equal(det.pose, exp.pose, `${where} pose`);
      assert.equal(det.closed, exp.closed, `${where} closed`);
      near(det.left.value, exp.left, `${where} left`);
      near(det.right.value, exp.right, `${where} right`);
      near(det.left.baseline, exp.baseL, `${where} baseL`);
      near(det.right.baseline, exp.baseR, `${where} baseR`);
      near(det.closedTime(t), exp.closedTime, `${where} closedTime`);
      near(pause, exp.pause, `${where} pause`);

      const ps = comp.pauseState(pause);
      assert.equal(ps[0], exp.pauseState[0], `${where} pauseState kind`);
      near(ps[1], exp.pauseState[1], `${where} pauseState progress`);
      assert.equal(comp.code, exp.code, `${where} code`);
      assert.equal(comp.text, exp.text, `${where} text`);
    });

    assert.ok(events > 0, "scenario produced no blink events, test is vacuous");
  });
}
