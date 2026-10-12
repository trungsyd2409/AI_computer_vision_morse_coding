// Tests for the sanitizer (shared vectors) and the AI reply timeline.
// Run: node --test tests/

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { sanitize } from "../public/js/sanitize.js";
import { buildTimeline, Player, START_DELAY_MS, END_PAD_MS } from "../public/js/player.js";
import { MORSE_TABLE } from "../public/js/morse.js";

const here = dirname(fileURLToPath(import.meta.url));
const vectors = JSON.parse(readFileSync(join(here, "sanitize_vectors.json"), "utf8"));

test("sanitize matches the shared vectors", () => {
  for (const v of vectors) {
    assert.equal(sanitize(v.in, v.max), v.out, JSON.stringify(v.in));
  }
});

test("sanitize output only has characters from the Morse table or spaces", () => {
  const out = sanitize("Tiếng Việt: Đẹp quá! 100% (ok) ~ <b>x</b> 😀", 200);
  for (const ch of out) {
    assert.ok(ch === " " || MORSE_TABLE[ch], `unexpected ${JSON.stringify(ch)}`);
  }
});

test("timeline follows the 1/3/1/3/7 unit rules", () => {
  const U = 100;
  const tl = buildTimeline("A B", U); // A = .-   B = -...
  const t0 = START_DELAY_MS;
  // A: dot, dash
  assert.deepEqual(tl[0], { t: t0, k: "close", sym: "." });
  assert.deepEqual(tl[1], { t: t0 + 1 * U, k: "open", sym: "." });
  assert.deepEqual(tl[2], { t: t0 + 2 * U, k: "close", sym: "-" });
  assert.deepEqual(tl[3], { t: t0 + 5 * U, k: "open", sym: "-" });
  // letter A is 3U after the last open
  assert.deepEqual(tl[4], { t: t0 + 8 * U, k: "letter", ch: "A" });
  // space is 7U after the last open, so 4U after the letter
  assert.deepEqual(tl[5], { t: t0 + 12 * U, k: "space" });
  // B starts one unit gap after the space point: next close follows at +0
  assert.equal(tl[6].k, "close");
  assert.equal(tl[6].t, t0 + 12 * U);
  const done = tl[tl.length - 1];
  assert.equal(done.k, "done");
  assert.equal(done.t, tl[tl.length - 2].t + END_PAD_MS);
});

test("timeline closed time: dot 1U, dash 3U", () => {
  const U = 80;
  const tl = buildTimeline("O", U); // ---
  const closes = tl.filter((e) => e.k === "close");
  const opens = tl.filter((e) => e.k === "open");
  assert.equal(closes.length, 3);
  closes.forEach((c, i) => assert.equal(opens[i].t - c.t, 3 * U));
});

test("player emits in order and flags late events", () => {
  const seen = [];
  const p = new Player((e) => seen.push(e));
  p.start("E", 100, 1000);
  p.tick(1000 + START_DELAY_MS - 1);
  assert.equal(seen.length, 0);
  p.tick(1000 + START_DELAY_MS + 10);
  assert.deepEqual(seen.map((e) => [e.k, e.late]), [["close", false]]);
  // jump far ahead: everything else is late
  p.tick(1000 + 60000);
  assert.deepEqual(seen.slice(1).map((e) => e.k), ["open", "letter", "done"]);
  assert.ok(seen.slice(1).every((e) => e.late));
  assert.equal(p.running, false);
});

test("player skip reveals text without eye events", () => {
  const seen = [];
  const p = new Player((e) => seen.push(e));
  p.start("HI YOU", 100, 0);
  p.skip();
  const kinds = new Set(seen.map((e) => e.k));
  assert.ok(!kinds.has("close") && !kinds.has("open"));
  assert.equal(seen.filter((e) => e.k === "letter").map((e) => e.ch).join(""), "HIYOU");
  assert.equal(seen.filter((e) => e.k === "space").length, 1);
  assert.equal(seen[seen.length - 1].k, "done");
});
