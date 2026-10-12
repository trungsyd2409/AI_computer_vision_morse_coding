// Visual helpers: the paper tape, the reference chart, the eye gauges that
// sit beside the user's eyes, and the auto-send countdown ring.

import { MORSE_TABLE, candidates } from "./morse.js";
import { t } from "./i18n.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(name, attrs = {}) {
  const el = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

// -- paper tape ---------------------------------------------------------------
//
// Marks are printed at a fixed "print head" near the right edge and the
// paper moves left as the message grows, like a real telegraph tape.
// Each letter sits under its own marks.

const MARK = 9;        // dot diameter and stroke width
const DASH = 27;       // dash length including the round caps
const SYMBOL_GAP = 8;
const LETTER_GAP = 24;
const WORD_GAP = 46;
const MARK_Y = 26;
const LETTER_Y = 56;
const HEAD_RATIO = 0.84;

export function tapeLayout(groups, pendingCode, live) {
  // Returns marks with x positions and letter labels, plus the total width.
  const marks = [];   // { x, w, kind: "done" | "pending" | "live" }
  const labels = [];  // { x, ch }
  let x = 0;
  const putSymbols = (code, kind) => {
    const start = x;
    for (const c of code) {
      const w = c === "." ? MARK : DASH;
      marks.push({ x, w, kind });
      x += w + SYMBOL_GAP;
    }
    return [start, x - SYMBOL_GAP];
  };
  groups.forEach((g) => {
    if (g.space) {
      x += WORD_GAP - LETTER_GAP;
      return;
    }
    const [a, b] = putSymbols(g.code, "done");
    labels.push({ x: (a + b) / 2, ch: g.ch });
    x += LETTER_GAP;
  });
  if (pendingCode) putSymbols(pendingCode, "pending");
  if (live) putSymbols(live, "live");
  return { marks, labels, width: x };
}

export class Tape {
  constructor(view, svg) {
    this.view = view;
    this.svg = svg;
    this.key = "";
    this.width = 0;
    this.content = svgEl("g", { class: "tape-content" });
    this.svg.append(this.content);
    this.head = svgEl("path", { class: "tape-head", d: `M0 4 V${LETTER_Y + 6}` });
    this.svg.append(this.head);
    this._resize = () => {
      this.width = this.view.clientWidth;
      this.svg.setAttribute("viewBox", `0 0 ${Math.max(1, this.width)} 72`);
      this.head.setAttribute("transform", `translate(${this._headX()} 0)`);
      this.key = "";
    };
    new ResizeObserver(this._resize).observe(this.view);
    this._resize();
  }

  _headX() {
    return Math.round(this.width * HEAD_RATIO);
  }

  // groups: [{ ch, code } | { space: true }]
  set(groups, pendingCode, live) {
    const key = JSON.stringify([groups.map((g) => g.ch || " ").join(""), pendingCode, live, this.width]);
    if (key === this.key) return;
    this.key = key;
    const layout = tapeLayout(groups, pendingCode, live);
    const frag = document.createDocumentFragment();
    for (const m of layout.marks) {
      const path = svgEl("path", {
        class: `mark ${m.kind}`,
        d: `M${m.x + MARK / 2} ${MARK_Y}h${m.w - MARK}${m.w === MARK ? ".01" : ""}`,
      });
      frag.append(path);
    }
    for (const l of layout.labels) {
      const text = svgEl("text", { class: "tape-letter", x: l.x, y: LETTER_Y, "text-anchor": "middle" });
      text.textContent = l.ch;
      frag.append(text);
    }
    this.content.replaceChildren(frag);
    // Right-align the newest mark to the print head. The paper slides, so
    // the move is a CSS transition on the group's transform.
    const shift = this._headX() - layout.width;
    this.content.style.transform = `translateX(${shift}px)`;
  }
}

// -- reference chart ---------------------------------------------------------------

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const DIGITS = "0123456789".split("");

function glyphPath(code) {
  let x = 3;
  let d = "";
  for (const c of code) {
    if (c === ".") { d += `M${x} 4h0.01`; x += 9; }
    else { d += `M${x} 4h8`; x += 16; }
  }
  return d;
}

export class Chart {
  constructor(grid, noteEl) {
    this.grid = grid;
    this.noteEl = noteEl;
    this.set = "letters";
    this.prefix = "";
    this.cells = new Map();
    this._build();
  }

  setSet(name) {
    if (name === this.set) return;
    this.set = name;
    this._build();
  }

  _build() {
    this.grid.replaceChildren();
    this.cells.clear();
    for (const ch of this.set === "letters" ? LETTERS : DIGITS) {
      const cell = document.createElement("div");
      cell.className = "cell";
      const label = document.createElement("b");
      label.textContent = ch;
      const svg = svgEl("svg", { viewBox: "0 0 82 8", "aria-hidden": "true" });
      svg.append(svgEl("path", { d: glyphPath(MORSE_TABLE[ch]) }));
      cell.append(label, svg);
      this.grid.append(cell);
      this.cells.set(ch, cell);
    }
    this.prefix = "";
    this.update("");
  }

  // Dims the characters that the code typed so far can no longer become.
  update(prefix) {
    this.prefix = prefix;
    for (const [ch, cell] of this.cells) {
      const code = MORSE_TABLE[ch];
      const reach = !prefix || code.startsWith(prefix);
      cell.classList.toggle("dim", !reach);
      cell.classList.toggle("reach", !!prefix && reach);
      cell.classList.toggle("exact", !!prefix && code === prefix);
    }
    this.noteEl.textContent = prefix
      ? t("chartMatch", { n: candidates(prefix).length })
      : t(this.set === "letters" ? "chartLetters" : "chartDigits");
  }

  refreshText() { this.update(this.prefix); }
}

// -- eye gauges ----------------------------------------------------------------------------
//
// MediaPipe eye landmarks: the eye that belongs to blendshape "Left" is
// 263/362, the one that belongs to "Right" is 33/133.
const EYE_LANDMARKS = { left: [263, 362], right: [33, 133] };

export class Chips {
  constructor(stage, video, chipL, chipR) {
    this.stage = stage;
    this.video = video;
    this.chips = { l: chipL, r: chipR };
    this.texts = { l: chipL.querySelector(".num"), r: chipR.querySelector(".num") };
    this.bars = { l: chipL.querySelector(".gauge i"), r: chipR.querySelector(".gauge i") };
  }

  hide() {
    this.chips.l.classList.remove("show");
    this.chips.r.classList.remove("show");
  }

  // landmarks: normalised points of the unmirrored video. `swap` says which
  // blendshape feeds the user's left gauge. `mirror` flips x like the video.
  update(landmarks, { leftValue, rightValue, swap, mirror }) {
    if (!landmarks) return this.hide();
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    const cw = this.stage.clientWidth;
    const ch = this.stage.clientHeight;
    if (!vw || !vh || !cw || !ch) return this.hide();

    // object-fit: cover math
    const scale = Math.max(cw / vw, ch / vh);
    const offX = (cw - vw * scale) / 2;
    const offY = (ch - vh * scale) / 2;
    const place = (indices) => {
      const a = landmarks[indices[0]];
      const b = landmarks[indices[1]];
      let nx = (a.x + b.x) / 2;
      const ny = (a.y + b.y) / 2;
      if (mirror) nx = 1 - nx;
      return [offX + nx * vw * scale, offY + ny * vh * scale];
    };

    const userLeft = swap ? EYE_LANDMARKS.right : EYE_LANDMARKS.left;
    const userRight = swap ? EYE_LANDMARKS.left : EYE_LANDMARKS.right;
    const sets = [
      ["l", place(userLeft), leftValue],
      ["r", place(userRight), rightValue],
    ];
    // Each gauge goes on the outer side of its eye so it never covers it.
    const centerX = (sets[0][1][0] + sets[1][1][0]) / 2;
    for (const [key, [x, y], value] of sets) {
      const chip = this.chips[key];
      const outward = x < centerX ? -1 : 1;
      const half = chip.offsetWidth / 2 || 26;
      const px = Math.min(Math.max(x + outward * (half + 30), half + 4), cw - half - 4);
      chip.style.transform = `translate(${px}px, ${y}px) translate(-50%, -50%)`;
      chip.classList.add("show");
      chip.classList.toggle("shut", value > 0.5);
      const open = Math.round((1 - value) * 100);
      this.texts[key].textContent = String(open);
      this.bars[key].style.height = `${open}%`;
    }
  }
}

// -- countdown ring -----------------------------------------------------------------------------

export class Ring {
  constructor(svg, fg, textEl) {
    this.svg = svg;
    this.fg = fg;
    this.textEl = textEl;
    this.visible = false;
  }

  hide() {
    if (!this.visible) return;
    this.visible = false;
    // SVG elements have no `hidden` property, so use the attribute.
    this.svg.toggleAttribute("hidden", true);
    this.textEl.hidden = true;
  }

  // progress 0..1 of the wait, seconds left to display
  show(progress, secondsLeft) {
    this.visible = true;
    this.svg.toggleAttribute("hidden", false);
    this.textEl.hidden = false;
    const p = Math.min(1, Math.max(0, progress));
    this.fg.style.strokeDashoffset = String(100 - p * 100);
    this.fg.classList.toggle("warn", p > 0.75);
    this.textEl.textContent = t("sendIn", { s: Math.max(0, secondsLeft).toFixed(1) });
  }
}
