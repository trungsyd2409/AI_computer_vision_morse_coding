// Small visual helpers: Morse strip paths, the reference chart, the eye chips
// that sit next to the user's eyes, and the auto-send countdown ring.

import { MORSE_TABLE, candidates } from "./morse.js";
import { t } from "./i18n.js";

const SVG_NS = "http://www.w3.org/2000/svg";

// Path data for the strip under each face. Dots are zero-length segments
// with round caps, dashes are 26 units long.
export function stripPaths(code, live) {
  let x = 7;
  let d = "";
  for (const c of code) {
    if (c === ".") { d += `M${x} 12h0.01`; x += 24; }
    else { d += `M${x} 12h26`; x += 50; }
  }
  let liveD = "";
  if (live === ".") liveD = `M${x} 12h0.01`;
  else if (live === "-") liveD = `M${x} 12h26`;
  return { d, liveD };
}

function glyphPath(code) {
  let x = 2;
  let d = "";
  for (const c of code) {
    if (c === ".") { d += `M${x} 4h0.01`; x += 6; }
    else { d += `M${x} 4h5`; x += 11; }
  }
  return d;
}

export class Strip {
  constructor(codeEl, liveEl) {
    this.codeEl = codeEl;
    this.liveEl = liveEl;
    this.key = "";
  }

  set(code, live) {
    const key = `${code}|${live}`;
    if (key === this.key) return;
    this.key = key;
    const { d, liveD } = stripPaths(code, live);
    this.codeEl.setAttribute("d", d);
    this.liveEl.setAttribute("d", liveD);
  }
}

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const DIGITS = "0123456789".split("");

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
      const svg = document.createElementNS(SVG_NS, "svg");
      svg.setAttribute("viewBox", "0 0 48 8");
      svg.setAttribute("aria-hidden", "true");
      const path = document.createElementNS(SVG_NS, "path");
      path.setAttribute("d", glyphPath(MORSE_TABLE[ch]));
      svg.append(path);
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
    let matches = 0;
    for (const [ch, cell] of this.cells) {
      const code = MORSE_TABLE[ch];
      const reach = !prefix || code.startsWith(prefix);
      const exact = !!prefix && code === prefix;
      if (reach) matches += 1;
      cell.classList.toggle("dim", !reach);
      cell.classList.toggle("reach", !!prefix && reach);
      cell.classList.toggle("exact", exact);
    }
    this.noteEl.textContent = prefix
      ? t("chartMatch", { n: candidates(prefix).length })
      : t(this.set === "letters" ? "chartLetters" : "chartDigits");
  }

  refreshText() { this.update(this.prefix); }
}

// MediaPipe eye landmarks: the eye that belongs to blendshape "Left" is
// 263/362, the one that belongs to "Right" is 33/133.
const EYE_LANDMARKS = { left: [263, 362], right: [33, 133] };

export class Chips {
  constructor(stage, video, chipL, chipR) {
    this.stage = stage;
    this.video = video;
    this.chips = { l: chipL, r: chipR };
    this.texts = { l: chipL.querySelector("span"), r: chipR.querySelector("span") };
  }

  hide() {
    this.chips.l.classList.remove("show");
    this.chips.r.classList.remove("show");
  }

  // landmarks: normalised points of the unmirrored video. `swap` says which
  // blendshape feeds the user's left chip. `mirror` flips x like the video.
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

    // The user's left eye follows blendshape Left (or Right when swapped).
    const userLeft = swap ? EYE_LANDMARKS.right : EYE_LANDMARKS.left;
    const userRight = swap ? EYE_LANDMARKS.left : EYE_LANDMARKS.right;
    const sets = [
      ["l", place(userLeft), leftValue],
      ["r", place(userRight), rightValue],
    ];
    // Put each chip on the outer side of its eye so it does not cover it.
    const centerX = (sets[0][1][0] + sets[1][1][0]) / 2;
    for (const [key, [x, y], value] of sets) {
      const chip = this.chips[key];
      const outward = x < centerX ? -1 : 1;
      const half = chip.offsetWidth / 2 || 36;
      const px = Math.min(Math.max(x + outward * (half + 26), half + 4), cw - half - 4);
      chip.style.transform = `translate(${px}px, ${y}px) translate(-50%, -50%)`;
      chip.classList.add("show");
      chip.classList.toggle("shut", value > 0.5);
      this.texts[key].textContent = `${Math.round((1 - value) * 100)}%`;
    }
  }
}

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
