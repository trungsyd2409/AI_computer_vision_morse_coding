// The face of Xeno Assistant. Two looks:
//  - designed: an original bakelite receiver with two brass-rimmed lenses.
//    Each eyelid slides down over its lens, so a blink reads as a shutter
//    rather than a squash;
//  - custom: two pictures supplied by the user (eyes open, eyes closed).
// Every eye closure is a Morse symbol, so the face never blinks on its own
// and has no idle animation. The lamp on top shows whether Xeno is busy.

const EYE = (id, cx) => `
  <g class="eye" data-eye="${id}" transform="translate(${cx} 128)">
    <circle r="40" fill="#0b0805"/>
    <g clip-path="url(#lens-clip)">
      <circle r="35" fill="#efe6d0"/>
      <circle class="iris" cx="2" cy="3" r="19" fill="#5a4326"/>
      <circle cx="2" cy="3" r="9" fill="#0b0805"/>
      <circle cx="-4" cy="-3" r="3.2" fill="#efe6d0" opacity="0.85"/>
      <g class="lid">
        <path d="M-40 -92 H40 V36 Q0 52 -40 36 Z" fill="#2b2118"/>
        <path d="M-40 36 Q0 52 40 36" fill="none" stroke="#120e0a" stroke-width="3" stroke-linecap="round"/>
      </g>
    </g>
    <circle r="38" fill="none" stroke="#b58a3c" stroke-width="5"/>
  </g>`;

const DESIGNED = `
<svg class="char" viewBox="0 0 320 290" aria-hidden="true">
  <defs><clipPath id="lens-clip"><circle r="35"/></clipPath></defs>
  <line x1="160" y1="38" x2="160" y2="20" stroke="#4a3b2c" stroke-width="4" stroke-linecap="round"/>
  <circle class="lamp-top" cx="160" cy="14" r="8"/>
  <rect x="30" y="38" width="260" height="226" rx="44" fill="#2b2118"/>
  <rect x="30" y="38" width="260" height="226" rx="44" fill="none" stroke="rgba(255,236,200,0.10)" stroke-width="2"/>
  <rect x="50" y="58" width="220" height="148" rx="30" fill="#120e0a"/>
  ${EYE("l", 112)}
  ${EYE("r", 208)}
  <g fill="#120e0a">
    <rect x="104" y="222" width="112" height="5" rx="2.5"/>
    <rect x="104" y="233" width="112" height="5" rx="2.5"/>
    <rect x="104" y="244" width="112" height="5" rx="2.5"/>
  </g>
  <circle cx="54" cy="246" r="4.5" fill="#4a3b2c"/>
  <circle cx="266" cy="246" r="4.5" fill="#4a3b2c"/>
</svg>`;

export class Character {
  constructor(stage) {
    this.stage = stage;
    this.mode = "designed";
    this.closed = false;
    this.state = "idle";
    this.images = { open: null, closed: null };
    this._build();
  }

  setMode(mode, images) {
    this.images = images || this.images;
    const usable = mode === "custom" && this.images.open && this.images.closed;
    const next = usable ? "custom" : "designed";
    if (next !== this.mode || !this.root) {
      this.mode = next;
      this._build();
    } else if (next === "custom") {
      this.imgOpen.src = this.images.open;
      this.imgClosed.src = this.images.closed;
    }
    this.setClosed(this.closed);
  }

  setClosed(closed) {
    this.closed = closed;
    if (this.mode === "custom") {
      this.imgOpen.classList.toggle("hide", closed);
      this.imgClosed.classList.toggle("hide", !closed);
    } else {
      this.eyes.forEach((eye) => eye.classList.toggle("shut", closed));
    }
  }

  // idle | thinking | talking
  setState(state) {
    this.state = state;
    if (this.svg) {
      this.svg.classList.toggle("thinking", state === "thinking");
      this.svg.classList.toggle("talking", state === "talking");
    }
  }

  // The lids need to finish moving inside one Morse unit, so tie their
  // speed to it. Fast replies (3x) have units as short as 30 ms.
  setUnit(unitMs) {
    const lid = Math.max(18, Math.min(70, Math.round(unitMs * 0.6)));
    this.stage.style.setProperty("--lid-ms", `${lid}ms`);
  }

  _build() {
    this.stage.replaceChildren();
    if (this.mode === "custom") {
      const wrap = document.createElement("div");
      wrap.className = "char-img";
      this.imgOpen = new Image();
      this.imgClosed = new Image();
      this.imgOpen.alt = "";
      this.imgClosed.alt = "";
      this.imgOpen.src = this.images.open;
      this.imgClosed.src = this.images.closed;
      wrap.append(this.imgOpen, this.imgClosed);
      this.stage.append(wrap);
      this.root = wrap;
      this.svg = null;
      this.eyes = [];
      return;
    }
    const holder = document.createElement("div");
    holder.innerHTML = DESIGNED.trim();
    this.svg = holder.firstElementChild;
    this.stage.append(this.svg);
    this.root = this.svg;
    this.eyes = [...this.svg.querySelectorAll(".eye")];
    this.setState(this.state);
  }
}
