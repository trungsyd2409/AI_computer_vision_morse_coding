// The face of Xeno Assistant. Two looks:
//  - designed: an original vector robot with smooth eyelids, a breathing
//    face and an antenna that lights up while the AI is busy;
//  - custom: two pictures supplied by the user (eyes open, eyes closed).
// Every eye closure means a Morse symbol, so the face never blinks on its
// own. Idle life comes from breathing and slow pupil drift only.

const SVG_NS = "http://www.w3.org/2000/svg";

const DESIGNED = `
<svg class="char" viewBox="0 0 320 260" aria-hidden="true">
  <line x1="160" y1="30" x2="160" y2="14" stroke="#3a4a44" stroke-width="3" stroke-linecap="round"/>
  <circle class="antenna" cx="160" cy="11" r="6"/>
  <g class="face">
    <rect x="40" y="30" width="240" height="205" rx="56" fill="#17201d" stroke="rgba(255,255,255,0.12)" stroke-width="1.5"/>
    <rect x="58" y="50" width="204" height="150" rx="40" fill="#0e1412"/>
    <g class="eye" data-eye="l">
      <ellipse cx="116" cy="110" rx="23" ry="28" fill="#e9efec"/>
      <circle class="pupil" cx="120" cy="112" r="10" fill="#0e1412"/>
    </g>
    <g class="eye" data-eye="r">
      <ellipse cx="204" cy="110" rx="23" ry="28" fill="#e9efec"/>
      <circle class="pupil" cx="208" cy="112" r="10" fill="#0e1412"/>
    </g>
    <path d="M140 166 Q160 176 180 166" fill="none" stroke="#3a4a44" stroke-width="3" stroke-linecap="round"/>
    <rect x="132" y="212" width="56" height="6" rx="3" fill="#0e1412"/>
  </g>
</svg>`;

export class Character {
  constructor(stage) {
    this.stage = stage;
    this.mode = "designed";
    this.closed = false;
    this.state = "idle";
    this.images = { open: null, closed: null };
    this._gazeTimer = null;
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

  _build() {
    clearInterval(this._gazeTimer);
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
    this.pupils = [...this.svg.querySelectorAll(".pupil")];
    this.setState(this.state);
    this._startGaze();
  }

  _startGaze() {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const spots = [[0, 0], [-4, -1], [4, 1], [2, -3], [-3, 2], [0, 3]];
    this._gazeTimer = setInterval(() => {
      if (this.closed || !this.pupils) return;
      const [x, y] = spots[Math.floor(Math.random() * spots.length)];
      this.pupils.forEach((p) => { p.style.transform = `translate(${x}px, ${y}px)`; });
    }, 3200);
  }
}
