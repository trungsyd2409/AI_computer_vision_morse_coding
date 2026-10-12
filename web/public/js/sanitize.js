// Cleans an AI reply so that every character exists in the Morse table.
// server.py has the same function; tests/sanitize_vectors.json holds cases
// that both implementations must pass.

const SMART = {
  "‘": "'", "’": "'", "‚": "'", "′": "'",
  "“": "", "”": "", "„": "", "″": "",
  "–": "-", "—": "-", "−": "-",
  "…": "...",
  " ": " ",
};

const ALLOWED = /[A-Z0-9 .,?!'/():=+\-@]/;

export function sanitize(text, maxChars = 40) {
  let s = String(text ?? "");
  s = s.replace(/[‘’‚′“”„″–—−… ]/g, (c) => SMART[c]);
  // Strip accents, then handle the letter D with stroke, which has no
  // decomposition.
  s = s.normalize("NFD").replace(/\p{M}+/gu, "");
  s = s.replace(/đ/g, "d").replace(/Đ/g, "D");
  s = s.toUpperCase();
  let out = "";
  for (const ch of s) {
    out += ALLOWED.test(ch) ? ch : " ";
  }
  out = out.replace(/ {2,}/g, " ").trim();

  if (out.length > maxChars) {
    let cut = out.slice(0, maxChars);
    // Do not end in the middle of a word when there is a space to cut at.
    if (out[maxChars] !== " ") {
      const last = cut.lastIndexOf(" ");
      if (last > 0) cut = cut.slice(0, last);
    }
    out = cut.trim();
  }
  return out;
}
