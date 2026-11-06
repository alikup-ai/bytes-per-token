// GPT-2 style byte-level BPE (the HuggingFace ByteLevel pre-tokenizer with
// use_regex, add_prefix_space false), encode and streaming decode.
// Shared by index.html and the Node check in check_bpe.mjs.
(function (root) {
  const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  function byteMaps() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
    const enc = {}, dec = {};
    bs.forEach((b, i) => { enc[b] = String.fromCharCode(cs[i]); dec[String.fromCharCode(cs[i])] = b; });
    return { enc, dec };
  }
  const MAPS = byteMaps();

  function makeTokenizer(spec) {
    const idOf = new Map();
    spec.tokens.forEach((t, i) => { if (t !== null) idOf.set(t, i); });
    const rank = new Map();
    spec.merges.forEach((m, i) => rank.set(m, i));
    const cache = new Map();
    const utf8 = new TextEncoder();

    function bpe(word) {
      if (cache.has(word)) return cache.get(word);
      let parts = Array.from(word);
      for (;;) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i + 1 < parts.length; i++) {
          const r = rank.get(parts[i] + " " + parts[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        const a = parts[best], b = parts[best + 1], merged = [];
        for (let i = 0; i < parts.length; i++) {   // merge every occurrence of the best pair, left to right
          if (i + 1 < parts.length && parts[i] === a && parts[i + 1] === b) { merged.push(a + b); i++; }
          else merged.push(parts[i]);
        }
        parts = merged;
      }
      cache.set(word, parts);
      return parts;
    }

    function encode(text) {
      const ids = [];
      for (const m of text.matchAll(PAT)) {
        let s = "";
        for (const byte of utf8.encode(m[0])) s += MAPS.enc[byte];
        for (const p of bpe(s)) {
          const id = idOf.get(p);
          if (id !== undefined) ids.push(id);
        }
      }
      return ids;
    }

    // Streaming decoder: bytes of partial UTF-8 characters wait for the next token.
    function decoder() {
      const td = new TextDecoder("utf-8");
      return (id) => {
        const t = spec.tokens[id];
        if (t === null || t === undefined || id === spec.eot) return "";
        const bytes = new Uint8Array(Array.from(t, (ch) => MAPS.dec[ch]));
        return td.decode(bytes, { stream: true });
      };
    }
    return { encode, decoder, label: spec.label, eot: spec.eot };
  }

  root.makeTokenizer = makeTokenizer;
})(typeof window !== "undefined" ? window : globalThis);
