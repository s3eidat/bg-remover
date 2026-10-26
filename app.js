/* Background Remover — U²-Net small (Apache-2.0) runs on your device with onnxruntime-web (WebAssembly).
   Photos are decoded, segmented and re-encoded locally. The only files fetched are this site's own engine and model. */
(function () {
  "use strict";
  const $ = S3.$, $$ = S3.$$;
  const SIZE = 320, MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
  const MASK_MAX = 1600;                 // working mask resolution (preview + export source)
  const MAX_SIDE = 12000, MAX_AREA = 36e6; // browser canvas limits for the export
  const MODEL_URL = "models/u2netp/u2netp.onnx", WASM_URL = "lib/ort-wasm-simd-threaded.wasm";
  const BANDS = 0.16;                    // soft ramp width around the threshold (keeps edges anti-aliased)

  const saved = S3.store("bgr") || {};
  function num(v, d) { return typeof v === "number" && isFinite(v) ? v : d; }
  const o = {
    threshold: num(saved.threshold, 50), shift: num(saved.shift, 0), feather: num(saved.feather, 1), smart: saved.smart !== false,
    bg: ["none", "color", "blur", "image"].indexOf(saved.bg) >= 0 && saved.bg !== "image" ? saved.bg : "none",
    color: /^#[0-9a-f]{6}$/i.test(saved.color || "") ? saved.color : "#ffffff", blur: num(saved.blur, 50),
    fmt: saved.fmt === "webp" ? "webp" : "png", brush: num(saved.brush, 40), soft: num(saved.soft, 50), ghost: saved.ghost !== false
  };
  function save() {
    S3.store("bgr", { threshold: o.threshold, shift: o.shift, feather: o.feather, smart: o.smart, bg: o.bg, color: o.color, blur: o.blur, fmt: o.fmt, brush: o.brush, soft: o.soft, ghost: o.ghost });
  }

  // =====================================================================
  // Pure maths (unit-tested in S3_SELFTEST)
  // =====================================================================
  // RGBA bytes (w*h) -> NCHW float32 [1,3,h,w], /255 then ImageNet mean/std.
  function preprocess(rgba, w, h) {
    const n = w * h, out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      const j = i * 4;
      out[i] = (rgba[j] / 255 - MEAN[0]) / STD[0];
      out[n + i] = (rgba[j + 1] / 255 - MEAN[1]) / STD[1];
      out[2 * n + i] = (rgba[j + 2] / 255 - MEAN[2]) / STD[2];
    }
    return out;
  }
  // min-max normalise to 0..1 (copy). A flat map becomes all zeros.
  function normalise(a) {
    let mi = Infinity, ma = -Infinity;
    for (let i = 0; i < a.length; i++) { const v = a[i]; if (v < mi) mi = v; if (v > ma) ma = v; }
    const r = ma - mi, out = new Float32Array(a.length);
    if (!(r > 1e-12)) return out;
    for (let i = 0; i < a.length; i++) out[i] = (a[i] - mi) / r;
    return out;
  }
  // Bilinear resize, pixel-centre aligned (what canvas/PIL do), edges clamped.
  function bilinear(src, sw, sh, dw, dh) {
    const out = new Float32Array(dw * dh), x0 = new Int32Array(dw), x1 = new Int32Array(dw), wx = new Float32Array(dw);
    for (let x = 0; x < dw; x++) {
      let sx = (x + 0.5) * sw / dw - 0.5; if (sx < 0) sx = 0;
      let a = Math.floor(sx); if (a > sw - 1) a = sw - 1;
      x0[x] = a; x1[x] = Math.min(sw - 1, a + 1); wx[x] = Math.min(1, sx - a);
    }
    for (let y = 0; y < dh; y++) {
      let sy = (y + 0.5) * sh / dh - 0.5; if (sy < 0) sy = 0;
      let b = Math.floor(sy); if (b > sh - 1) b = sh - 1;
      const r0 = b * sw, r1 = Math.min(sh - 1, b + 1) * sw, wy = Math.min(1, sy - b), row = y * dw;
      for (let x = 0; x < dw; x++) {
        const t = src[r0 + x0[x]] + (src[r0 + x1[x]] - src[r0 + x0[x]]) * wx[x];
        const u = src[r1 + x0[x]] + (src[r1 + x1[x]] - src[r1 + x0[x]]) * wx[x];
        out[row + x] = t + (u - t) * wy;
      }
    }
    return out;
  }
  // Box filter, separable running sums, edges clamped. O(n) for any radius.
  function boxLine(src, dst, start, step, n, r) {
    const win = 2 * r + 1;
    let sum = 0;
    for (let k = -r; k <= r; k++) sum += src[start + Math.min(n - 1, Math.max(0, k)) * step];
    for (let i = 0; i < n; i++) {
      dst[start + i * step] = sum / win;
      sum += src[start + Math.min(n - 1, i + r + 1) * step] - src[start + Math.max(0, i - r) * step];
    }
  }
  function box(a, w, h, r, tmp) {
    if (r < 1) return Float32Array.from(a);
    const t = tmp || new Float32Array(a.length), out = new Float32Array(a.length);
    for (let y = 0; y < h; y++) boxLine(a, t, y * w, 1, w, r);
    for (let x = 0; x < w; x++) boxLine(t, out, x, w, h, r);
    return out;
  }
  function blur3(a, w, h, r) { // three box passes ≈ gaussian
    if (r < 1) return a;
    const t = new Float32Array(a.length);
    return box(box(box(a, w, h, r, t), w, h, r, t), w, h, r, t);
  }
  // Guided filter (He et al.): smooths p while keeping the edges of guide I. Snaps the coarse mask to real edges.
  function guided(I, p, w, h, r, eps) {
    const n = w * h, Ip = new Float32Array(n), II = new Float32Array(n), t = new Float32Array(n);
    for (let i = 0; i < n; i++) { Ip[i] = I[i] * p[i]; II[i] = I[i] * I[i]; }
    const mI = box(I, w, h, r, t), mP = box(p, w, h, r, t), mIp = box(Ip, w, h, r, t), mII = box(II, w, h, r, t);
    for (let i = 0; i < n; i++) {
      const a = (mIp[i] - mI[i] * mP[i]) / (mII[i] - mI[i] * mI[i] + eps);
      Ip[i] = a; II[i] = mP[i] - a * mI[i];
    }
    const ma = box(Ip, w, h, r, t), mb = box(II, w, h, r, t), q = new Float32Array(n);
    for (let i = 0; i < n; i++) { const v = ma[i] * I[i] + mb[i]; q[i] = v < 0 ? 0 : v > 1 ? 1 : v; }
    return q;
  }
  // Square dilate (max) / erode (min) with a sliding-window deque: O(n) for any radius.
  function morphLine(src, dst, start, step, n, r, isMax, q) {
    let head = 0, tail = 0, next = 0;
    for (let i = 0; i < n; i++) {
      const hi = Math.min(n - 1, i + r);
      while (next <= hi) {
        const v = src[start + next * step];
        while (tail > head && (isMax ? src[start + q[tail - 1] * step] <= v : src[start + q[tail - 1] * step] >= v)) tail--;
        q[tail++] = next++;
      }
      while (q[head] < i - r) head++;
      dst[start + i * step] = src[start + q[head] * step];
    }
  }
  function morph(a, w, h, r, isMax) {
    if (r < 1) return a;
    const t = new Float32Array(a.length), out = new Float32Array(a.length), q = new Int32Array(Math.max(w, h) + 1);
    for (let y = 0; y < h; y++) morphLine(a, t, y * w, 1, w, r, isMax, q);
    for (let x = 0; x < w; x++) morphLine(t, out, x, w, h, r, isMax, q);
    return out;
  }
  // The whole cut-out pipeline at w×h (original image is W wide; slider pixels are original-image pixels).
  function computeMask(raw, guide, w, h, W, opt) {
    let m = bilinear(raw, SIZE, SIZE, w, h);
    const scale = Math.max(w, h) / SIZE;
    if (opt.smart && guide && scale > 1.4) m = guided(guide, m, w, h, Math.max(2, Math.min(24, Math.round(scale * 2))), 2e-3);
    const t = opt.threshold / 100;
    for (let i = 0; i < m.length; i++) { const v = (m[i] - t) / BANDS + 0.5; m[i] = v < 0 ? 0 : v > 1 ? 1 : v; }
    const k = w / W, sh = Math.round(opt.shift * k);
    if (sh) m = morph(m, w, h, Math.abs(sh), sh > 0);
    const f = Math.round(opt.feather * k * 0.6);
    if (opt.feather > 0) m = f >= 1 ? blur3(m, w, h, f) : blur3(m, w, h, 1).map((v, i) => (v + m[i]) / 2);
    return m;
  }
  function fitSize(w, h, max, area) {
    let s = 1;
    if (max && Math.max(w, h) > max) s = max / Math.max(w, h);
    if (area && w * h * s * s > area) s = Math.sqrt(area / (w * h));
    return { w: Math.max(1, Math.round(w * s)), h: Math.max(1, Math.round(h * s)) };
  }

  // =====================================================================
  // Canvas helpers
  // =====================================================================
  function canvas(w, h) { const c = document.createElement("canvas"); c.width = Math.max(1, w); c.height = Math.max(1, h); return c; }
  function ctx2d(c) { return c.getContext("2d", { willReadFrequently: false }); }
  // High-quality downscale: halve step by step, then one smooth draw (optionally onto a solid colour).
  function resample(src, sw, sh, tw, th, bg) {
    let cur = src, cw = sw, ch = sh;
    while (cw / 2 >= tw && ch / 2 >= th) {
      const c = canvas(Math.round(cw / 2), Math.round(ch / 2)), x = ctx2d(c);
      x.imageSmoothingQuality = "high"; x.drawImage(cur, 0, 0, c.width, c.height);
      cur = c; cw = c.width; ch = c.height;
    }
    const out = canvas(tw, th), x = ctx2d(out);
    if (bg) { x.fillStyle = bg; x.fillRect(0, 0, tw, th); }
    x.imageSmoothingQuality = "high"; x.drawImage(cur, 0, 0, tw, th);
    return out;
  }
  function pixels(c) { return c.getContext("2d").getImageData(0, 0, c.width, c.height).data; }
  function grayOf(c) {
    const p = pixels(c), n = c.width * c.height, g = new Float32Array(n);
    for (let i = 0; i < n; i++) g[i] = (0.299 * p[i * 4] + 0.587 * p[i * 4 + 1] + 0.114 * p[i * 4 + 2]) / 255;
    return g;
  }
  function alphaCanvas(m, w, h) {
    const c = canvas(w, h), x = c.getContext("2d"), d = x.createImageData(w, h), p = d.data;
    for (let i = 0; i < m.length; i++) { const j = i * 4; p[j] = p[j + 1] = p[j + 2] = 255; p[j + 3] = Math.round(m[i] * 255); }
    x.putImageData(d, 0, 0);
    return c;
  }
  function toBlob(c, type, q) {
    return new Promise((res, rej) => c.toBlob(b => b ? res(b) : rej(new Error("The browser could not encode this image (too large?)")), type, q));
  }
  function loadImage(blob) {
    const url = URL.createObjectURL(blob), img = new Image();
    img.src = url;
    return img.decode().then(() => {
      if (!img.naturalWidth) throw new Error("empty");
      img._url = url; return img;
    }).catch(() => { URL.revokeObjectURL(url); throw new Error("Can't read this image (unsupported or damaged)"); });
  }
  const nextFrame = () => new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));

  // ---------- brush (strokes are stored in original-image pixels; rasterised at any scale) ----------
  const sprites = new Map();
  function sprite(rad, hard) {
    const key = rad.toFixed(1) + "|" + hard.toFixed(2);
    let c = sprites.get(key);
    if (c) return c;
    const s = Math.ceil(rad * 2) + 2, x = ctx2d(c = canvas(s, s)), m = s / 2;
    const g = x.createRadialGradient(m, m, 0, m, m, rad);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(Math.min(0.98, Math.max(0, hard)), "rgba(255,255,255,1)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    x.fillStyle = g; x.beginPath(); x.arc(m, m, rad, 0, Math.PI * 2); x.fill();
    if (sprites.size > 60) sprites.delete(sprites.keys().next().value);
    sprites.set(key, c);
    return c;
  }
  // Paint stroke points [from..] onto an alpha canvas drawn at scale k (canvas px per image px).
  function paintStroke(c, s, k, from) {
    const rad = Math.max(0.6, s.size * k / 2), spr = sprite(rad, s.hard), d = spr.width / 2, step = Math.max(0.5, rad * 0.18);
    c.save();
    c.globalCompositeOperation = s.mode === "erase" ? "destination-out" : "source-over";
    let left;
    if (!from) { c.drawImage(spr, s.pts[0][0] * k - d, s.pts[0][1] * k - d); left = step; from = 1; } else left = s._left;
    for (let i = from; i < s.pts.length; i++) {
      const ax = s.pts[i - 1][0] * k, ay = s.pts[i - 1][1] * k, bx = s.pts[i][0] * k, by = s.pts[i][1] * k;
      const len = Math.hypot(bx - ax, by - ay);
      let t = left;
      while (t <= len) { const f = t / len; c.drawImage(spr, ax + (bx - ax) * f - d, ay + (by - ay) * f - d); t += step; }
      left = t - len;
    }
    s._left = left;
    c.restore();
  }

  // =====================================================================
  // The model
  // =====================================================================
  let modelP = null, session = null;
  async function fetchBytes(url, onProg) {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not load " + url + " (HTTP " + res.status + ")");
    const total = +res.headers.get("content-length") || 0;
    if (!res.body || !res.body.getReader) { const b = await res.arrayBuffer(); onProg(b.byteLength, b.byteLength); return b; }
    const reader = res.body.getReader(), chunks = [];
    let got = 0;
    for (;;) {
      const r = await reader.read();
      if (r.done) break;
      chunks.push(r.value); got += r.value.length; onProg(got, total);
    }
    const out = new Uint8Array(got);
    let off = 0;
    chunks.forEach(c => { out.set(c, off); off += c.length; });
    return out.buffer;
  }
  function loadModel() {
    if (modelP) return modelP;
    modelP = (async () => {
      if (!window.ort) throw new Error("The AI engine did not load. Reload the page and try again.");
      const size = { w: 11.3e6, m: 4.6e6 }, got = { w: 0, m: 0 };
      const tick = () => busy("Loading the AI model…", (got.w + got.m) / (size.w + size.m),
        S3.fmtBytes(got.w + got.m) + " of " + S3.fmtBytes(size.w + size.m) + " · first time only, then it works offline");
      tick();
      const both = await Promise.all([
        fetchBytes(WASM_URL, (g, t) => { got.w = g; if (t) size.w = t; tick(); }),
        fetchBytes(MODEL_URL, (g, t) => { got.m = g; if (t) size.m = t; tick(); })
      ]);
      busy("Starting the AI engine…", -1, "Compiling WebAssembly on your device");
      await nextFrame();
      // GitHub Pages is not cross-origin isolated: single thread, no proxy worker. Engine files come from ./lib only.
      // wasmPaths must be an absolute URL: ort uses it as an import() base, and a bare "lib/" is not a valid module specifier.
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = new URL("lib/", location.href).href;
      ort.env.wasm.wasmBinary = both[0];
      ort.env.logLevel = "error";
      session = await ort.InferenceSession.create(both[1], { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      return session;
    })();
    modelP.catch(() => { modelP = null; });
    return modelP;
  }
  // Run U²-Net on any drawable. Returns the normalised 320×320 mask (Float32, 0..1).
  async function segment(img, W, H) {
    const s = await loadModel();
    const c = resample(img, W, H, SIZE, SIZE, "#ffffff");
    const input = new ort.Tensor("float32", preprocess(pixels(c), SIZE, SIZE), [1, 3, SIZE, SIZE]);
    const feeds = {};
    feeds[s.inputNames[0]] = input;
    const out = await s.run(feeds);
    const raw = normalise(out[s.outputNames[0]].data);
    Object.keys(out).forEach(k => { if (out[k].dispose) out[k].dispose(); });
    return raw;
  }

  // =====================================================================
  // Items (batch) and per-item caches
  // =====================================================================
  let items = [], sel = null, nextId = 1, queue = Promise.resolve(), bgImg = null;
  function maskKey() { return [o.threshold, o.shift, o.feather, o.smart].join("|"); }
  function prepare(it) { // working copies at mask resolution (once per item)
    if (it.guide) return;
    const f = fitSize(it.W, it.H, MASK_MAX);
    it.mw = f.w; it.mh = f.h;
    it.small = resample(it.img, it.W, it.H, f.w, f.h);
    it.guide = grayOf(it.small);
  }
  function ensureBase(it) {
    const key = maskKey();
    if (it.baseKey === key && it.baseCv) return it.baseCv;
    prepare(it);
    it.baseCv = alphaCanvas(computeMask(it.raw, it.guide, it.mw, it.mh, it.W, o), it.mw, it.mh);
    it.baseKey = key; it.maskCv = null;
    return it.baseCv;
  }
  // base + brush strokes at mask resolution
  function ensureMask(it) {
    ensureBase(it);
    if (it.maskCv && it.maskVer === it.strokes.length + "|" + it.baseKey) return it.maskCv;
    const c = it.maskCv && it.maskCv.width === it.mw ? it.maskCv : canvas(it.mw, it.mh), x = ctx2d(c);
    x.globalCompositeOperation = "copy"; x.drawImage(it.baseCv, 0, 0); x.globalCompositeOperation = "source-over";
    it.strokes.forEach(s => paintStroke(x, s, it.mw / it.W, 0));
    it.maskCv = c; it.maskVer = it.strokes.length + "|" + it.baseKey;
    return c;
  }
  // Blurred background with the subject's colours pushed out (normalised convolution), at a small size.
  function blurBg(it) {
    const mask = ensureMask(it), key = it.maskVer + "|" + o.blur;
    if (it.blurKey === key) return it.blurCv;
    const f = fitSize(it.W, it.H, 360), w = f.w, h = f.h, n = w * h;
    const p = pixels(resample(it.small, it.mw, it.mh, w, h)), a = pixels(resample(mask, it.mw, it.mh, w, h));
    const r = Math.max(1, Math.round(o.blur / 100 * 14 * Math.max(w, h) / 360));
    const wt = new Float32Array(n), ch = [new Float32Array(n), new Float32Array(n), new Float32Array(n)], pl = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
    for (let i = 0; i < n; i++) {
      const bgw = Math.pow(1 - a[i * 4 + 3] / 255, 2);
      wt[i] = bgw;
      for (let c = 0; c < 3; c++) { pl[c][i] = p[i * 4 + c]; ch[c][i] = p[i * 4 + c] * bgw; }
    }
    const W2 = blur3(wt, w, h, r), out = canvas(w, h), x = out.getContext("2d"), d = x.createImageData(w, h);
    const bl = ch.map(c => blur3(c, w, h, r)), pb = pl.map(c => blur3(c, w, h, r)), eps = 0.03;
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) d.data[i * 4 + c] = (bl[c][i] + pb[c][i] * eps) / (W2[i] + eps);
    for (let i = 0; i < n; i++) d.data[i * 4 + 3] = 255;
    x.putImageData(d, 0, 0);
    it.blurCv = out; it.blurKey = key;
    return out;
  }
  function paintBg(x, w, h, it) {
    x.imageSmoothingQuality = "high";
    if (o.bg === "color") { x.fillStyle = o.color; x.fillRect(0, 0, w, h); }
    else if (o.bg === "blur") x.drawImage(blurBg(it), 0, 0, w, h);
    else if (o.bg === "image" && bgImg) {
      const iw = bgImg.naturalWidth, ih = bgImg.naturalHeight, s = Math.max(w / iw, h / ih), dw = iw * s, dh = ih * s;
      x.drawImage(bgImg, (w - dw) / 2, (h - dh) / 2, dw, dh);
    }
  }
  // Composite at w×h into ctx x. img: source drawable; mask: alpha drawable; fg: scratch canvas of w×h.
  function compose(x, w, h, it, img, mask, fg, ghost) {
    x.clearRect(0, 0, w, h);
    paintBg(x, w, h, it);
    if (ghost) { x.globalAlpha = 0.28; x.drawImage(img, 0, 0, w, h); x.globalAlpha = 1; }
    const f = fg.getContext("2d");
    f.globalCompositeOperation = "copy"; f.imageSmoothingQuality = "high"; f.drawImage(img, 0, 0, w, h);
    f.globalCompositeOperation = "destination-in"; f.drawImage(mask, 0, 0, w, h);
    f.globalCompositeOperation = "source-over";
    x.drawImage(fg, 0, 0);
  }
  function outDims(it) { return fitSize(it.W, it.H, MAX_SIDE, MAX_AREA); }
  // Full-resolution mask: upscale the working mask smoothly, then re-rasterise brush strokes crisply at full size.
  function fullMask(it, w, h) {
    const m = canvas(w, h), x = ctx2d(m);
    x.imageSmoothingQuality = "high"; x.drawImage(ensureBase(it), 0, 0, w, h);
    it.strokes.forEach(s => paintStroke(x, s, w / it.W, 0));
    return m;
  }
  function renderFull(it) {
    const d = outDims(it), out = canvas(d.w, d.h);
    compose(out.getContext("2d"), d.w, d.h, it, it.img, fullMask(it, d.w, d.h), canvas(d.w, d.h), false);
    return out;
  }
  function renderMask(it) {
    const d = outDims(it), out = canvas(d.w, d.h), x = out.getContext("2d");
    x.fillStyle = "#000"; x.fillRect(0, 0, d.w, d.h); x.drawImage(fullMask(it, d.w, d.h), 0, 0);
    return out;
  }
  function baseName(it) { return it.name.replace(/\.[^.]+$/, "") || "image"; }
  function exportType() { return o.fmt === "webp" && webpOk ? "image/webp" : "image/png"; }
  async function exportBlob(it) { const t = exportType(); return toBlob(renderFull(it), t, t === "image/webp" ? 0.95 : undefined); }
  function exportName(it) { return baseName(it) + (o.bg === "none" ? "-no-bg." : "-cutout.") + (exportType() === "image/webp" ? "webp" : "png"); }

  // =====================================================================
  // Stage (preview + brush)
  // =====================================================================
  const stage = $("#stage"), wrap = $("#stageWrap"), cursor = $("#cursor"), sx = stage.getContext("2d");
  let view = null, mode = "compare", split = 0.5, drawQ = false, stroke = null;
  function layout() {
    if (!sel || !sel.img) { stage.width = stage.height = 0; stage.style.width = stage.style.height = "0"; view = null; return; }
    const W = sel.W, H = sel.H, cw = Math.max(200, wrap.clientWidth - 2), maxH = Math.max(300, Math.min(window.innerHeight * 0.6, 600));
    const s = Math.min(cw / W, maxH / H, 2), dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = Math.max(1, Math.round(W * s)), cssH = Math.max(1, Math.round(H * s));
    stage.style.width = cssW + "px"; stage.style.height = cssH + "px";
    stage.width = Math.round(cssW * dpr); stage.height = Math.round(cssH * dpr);
    view = { css: s, it: sel, img: resample(sel.img, W, H, stage.width, stage.height), fg: canvas(stage.width, stage.height) };
    draw();
  }
  function draw() {
    drawQ = false;
    if (!view || !sel) return;
    const w = stage.width, h = stage.height, img = view.img;
    if (!sel.raw) { sx.clearRect(0, 0, w, h); sx.drawImage(img, 0, 0); return; }
    compose(sx, w, h, sel, img, ensureMask(sel), view.fg, mode !== "compare" && o.ghost);
    if (mode === "compare") {
      const X = Math.round(split * w), dpr = w / parseFloat(stage.style.width);
      sx.save(); sx.beginPath(); sx.rect(0, 0, X, h); sx.clip(); sx.clearRect(0, 0, X, h); sx.drawImage(img, 0, 0); sx.restore();
      sx.fillStyle = "#fff"; sx.fillRect(X - dpr, 0, 2 * dpr, h);
      sx.beginPath(); sx.arc(X, h / 2, 15 * dpr, 0, Math.PI * 2); sx.fill();
      sx.strokeStyle = "rgba(0,0,0,.35)"; sx.lineWidth = dpr; sx.stroke();
      sx.fillStyle = "#1a1e23"; sx.beginPath();
      sx.moveTo(X - 4 * dpr, h / 2 - 5 * dpr); sx.lineTo(X - 9 * dpr, h / 2); sx.lineTo(X - 4 * dpr, h / 2 + 5 * dpr);
      sx.moveTo(X + 4 * dpr, h / 2 - 5 * dpr); sx.lineTo(X + 9 * dpr, h / 2); sx.lineTo(X + 4 * dpr, h / 2 + 5 * dpr); sx.fill();
      label("Original", 10 * dpr, dpr, X > 90 * dpr); label("Cut-out", w - 10 * dpr, dpr, w - X > 90 * dpr, true);
    }
  }
  function label(t, x, dpr, show, right) {
    if (!show) return;
    sx.save(); sx.font = "600 " + 11.5 * dpr + "px Inter, system-ui, sans-serif";
    const tw = sx.measureText(t).width + 16 * dpr, X = right ? x - tw : x, Y = 10 * dpr;
    sx.fillStyle = "rgba(0,0,0,.6)"; sx.beginPath();
    if (sx.roundRect) sx.roundRect(X, Y, tw, 22 * dpr, 11 * dpr); else sx.rect(X, Y, tw, 22 * dpr);
    sx.fill(); sx.fillStyle = "#fff"; sx.textBaseline = "middle"; sx.fillText(t, X + 8 * dpr, Y + 11 * dpr); sx.restore();
  }
  function redraw() { if (!drawQ) { drawQ = true; requestAnimationFrame(draw); } }
  function toImg(e) { const r = stage.getBoundingClientRect(); return [(e.clientX - r.left) / view.css, (e.clientY - r.top) / view.css]; }
  function moveCursor(e) {
    if (mode === "compare" || !view) { cursor.hidden = true; return; }
    const r = stage.getBoundingClientRect(), wr = wrap.getBoundingClientRect();
    cursor.hidden = false;
    cursor.style.width = cursor.style.height = o.brush + "px";
    cursor.style.left = (e.clientX - wr.left) + "px"; cursor.style.top = (e.clientY - wr.top) + "px";
    cursor.className = "bg-cursor " + mode;
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) cursor.hidden = true;
  }
  stage.addEventListener("pointerdown", e => {
    if (!view || !sel || !sel.raw || e.button > 0) return;
    stage.focus({ preventScroll: true });
    stage.setPointerCapture(e.pointerId);
    if (mode === "compare") { const r = stage.getBoundingClientRect(); split = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)); redraw(); return; }
    const mask = ensureMask(sel);
    stroke = { mode: mode, size: o.brush / view.css, hard: 1 - o.soft / 100, pts: [toImg(e)] };
    paintStroke(ctx2d(mask), stroke, sel.mw / sel.W, 0);
    redraw();
  });
  stage.addEventListener("pointermove", e => {
    moveCursor(e);
    if (!view || !sel || !stage.hasPointerCapture(e.pointerId)) return;
    if (mode === "compare") { const r = stage.getBoundingClientRect(); split = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)); redraw(); return; }
    if (!stroke) return;
    const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
    const from = stroke.pts.length;
    (evs.length ? evs : [e]).forEach(ev => stroke.pts.push(toImg(ev)));
    paintStroke(ctx2d(sel.maskCv), stroke, sel.mw / sel.W, from);
    redraw();
  });
  function endStroke() {
    if (!stroke || !sel) { stroke = null; return; }
    sel.undo.push(sel.strokes.slice());
    if (sel.undo.length > 80) sel.undo.shift();
    sel.strokes = sel.strokes.concat([stroke]);
    sel.maskVer = sel.strokes.length + "|" + sel.baseKey; // the live mask already contains it
    stroke = null; ui(); thumb(sel);
  }
  stage.addEventListener("pointerup", endStroke);
  stage.addEventListener("pointercancel", endStroke);
  stage.addEventListener("pointerleave", () => { cursor.hidden = true; });
  stage.addEventListener("keydown", e => {
    if (mode !== "compare") return;
    const d = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
    if (d) { e.preventDefault(); split = Math.max(0, Math.min(1, split + d * (e.shiftKey ? 0.1 : 0.02))); redraw(); }
  });
  function doUndo() {
    if (!sel || !sel.undo.length) return;
    sel.strokes = sel.undo.pop(); sel.maskCv = null; ui(); draw(); thumb(sel);
  }
  function setMode(v) {
    mode = v; press($("#mode"), v);
    stage.classList.toggle("is-brush", v !== "compare"); stage.classList.toggle("is-pan", v === "compare");
    if (v === "compare") cursor.hidden = true;
    $("#ghostWrap").hidden = v === "compare";
    redraw();
  }

  // =====================================================================
  // UI
  // =====================================================================
  const list = $("#list");
  let webpOk = true;
  function press(el, v) { $$("button", el).forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === v))); }
  function seg(el, value, onPick) { press(el, value); $$("button", el).forEach(b => b.addEventListener("click", () => onPick(b.dataset.v))); }

  let busyOn = false;
  function busy(text, frac, sub) {
    busyOn = true;
    $("#busy").hidden = false; $("#busyText").textContent = text; $("#busySub").textContent = sub || "";
    const bar = $("#busyBar"), prog = bar.parentNode;
    prog.classList.toggle("indet", frac < 0);
    if (frac >= 0) bar.style.width = Math.round(Math.min(1, frac) * 100) + "%";
  }
  function idle() { busyOn = false; $("#busy").hidden = true; }

  function ui() {
    const ready = !!(sel && sel.raw), any = items.some(it => it.raw);
    $("#save").disabled = $("#copy").disabled = $("#mask").disabled = !ready;
    $("#zip").disabled = !any;
    $("#zip").hidden = items.length < 2;
    $("#undo").disabled = !(sel && sel.undo.length);
    $("#reset").disabled = !(sel && sel.strokes.length);
    $("#empty").hidden = items.length > 0;
    wrap.classList.toggle("has-img", items.length > 0);
    list.hidden = items.length < 2;
    $("#add").hidden = !items.length;
    $("#colorRow").hidden = o.bg !== "color"; $("#blurRow").hidden = o.bg !== "blur"; $("#imageRow").hidden = o.bg !== "image";
    $$("#swatches .bg-sw").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.c === o.color.toLowerCase())));
    if (sel) {
      const d = outDims(sel), scaled = d.w !== sel.W;
      $("#outSize").textContent = d.w + "×" + d.h + (scaled ? " (browser limit)" : "");
      $("#meta").textContent = sel.name + " · " + sel.W + "×" + sel.H + (sel.err ? " · " + sel.err : sel.ms ? " · cut out in " + (sel.ms / 1000).toFixed(1) + " s" : "") +
        (sel.strokes.length ? " · " + sel.strokes.length + (sel.strokes.length === 1 ? " edit" : " edits") : "");
    } else { $("#outSize").textContent = ""; $("#meta").textContent = ""; }
    if (sel && !sel.raw && !sel.err && !busyOn) busy("Waiting in line…", -1, "");
    if (sel && (sel.raw || sel.err) && !exporting) idle();
  }

  function addFiles(files) {
    const fresh = files.filter(f => /^image\//.test(f.type) || /\.(jpe?g|png|webp|gif|bmp|avif)$/i.test(f.name))
      .map(f => ({ id: nextId++, file: f, name: f.name || "image.png", strokes: [], undo: [] }));
    if (!fresh.length) { if (files.length) S3.toast("Those don't look like images", "error"); return Promise.resolve(); }
    items = items.concat(fresh);
    fresh.forEach(it => {
      it.el = S3.h("li", { "class": "bg-item", tabindex: "0", title: it.name });
      list.appendChild(it.el); renderItem(it);
    });
    if (!sel || !sel.raw) select(fresh[0]);
    ui();
    return Promise.all(fresh.map(it => (queue = queue.then(() => processItem(it)))));
  }
  let processing = false, exporting = false;
  async function processItem(it) {
    if (items.indexOf(it) < 0) return;
    processing = true;
    try {
      if (!it.img) {
        const img = await loadImage(it.file);
        it.img = img; it.W = img.naturalWidth; it.H = img.naturalHeight;
        if (sel === it) layout();
      }
      if (!session) await loadModel();
      if (sel === it) busyForSel("Removing background…");
      await nextFrame();
      const t0 = performance.now();
      it.raw = await segment(it.img, it.W, it.H);
      prepare(it); ensureMask(it);
      it.ms = performance.now() - t0;
      it.err = null;
    } catch (e) {
      it.err = e && e.message || "Failed"; S3.toast(it.err, "error");
    } finally {
      processing = false;
      if (sel === it || !sel || sel.raw) idle();
      renderItem(it); thumb(it);
      if (sel === it) { if (!view) layout(); else draw(); }
      ui();
    }
  }
  function busyForSel(t) {
    const left = items.filter(x => !x.raw && !x.err).length;
    busy(t, -1, left > 1 ? left + " images left · runs on your device" : "Runs on your device · nothing is uploaded");
  }
  function renderItem(it) {
    const st = it.err ? '<span class="chip bad bg-st">Error</span>' : it.raw ? "" : '<span class="chip bg-st">…</span>';
    it.el.innerHTML = '<span class="bg-thumb checker"></span>' + st + '<span class="bg-name">' + S3.esc(it.name) + "</span>" +
      '<button class="btn ghost small icon bg-rm" data-rm aria-label="Remove ' + S3.esc(it.name) + '"><svg class="i"><use href="#i-x"/></svg></button>';
    it.el.classList.toggle("is-sel", it === sel);
    if (it.thumb) it.el.replaceChild(it.thumb, it.el.firstChild);
  }
  function thumb(it) {
    if (!it.img || !it.el) return;
    const f = fitSize(it.W, it.H, 160), c = it.thumb || canvas(f.w, f.h);
    c.width = f.w; c.height = f.h; c.className = "bg-thumb checker";
    const small = resample(it.small || it.img, it.small ? it.mw : it.W, it.small ? it.mh : it.H, f.w, f.h);
    if (it.raw) compose(c.getContext("2d"), f.w, f.h, it, small, ensureMask(it), canvas(f.w, f.h), false);
    else c.getContext("2d").drawImage(small, 0, 0);
    it.thumb = c;
    if (it.el.firstChild !== c) it.el.replaceChild(c, it.el.firstChild);
  }
  function select(it) {
    const old = sel; sel = it;
    if (old && old.el) old.el.classList.remove("is-sel");
    if (it && it.el) it.el.classList.add("is-sel");
    if (it && it.img) layout(); else { view = null; layout(); }
    if (it && !it.raw && !it.err) busyForSel(session ? "Waiting in line…" : "Preparing…");
    ui();
  }
  function remove(it) {
    items = items.filter(x => x !== it); it.el.remove();
    if (it.img) URL.revokeObjectURL(it.img._url);
    if (sel === it) { sel = null; select(items[0] || null); }
    if (!items.length) idle();
    ui();
  }
  list.addEventListener("click", e => {
    const li = e.target.closest(".bg-item"); if (!li) return;
    const it = items.filter(x => x.el === li)[0]; if (!it) return;
    if (e.target.closest("[data-rm]")) { remove(it); return; }
    if (it !== sel) select(it);
  });
  list.addEventListener("keydown", e => { if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("bg-item")) { e.preventDefault(); e.target.click(); } });

  // ---------- controls ----------
  function refreshAll() { // mask settings or background changed
    save();
    if (sel && sel.raw) draw();
    items.forEach(it => { if (it !== sel && it.raw) it.thumbDirty = true; });
    clearTimeout(refreshAll.t);
    refreshAll.t = setTimeout(() => items.forEach(it => { if (it.raw && (it.thumbDirty || it === sel)) { it.thumbDirty = false; thumb(it); } }), 350);
    ui();
  }
  function slider(id, key, fmt, onChange) {
    const el = $("#" + id), out = $("#" + id + "V");
    el.value = o[key]; out.textContent = fmt(o[key]);
    el.addEventListener("input", () => { o[key] = +el.value; out.textContent = fmt(o[key]); onChange(); });
  }
  const live = (() => { let q = false; return () => { if (q) return; q = true; requestAnimationFrame(() => { q = false; refreshAll(); }); }; })();
  slider("threshold", "threshold", v => v, live);
  slider("shift", "shift", v => (v > 0 ? "+" : "") + v + " px", live);
  slider("feather", "feather", v => v + " px", live);
  slider("blur", "blur", v => v, live);
  slider("brush", "brush", v => v + " px", save);
  slider("soft", "soft", v => v + "%", save);
  $("#smart").checked = o.smart;
  $("#smart").addEventListener("change", e => { o.smart = e.target.checked; refreshAll(); });
  $("#ghost").checked = o.ghost;
  $("#ghost").addEventListener("change", e => { o.ghost = e.target.checked; save(); redraw(); });
  seg($("#mode"), mode, setMode);
  seg($("#fmt"), o.fmt, v => { o.fmt = v; press($("#fmt"), v); save(); });
  seg($("#bg"), o.bg, v => {
    if (v === "image" && !bgImg) { $("#bgPick input").click(); }
    o.bg = v; press($("#bg"), v); refreshAll();
  });
  const SW = ["#ffffff", "#000000", "#f1f3f6", "#ffd84d", "#ff6b6b", "#3dd6d0", "#4f7cff", "#9b6bff"];
  SW.forEach(c => {
    const b = S3.h("button", { "class": "bg-sw", type: "button", "data-c": c, "aria-label": "Background " + c, title: c });
    b.style.background = c;
    b.addEventListener("click", () => { o.color = c; $("#color").value = c; refreshAll(); });
    $("#swatches").appendChild(b);
  });
  $("#color").value = o.color;
  $("#color").addEventListener("input", e => { o.color = e.target.value.toLowerCase(); live(); });
  $("#bgPick input").addEventListener("change", async e => {
    const f = e.target.files[0]; e.target.value = "";
    if (!f) return;
    try {
      const im = await loadImage(f);
      if (bgImg) URL.revokeObjectURL(bgImg._url);
      bgImg = im; $("#bgName").textContent = f.name; o.bg = "image"; press($("#bg"), "image"); refreshAll();
    } catch (err) { S3.toast(err.message, "error"); }
  });
  $("#undo").addEventListener("click", doUndo);
  $("#reset").addEventListener("click", () => {
    if (!sel || !sel.strokes.length) return;
    sel.undo.push(sel.strokes.slice()); sel.strokes = []; sel.maskCv = null; ui(); draw(); thumb(sel);
  });
  document.addEventListener("keydown", e => {
    if (e.target.closest && e.target.closest("input, textarea, select")) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); doUndo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "c" || k === "e" || k === "r") setMode({ c: "compare", e: "erase", r: "restore" }[k]);
    else if (k === "[" || k === "]") {
      o.brush = Math.max(4, Math.min(200, o.brush + (k === "]" ? 1 : -1) * Math.max(2, Math.round(o.brush * 0.15))));
      $("#brush").value = o.brush; $("#brushV").textContent = o.brush + " px"; save();
      cursor.style.width = cursor.style.height = o.brush + "px";
    }
  });

  S3.drop($("#drop"), addFiles, { accept: "image/*,.jpg,.jpeg,.png,.webp,.gif,.bmp,.avif", multiple: true });
  $("#add input").addEventListener("change", e => { const fs = Array.prototype.slice.call(e.target.files); e.target.value = ""; addFiles(fs); });
  wrap.addEventListener("dragover", e => { if (items.length) e.preventDefault(); });
  wrap.addEventListener("drop", e => { if (!items.length) return; e.preventDefault(); addFiles(Array.prototype.slice.call((e.dataTransfer && e.dataTransfer.files) || [])); });
  document.addEventListener("paste", e => {
    const fs = Array.prototype.slice.call((e.clipboardData && e.clipboardData.files) || []).filter(f => /^image\//.test(f.type));
    if (fs.length) { e.preventDefault(); addFiles(fs.map((f, i) => f.name && f.name !== "image.png" ? f : new File([f], "pasted-" + (i + 1) + ".png", { type: f.type }))); }
  });
  $("#sample").addEventListener("click", e => { e.preventDefault(); loadSample(); });

  $("#save").addEventListener("click", async () => {
    if (!sel || !sel.raw) return;
    try { S3.download(await exportBlob(sel), exportName(sel)); S3.done(); } catch (err) { S3.toast(err.message, "error"); }
  });
  $("#mask").addEventListener("click", async () => {
    if (!sel || !sel.raw) return;
    try { S3.download(await toBlob(renderMask(sel), "image/png"), baseName(sel) + "-mask.png"); S3.done(); } catch (err) { S3.toast(err.message, "error"); }
  });
  $("#copy").addEventListener("click", async () => {
    if (!sel || !sel.raw) return;
    try {
      const b = await toBlob(renderFull(sel), "image/png");
      await navigator.clipboard.write([new ClipboardItem({ "image/png": b })]);
      S3.toast("Cut-out copied ✓"); S3.done();
    } catch (err) { S3.toast("Your browser blocked copying images. Use Download instead.", "error"); }
  });
  $("#zip").addEventListener("click", async () => {
    const done = items.filter(it => it.raw);
    if (!done.length) return;
    const btn = $("#zip"), used = {}, files = [];
    btn.disabled = true; exporting = true;
    try {
      for (let i = 0; i < done.length; i++) {
        busy("Exporting " + (i + 1) + " of " + done.length + "…", i / done.length, "Full resolution");
        await nextFrame();
        let n = exportName(done[i]);
        while (used[n.toLowerCase()]) n = n.replace(/(\.[^.]+)$/, "-" + (used[n.toLowerCase()]++) + "$1");
        used[n.toLowerCase()] = 1;
        files.push({ name: n, data: new Uint8Array(await (await exportBlob(done[i])).arrayBuffer()) });
      }
      S3.download(S3.zip(files), "background-removed.zip"); S3.done();
    } catch (err) { S3.toast(err.message, "error"); }
    finally { exporting = false; idle(); btn.disabled = false; ui(); }
  });

  if (window.ResizeObserver) new ResizeObserver(S3.debounce(() => { if (sel && sel.img) layout(); }, 80)).observe(wrap);
  else window.addEventListener("resize", S3.debounce(() => { if (sel && sel.img) layout(); }, 100));
  toBlob(canvas(2, 2), "image/webp", 0.8).then(b => { webpOk = b.type === "image/webp"; if (!webpOk) { $("#fmt [data-v=webp]").hidden = true; o.fmt = "png"; press($("#fmt"), "png"); } }).catch(() => {});
  setMode("compare");
  ui();

  // =====================================================================
  // Generated sample photo (drawn locally, no real person, no remote files)
  // =====================================================================
  function rng(seed) { let s = seed; return () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff; }
  function drawPerson(x, cx, base, s, shirt) {
    // torso / shoulders
    let g = x.createLinearGradient(cx - 1.4 * s, 0, cx + 1.4 * s, 0);
    g.addColorStop(0, shirt[0]); g.addColorStop(0.5, shirt[1]); g.addColorStop(1, shirt[0]);
    x.fillStyle = g; x.beginPath();
    x.moveTo(cx - 1.55 * s, base);
    x.bezierCurveTo(cx - 1.55 * s, base - 0.75 * s, cx - 1.25 * s, base - 1.15 * s, cx - 0.5 * s, base - 1.25 * s);
    x.lineTo(cx + 0.5 * s, base - 1.25 * s);
    x.bezierCurveTo(cx + 1.25 * s, base - 1.15 * s, cx + 1.55 * s, base - 0.75 * s, cx + 1.55 * s, base);
    x.closePath(); x.fill();
    // neck
    x.fillStyle = "#d9a07b"; x.beginPath(); x.moveTo(cx - 0.24 * s, base - 1.5 * s); x.lineTo(cx + 0.24 * s, base - 1.5 * s);
    x.lineTo(cx + 0.3 * s, base - 1.2 * s); x.quadraticCurveTo(cx, base - 1.02 * s, cx - 0.3 * s, base - 1.2 * s); x.fill();
    // head
    g = x.createRadialGradient(cx - 0.2 * s, base - 2.1 * s, 0.1 * s, cx, base - 1.95 * s, 0.75 * s);
    g.addColorStop(0, "#f7d2b4"); g.addColorStop(1, "#d99a74");
    x.fillStyle = g; x.beginPath(); x.ellipse(cx, base - 1.95 * s, 0.52 * s, 0.64 * s, 0, 0, Math.PI * 2); x.fill();
    // hair
    x.fillStyle = "#2b1c16"; x.beginPath();
    x.ellipse(cx, base - 2.22 * s, 0.6 * s, 0.48 * s, 0, Math.PI * 0.95, Math.PI * 2.05); x.fill();
    x.beginPath(); x.ellipse(cx - 0.42 * s, base - 2.0 * s, 0.16 * s, 0.36 * s, 0.25, 0, Math.PI * 2); x.fill();
    x.beginPath(); x.ellipse(cx + 0.42 * s, base - 2.0 * s, 0.16 * s, 0.36 * s, -0.25, 0, Math.PI * 2); x.fill();
    // face
    x.fillStyle = "#2b1c16";
    x.beginPath(); x.arc(cx - 0.19 * s, base - 1.95 * s, 0.045 * s, 0, Math.PI * 2); x.arc(cx + 0.19 * s, base - 1.95 * s, 0.045 * s, 0, Math.PI * 2); x.fill();
    x.strokeStyle = "#9c4f3c"; x.lineWidth = 0.035 * s; x.lineCap = "round";
    x.beginPath(); x.arc(cx, base - 1.8 * s, 0.17 * s, Math.PI * 0.2, Math.PI * 0.8); x.stroke();
    // collar
    x.fillStyle = "rgba(255,255,255,.85)"; x.beginPath(); x.moveTo(cx - 0.32 * s, base - 1.22 * s); x.lineTo(cx, base - 0.9 * s);
    x.lineTo(cx + 0.32 * s, base - 1.22 * s); x.lineTo(cx + 0.18 * s, base - 1.25 * s); x.lineTo(cx, base - 1.05 * s); x.lineTo(cx - 0.18 * s, base - 1.25 * s); x.fill();
  }
  // Busy, textured background (stripes + noise) so the model has to find the subject, not just a flat colour.
  function testScene(w, h) {
    const c = canvas(w, h), x = c.getContext("2d"), r = rng(11);
    const g = x.createLinearGradient(0, 0, w, h); g.addColorStop(0, "#5d8a68"); g.addColorStop(1, "#2f5a6b");
    x.fillStyle = g; x.fillRect(0, 0, w, h);
    x.strokeStyle = "rgba(255,255,255,.10)"; x.lineWidth = w / 60;
    for (let i = -h; i < w; i += w / 18) { x.beginPath(); x.moveTo(i, 0); x.lineTo(i + h, h); x.stroke(); }
    for (let i = 0; i < 120; i++) { x.fillStyle = "rgba(" + (r() > 0.5 ? "255,255,255" : "0,0,0") + "," + (0.04 + r() * 0.08) + ")"; x.fillRect(r() * w, r() * h, w * 0.05 * r(), h * 0.05 * r()); }
    drawPerson(x, w / 2, h, h * 0.36, ["#b8322b", "#e0574c"]);
    return c;
  }
  function demoScene(w, h) {
    const c = canvas(w, h), x = c.getContext("2d"), r = rng(5);
    let g = x.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, "#87b7c9"); g.addColorStop(0.55, "#c9d8b6"); g.addColorStop(1, "#6f8f5a");
    x.fillStyle = g; x.fillRect(0, 0, w, h);
    for (let i = 0; i < 38; i++) { // soft bokeh
      const bx = r() * w, by = r() * h * 0.8, br = (0.03 + r() * 0.08) * w;
      g = x.createRadialGradient(bx, by, 0, bx, by, br);
      const col = ["255,244,214", "214,240,255", "255,220,180", "200,235,190"][i % 4];
      g.addColorStop(0, "rgba(" + col + ",.55)"); g.addColorStop(0.7, "rgba(" + col + ",.25)"); g.addColorStop(1, "rgba(" + col + ",0)");
      x.fillStyle = g; x.beginPath(); x.arc(bx, by, br, 0, Math.PI * 2); x.fill();
    }
    x.fillStyle = "rgba(60,90,50,.35)"; // distant hedge
    x.beginPath(); x.moveTo(0, h * 0.72);
    for (let i = 0; i <= 24; i++) x.lineTo(w * i / 24, h * (0.66 + 0.04 * Math.sin(i * 1.7)));
    x.lineTo(w, h); x.lineTo(0, h); x.fill();
    drawPerson(x, w * 0.5, h, h * 0.31, ["#c98a1e", "#efb847"]);
    return c;
  }
  async function loadSample() {
    const b = await toBlob(demoScene(1200, 800), "image/jpeg", 0.92);
    return addFiles([new File([b], "sample-portrait.jpg", { type: "image/jpeg" })]);
  }

  window.S3_DEMO = async function () {
    await loadSample();
    split = 0.5; setMode("compare");
  };

  // =====================================================================
  // Self-test (scripts/smoke.py): unit maths + a REAL run of the model on a generated photo.
  // =====================================================================
  window.S3_SELFTEST = async function () {
    const near = (a, b, e) => Math.abs(a - b) <= (e || 1e-4);
    // 1) pre-processing: 2×1 image, pixel A = (255,0,128), pixel B = (0,255,255); NCHW layout
    const pre = preprocess(new Uint8ClampedArray([255, 0, 128, 255, 0, 255, 255, 255]), 2, 1);
    const ok1 = pre.length === 6 &&
      near(pre[0], (1 - 0.485) / 0.229) && near(pre[1], (0 - 0.485) / 0.229) &&        // R plane
      near(pre[2], (0 - 0.456) / 0.224) && near(pre[3], (1 - 0.456) / 0.224) &&        // G plane
      near(pre[4], (128 / 255 - 0.406) / 0.225) && near(pre[5], (1 - 0.406) / 0.225) && // B plane
      near(pre[0], 2.24891, 1e-4) && near(pre[2], -2.03571, 1e-4);
    // 2) min-max normalise, bilinear, box, morph
    const nm = normalise(new Float32Array([2, 4, 6, 10])), flat = normalise(new Float32Array([3, 3]));
    const bl = bilinear(new Float32Array([0, 1, 0, 1]), 2, 2, 4, 4);         // columns 0|1 → 0, .25, .75, 1
    const bx = box(new Float32Array([0, 0, 9, 0, 0]), 5, 1, 1);               // spreads 9 into 3s
    const di = morph(new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0]), 3, 3, 1, true), er = morph(new Float32Array([1, 1, 1, 1, 1, 1, 1, 1, 0]), 3, 3, 1, false);
    const ok2 = near(nm[0], 0) && near(nm[1], 0.25) && near(nm[3], 1) && flat[0] === 0 &&
      near(bl[0], 0) && near(bl[1], 0.25) && near(bl[2], 0.75) && near(bl[3], 1) && near(bl[13], 0.25) &&
      near(bx[1], 3) && near(bx[2], 3) && near(bx[3], 3) && near(bx[0], 0) &&
      Array.prototype.every.call(di, v => v === 1) && er[4] === 0 && er[0] === 1;
    // 3) threshold ramp + edge shift on a synthetic raw mask (square in the middle)
    const raw = new Float32Array(SIZE * SIZE);
    for (let y = 100; y < 220; y++) for (let x = 100; x < 220; x++) raw[y * SIZE + x] = 1;
    const base = computeMask(raw, null, SIZE, SIZE, SIZE, { threshold: 50, shift: 0, feather: 0, smart: false });
    const grow = computeMask(raw, null, SIZE, SIZE, SIZE, { threshold: 50, shift: 6, feather: 0, smart: false });
    const shrink = computeMask(raw, null, SIZE, SIZE, SIZE, { threshold: 50, shift: -6, feather: 0, smart: false });
    const at = (m, x, y) => m[y * SIZE + x];
    const ok3 = at(base, 160, 160) === 1 && at(base, 10, 10) === 0 && at(base, 96, 160) === 0 &&
      at(grow, 96, 160) === 1 && at(grow, 90, 160) === 0 && at(shrink, 103, 160) === 0 && at(shrink, 108, 160) === 1;
    // 4) brush: an erase dab clears alpha at its centre, a restore dab brings it back
    const bc = canvas(40, 40), bctx = bc.getContext("2d");
    bctx.fillStyle = "#fff"; bctx.fillRect(0, 0, 40, 40);
    paintStroke(bctx, { mode: "erase", size: 16, hard: 1, pts: [[10, 20], [30, 20]] }, 1, 0);
    const a1 = bctx.getImageData(20, 20, 1, 1).data[3], a2 = bctx.getImageData(20, 2, 1, 1).data[3];
    paintStroke(bctx, { mode: "restore", size: 8, hard: 1, pts: [[20, 20]] }, 1, 0);
    const a3 = bctx.getImageData(20, 20, 1, 1).data[3];
    const ok4 = a1 === 0 && a2 === 255 && a3 === 255;
    // 5) the real model on a generated person-on-texture photo
    const W = 480, H = 360, scene = testScene(W, H);
    const m = await segment(scene, W, H);
    const mean = (x0, y0, x1, y1) => { let s = 0, n = 0; for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { s += m[y * SIZE + x]; n++; } return s / n; };
    // subject: head centre ≈ (0.5, 0.32), torso ≈ (0.5, 0.9) of the frame; corners are background
    const head = mean(152, 92, 168, 112), torso = mean(140, 280, 180, 300);
    const corners = [mean(0, 0, 24, 24), mean(296, 0, 320, 24), mean(0, 296, 24, 320), mean(296, 296, 320, 320)];
    const ok5 = head > 0.7 && torso > 0.7 && corners.every(v => v < 0.1);
    // 6) full composite: alpha follows the mask; colour background fills where the subject isn't
    const it = { raw: m, img: scene, W, H, strokes: [], undo: [], name: "t.png" };
    const saveBg = o.bg, saveColor = o.color;
    o.bg = "none";
    const cut = renderFull(it), cp = cut.getContext("2d");
    const aIn = cp.getImageData(W / 2, Math.round(H * 0.32), 1, 1).data[3], aOut = cp.getImageData(5, 5, 1, 1).data[3];
    o.bg = "color"; o.color = "#ff00ff";
    const col = renderFull(it).getContext("2d").getImageData(5, 5, 1, 1).data;
    o.bg = saveBg; o.color = saveColor;
    const ok6 = cut.width === W && cut.height === H && aIn > 240 && aOut < 15 && col[0] === 255 && col[1] === 0 && col[2] === 255 && col[3] === 255;
    return (ok1 && ok2 && ok3 && ok4 && ok5 && ok6) ||
      { ok1, ok2, ok3, ok4, ok5, ok6, head: +head.toFixed(3), torso: +torso.toFixed(3), corners: corners.map(v => +v.toFixed(3)), a: [a1, a2, a3], aIn, aOut };
  };
})();
