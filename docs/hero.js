/* Character-art hero — Moonshot-style eclipse + glyph wordmark.
 *
 *   · A big radial glow + an eclipse: a faint ring, a bright lower crescent, and
 *     a small light that orbits the rim (loops slowly).
 *   · The "AlphaZero 五子棋" wordmark rendered as a static field of glyphs with a
 *     gentle brightness wave sweeping across, that scatter + brighten around the
 *     cursor.
 *
 * Self-contained 2D canvas — no dependencies, no build step, no watermark — so it
 * ships on GitHub Pages as static files. Progressive enhancement: under
 * prefers-reduced-motion or if 2D canvas is unavailable, the static CSS hero
 * stays. The real <h1> remains in the DOM for accessibility (hidden when live).
 */
(() => {
  const hero = document.querySelector(".hero");
  if (!hero) return;
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const WORDMARK = "AlphaZero 五子棋";
  const RAMP = " .·:-=+*o&%#@"; // sparse → dense
  const DPR = Math.min(window.devicePixelRatio || 1, 2);

  const canvas = document.createElement("canvas");
  canvas.className = "hero-canvas";
  canvas.setAttribute("aria-hidden", "true");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const buf = document.createElement("canvas");
  const bctx = buf.getContext("2d");
  if (!bctx) return;

  let W = 0, H = 0, cell = 8, cols = 0, rows = 0;
  let density = new Float32Array(0);
  let haloCx = 0, haloCy = 0, haloR = 1, ringR = 1;

  function build() {
    const rect = hero.getBoundingClientRect();
    W = canvas.width = buf.width = Math.max(1, Math.round(rect.width * DPR));
    H = canvas.height = buf.height = Math.max(1, Math.round(rect.height * DPR));
    cell = Math.max(6, Math.round(8 * DPR));

    let cy = 0.5;
    const titleEl = hero.querySelector(".hero-title");
    if (titleEl && rect.height > 0) {
      const tr = titleEl.getBoundingClientRect();
      cy = ((tr.top + tr.height / 2) - rect.top) / rect.height;
    }

    bctx.clearRect(0, 0, W, H);
    bctx.fillStyle = "#fff";
    bctx.textAlign = "center";
    bctx.textBaseline = "middle";
    let fs = Math.min(H * 0.17, W * 0.078);
    const font = (px) => `800 ${px}px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif`;
    bctx.font = font(fs);
    while (fs > 10 && bctx.measureText(WORDMARK).width > W * 0.82) {
      fs -= 2;
      bctx.font = font(fs);
    }
    bctx.fillText(WORDMARK, W / 2, H * cy);

    cols = Math.ceil(W / cell);
    rows = Math.ceil(H / cell);
    density = new Float32Array(cols * rows);
    const data = bctx.getImageData(0, 0, W, H).data;
    const half = cell >> 1;
    for (let ry = 0; ry < rows; ry++) {
      const py = Math.min(H - 1, ry * cell + half);
      for (let rx = 0; rx < cols; rx++) {
        const px = Math.min(W - 1, rx * cell + half);
        density[ry * cols + rx] = data[(py * W + px) * 4 + 3] / 255;
      }
    }

    haloCx = W / 2;
    haloCy = H * cy;
    haloR = Math.min(W, H) * 0.54;
    ringR = Math.min(W, H) * 0.2;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
  }

  const mouse = { x: -0.5, y: -0.5, tx: -0.5, ty: -0.5 };
  hero.addEventListener("pointermove", (e) => {
    const rect = hero.getBoundingClientRect();
    mouse.tx = (e.clientX - rect.left) / rect.width;
    mouse.ty = (e.clientY - rect.top) / rect.height;
  });
  hero.addEventListener("pointerleave", () => { mouse.tx = -0.5; mouse.ty = -0.5; });

  let running = true, startT = null, raf = 0;

  function drawEclipse(t, intro) {
    // Big soft glow + a tighter bright core (the "大光晕").
    let g = ctx.createRadialGradient(haloCx, haloCy + haloR * 0.05, 0, haloCx, haloCy, haloR);
    g.addColorStop(0, "rgba(46,143,255,0.22)");
    g.addColorStop(0.4, "rgba(31,182,201,0.07)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
    g = ctx.createRadialGradient(haloCx, haloCy, 0, haloCx, haloCy, ringR * 1.3);
    g.addColorStop(0, "rgba(120,170,255,0.16)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    ctx.save();
    ctx.globalAlpha = intro;
    // Faint full ring.
    ctx.strokeStyle = "rgba(150,185,255,0.32)";
    ctx.lineWidth = 1.2 * DPR;
    ctx.beginPath();
    ctx.arc(haloCx, haloCy, ringR, 0, Math.PI * 2);
    ctx.stroke();
    // Bright lower crescent (the eclipse highlight).
    ctx.shadowColor = "rgba(150,190,255,0.9)";
    ctx.shadowBlur = 20 * DPR;
    ctx.strokeStyle = "rgba(255,255,255,0.92)";
    ctx.lineWidth = 2 * DPR;
    ctx.beginPath();
    ctx.arc(haloCx, haloCy, ringR, Math.PI * 0.16, Math.PI * 0.84);
    ctx.stroke();
    // A small light orbiting the rim (loops slowly).
    const ang = -Math.PI * 0.4 + Math.sin(t * 0.45) * 0.28;
    ctx.shadowBlur = 14 * DPR;
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.beginPath();
    ctx.arc(haloCx + Math.cos(ang) * ringR, haloCy + Math.sin(ang) * ringR, 2.6 * DPR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawWordmark(t, mx, my, intro) {
    ctx.font = `${Math.round(cell * 1.05)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    const R = 135 * DPR, R2 = R * R, rampMax = RAMP.length - 1;
    for (let ry = 0; ry < rows; ry++) {
      for (let rx = 0; rx < cols; rx++) {
        const d = density[ry * cols + rx];
        if (d < 0.18) continue;
        // Static positions; only a gentle brightness wave sweeps across (no drift).
        let x = rx * cell + (cell >> 1);
        let y = ry * cell + (cell >> 1);
        const wave = 0.5 + 0.5 * Math.sin(rx * 0.12 - t * 1.5);
        let bright = d * (0.74 + 0.26 * wave);

        const dx = x - mx, dy = y - my, dist2 = dx * dx + dy * dy;
        if (dist2 < R2) {
          const dist = Math.sqrt(dist2) || 1;
          const f = (R - dist) / R;
          const push = f * f * 30 * DPR;
          x += (dx / dist) * push;
          y += (dy / dist) * push;
          bright = Math.min(1, bright + f * 0.8);
        }
        const hb = 1 - Math.min(1, Math.hypot(x - haloCx, y - haloCy) / haloR);
        bright = Math.min(1, bright + hb * 0.2) * intro;

        const ch = RAMP[Math.max(1, Math.round(bright * rampMax))];
        if (ch === " ") continue;
        const rr = Math.round(46 + 194 * bright);
        const gg = Math.round(143 + 105 * bright);
        ctx.fillStyle = `rgba(${rr},${gg},255,${0.32 + 0.68 * bright})`;
        ctx.fillText(ch, x, y);
      }
    }
  }

  function frame(now) {
    if (!running) return;
    if (startT === null) startT = now;
    const t = (now - startT) / 1000;
    const intro = Math.min(1, t / 1.2);
    mouse.x += (mouse.tx - mouse.x) * 0.08;
    mouse.y += (mouse.ty - mouse.y) * 0.08;
    const mx = mouse.x * W, my = mouse.y * H;

    ctx.fillStyle = "#050506";
    ctx.fillRect(0, 0, W, H);
    drawEclipse(t, intro);
    drawWordmark(t, mx, my, intro);

    raf = requestAnimationFrame(frame);
  }

  function play() { if (!(running && raf)) { running = true; startT = null; raf = requestAnimationFrame(frame); } }
  function pause() { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; }

  if ("IntersectionObserver" in window) {
    new IntersectionObserver((es) => { es[0].isIntersecting ? play() : pause(); }, { threshold: 0.02 }).observe(hero);
  }
  document.addEventListener("visibilitychange", () => { document.hidden ? pause() : play(); });
  let rt = 0;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(build, 120); }, { passive: true });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(build);

  hero.insertBefore(canvas, hero.firstChild);
  hero.classList.add("hero--bg");
  build();
  play();
})();
