/* Editorial hero + scroll-reveal — "Paper Studio" light theme.
 *
 * Replaces the old dark character-art canvas. Motion here is deliberately quiet
 * (per the minimalist protocol): a staggered fade-up for the hero on load, and
 * IntersectionObserver scroll-entry for the major content blocks — translateY(12px)
 * + opacity resolving over 600ms with cubic-bezier(0.16, 1, 0.3, 1). Animates only
 * transform/opacity, never layout.
 *
 * Progressive enhancement: under prefers-reduced-motion, with JS disabled, or
 * without IntersectionObserver, every element stays fully visible — the hidden
 * (opacity:0) state is injected from here, so it can never strand no-JS content.
 */
(() => {
  const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduce || !("IntersectionObserver" in window)) return;

  const EASE = "cubic-bezier(0.16, 1, 0.3, 1)";
  const style = document.createElement("style");
  style.textContent = `
    .reveal {
      opacity: 0;
      transform: translateY(12px);
      transition: opacity 600ms ${EASE}, transform 600ms ${EASE};
      transition-delay: var(--reveal-delay, 0ms);
      will-change: transform;
    }
    .reveal.in { opacity: 1; transform: none; }
  `;
  document.head.appendChild(style);

  const arm = (el, delay) => {
    el.classList.add("reveal");
    if (delay) el.style.setProperty("--reveal-delay", `${delay}ms`);
  };

  // Hero — staggered fade-up on first paint.
  const heroBits = [".hero-kicker", ".hero-title", ".hero-sub", ".hero-cta"]
    .map((s) => document.querySelector(s))
    .filter(Boolean);
  heroBits.forEach((el, i) => arm(el, i * 90));
  requestAnimationFrame(() =>
    requestAnimationFrame(() => heroBits.forEach((el) => el.classList.add("in")))
  );

  // Content blocks — reveal as they scroll into view, then stop observing.
  const blocks = Array.from(
    document.querySelectorAll(
      ".board-pane > .topbar, .board-area, .panel.training, .side-pane > .panel, .analysis-details"
    )
  );
  const io = new IntersectionObserver(
    (entries, obs) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        e.target.classList.add("in");
        obs.unobserve(e.target);
      }
    },
    { threshold: 0.08, rootMargin: "0px 0px -8% 0px" }
  );
  blocks.forEach((el) => {
    arm(el, 0);
    io.observe(el);
  });
})();
