// CAVYAA — App interactions & animations

// Register service worker (PWA)
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

document.addEventListener('DOMContentLoaded', () => {
  animateScoreRing();
  animateProgressBars();
  animateMetricBars();
  animateCounters();
  setupFilterChips();
  animateWeekCheckmarks();
  initLandingOrbs();
});

/* ── Score ring ──────────────────────────────────────────── */
function animateScoreRing() {
  const fill = document.querySelector('.score-ring-fill');
  const glow = document.querySelector('.score-ring-glow');
  if (!fill) return;
  const score       = parseInt(fill.dataset.score || 0);
  const circumference = 440;
  const offset      = circumference - (score / 100) * circumference;
  const color       = score >= 80 ? '#ffffff'
                    : score >= 60 ? '#FDE68A'
                    :               '#FCA5A5';
  fill.setAttribute('stroke', color);
  if (glow) glow.setAttribute('stroke', color);
  requestAnimationFrame(() => {
    setTimeout(() => {
      fill.style.strokeDashoffset = offset;
      if (glow) glow.style.strokeDashoffset = offset;
    }, 120);
  });
}

/* ── Progress bars ───────────────────────────────────────── */
function animateProgressBars() {
  document.querySelectorAll('.progress-bar-fill[data-pct]').forEach((bar, i) => {
    const pct = bar.dataset.pct;
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = pct + '%'; }, 150 + i * 40);
  });
}

/* ── Metric bars (doctor detail) ─────────────────────────── */
function animateMetricBars() {
  document.querySelectorAll('.metric-bar-fill[data-pct]').forEach((bar, i) => {
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = bar.dataset.pct + '%'; }, 250 + i * 60);
  });
}

/* ── Animated number counters ────────────────────────────── */
function animateCounters() {
  document.querySelectorAll('[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count);
    const duration = 900;
    const start = performance.now();
    const suffix = el.dataset.suffix || '';
    function step(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      el.textContent = Math.round(eased * target) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }
    // Delay counters to match card animation
    setTimeout(() => requestAnimationFrame(step), 300);
  });
}

/* ── Week dot checkmarks (draw-in SVG) ───────────────────── */
function animateWeekCheckmarks() {
  document.querySelectorAll('.check-path').forEach((path, i) => {
    path.style.strokeDasharray = '200';
    path.style.strokeDashoffset = '200';
    setTimeout(() => {
      path.style.transition = 'stroke-dashoffset 0.4s cubic-bezier(0.4,0,0.2,1)';
      path.style.strokeDashoffset = '0';
    }, 400 + i * 80);
  });
}

/* ── Filter chips ────────────────────────────────────────── */
function setupFilterChips() {
  document.querySelectorAll('.filter-chip[data-filter]').forEach(chip => {
    chip.addEventListener('click', () => {
      const group = chip.dataset.group || 'default';
      document.querySelectorAll(`.filter-chip[data-group="${group}"]`)
              .forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      filterPatients(chip.dataset.filter);
    });
  });
}
function filterPatients(filter) {
  document.querySelectorAll('.patient-card[data-risk]').forEach(card => {
    const show = filter === 'all' || card.dataset.risk === filter;
    card.style.display = show ? '' : 'none';
    if (show) {
      card.style.animation = 'fadeUp 0.3s cubic-bezier(0.4,0,0.2,1) both';
      card.addEventListener('animationend', () => { card.style.animation = ''; }, { once: true });
    }
  });
}

/* ── Subtle landing background orbs (extra JS-driven movement) ─ */
function initLandingOrbs() {
  const orb = document.querySelector('.orb-3');
  if (!orb) return;
  // Already animated via CSS; no JS needed here. Reserved for future interaction.
}

/* ── Landing role pill CTA update ────────────────────────── */
function selectRole(role) {
  document.querySelectorAll('.role-pill').forEach(p => p.classList.remove('active'));
  event.currentTarget.classList.add('active');
  const btn = document.getElementById('ctaBtn');
  if (role === 'doctor') {
    btn.href = '/select/doctor';
    btn.textContent = 'Enter Doctor Portal';
  } else {
    btn.href = '/select/patient';
    btn.textContent = 'Start My Journey';
  }
}
