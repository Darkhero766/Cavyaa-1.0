// CAVYAA app interactions

// Register service worker for installable PWA
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch(() => {});
}

document.addEventListener('DOMContentLoaded', () => {
  animateScoreRing();
  animateProgressBars();
  animateMetricBars();
  setupFilterChips();
});

// Animate SVG score ring
function animateScoreRing() {
  const fill = document.querySelector('.score-ring-fill');
  if (!fill) return;
  const score = parseInt(fill.dataset.score || 0);
  const circumference = 440;
  const offset = circumference - (score / 100) * circumference;
  requestAnimationFrame(() => {
    setTimeout(() => { fill.style.strokeDashoffset = offset; }, 80);
  });
}

// Animate progress bars
function animateProgressBars() {
  document.querySelectorAll('.progress-bar-fill[data-pct]').forEach(bar => {
    const pct = bar.dataset.pct;
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = pct + '%'; }, 120);
  });
}

// Animate metric bars in doctor detail
function animateMetricBars() {
  document.querySelectorAll('.metric-bar-fill[data-pct]').forEach(bar => {
    const pct = bar.dataset.pct;
    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = pct + '%'; }, 200);
  });
}

// Filter chips
function setupFilterChips() {
  document.querySelectorAll('.filter-chip[data-filter]').forEach(chip => {
    chip.addEventListener('click', () => {
      const group = chip.dataset.group || 'default';
      document.querySelectorAll(`.filter-chip[data-group="${group}"]`).forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      const filter = chip.dataset.filter;
      filterPatients(filter);
    });
  });
}

function filterPatients(filter) {
  document.querySelectorAll('.patient-card[data-risk]').forEach(card => {
    if (filter === 'all') {
      card.style.display = '';
    } else {
      card.style.display = card.dataset.risk === filter ? '' : 'none';
    }
  });
}

// Score color based on value
function scoreColor(score) {
  if (score >= 80) return '#10B981';
  if (score >= 60) return '#F59E0B';
  return '#EF4444';
}
