const status = document.getElementById('dashboard-status');

async function loadDashboard() {
  try {
    const response = await fetch('/api/metadata');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analytics unavailable');
    status.textContent = `${data.quality.rows.toLocaleString()} validated records · ${data.quality.columns - 1} predictive features`;
    renderQuality(data.quality);
  } catch (error) {
    status.textContent = 'Analytics are unavailable until the verified dataset is loaded.';
  }
}

function renderQuality(quality) {
  const target = document.getElementById('target-stats');
  if (!target) return;
  const counts = quality.class_counts || {};
  target.innerHTML = `<div><strong>${counts['0'] || 0}</strong><span>Outcome 0</span></div><div><strong>${counts['1'] || 0}</strong><span>Outcome 1</span></div>`;
}

async function loadBenchmark() {
  const statusEl = document.getElementById('benchmark-status');
  const table = document.getElementById('benchmark-table');
  if (!statusEl || !table) return;
  try {
    const response = await fetch('/api/benchmark');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Benchmark unavailable');
    statusEl.textContent = `Selected model: ${data.selected_model || 'pending'} · ${data.selection_metric || 'ROC-AUC'} · version ${data.model_version || '—'}`;
    if (!Array.isArray(data.models) || data.models.length === 0) {
      table.textContent = 'No benchmark results are available yet.';
      return;
    }
    const headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'];
    table.innerHTML = `<div class="benchmark-row benchmark-head">${headers.map((h) => `<span>${h}</span>`).join('')}</div>` +
      data.models.map((row) => `<div class="benchmark-row"><strong>${escapeHtml(row.model)}</strong><span>${fmt(row.accuracy)}</span><span>${fmt(row.precision)}</span><span>${fmt(row.recall)}</span><span>${fmt(row.f1)}</span><span>${fmt(row.roc_auc)}</span><span>${fmt(row.pr_auc)}</span></div>`).join('');
  } catch (error) {
    statusEl.textContent = 'Benchmark artifacts are not available yet. Run the verified training pipeline first.';
  }
}

function fmt(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}

loadDashboard();
loadBenchmark();
