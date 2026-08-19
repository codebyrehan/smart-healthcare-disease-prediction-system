const status = document.getElementById('dashboard-status');

async function loadDashboard() {
  try {
    const response = await fetch('/api/metadata');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analytics unavailable');
    const quality = data.quality_summary || {};
    status.textContent = `${quality.rows?.toLocaleString?.() || 0} validated records · ${quality.features || data.features?.length || 0} predictive features`;
    renderQuality(data.quality, quality);
  } catch (error) {
    status.textContent = 'Analytics are unavailable until the verified dataset is loaded.';
  }
}

function renderQuality(quality, summary) {
  const target = document.getElementById('target-stats');
  if (!target) return;
  const counts = quality?.class_counts || {};
  target.innerHTML = `<div><strong>${counts['0'] || 0}</strong><span>Outcome 0</span></div><div><strong>${counts['1'] || 0}</strong><span>Outcome 1</span></div><div><strong>${summary?.missing_values ?? '—'}</strong><span>Missing values</span></div><div><strong>${summary?.duplicate_rows ?? '—'}</strong><span>Duplicate rows</span></div>`;
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
    table.innerHTML = `<div class="benchmark-row benchmark-head">${headers.map((h) => `<span>${escapeHtml(h)}</span>`).join('')}</div>` +
      data.models.map((row) => `<div class="benchmark-row"><strong>${escapeHtml(row.model)}</strong><span>${fmt(row.accuracy)}</span><span>${fmt(row.precision)}</span><span>${fmt(row.recall)}</span><span>${fmt(row.f1)}</span><span>${fmt(row.roc_auc)}</span><span>${fmt(row.pr_auc)}</span></div>`).join('');
  } catch (error) {
    statusEl.textContent = 'Benchmark artifacts are not available yet. Run the verified training pipeline first.';
  }
}

async function loadExplainability() {
  const statusEl = document.getElementById('explainability-status');
  const list = document.getElementById('importance-list');
  if (!statusEl || !list) return;
  try {
    const response = await fetch('/api/explainability');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Explainability unavailable');
    statusEl.textContent = `Verified global importance · ${escapeHtml(data.model)}`;
    list.innerHTML = data.importance.map((item) => `<div class="importance-row"><span>${escapeHtml(item.feature)}</span><div class="bar"><i style="width:${Math.min(100, Math.max(0, Number(item.importance) * 100))}%"></i></div><strong>${(Number(item.importance) * 100).toFixed(1)}%</strong></div>`).join('');
  } catch (error) {
    statusEl.textContent = 'Verified model explainability is unavailable until a compatible trained model is loaded.';
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
loadExplainability();
