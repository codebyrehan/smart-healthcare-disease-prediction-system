const status = document.getElementById('dashboard-status');

async function loadDashboard() {
  try {
    const response = await fetch('/api/metadata');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analytics unavailable');
    const quality = data.quality_summary || {};
    status.textContent = `${quality.rows?.toLocaleString?.() || 0} validated records · ${quality.features || data.features?.length || 0} predictive features`;
    renderQuality(data.quality, quality);
    populateSensitivityFeatures(data.features || []);
  } catch (error) {
    status.textContent = 'Analytics are unavailable until the verified dataset is loaded.';
  }
}

function renderQuality(quality, summary) {
  const target = document.getElementById('target-stats');
  const details = document.getElementById('quality-details');
  if (target) {
    const counts = quality?.class_counts || {};
    target.innerHTML = `<div><strong>${counts['0'] || 0}</strong><span>Outcome 0</span></div><div><strong>${counts['1'] || 0}</strong><span>Outcome 1</span></div><div><strong>${summary?.missing_values ?? '—'}</strong><span>Missing values</span></div><div><strong>${summary?.duplicate_rows ?? '—'}</strong><span>Duplicate rows</span></div>`;
  }
  if (details) {
    const ratio = Number(summary?.finite_value_ratio);
    details.innerHTML = `<div><strong>${summary?.rows ?? '—'}</strong><span>Validated rows</span></div><div><strong>${Number.isFinite(ratio) ? `${(ratio * 100).toFixed(1)}%` : '—'}</strong><span>Finite numeric values</span></div>`;
  }
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

async function loadExperiments() {
  const statusEl = document.getElementById('experiment-status');
  const table = document.getElementById('experiment-table');
  if (!statusEl || !table) return;
  try {
    const response = await fetch('/api/experiments');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Experiment history unavailable');
    const records = Array.isArray(data.experiments) ? data.experiments : [];
    statusEl.textContent = records.length ? `${records.length} reproducible experiment record${records.length === 1 ? '' : 's'}` : 'No experiment runs have been recorded yet.';
    if (!records.length) {
      table.innerHTML = '<div class="empty">Run the verified experiment pipeline to populate this lab.</div>';
      return;
    }
    const headers = ['Experiment', 'Model', 'Dataset', 'Features', 'ROC-AUC', 'F1', 'Recorded (UTC)'];
    table.innerHTML = `<div class="benchmark-row benchmark-head">${headers.map((h) => `<span>${escapeHtml(h)}</span>`).join('')}</div>` + records.slice().reverse().map((row) => `<div class="benchmark-row"><strong>${escapeHtml(row.experiment_id)}</strong><span>${escapeHtml(row.model_name)}</span><span>${escapeHtml(row.dataset_version)}</span><span>${escapeHtml(row.feature_count)}</span><span>${fmt(row.metrics?.roc_auc)}</span><span>${fmt(row.metrics?.f1)}</span><span>${escapeHtml(formatUtc(row.timestamp_utc))}</span></div>`).join('');
  } catch (error) {
    statusEl.textContent = 'Experiment registry is unavailable.';
    table.innerHTML = '';
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

let baselineFeatures = null;
function populateSensitivityFeatures(features) {
  const select = document.getElementById('sensitivity-feature');
  if (!select) return;
  select.innerHTML = features.map((feature) => `<option value="${escapeHtml(feature)}">${escapeHtml(feature)}</option>`).join('');
}

function getBaselineFromForm() {
  const fields = document.querySelectorAll('#fields input, #fields select');
  const baseline = {};
  fields.forEach((field) => { if (field.name) baseline[field.name] = Number(field.value); });
  return baseline;
}

async function runSensitivity() {
  const statusEl = document.getElementById('sensitivity-status');
  const resultsEl = document.getElementById('sensitivity-results');
  const feature = document.getElementById('sensitivity-feature')?.value;
  const start = Number(document.getElementById('sensitivity-start')?.value);
  const end = Number(document.getElementById('sensitivity-end')?.value);
  const steps = Number(document.getElementById('sensitivity-steps')?.value);
  if (!feature || !Number.isFinite(start) || !Number.isFinite(end) || !Number.isInteger(steps) || steps < 2 || steps > 25) {
    statusEl.textContent = 'Enter valid start, end, and step values.';
    return;
  }
  const baseline = baselineFeatures || getBaselineFromForm();
  if (Object.keys(baseline).length !== 8 || Object.values(baseline).some((value) => !Number.isFinite(value))) {
    statusEl.textContent = 'Complete a valid prediction form before running sensitivity analysis.';
    return;
  }
  const values = Array.from({length: steps}, (_, index) => start + ((end - start) * index / (steps - 1)));
  statusEl.textContent = 'Running verified model sensitivity…';
  try {
    const response = await fetch('/api/sensitivity', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({feature, baseline, values})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Sensitivity analysis unavailable');
    resultsEl.innerHTML = data.results.map((row) => `<div class="sensitivity-row"><span>${Number(row.value).toFixed(2)}</span><div class="bar"><i style="width:${Math.min(100, Math.max(0, Number(row.probability) * 100))}%"></i></div><strong>${(Number(row.probability) * 100).toFixed(1)}%</strong></div>`).join('');
    statusEl.textContent = `Sensitivity results for ${escapeHtml(data.feature)}. Model behavior only; not causal or clinical advice.`;
  } catch (error) {
    resultsEl.textContent = '';
    statusEl.textContent = error.message || 'Sensitivity analysis unavailable.';
  }
}

function fmt(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—';
}

function formatUtc(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toISOString().replace('T', ' ').replace('.000Z', ' UTC');
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}

window.setPredictionBaseline = (features) => { baselineFeatures = {...features}; };
document.getElementById('run-sensitivity')?.addEventListener('click', runSensitivity);
loadDashboard();
loadBenchmark();
loadExplainability();
loadExperiments();
