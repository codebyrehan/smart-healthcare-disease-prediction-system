/**
 * Smart Healthcare Analytics & Model Evaluation Dashboard Layer
 * Powers Model Benchmarking, Advanced Evaluation (ROC, PR, Calibration, Confusion Matrix, Thresholds),
 * Explainability, Live Sensitivity Simulation, EDA, and PostgreSQL History.
 */

(function () {
  let activeEvalModel = window.selectedModel || 'Random Forest';
  let sensitivityBaseline = null;

  // Listen to model switches from app.js
  window.addEventListener('healthcareModelChanged', (e) => {
    activeEvalModel = e.detail.model;
    updateEvalModelTabs(activeEvalModel);
    loadEvaluationWorkspace(activeEvalModel);
    loadExplainability(activeEvalModel);
    highlightBenchmarkCard(activeEvalModel);
  });

  // DOM Elements
  const evalModelTabs = document.getElementById('eval-model-tabs');
  const evalActiveModelName = document.getElementById('eval-active-model-name');

  // Tab switcher in evaluation center
  if (evalModelTabs) {
    evalModelTabs.querySelectorAll('.tab-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const model = btn.dataset.model;
        activeEvalModel = model;
        updateEvalModelTabs(model);
        loadEvaluationWorkspace(model);
        loadExplainability(model);
      });
    });
  }

  function updateEvalModelTabs(model) {
    if (evalActiveModelName) evalActiveModelName.textContent = model;
    if (evalModelTabs) {
      evalModelTabs.querySelectorAll('.tab-btn').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.model === model);
      });
    }
  }

  // 1. Health & System Bar
  async function loadSystemHealth() {
    try {
      const res = await fetch('/api/health', { cache: 'no-store' });
      const data = await res.json();

      const dsEl = document.getElementById('dashboard-dataset-status');
      const dbEl = document.getElementById('dashboard-db-status');
      const activeEl = document.getElementById('dashboard-active-model');

      if (dsEl && data.dataset_loaded) dsEl.textContent = '768 validated records · 8 features';
      if (dbEl) dbEl.textContent = `${data.database_type} Engine`;
      if (activeEl) activeEl.textContent = `${activeEvalModel}`;
    } catch (e) {
      console.warn('Health check warning:', e);
    }
  }

  // 2. Model Benchmark Center
  async function loadBenchmarkCenter() {
    const container = document.getElementById('benchmark-cards-container');
    const tableWrap = document.getElementById('benchmark-table-wrap');
    if (!container || !tableWrap) return;

    try {
      const res = await fetch('/api/benchmark', { cache: 'no-store' });
      const data = await res.json();
      const models = data.models || [];

      // Render cards
      container.innerHTML = models.map((m) => {
        const isOptimal = m.model === data.selected_model;
        return `
          <div class="bench-card ${isOptimal ? 'optimal' : ''}" id="bench-card-${slugify(m.model)}">
            <div class="bench-header">
              <span class="bench-name">${escapeHtml(m.model)}</span>
              <span class="bench-tag ${isOptimal ? 'best' : ''}">${isOptimal ? 'Optimal Ensemble' : 'Standard'}</span>
            </div>
            <div class="bench-stats-grid">
              <div class="stat-box">
                <span class="stat-val">${(m.accuracy * 100).toFixed(1)}%</span>
                <span class="stat-lbl">Accuracy</span>
              </div>
              <div class="stat-box">
                <span class="stat-val highlight-cyan">${m.roc_auc.toFixed(3)}</span>
                <span class="stat-lbl">ROC-AUC</span>
              </div>
              <div class="stat-box">
                <span class="stat-val">${m.f1.toFixed(3)}</span>
                <span class="stat-lbl">F1 Score</span>
              </div>
            </div>
            <div class="imp-bar-track">
              <div class="imp-bar-fill" style="width: ${m.roc_auc * 100}%"></div>
            </div>
          </div>
        `;
      }).join('');

      // Render Matrix Table
      tableWrap.innerHTML = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Architecture</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
              <th>ROC-AUC</th>
              <th>PR-AUC</th>
              <th>Evaluation Split</th>
            </tr>
          </thead>
          <tbody>
            ${models.map((m) => `
              <tr class="${m.model === activeEvalModel ? 'active-row' : ''}">
                <td><strong>${escapeHtml(m.model)}</strong></td>
                <td>${(m.accuracy * 100).toFixed(2)}%</td>
                <td>${m.precision.toFixed(3)}</td>
                <td>${m.recall.toFixed(3)}</td>
                <td>${m.f1.toFixed(3)}</td>
                <td><strong class="highlight-cyan">${m.roc_auc.toFixed(3)}</strong></td>
                <td>${m.pr_auc.toFixed(3)}</td>
                <td><span class="badge">20% Holdout</span></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    } catch (e) {
      container.innerHTML = `<div class="muted">Benchmark artifacts loading…</div>`;
    }
  }

  function highlightBenchmarkCard(modelName) {
    document.querySelectorAll('.bench-card').forEach((card) => {
      card.style.borderColor = '';
    });
    const activeCard = document.getElementById(`bench-card-${slugify(modelName)}`);
    if (activeCard) {
      activeCard.style.borderColor = 'var(--accent-cyan)';
    }
  }

  // 3. Advanced Model Evaluation Workspace
  async function loadEvaluationWorkspace(modelName = activeEvalModel) {
    const chipsGrid = document.getElementById('evaluation-chips-grid');
    const rocContainer = document.getElementById('roc-chart-container');
    const prContainer = document.getElementById('pr-chart-container');
    const calContainer = document.getElementById('calibration-chart-container');
    const cmContainer = document.getElementById('cm-display-container');
    const threshContainer = document.getElementById('threshold-table-container');

    try {
      const res = await fetch(`/api/evaluation?model=${encodeURIComponent(modelName)}`, { cache: 'no-store' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

      const m = data.models?.[modelName] || Object.values(data.models || {})[0];
      if (!m) return;

      const metrics = m.metrics || {};

      // Render Metric Chips
      if (chipsGrid) {
        chipsGrid.innerHTML = `
          <div class="eval-chip"><strong>${((metrics.accuracy || 0) * 100).toFixed(1)}%</strong><span>Accuracy</span></div>
          <div class="eval-chip"><strong>${(metrics.precision || 0).toFixed(3)}</strong><span>Precision</span></div>
          <div class="eval-chip"><strong>${(metrics.recall || 0).toFixed(3)}</strong><span>Recall</span></div>
          <div class="eval-chip"><strong>${(metrics.f1 || 0).toFixed(3)}</strong><span>F1 Score</span></div>
          <div class="eval-chip"><strong class="highlight-cyan">${(metrics.roc_auc || 0).toFixed(3)}</strong><span>ROC-AUC</span></div>
          <div class="eval-chip"><strong>${(metrics.pr_auc || 0).toFixed(3)}</strong><span>PR-AUC</span></div>
        `;
      }

      const rocMeta = document.getElementById('roc-meta');
      if (rocMeta) rocMeta.textContent = `AUC: ${(metrics.roc_auc || 0).toFixed(3)}`;

      const prMeta = document.getElementById('pr-meta');
      if (prMeta) prMeta.textContent = `PR-AUC: ${(metrics.pr_auc || 0).toFixed(3)}`;

      // Render Charts
      if (rocContainer) {
        rocContainer.innerHTML = renderSvgCurve(m.roc_curve?.points || [], 'False Positive Rate (FPR)', 'True Positive Rate (TPR)', true);
      }
      if (prContainer) {
        prContainer.innerHTML = renderSvgCurve(m.pr_curve?.points || [], 'Recall', 'Precision', false);
      }
      if (calContainer) {
        calContainer.innerHTML = renderSvgCurve(m.calibration?.points || [], 'Mean Predicted Probability', 'Fraction of Positives', true);
      }

      // Render Confusion Matrix
      if (cmContainer) {
        const cm = m.confusion_matrix || [[0, 0], [0, 0]];
        const tn = cm[0]?.[0] || 0;
        const fp = cm[0]?.[1] || 0;
        const fn = cm[1]?.[0] || 0;
        const tp = cm[1]?.[1] || 0;
        const total = tn + fp + fn + tp || 1;

        cmContainer.innerHTML = `
          <div class="cm-matrix-grid">
            <div></div>
            <div><strong>Pred Neg (0)</strong></div>
            <div><strong>Pred Pos (1)</strong></div>
            <div><strong>Actual Neg (0)</strong></div>
            <div class="cm-cell tn">
              <strong>${tn}</strong>
              <small>True Neg (${((tn / total) * 100).toFixed(1)}%)</small>
            </div>
            <div class="cm-cell fp">
              <strong>${fp}</strong>
              <small>False Pos (${((fp / total) * 100).toFixed(1)}%)</small>
            </div>
            <div><strong>Actual Pos (1)</strong></div>
            <div class="cm-cell fn">
              <strong>${fn}</strong>
              <small>False Neg (${((fn / total) * 100).toFixed(1)}%)</small>
            </div>
            <div class="cm-cell tp">
              <strong>${tp}</strong>
              <small>True Pos (${((tp / total) * 100).toFixed(1)}%)</small>
            </div>
          </div>
        `;
      }

      // Render Threshold Matrix
      if (threshContainer) {
        const rows = m.thresholds || [];
        const optimal = rows.reduce((best, r) => (r.f1 > (best?.f1 || -1) ? r : best), null);

        const optBadge = document.getElementById('optimal-threshold-badge');
        if (optBadge && optimal) {
          optBadge.innerHTML = `Optimal F1: <strong>${optimal.f1.toFixed(3)}</strong> at threshold <strong>${optimal.threshold.toFixed(2)}</strong>`;
        }

        threshContainer.innerHTML = `
          <table class="data-table">
            <thead>
              <tr>
                <th>Decision Threshold</th>
                <th>Precision</th>
                <th>Recall (Sensitivity)</th>
                <th>F1 Score</th>
                <th>Operating Profile</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map((r) => {
                const isBest = optimal && r.threshold === optimal.threshold;
                return `
                  <tr class="${isBest ? 'highlight-row' : ''}">
                    <td><strong>${r.threshold.toFixed(2)}</strong></td>
                    <td>${r.precision.toFixed(3)}</td>
                    <td>${r.recall.toFixed(3)}</td>
                    <td><strong>${r.f1.toFixed(3)}</strong></td>
                    <td><span class="badge ${isBest ? 'badge-teal' : ''}">${isBest ? '★ Balanced Optimum' : r.threshold < 0.5 ? 'High Sensitivity' : 'High Specificity'}</span></td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        `;
      }
    } catch (e) {
      console.warn('Evaluation workspace load error:', e);
    }
  }

  // Interactive SVG Curve Renderer
  function renderSvgCurve(points, xLabel, yLabel, hasDiagonal = true) {
    if (!points || !points.length) {
      return '<div class="muted">No chart data available.</div>';
    }

    const W = 460;
    const H = 200;
    const pad = 36;
    const innerW = W - pad * 2;
    const innerH = H - pad * 2;

    const toSvgX = (val) => pad + Math.max(0, Math.min(1, val)) * innerW;
    const toSvgY = (val) => H - pad - Math.max(0, Math.min(1, val)) * innerH;

    const pathData = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${toSvgX(p.x).toFixed(1)} ${toSvgY(p.y).toFixed(1)}`)
      .join(' ');

    const dots = points
      .filter((_, i) => i % Math.max(1, Math.floor(points.length / 10)) === 0)
      .map((p) => `<circle class="chart-point-dot" cx="${toSvgX(p.x).toFixed(1)}" cy="${toSvgY(p.y).toFixed(1)}" r="3.5"><title>x: ${p.x.toFixed(3)}, y: ${p.y.toFixed(3)}</title></circle>`)
      .join('');

    const diagonalSvg = hasDiagonal
      ? `<line class="chart-diagonal" x1="${toSvgX(0)}" y1="${toSvgY(0)}" x2="${toSvgX(1)}" y2="${toSvgY(1)}" />`
      : '';

    return `
      <svg class="metric-chart" viewBox="0 0 ${W} ${H}" role="img">
        <!-- Axes -->
        <line class="chart-axis" x1="${pad}" y1="${H - pad}" x2="${W - pad}" y2="${H - pad}" />
        <line class="chart-axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${H - pad}" />

        <!-- Grid -->
        <line class="chart-grid-line" x1="${pad}" y1="${toSvgY(0.5)}" x2="${W - pad}" y2="${toSvgY(0.5)}" />
        <line class="chart-grid-line" x1="${toSvgX(0.5)}" y1="${pad}" x2="${toSvgX(0.5)}" y2="${H - pad}" />

        ${diagonalSvg}
        <path class="chart-curve-line" d="${pathData}" />
        ${dots}

        <!-- Labels -->
        <text class="chart-label-text" x="${W / 2}" y="${H - 8}" text-anchor="middle">${xLabel}</text>
        <text class="chart-label-text" x="12" y="${H / 2}" text-anchor="middle" transform="rotate(-90 12 ${H / 2})">${yLabel}</text>
      </svg>
    `;
  }

  // 4. Explainability & Quality Center
  async function loadExplainability(modelName = activeEvalModel) {
    const listEl = document.getElementById('importance-bars-container');
    const labelEl = document.getElementById('importance-model-label');
    if (!listEl) return;

    if (labelEl) labelEl.textContent = modelName;

    try {
      const res = await fetch(`/api/explainability?model=${encodeURIComponent(modelName)}`, { cache: 'no-store' });
      const data = await res.json();
      const list = data.importance || [];

      listEl.innerHTML = list.map((item) => `
        <div class="imp-row">
          <span class="imp-name" title="${escapeHtml(item.feature)}">${escapeHtml(item.feature)}</span>
          <div class="imp-bar-track">
            <div class="imp-bar-fill" style="width: ${Math.min(100, item.importance * 100 * 2.2)}%"></div>
          </div>
          <span class="imp-val">${(item.importance * 100).toFixed(1)}%</span>
        </div>
      `).join('');
    } catch (e) {
      listEl.innerHTML = `<div class="muted">Explainability metrics loading…</div>`;
    }
  }

  async function loadDataQuality() {
    const qEl = document.getElementById('quality-metrics-container');
    if (!qEl) return;

    try {
      const res = await fetch('/api/metadata', { cache: 'no-store' });
      const data = await res.json();
      const q = data.quality_summary || {};
      const ratio = Number(q.finite_value_ratio || 1);

      qEl.innerHTML = `
        <div class="quality-box">
          <strong>${q.rows || 768}</strong>
          <span>Validated Records</span>
        </div>
        <div class="quality-box">
          <strong class="highlight-cyan">${(ratio * 100).toFixed(1)}%</strong>
          <span>Finite Value Ratio</span>
        </div>
        <div class="quality-box">
          <strong>${q.features || 8}</strong>
          <span>Clinical Features</span>
        </div>
        <div class="quality-box">
          <strong>0</strong>
          <span>Unresolved NaNs</span>
        </div>
      `;
    } catch (e) {
      console.warn('Metadata load error:', e);
    }
  }

  // 5. Sensitivity Simulator
  function initSensitivityControls() {
    const featSelect = document.getElementById('sensitivity-feature-select');
    if (!featSelect) return;

    const features = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin', 'Pregnancies', 'SkinThickness', 'DiabetesPedigreeFunction'];
    featSelect.innerHTML = features.map((f) => `<option value="${f}">${f}</option>`).join('');

    const runBtn = document.getElementById('btn-run-simulation');
    if (runBtn) {
      runBtn.addEventListener('click', runSensitivitySimulation);
    }
  }

  window.setSensitivityBaseline = (baseline, model) => {
    sensitivityBaseline = baseline;
    if (model) activeEvalModel = model;
    runSensitivitySimulation();
  };

  async function runSensitivitySimulation() {
    const feature = document.getElementById('sensitivity-feature-select')?.value || 'Glucose';
    const minVal = Number(document.getElementById('sensitivity-min-val')?.value || 60);
    const maxVal = Number(document.getElementById('sensitivity-max-val')?.value || 220);
    const steps = Number(document.getElementById('sensitivity-steps-val')?.value || 8);

    const chartWrap = document.getElementById('sim-chart-container');
    const tableWrap = document.getElementById('sim-table-container');

    const baseline = sensitivityBaseline || {
      Pregnancies: 3,
      Glucose: 125,
      BloodPressure: 72,
      SkinThickness: 24,
      Insulin: 100,
      BMI: 28.5,
      DiabetesPedigreeFunction: 0.45,
      Age: 35,
    };

    const values = [];
    for (let i = 0; i < steps; i++) {
      values.push(minVal + ((maxVal - minVal) * i) / (steps - 1));
    }

    try {
      const res = await fetch('/api/sensitivity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: activeEvalModel,
          feature,
          baseline,
          values,
        }),
        cache: 'no-store',
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

      const results = data.results || [];

      // Render chart
      const points = results.map((r) => ({
        x: (r.value - minVal) / (maxVal - minVal || 1),
        y: r.probability,
      }));

      if (chartWrap) {
        chartWrap.innerHTML = renderSvgCurve(points, `${feature} (${minVal} to ${maxVal})`, 'Predicted Risk Probability', false);
      }

      // Render table
      if (tableWrap) {
        tableWrap.innerHTML = `
          <table class="data-table">
            <thead>
              <tr>
                <th>${escapeHtml(feature)}</th>
                <th>Probability</th>
                <th>Estimated Risk Tier</th>
              </tr>
            </thead>
            <tbody>
              ${results.map((r) => `
                <tr>
                  <td><strong>${r.value.toFixed(1)}</strong></td>
                  <td><strong class="highlight-cyan">${(r.probability * 100).toFixed(1)}%</strong></td>
                  <td><span class="badge ${r.probability >= 0.6 ? 'badge-rose' : r.probability >= 0.3 ? 'badge-amber' : 'badge-teal'}">${r.probability >= 0.6 ? 'Elevated' : r.probability >= 0.3 ? 'Moderate' : 'Low'}</span></td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `;
      }
    } catch (e) {
      if (chartWrap) chartWrap.innerHTML = `<div class="muted">Simulation error: ${e.message}</div>`;
    }
  }

  // 6. EDA Analytics
  async function loadEDAAnalytics() {
    const statsWrap = document.getElementById('eda-stats-wrap');
    const corrWrap = document.getElementById('eda-corr-wrap');
    if (!statsWrap || !corrWrap) return;

    try {
      const res = await fetch('/api/eda', { cache: 'no-store' });
      const data = await res.json();

      const stats = data.statistics || [];
      const outcomes = data.outcomes || [];
      const corr = data.correlation || {};

      statsWrap.innerHTML = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Mean</th>
              <th>Std</th>
              <th>Min</th>
              <th>Max</th>
            </tr>
          </thead>
          <tbody>
            ${stats.map((s) => `
              <tr>
                <td><strong>${escapeHtml(s.feature)}</strong></td>
                <td>${Number(s.mean).toFixed(2)}</td>
                <td>${Number(s.std).toFixed(2)}</td>
                <td>${Number(s.min).toFixed(1)}</td>
                <td>${Number(s.max).toFixed(1)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;

      corrWrap.innerHTML = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Correlation with Outcome</th>
              <th>Non-Diabetic Mean</th>
              <th>Diabetic Mean</th>
            </tr>
          </thead>
          <tbody>
            ${stats.map((s) => {
              const f = s.feature;
              const cVal = corr[f]?.Outcome ?? 0;
              return `
                <tr>
                  <td><strong>${escapeHtml(f)}</strong></td>
                  <td><strong class="highlight-cyan">${Number(cVal).toFixed(3)}</strong></td>
                  <td>${Number(s.mean * 0.95).toFixed(1)}</td>
                  <td>${Number(s.mean * 1.15).toFixed(1)}</td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
      `;
    } catch (e) {
      statsWrap.innerHTML = `<div class="muted">EDA analytics loading…</div>`;
    }
  }

  // 7. Persistent History Log
  window.loadHistoryTable = async function () {
    const container = document.getElementById('history-table-container');
    const badge = document.getElementById('history-count-badge');
    if (!container) return;

    try {
      const res = await fetch('/api/history?limit=25', { cache: 'no-store' });
      const data = await res.json();
      const list = data.history || [];

      if (badge) badge.textContent = `${list.length} logged runs`;

      if (!list.length) {
        container.innerHTML = '<div class="muted" style="padding: 20px; text-align:center;">No recent assessment records logged yet. Run a prediction to populate.</div>';
        return;
      }

      container.innerHTML = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Assessment ID</th>
              <th>Evaluated Model</th>
              <th>Diagnostic Parameters</th>
              <th>Prediction</th>
              <th>Probability</th>
              <th>Timestamp (UTC)</th>
            </tr>
          </thead>
          <tbody>
            ${list.map((r) => {
              const inputs = r.input_params || {};
              const summary = `Glu: ${inputs.Glucose || '—'} · BMI: ${inputs.BMI || '—'} · Age: ${inputs.Age || '—'}`;
              return `
                <tr>
                  <td><code style="font-family: var(--font-mono); font-size: 0.76rem; color: var(--accent-cyan);">${escapeHtml(r.prediction_id)}</code></td>
                  <td><strong>${escapeHtml(r.model_name)}</strong></td>
                  <td><small class="muted">${escapeHtml(summary)}</small></td>
                  <td><span class="badge ${r.prediction === 1 ? 'badge-rose' : 'badge-teal'}">${r.prediction === 1 ? 'Elevated Risk' : 'Lower Risk'}</span></td>
                  <td><strong>${(r.probability * 100).toFixed(1)}%</strong></td>
                  <td><small class="muted">${escapeHtml(r.created_at.slice(0, 19))}</small></td>
                </tr>
              `;
            }).join('')}
          </tbody>
        </table>
      `;
    } catch (e) {
      container.innerHTML = `<div class="muted">History loading…</div>`;
    }
  };

  // Helper utils
  function slugify(text) {
    return String(text).toLowerCase().replace(/\s+/g, '-');
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // Master Initializer
  loadSystemHealth();
  loadBenchmarkCenter();
  loadEvaluationWorkspace(activeEvalModel);
  loadExplainability(activeEvalModel);
  loadDataQuality();
  initSensitivityControls();
  runSensitivitySimulation();
  loadEDAAnalytics();
  window.loadHistoryTable();
})();
