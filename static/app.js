/**
 * Smart Healthcare Risk Prediction Application Layer
 * Manages patient inputs, clinical presets, model switching, biometric risk ring, and PDF export.
 */

(function () {
  // Clinical Parameter Specifications
  const PARAM_SPECS = [
    { name: 'Pregnancies', label: 'Pregnancies', min: 0, max: 20, step: 1, defaultVal: 1, unit: 'Count', desc: 'Number of times pregnant' },
    { name: 'Glucose', label: 'Plasma Glucose', min: 40, max: 300, step: 1, defaultVal: 120, unit: 'mg/dL', desc: '2h oral glucose tolerance test' },
    { name: 'BloodPressure', label: 'Diastolic BP', min: 30, max: 200, step: 1, defaultVal: 72, unit: 'mm Hg', desc: 'Diastolic blood pressure' },
    { name: 'SkinThickness', label: 'Skinfold Thickness', min: 0, max: 100, step: 1, defaultVal: 24, unit: 'mm', desc: 'Triceps skin fold thickness' },
    { name: 'Insulin', label: 'Serum Insulin', min: 0, max: 900, step: 1, defaultVal: 90, unit: 'μU/mL', desc: '2-Hour serum insulin' },
    { name: 'BMI', label: 'Body Mass Index', min: 10.0, max: 70.0, step: 0.1, defaultVal: 28.5, unit: 'kg/m²', desc: 'Weight in kg/(height in m)²' },
    { name: 'DiabetesPedigreeFunction', label: 'Diabetes Pedigree', min: 0.05, max: 3.0, step: 0.01, defaultVal: 0.38, unit: 'Score', desc: 'Genetic diabetes pedigree function' },
    { name: 'Age', label: 'Age', min: 18, max: 120, step: 1, defaultVal: 33, unit: 'Years', desc: 'Patient age' },
  ];

  // Clinical Personas
  const PRESETS = {
    low: { Pregnancies: 1, Glucose: 88, BloodPressure: 66, SkinThickness: 20, Insulin: 60, BMI: 22.4, DiabetesPedigreeFunction: 0.24, Age: 25 },
    borderline: { Pregnancies: 3, Glucose: 128, BloodPressure: 74, SkinThickness: 28, Insulin: 110, BMI: 29.5, DiabetesPedigreeFunction: 0.48, Age: 38 },
    elevated: { Pregnancies: 6, Glucose: 168, BloodPressure: 82, SkinThickness: 35, Insulin: 220, BMI: 36.8, DiabetesPedigreeFunction: 0.85, Age: 52 },
  };

  // State
  window.selectedModel = 'Random Forest';
  let lastAssessmentResult = null;

  // DOM Elements
  const fieldsContainer = document.getElementById('fields');
  const predictionForm = document.getElementById('prediction-form');
  const formError = document.getElementById('form-error');
  const predictBtn = document.getElementById('predict-button');
  const modelPills = document.querySelectorAll('.model-pill');
  const presetBtns = document.querySelectorAll('.preset-btn');
  const evaluatedModelTag = document.getElementById('evaluated-model-tag');
  const riskRingFill = document.getElementById('risk-ring-fill');
  const riskScoreValue = document.getElementById('risk-score-value');
  const riskScoreLabel = document.getElementById('risk-score-label');
  const riskTierBadge = document.getElementById('risk-tier-badge');
  const riskTierText = document.getElementById('risk-tier-text');
  const factorsList = document.getElementById('factors-list');
  const btnExportPdf = document.getElementById('btn-export-pdf');

  // Populate Input Fields
  function initFormFields() {
    if (!fieldsContainer) return;
    fieldsContainer.innerHTML = PARAM_SPECS.map((spec) => `
      <div class="form-group">
        <label class="form-label" for="inp-${spec.name}">
          <span>${spec.label}</span>
          <span class="unit">${spec.unit}</span>
        </label>
        <input 
          type="number" 
          id="inp-${spec.name}" 
          name="${spec.name}" 
          class="form-input" 
          min="${spec.min}" 
          max="${spec.max}" 
          step="${spec.step}" 
          value="${spec.defaultVal}" 
          required
        >
      </div>
    `).join('');
  }

  // Model Switching Handlers
  function setModel(modelName) {
    window.selectedModel = modelName;
    modelPills.forEach((btn) => {
      const active = btn.dataset.model === modelName;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-checked', active ? 'true' : 'false');
    });

    if (evaluatedModelTag) {
      evaluatedModelTag.textContent = modelName;
    }

    const activeModelIndicator = document.getElementById('dashboard-active-model');
    if (activeModelIndicator) {
      activeModelIndicator.textContent = `${modelName}`;
    }

    // Broadcast model change to dashboard charts
    window.dispatchEvent(new CustomEvent('healthcareModelChanged', { detail: { model: modelName } }));
  }

  modelPills.forEach((btn) => {
    btn.addEventListener('click', () => setModel(btn.dataset.model));
  });

  // Preset Handlers
  presetBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const presetKey = btn.dataset.preset;
      if (presetKey === 'reset') {
        PARAM_SPECS.forEach((s) => {
          const inp = document.getElementById(`inp-${s.name}`);
          if (inp) inp.value = s.defaultVal;
        });
      } else if (PRESETS[presetKey]) {
        const values = PRESETS[presetKey];
        Object.entries(values).forEach(([k, v]) => {
          const inp = document.getElementById(`inp-${k}`);
          if (inp) inp.value = v;
        });
      }
    });
  });

  // Extract Form Data
  function getFormData() {
    const data = {};
    PARAM_SPECS.forEach((s) => {
      const inp = document.getElementById(`inp-${s.name}`);
      data[s.name] = inp ? Number(inp.value) : s.defaultVal;
    });
    return data;
  }

  // Update 3D Biometric Signal Display
  function renderBiometricResult(result) {
    lastAssessmentResult = result;
    const probability = Number(result.probability || 0);
    const percentage = (probability * 100).toFixed(1);
    const isElevated = result.prediction === 1;

    // SVG Ring circumference: 2 * PI * 90 ≈ 565.48
    const circumference = 565.48;
    const offset = circumference - (probability * circumference);

    if (riskRingFill) {
      riskRingFill.style.strokeDashoffset = offset;
      if (probability < 0.30) {
        riskRingFill.style.stroke = '#10b981'; // Emerald
      } else if (probability < 0.60) {
        riskRingFill.style.stroke = '#f59e0b'; // Amber
      } else {
        riskRingFill.style.stroke = '#f43f5e'; // Rose
      }
    }

    if (riskScoreValue) {
      riskScoreValue.textContent = `${percentage}%`;
    }

    if (riskScoreLabel) {
      riskScoreLabel.textContent = `${result.model} Signal`;
    }

    if (riskTierBadge && riskTierText) {
      riskTierBadge.className = 'risk-tier-badge';
      if (probability < 0.30) {
        riskTierBadge.classList.add('tier-low');
        riskTierText.textContent = 'Low Diabetes Likelihood';
      } else if (probability < 0.60) {
        riskTierBadge.classList.add('tier-moderate');
        riskTierText.textContent = 'Moderate Risk Indicator';
      } else {
        riskTierBadge.classList.add('tier-elevated');
        riskTierText.textContent = 'Elevated Diabetes Risk';
      }
    }

    if (factorsList) {
      const factors = result.contributing_factors || [];
      if (factors.length) {
        factorsList.innerHTML = factors.map((f) => `<li>${escapeHtml(f)}</li>`).join('');
      } else {
        factorsList.innerHTML = '<li>Measured values align with non-elevated baseline reference.</li>';
      }
    }

    // Pass baseline to sensitivity engine
    if (window.setSensitivityBaseline) {
      window.setSensitivityBaseline(result.inputs, result.model);
    }
  }

  // Handle Prediction Submission
  if (predictionForm) {
    predictionForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      formError.textContent = '';

      const inputs = getFormData();
      const payload = {
        model: window.selectedModel,
        ...inputs,
      };

      if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.innerHTML = `<span>Evaluating with ${escapeHtml(window.selectedModel)}…</span>`;
      }

      try {
        const startTime = performance.now();
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify(payload),
          cache: 'no-store',
        });
        const latency = Math.round(performance.now() - startTime);
        const latTag = document.getElementById('latency-tag');
        if (latTag) latTag.textContent = `${latency} ms`;

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Prediction calculation failed.');
        }

        renderBiometricResult(data);

        // Refresh persistent history table
        if (window.loadHistoryTable) {
          window.loadHistoryTable();
        }
      } catch (err) {
        formError.textContent = err.message || 'Error executing assessment.';
      } finally {
        if (predictBtn) {
          predictBtn.disabled = false;
          predictBtn.innerHTML = `<span>Evaluate Patient Risk</span><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>`;
        }
      }
    });
  }

  // Export PDF Report Action
  if (btnExportPdf) {
    btnExportPdf.addEventListener('click', async () => {
      const inputs = getFormData();
      const payload = lastAssessmentResult || {
        model: window.selectedModel,
        prediction: 0,
        probability: 0.25,
        risk_level: 'Low',
        inputs,
      };

      try {
        btnExportPdf.disabled = true;
        btnExportPdf.querySelector('span').textContent = 'Generating PDF…';

        const res = await fetch('/api/export/pdf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error('PDF export failed.');

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `smart-healthcare-report-${window.selectedModel.toLowerCase().replace(/\s+/g, '-')}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } catch (err) {
        alert(`Export failed: ${err.message}`);
      } finally {
        btnExportPdf.disabled = false;
        btnExportPdf.querySelector('span').textContent = 'Export Clinical Dossier (PDF)';
      }
    });
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // Initialize
  initFormFields();
})();
