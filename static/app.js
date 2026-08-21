// Prediction workflow with real model switching.
(function () {
  const form = document.getElementById('prediction-form');
  const fields = document.getElementById('fields');
  const result = document.getElementById('result');
  const error = document.getElementById('form-error');
  if (!form || !fields || !result) return;

  const definitions = [
    ['Pregnancies', 0, 17, 1], ['Glucose', 50, 250, 1], ['BloodPressure', 30, 150, 1],
    ['SkinThickness', 0, 100, 1], ['Insulin', 0, 900, 1], ['BMI', 10, 70, 0.1],
    ['DiabetesPedigreeFunction', 0, 3, 0.01], ['Age', 18, 100, 1]
  ];

  const modelWrap = document.createElement('label');
  modelWrap.className = 'model-selector';
  modelWrap.innerHTML = '<span>Prediction Model</span><select id="model-select" name="model" aria-label="Prediction model"><option>Logistic Regression</option><option>Decision Tree</option><option>Random Forest</option></select><small id="model-help">Choose which trained model will generate this prediction.</small>';
  form.insertBefore(modelWrap, fields);

  fields.innerHTML = definitions.map(([name, min, max, step]) =>
    `<label>${name}<input required name="${name}" type="number" min="${min}" max="${max}" step="${step}"></label>`
  ).join('');

  const modelSelect = document.getElementById('model-select');
  modelSelect.addEventListener('change', () => {
    result.className = 'result';
    result.innerHTML = `<strong>${modelSelect.value}</strong><span>Ready to run a prediction with this model.</span>`;
    error.textContent = '';
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    error.textContent = '';
    const payload = {};
    new FormData(form).forEach((value, key) => { payload[key] = key === 'model' ? value : Number(value); });
    result.className = 'result loading'; result.innerHTML = `<strong>Analyzing…</strong><span>${modelSelect.value}</span>`;
    try {
      const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Prediction failed.');
      const probability = Number(data.probability ?? data.risk_probability ?? 0);
      const positive = data.prediction === 1 || data.prediction === '1';
      result.className = `result ${positive ? 'higher' : 'lower'}`;
      result.innerHTML = `<strong>${positive ? 'Higher predicted risk' : 'Lower predicted risk'}</strong><span>${escapeHtml(data.model || modelSelect.value)}</span><span>Model probability: ${(probability * 100).toFixed(1)}%</span>`;
      if (window.setPredictionBaseline) window.setPredictionBaseline(payload);
    } catch (e) {
      result.className = 'result empty'; result.textContent = 'Prediction unavailable.'; error.textContent = e.message || 'Prediction failed.';
    }
  });

  function escapeHtml(v) { return String(v).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
})();
