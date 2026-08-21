// Prediction workflow with explicit model switching.
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
  const models = ['Logistic Regression', 'Decision Tree', 'Random Forest'];

  const modelWrap = document.createElement('div');
  modelWrap.className = 'model-selector';
  modelWrap.innerHTML = `<span>Prediction Model</span><div class="model-options" role="radiogroup" aria-label="Prediction model">${models.map((name, i) => `<button type="button" class="model-option${i === 0 ? ' active' : ''}" data-model="${escapeHtml(name)}" role="radio" aria-checked="${i === 0 ? 'true' : 'false'}">${escapeHtml(name)}</button>`).join('')}</div><select id="model-select" name="model" aria-label="Prediction model" class="model-select-fallback">${models.map((name, i) => `<option value="${escapeHtml(name)}"${i === 0 ? ' selected' : ''}>${escapeHtml(name)}</option>`).join('')}</select><small id="model-help">Selected model is sent directly to the prediction API.</small>`;
  form.insertBefore(modelWrap, fields);

  fields.innerHTML = definitions.map(([name, min, max, step]) =>
    `<label>${name}<input required name="${name}" type="number" min="${min}" max="${max}" step="${step}"></label>`
  ).join('');

  const modelSelect = document.getElementById('model-select');
  const modelButtons = [...modelWrap.querySelectorAll('.model-option')];
  let selectedModel = modelSelect.value;

  function setModel(model) {
    if (!models.includes(model)) return;
    selectedModel = model;
    modelSelect.value = model;
    modelButtons.forEach(btn => {
      const active = btn.dataset.model === model;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-checked', active ? 'true' : 'false');
    });
    result.className = 'result';
    result.innerHTML = `<strong>${escapeHtml(model)}</strong><span>Ready to run a prediction with this model.</span>`;
    error.textContent = '';
  }

  modelButtons.forEach(btn => btn.addEventListener('click', () => setModel(btn.dataset.model)));
  modelSelect.addEventListener('change', () => setModel(modelSelect.value));

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    error.textContent = '';
    const payload = {};
    new FormData(form).forEach((value, key) => { payload[key] = key === 'model' ? value : Number(value); });
    payload.model = selectedModel;
    result.className = 'result loading';
    result.innerHTML = `<strong>Analyzing…</strong><span>${escapeHtml(selectedModel)}</span>`;
    try {
      const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json', 'Accept': 'application/json'}, body: JSON.stringify(payload), cache: 'no-store' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Prediction failed.');
      const probability = Number(data.probability ?? data.risk_probability ?? 0);
      const positive = data.prediction === 1 || data.prediction === '1';
      const actualModel = data.model || selectedModel;
      result.className = `result ${positive ? 'higher' : 'lower'}`;
      result.innerHTML = `<strong>${positive ? 'Higher predicted risk' : 'Lower predicted risk'}</strong><span>${escapeHtml(actualModel)}</span><span>Model probability: ${(probability * 100).toFixed(1)}%</span>`;
      if (window.setPredictionBaseline) window.setPredictionBaseline(payload);
    } catch (e) {
      result.className = 'result empty';
      result.textContent = 'Prediction unavailable.';
      error.textContent = e.message || 'Prediction failed.';
    }
  });

  function escapeHtml(v) { return String(v).replace(/[&<>\"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[c])); }
})();
