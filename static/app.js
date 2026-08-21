// Shared browser helpers and prediction workflow.
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
  fields.innerHTML = definitions.map(([name, min, max, step]) =>
    `<label>${name}<input required name="${name}" type="number" min="${min}" max="${max}" step="${step}"></label>`
  ).join('');

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    error.textContent = '';
    const payload = {};
    new FormData(form).forEach((value, key) => { payload[key] = Number(value); });
    result.className = 'result'; result.textContent = 'Analyzing…';
    try {
      const response = await fetch('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Prediction failed.');
      const probability = Number(data.probability ?? data.risk_probability ?? 0);
      const label = data.prediction === 1 || data.prediction === '1' ? 'Higher predicted risk' : 'Lower predicted risk';
      result.innerHTML = `<strong>${label}</strong><span>Model probability: ${(probability * 100).toFixed(1)}%</span>`;
      if (window.setPredictionBaseline) window.setPredictionBaseline(payload);
    } catch (e) {
      result.className = 'result empty'; result.textContent = 'Prediction unavailable.'; error.textContent = e.message || 'Prediction failed.';
    }
  });
})();
