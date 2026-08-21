const features = [
  ["Pregnancies", 0, 30, 1], ["Glucose", 1, 400, 1], ["BloodPressure", 1, 250, 1],
  ["SkinThickness", 0, 150, 1], ["Insulin", 0, 1000, 1], ["BMI", 1, 100, 0.1],
  ["DiabetesPedigreeFunction", 0, 5, 0.01], ["Age", 1, 120, 1]
];
const models = ["Logistic Regression", "Decision Tree", "Random Forest"];
const fields = document.getElementById("fields");
const form = document.getElementById("prediction-form");
const result = document.getElementById("result");
const error = document.getElementById("form-error");
const button = document.getElementById("predict-button");

if (fields && form && result) {
  const modelWrap = document.createElement("div");
  modelWrap.className = "model-selector";
  modelWrap.innerHTML = `<span>Prediction Model</span><div class="model-options" role="radiogroup" aria-label="Prediction model">${models.map((name, index) => `<button type="button" class="model-option${index === 0 ? " active" : ""}" data-model="${escapeHtml(name)}" role="radio" aria-checked="${index === 0 ? "true" : "false"}">${escapeHtml(name)}</button>`).join("")}</div><select id="model-select" name="model" class="model-select-native" aria-label="Prediction model">${models.map((name, index) => `<option value="${escapeHtml(name)}"${index === 0 ? " selected" : ""}>${escapeHtml(name)}</option>`).join("")}</select><small>Choose the trained model used for this prediction.</small>`;
  form.insertBefore(modelWrap, fields);

  fields.innerHTML = features.map(([name, min, max, step]) =>
    `<label><span>${name}</span><input name="${name}" type="number" min="${min}" max="${max}" step="${step}" required inputmode="decimal"></label>`
  ).join("");

  const modelSelect = document.getElementById("model-select");
  const modelButtons = [...modelWrap.querySelectorAll(".model-option")];

  function setModel(model) {
    if (!models.includes(model)) return;
    modelSelect.value = model;
    modelButtons.forEach((btn) => {
      const active = btn.dataset.model === model;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-checked", active ? "true" : "false");
    });
    result.className = "result";
    result.innerHTML = `<strong>${escapeHtml(model)}</strong><span>Ready to run a prediction with this model.</span>`;
    error.textContent = "";
  }

  modelButtons.forEach((btn) => btn.addEventListener("click", () => setModel(btn.dataset.model)));
  modelSelect.addEventListener("change", () => setModel(modelSelect.value));

  window.getPredictionBaseline = function () {
    const payload = Object.fromEntries(new FormData(form).entries());
    const model = payload.model || modelSelect.value;
    delete payload.model;
    return Object.fromEntries(Object.entries(payload).map(([key, value]) => [key, Number(value)]));
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    error.textContent = "";
    button.disabled = true;
    button.textContent = "Analyzing…";
    result.className = "result loading";
    const selectedModel = modelSelect.value;
    result.textContent = `Evaluating ${selectedModel}…`;

    const numericPayload = window.getPredictionBaseline();
    const payload = { ...numericPayload, model: selectedModel };

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify(payload),
        cache: "no-store"
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction request failed.");
      const probability = Number(data.probability || 0);
      result.className = `result ${data.prediction ? "higher" : "lower"}`;
      result.innerHTML = `<strong>${escapeHtml(data.label || (data.prediction ? "Higher predicted risk" : "Lower predicted risk"))}</strong><span>${escapeHtml(data.model || selectedModel)}</span><span>Model probability: ${(probability * 100).toFixed(1)}%</span><small>Classification threshold: ${(Number(data.threshold || 0.5) * 100).toFixed(0)}%</small>`;
      window.dispatchEvent(new CustomEvent("prediction-ready", { detail: { ...numericPayload, model: data.model || selectedModel } }));
      if (typeof window.setPredictionBaseline === "function") window.setPredictionBaseline({ ...numericPayload, model: data.model || selectedModel });
    } catch (err) {
      result.className = "result empty";
      result.textContent = "Unable to generate a prediction.";
      error.textContent = err.message;
    } finally {
      button.disabled = false;
      button.textContent = "Analyze prediction";
    }
  });
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[char]));
}