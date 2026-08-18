const features = [
  ["Pregnancies", 0, 30, 1], ["Glucose", 1, 400, 1], ["BloodPressure", 1, 250, 1],
  ["SkinThickness", 0, 150, 1], ["Insulin", 0, 1000, 1], ["BMI", 1, 100, 0.1],
  ["DiabetesPedigreeFunction", 0, 5, 0.01], ["Age", 1, 120, 1]
];
const fields = document.getElementById("fields");
const form = document.getElementById("prediction-form");
const result = document.getElementById("result");
const error = document.getElementById("form-error");
const button = document.getElementById("predict-button");

features.forEach(([name, min, max, step]) => {
  const wrapper = document.createElement("label");
  wrapper.innerHTML = `<span>${name}</span><input name="${name}" type="number" min="${min}" max="${max}" step="${step}" required inputmode="decimal">`;
  fields.appendChild(wrapper);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.textContent = "";
  button.disabled = true;
  button.textContent = "Analyzing…";
  result.className = "result loading";
  result.textContent = "Validating inputs and evaluating the model…";

  const payload = Object.fromEntries(new FormData(form).entries());
  Object.keys(payload).forEach((key) => { payload[key] = Number(payload[key]); });

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction request failed.");
    result.className = `result ${data.prediction ? "higher" : "lower"}`;
    result.innerHTML = `<strong>${data.label}</strong><span>Model probability: ${(data.probability * 100).toFixed(1)}%</span><small>Classification threshold: ${(data.threshold * 100).toFixed(0)}%</small>`;
  } catch (err) {
    result.className = "result empty";
    result.textContent = "Unable to generate a prediction.";
    error.textContent = err.message;
  } finally {
    button.disabled = false;
    button.textContent = "Analyze prediction";
  }
});
