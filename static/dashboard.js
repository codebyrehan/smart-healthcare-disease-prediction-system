const chartCanvas = document.getElementById('metrics-chart');
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

loadDashboard();
