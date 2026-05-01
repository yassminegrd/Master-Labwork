/* ─── Dysgraphia Detection System — Frontend Script ─── */

'use strict';

// ── State
let selectedFile = null;
let metricsLoaded = false;

// ── DOM refs (lazy)
const $ = id => document.getElementById(id);

// ─────────────────────────────────────────────
// Page Switching
// ─────────────────────────────────────────────
function switchPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

  const page = document.getElementById(`page-${name}`);
  if (page) page.classList.add('active');

  const link = document.querySelector(`[data-page="${name}"]`);
  if (link) link.classList.add('active');

  if (name === 'evaluate' && !metricsLoaded) {
    loadMetrics();
  }
}

// ─────────────────────────────────────────────
// File Upload — Drop Zone
// ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const dropZone = $('dropZone');
  const fileInput = $('fileInput');

  if (!dropZone || !fileInput) return;

  // Click to open file dialog
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });

  // File input change
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  // Drag events
  ['dragenter', 'dragover'].forEach(evt =>
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    })
  );
  dropZone.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
});

function handleFile(file) {
  const allowed = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp',
                   'image/gif', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showToast('Unsupported file type. Please upload PNG, JPG, or BMP.', 'error');
    return;
  }
  selectedFile = file;
  showPreview(file);
  resetResult();
  $('predictBtn').disabled = false;
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const img = $('previewImg');
    img.src = e.target.result;
    img.onload = () => {
      $('previewInfo').textContent =
        `${file.name} · ${img.naturalWidth}×${img.naturalHeight}px · ${formatBytes(file.size)}`;
    };
    $('previewSection').classList.remove('hidden');
    $('dropZone').classList.add('hidden');
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  selectedFile = null;
  $('fileInput').value = '';
  $('previewSection').classList.add('hidden');
  $('dropZone').classList.remove('hidden');
  $('predictBtn').disabled = true;
  resetResult();
}

// ─────────────────────────────────────────────
// Prediction
// ─────────────────────────────────────────────
async function runPrediction() {
  if (!selectedFile) return;

  const btn = $('predictBtn');
  const btnText = $('btnText');
  const btnSpinner = $('btnSpinner');

  btn.disabled = true;
  btnText.textContent = 'Analysing…';
  btnSpinner.classList.remove('hidden');

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      showError(data.error);
      return;
    }

    displayResult(data);
  } catch (err) {
    showError('Network error. Please try again.');
    console.error(err);
  } finally {
    btn.disabled = false;
    btnText.textContent = 'Analyse Handwriting';
    btnSpinner.classList.add('hidden');
  }
}

function displayResult(data) {
  const isDys = data.prediction === 1;
  const section = $('resultSection');
  const content = $('resultContent');
  const placeholder = $('resultPlaceholder');

  // Show section
  section.classList.remove('hidden');
  placeholder.classList.add('hidden');
  content.classList.remove('hidden');

  // Banner
  const banner = $('resultBanner');
  banner.className = `result-banner ${isDys ? 'dysgraphia' : 'normal'}`;

  const icon = $('resultIcon');
  icon.className = `result-icon ${isDys ? 'dysgraphia' : 'normal'}`;
  icon.textContent = isDys ? '⚠' : '✓';

  $('resultLabel').textContent = data.label;
  $('resultLabel').style.color = isDys ? 'var(--red)' : 'var(--green)';
  $('resultConfidence').textContent = `Confidence: ${data.confidence.toFixed(1)}%`;

  // Probability bars (animated)
  setTimeout(() => {
    const pn = data.probabilities.normal;
    const pd = data.probabilities.dysgraphia;
    $('probNormal').style.width = `${pn}%`;
    $('probDys').style.width    = `${pd}%`;
    $('probNormalVal').textContent = `${pn.toFixed(1)}%`;
    $('probDysVal').textContent    = `${pd.toFixed(1)}%`;
  }, 80);

  // Model badge
  $('modelBadge').innerHTML =
    `Model used: <strong>${escHtml(data.model_used)}</strong>`;

  // Features grid
  const grid = $('featureGrid');
  grid.innerHTML = '';
  (data.features || []).forEach(f => {
    const el = document.createElement('div');
    el.className = 'feature-item';
    el.innerHTML = `
      <div class="feat-name">${escHtml(f.name)}</div>
      <div class="feat-val">${Number(f.value).toFixed(4)}</div>
    `;
    grid.appendChild(el);
  });

  // Alert
  const alert = $('resultAlert');
  if (isDys) {
    alert.className = 'alert show danger';
    alert.textContent = '⚠ Potential signs of dysgraphia detected. This is an academic tool — consult a specialist for clinical assessment.';
  } else {
    alert.className = 'alert show success';
    alert.textContent = '✓ No significant signs of dysgraphia detected in this sample.';
  }

  // Scroll into view
  section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(msg) {
  const section = $('resultSection');
  section.classList.remove('hidden');
  $('resultPlaceholder').classList.add('hidden');
  $('resultContent').classList.remove('hidden');

  const banner = $('resultBanner');
  banner.className = 'result-banner dysgraphia';
  $('resultIcon').className = 'result-icon dysgraphia';
  $('resultIcon').textContent = '✕';
  $('resultLabel').textContent = 'Analysis Failed';
  $('resultLabel').style.color = 'var(--red)';
  $('resultConfidence').textContent = msg;

  $('probNormal').style.width = '0%';
  $('probDys').style.width    = '0%';
  $('probNormalVal').textContent = '—';
  $('probDysVal').textContent    = '—';
  $('featureGrid').innerHTML = '';
  $('modelBadge').innerHTML = '';

  const alert = $('resultAlert');
  alert.className = 'alert show danger';
  alert.textContent = `Error: ${msg}`;
}

function resetResult() {
  $('resultSection').classList.add('hidden');
  $('resultContent').classList.add('hidden');
  $('resultPlaceholder').classList.remove('hidden');
}

// ─────────────────────────────────────────────
// Evaluation Metrics
// ─────────────────────────────────────────────
async function loadMetrics() {
  $('metricsLoading').style.display = 'flex';
  $('metricsContent').classList.add('hidden');

  try {
    const res = await fetch('/api/metrics');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderMetrics(data);
    metricsLoaded = true;
  } catch (err) {
    $('metricsLoading').innerHTML =
      `<p style="color:var(--red)">Failed to load metrics: ${escHtml(err.message)}</p>`;
  }
}

function renderMetrics(data) {
  const models = data.models;
  const best = data.best_model;

  // Summary cards
  const bestModel = models.find(m => m.name === best) || models[0];
  const summary = [
    { label: 'Best Model', value: best, sub: 'Highest F1-Score', cls: 'best' },
    { label: 'Best Accuracy', value: pct(bestModel.accuracy), sub: `${best}`, cls: 'green-bar' },
    { label: 'Best F1-Score', value: pct(bestModel.f1_score), sub: `${best}`, cls: 'blue-bar' },
    { label: 'Best Precision', value: pct(bestModel.precision), sub: `${best}`, cls: 'amber-bar' },
  ];
  $('metricsSummary').innerHTML = summary.map(s => `
    <div class="summary-card ${s.cls}">
      <div class="summary-label">${s.label}</div>
      <div class="summary-value">${s.value}</div>
      <div class="summary-sub">${s.sub}</div>
    </div>
  `).join('');

  // Model cards
  $('modelsGrid').innerHTML = models.map(m => `
    <div class="model-card ${m.name === best ? 'best-model' : ''}">
      <div class="model-card-header">
        <div class="model-name">${escHtml(m.name)}</div>
        ${m.name === best ? '<span class="best-badge">Best</span>' : ''}
      </div>
      <div class="metric-rows">
        ${['accuracy', 'precision', 'recall', 'f1_score'].map(k => `
          <div class="metric-row">
            <div class="metric-label-row">
              <span class="metric-name">${metricLabel(k)}</span>
              <span class="metric-val">${pct(m[k])}</span>
            </div>
            <div class="metric-track">
              <div class="metric-fill" style="width:${(m[k]*100).toFixed(1)}%"></div>
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');

  // Comparison table
  $('tableBody').innerHTML = models.map(m => `
    <tr class="${m.name === best ? 'best-row' : ''}">
      <td>${escHtml(m.name)}</td>
      <td>${pct(m.accuracy)}</td>
      <td>${pct(m.precision)}</td>
      <td>${pct(m.recall)}</td>
      <td>${pct(m.f1_score)}</td>
      <td>
        <span class="status-badge ${m.name === best ? 'best' : 'other'}">
          ${m.name === best ? '★ Best' : 'Trained'}
        </span>
      </td>
    </tr>
  `).join('');

  // Confusion matrices
  $('cmGrid').innerHTML = models.map(m => {
    const cm = m.confusion_matrix;
    const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
    return `
      <div class="cm-card">
        <div class="cm-title">${escHtml(m.name)}</div>
        <table class="cm-table">
          <thead>
            <tr>
              <th></th>
              <th>Pred. Normal</th>
              <th>Pred. Dys.</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="row-label">Act. Normal</td>
              <td class="tn">${tn}</td>
              <td class="fp">${fp}</td>
            </tr>
            <tr>
              <td class="row-label">Act. Dys.</td>
              <td class="fn">${fn}</td>
              <td class="tp">${tp}</td>
            </tr>
          </tbody>
        </table>
      </div>
    `;
  }).join('');

  // Dataset info
  $('datasetCard').innerHTML = `
    <div class="ds-item">
      <span class="ds-label">Total Samples</span>
      <span class="ds-value">${data.dataset_size}</span>
    </div>
    <div class="ds-item">
      <span class="ds-label">Training Set</span>
      <span class="ds-value">${data.train_size}</span>
    </div>
    <div class="ds-item">
      <span class="ds-label">Test Set</span>
      <span class="ds-value">${data.test_size}</span>
    </div>
    <div class="ds-item">
      <span class="ds-label">Split Ratio</span>
      <span class="ds-value">80/20</span>
    </div>
    <div class="ds-item">
      <span class="ds-label">Features</span>
      <span class="ds-value">${(data.feature_names || []).length}</span>
    </div>
  `;

  $('metricsLoading').style.display = 'none';
  $('metricsContent').classList.remove('hidden');
}

// ─────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────
function pct(val) {
  return (val * 100).toFixed(2) + '%';
}

function metricLabel(key) {
  return { accuracy: 'Accuracy', precision: 'Precision',
           recall: 'Recall', f1_score: 'F1-Score' }[key] || key;
}

function escHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function showToast(msg, type) {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.style.cssText = `
    position:fixed; bottom:24px; right:24px;
    background:${type === 'error' ? 'var(--surface)' : 'var(--surface)'};
    border:1px solid ${type === 'error' ? 'var(--red)' : 'var(--green)'};
    color:${type === 'error' ? 'var(--red)' : 'var(--green)'};
    padding:12px 18px; border-radius:10px; font-size:0.88rem; font-weight:500;
    box-shadow:var(--shadow); z-index:999; animation:fadeIn 0.2s ease;
    max-width:320px;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}
