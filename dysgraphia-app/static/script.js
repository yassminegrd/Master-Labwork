/* ================================================================
   script.js — Frontend JavaScript
   DysDetect | APM.02 | Master 1 STIC | 2025-2026
   Dr. NECIBI Khaled | Université Constantine 2
   ================================================================ */

'use strict';

// ── State
let selectedFile = null;
let selectedModel = 'cnn';    // 'cnn' or 'ml'
let metricsLoaded = false;
let dashLoaded = false;
let barChart = null;
let histChart = null;

// ── DOM helper
const $ = id => document.getElementById(id);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PAGE NAVIGATION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function switchPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  const page = $(`page-${name}`);
  if (page) page.classList.add('active');
  const link = document.querySelector(`[data-page="${name}"]`);
  if (link) link.classList.add('active');

  // Lazy-load data
  if (name === 'evaluate' && !metricsLoaded) loadMetrics();
  if (name === 'dashboard' && !dashLoaded)   loadDashboard();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MODEL SELECTOR
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function selectModel(model) {
  selectedModel = model;
  document.querySelectorAll('.pill').forEach(p => {
    p.classList.toggle('active', p.dataset.model === model);
  });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FILE UPLOAD + DRAG & DROP
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
document.addEventListener('DOMContentLoaded', () => {
  const dropZone = $('dropZone');
  const fileInput = $('fileInput');
  if (!dropZone) return;

  // Click → open file dialog
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') fileInput.click();
  });

  // File selected via dialog
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  // Drag events
  ['dragenter', 'dragover'].forEach(evt =>
    dropZone.addEventListener(evt, e => {
      e.preventDefault(); dropZone.classList.add('drag-over');
    })
  );
  ['dragleave', 'drop'].forEach(evt =>
    dropZone.addEventListener(evt, e => {
      e.preventDefault(); dropZone.classList.remove('drag-over');
    })
  );
  dropZone.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
});

function handleFile(file) {
  const allowed = ['image/png', 'image/jpeg', 'image/jpg',
                   'image/bmp', 'image/gif', 'image/webp'];
  if (!allowed.includes(file.type)) {
    showToast('Type de fichier non supporté. Utilisez PNG, JPG ou BMP.', 'error');
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
        `${file.name} · ${img.naturalWidth}×${img.naturalHeight}px · ${fmtBytes(file.size)}`;
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PREDICTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function runPrediction() {
  if (!selectedFile) return;

  const btn       = $('predictBtn');
  const btnText   = $('btnText');
  const spinner   = $('btnSpinner');

  btn.disabled = true;
  btnText.textContent = 'Analyse en cours…';
  spinner.classList.remove('hidden');

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('model', selectedModel);   // send chosen model to Flask

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) { showError(data.error); return; }
    displayResult(data);
  } catch (err) {
    showError('Erreur réseau. Veuillez réessayer.');
    console.error(err);
  } finally {
    btn.disabled = false;
    btnText.textContent = "Analyser l'écriture";
    spinner.classList.add('hidden');
  }
}

function displayResult(data) {
  const isDys = data.prediction === 1;

  // Show section
  $('resultSection').classList.remove('hidden');
  $('resultPlaceholder').classList.add('hidden');
  $('resultContent').classList.remove('hidden');

  // Banner
  const banner = $('resultBanner');
  banner.className = `result-banner ${isDys ? 'dysgraphia' : 'normal'}`;
  const icon = $('resultIcon');
  icon.className = `result-icon ${isDys ? 'dysgraphia' : 'normal'}`;
  icon.textContent = isDys ? '⚠' : '✓';
  $('resultLabel').textContent = data.label;
  $('resultLabel').style.color = isDys ? 'var(--red)' : 'var(--green)';
  $('resultConfidence').textContent = `Confiance : ${data.confidence.toFixed(1)}%`;
  $('resultModelTag').textContent   = `Modèle : ${data.model_used}`;

  // Probability bars (animated after tiny delay)
  setTimeout(() => {
    $('probNormal').style.width = `${data.probabilities.normal}%`;
    $('probDys').style.width    = `${data.probabilities.dysgraphia}%`;
    $('probNormalVal').textContent = `${data.probabilities.normal.toFixed(1)}%`;
    $('probDysVal').textContent    = `${data.probabilities.dysgraphia.toFixed(1)}%`;
  }, 60);

  // Feature grid
  const grid = $('featureGrid');
  grid.innerHTML = '';
  (data.features || []).forEach(f => {
    const el = document.createElement('div');
    el.className = 'feature-item';
    el.innerHTML = `<div class="feat-name">${esc(f.name)}</div>
                    <div class="feat-val">${Number(f.value).toFixed(4)}</div>`;
    grid.appendChild(el);
  });

  // Alert message
  const alert = $('resultAlert');
  if (isDys) {
    alert.className = 'alert show danger';
    alert.textContent = '⚠ Signes potentiels de dysgraphie détectés. '
      + "Ceci est un outil académique — consultez un spécialiste pour un diagnostic clinique.";
  } else {
    alert.className = 'alert show success';
    alert.textContent = '✓ Aucun signe significatif de dysgraphie détecté dans cet échantillon.';
  }

  $('resultSection').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(msg) {
  $('resultSection').classList.remove('hidden');
  $('resultPlaceholder').classList.add('hidden');
  $('resultContent').classList.remove('hidden');

  $('resultBanner').className = 'result-banner dysgraphia';
  $('resultIcon').className   = 'result-icon dysgraphia';
  $('resultIcon').textContent = '✕';
  $('resultLabel').textContent = 'Analyse échouée';
  $('resultLabel').style.color = 'var(--red)';
  $('resultConfidence').textContent = msg;
  $('resultModelTag').textContent   = '';
  $('probNormal').style.width = '0%';
  $('probDys').style.width    = '0%';
  $('probNormalVal').textContent = '—';
  $('probDysVal').textContent    = '—';
  $('featureGrid').innerHTML = '';

  const alert = $('resultAlert');
  alert.className = 'alert show danger';
  alert.textContent = `Erreur : ${msg}`;
}

function resetResult() {
  $('resultSection').classList.add('hidden');
  $('resultContent').classList.add('hidden');
  $('resultPlaceholder').classList.remove('hidden');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DASHBOARD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function loadDashboard() {
  $('dashLoading').style.display = 'flex';
  $('dashContent').classList.add('hidden');

  try {
    const res  = await fetch('/api/dashboard');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderDashboard(data);
    dashLoaded = true;
  } catch (err) {
    $('dashLoadingMsg').textContent = `Erreur : ${err.message}`;
  }
}

function renderDashboard(data) {
  const models = data.models;
  const best   = data.best_model;
  const ds     = data.dataset || {};

  // ── Summary cards
  const bestM = models.find(m => m.name === best) || models[0];
  const summaryItems = [
    { label: 'Meilleur Modèle',  value: best,              sub: 'F1-Score le plus élevé', cls: 'c-accent' },
    { label: 'Meilleure Acc.',   value: pct(bestM.accuracy), sub: best,                   cls: 'c-green' },
    { label: 'Meilleur F1',      value: pct(bestM.f1_score), sub: best,                   cls: 'c-blue' },
    { label: 'Modèles Testés',   value: models.length,     sub: 'CNN + ML classiques',    cls: 'c-amber' },
    { label: 'Augmentation',     value: `×${ds.augmentation || '—'}`, sub: 'Facteur d\'augm.', cls: 'c-cyan' },
  ];
  $('summaryRow').innerHTML = summaryItems.map(s => `
    <div class="summary-card ${s.cls}">
      <div class="sc-label">${s.label}</div>
      <div class="sc-value">${s.value}</div>
      <div class="sc-sub">${s.sub}</div>
    </div>
  `).join('');

  // ── Bar chart: all models vs all metrics
  const labels  = models.map(m => m.name);
  const accData = models.map(m => (m.accuracy  * 100).toFixed(1));
  const precData= models.map(m => (m.precision * 100).toFixed(1));
  const recData = models.map(m => (m.recall    * 100).toFixed(1));
  const f1Data  = models.map(m => (m.f1_score  * 100).toFixed(1));

  if (barChart) barChart.destroy();
  const bCtx = document.getElementById('barChart').getContext('2d');
  barChart = new Chart(bCtx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Accuracy',   data: accData,  backgroundColor: 'rgba(99,102,241,0.75)'  },
        { label: 'Précision',  data: precData, backgroundColor: 'rgba(168,85,247,0.75)'  },
        { label: 'Rappel',     data: recData,  backgroundColor: 'rgba(6,182,212,0.75)'   },
        { label: 'F1-Score',   data: f1Data,   backgroundColor: 'rgba(34,197,94,0.75)'   },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: '#8892aa', font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: '#8892aa' }, grid: { color: '#1e2540' } },
        y: { min: 0, max: 100, ticks: { color: '#8892aa', callback: v => v + '%' }, grid: { color: '#1e2540' } },
      },
    },
  });

  // ── Training history chart (CNN only)
  if (data.cnn_history && data.cnn_available) {
    $('historyCard').style.display = '';
    const h = data.cnn_history;
    const epochs = h.accuracy.map((_, i) => i + 1);
    if (histChart) histChart.destroy();
    const hCtx = document.getElementById('historyChart').getContext('2d');
    histChart = new Chart(hCtx, {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          { label: 'Train Acc',   data: h.accuracy.map(v => (v*100).toFixed(2)),     borderColor: '#6366f1', tension: 0.3, fill: false, pointRadius: 2 },
          { label: 'Val Acc',     data: h.val_accuracy.map(v => (v*100).toFixed(2)), borderColor: '#22c55e', tension: 0.3, fill: false, pointRadius: 2 },
          { label: 'Train Loss',  data: h.loss.map(v => v.toFixed(4)),               borderColor: '#ef4444', tension: 0.3, fill: false, pointRadius: 2, yAxisID: 'y2' },
          { label: 'Val Loss',    data: h.val_loss.map(v => v.toFixed(4)),            borderColor: '#f59e0b', tension: 0.3, fill: false, pointRadius: 2, yAxisID: 'y2' },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#8892aa', font: { size: 10 } } } },
        scales: {
          x: { ticks: { color: '#8892aa' }, grid: { color: '#1e2540' } },
          y:  { position: 'left',  ticks: { color: '#8892aa', callback: v => v + '%' }, grid: { color: '#1e2540' }, title: { display: true, text: 'Accuracy (%)', color: '#5a6480' } },
          y2: { position: 'right', ticks: { color: '#8892aa' }, grid: { drawOnChartArea: false }, title: { display: true, text: 'Loss', color: '#5a6480' } },
        },
      },
    });
  } else {
    $('historyCard').innerHTML = `
      <div class="card-header"><h3>Historique CNN</h3></div>
      <div style="padding:24px 24px 28px;text-align:center;color:var(--text-3);font-size:0.86rem">
        CNN non encore entraîné.<br/>
        <button class="btn btn-accent btn-sm" style="margin-top:12px" onclick="retrainCNN()">
          Entraîner le CNN maintenant
        </button>
      </div>`;
  }

  // ── Comparison table
  $('dashTableBody').innerHTML = models.map(m => {
    const isBest = m.name === best;
    const type   = m.name === 'CNN' ? 'dl' : 'ml';
    const typeLabel = m.name === 'CNN' ? 'Deep Learning' : 'ML Classique';
    return `
      <tr class="${isBest ? 'best-row' : ''}">
        <td>${esc(m.name)}</td>
        <td><span class="type-badge ${type}">${typeLabel}</span></td>
        <td>${pct(m.accuracy)}</td>
        <td>${pct(m.precision)}</td>
        <td>${pct(m.recall)}</td>
        <td>${pct(m.f1_score)}</td>
        <td><span class="status-badge ${isBest ? 'best' : 'other'}">${isBest ? '★ Meilleur' : 'Entraîné'}</span></td>
      </tr>`;
  }).join('');

  // ── Confusion matrices
  $('dashCmGrid').innerHTML = models.map(m => buildCmHtml(m)).join('');

  // ── Dataset info
  if (Object.keys(ds).length) {
    $('datasetInfo').innerHTML = `
      <div class="ds-item"><span class="ds-label">Total Samples</span><span class="ds-value">${ds.size || '—'}</span></div>
      <div class="ds-item"><span class="ds-label">Train</span><span class="ds-value">${ds.train || '—'}</span></div>
      <div class="ds-item"><span class="ds-label">Test</span><span class="ds-value">${ds.test || '—'}</span></div>
      <div class="ds-item"><span class="ds-label">Features</span><span class="ds-value">${ds.features || 12}</span></div>
      <div class="ds-item"><span class="ds-label">Augmentation</span><span class="ds-value">×${ds.augmentation || '—'}</span></div>
      <div class="ds-item"><span class="ds-label">Split</span><span class="ds-value">80/20</span></div>`;
  }

  $('dashLoading').style.display = 'none';
  $('dashContent').classList.remove('hidden');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// EVALUATE PAGE (ML only)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function loadMetrics() {
  $('metricsLoading').style.display = 'flex';
  $('metricsContent').classList.add('hidden');
  try {
    const res  = await fetch('/api/metrics');
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderMetrics(data);
    metricsLoaded = true;
  } catch (err) {
    $('metricsLoading').innerHTML =
      `<p style="color:var(--red)">Erreur : ${esc(err.message)}</p>`;
  }
}

function renderMetrics(data) {
  const models = data.models;
  const best   = data.best_model;

  // Summary cards
  const bestM = models.find(m => m.name === best) || models[0];
  $('metricsSummary').innerHTML = [
    { label: 'Meilleur Modèle',   value: best,               sub: 'F1-Score le plus élevé', cls: 'c-accent' },
    { label: 'Meilleure Accuracy', value: pct(bestM.accuracy), sub: best,                    cls: 'c-green' },
    { label: 'Meilleur F1-Score',  value: pct(bestM.f1_score), sub: best,                    cls: 'c-blue' },
    { label: 'Taille Dataset',     value: data.dataset_size,  sub: 'après augmentation',     cls: 'c-amber' },
  ].map(s => `
    <div class="summary-card ${s.cls}">
      <div class="sc-label">${s.label}</div>
      <div class="sc-value">${s.value}</div>
      <div class="sc-sub">${s.sub}</div>
    </div>`).join('');

  // Model cards
  $('modelsGrid').innerHTML = models.map(m => `
    <div class="model-card ${m.name === best ? 'best-model' : ''}">
      <div class="model-card-header">
        <div class="model-name">${esc(m.name)}</div>
        ${m.name === best ? '<span class="best-badge">Meilleur</span>' : ''}
      </div>
      <div class="metric-rows">
        ${['accuracy','precision','recall','f1_score'].map(k => `
          <div class="metric-row">
            <div class="metric-label-row">
              <span class="metric-name">${mlabel(k)}</span>
              <span class="metric-val">${pct(m[k])}</span>
            </div>
            <div class="metric-track">
              <div class="metric-fill" style="width:${(m[k]*100).toFixed(1)}%"></div>
            </div>
          </div>`).join('')}
      </div>
    </div>`).join('');

  // Table
  $('tableBody').innerHTML = models.map(m => `
    <tr class="${m.name === best ? 'best-row' : ''}">
      <td>${esc(m.name)}</td>
      <td>${pct(m.accuracy)}</td>
      <td>${pct(m.precision)}</td>
      <td>${pct(m.recall)}</td>
      <td>${pct(m.f1_score)}</td>
      <td><span class="status-badge ${m.name === best ? 'best' : 'other'}">
        ${m.name === best ? '★ Meilleur' : 'Entraîné'}
      </span></td>
    </tr>`).join('');

  // Confusion matrices
  $('cmGrid').innerHTML = models.map(m => buildCmHtml(m)).join('');

  // Dataset info
  $('datasetCard').innerHTML = `
    <div class="ds-item"><span class="ds-label">Total Samples</span><span class="ds-value">${data.dataset_size}</span></div>
    <div class="ds-item"><span class="ds-label">Train</span><span class="ds-value">${data.train_size}</span></div>
    <div class="ds-item"><span class="ds-label">Test</span><span class="ds-value">${data.test_size}</span></div>
    <div class="ds-item"><span class="ds-label">Features</span><span class="ds-value">${(data.feature_names||[]).length}</span></div>
    <div class="ds-item"><span class="ds-label">Augmentation</span><span class="ds-value">×${data.augmentation_factor||1}</span></div>`;

  $('metricsLoading').style.display = 'none';
  $('metricsContent').classList.remove('hidden');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RETRAIN CNN
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function retrainCNN() {
  const btn = $('retrainCNNBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Entraînement…'; }
  showToast('Entraînement CNN lancé — cela peut prendre plusieurs minutes.', 'info');
  try {
    const res  = await fetch('/api/retrain-cnn', { method: 'POST' });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    showToast(`CNN entraîné ! Accuracy: ${(data.accuracy*100).toFixed(1)}%  F1: ${(data.f1_score*100).toFixed(1)}%`, 'success');
    dashLoaded = false;
    loadDashboard();
  } catch (err) {
    showToast(`Erreur CNN : ${err.message}`, 'error');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Entraîner CNN'; }
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/** Build confusion matrix HTML card */
function buildCmHtml(m) {
  const cm = m.confusion_matrix;
  if (!cm) return '';
  const tn = cm[0][0], fp = cm[0][1], fn = cm[1][0], tp = cm[1][1];
  return `
    <div class="cm-card">
      <div class="cm-title">${esc(m.name)}</div>
      <table class="cm-table">
        <thead>
          <tr><th></th><th>Prédit Normal</th><th>Prédit Dysgraphie</th></tr>
        </thead>
        <tbody>
          <tr><td class="row-label">Réel Normal</td><td class="tn">${tn}</td><td class="fp">${fp}</td></tr>
          <tr><td class="row-label">Réel Dysgraphie</td><td class="fn">${fn}</td><td class="tp">${tp}</td></tr>
        </tbody>
      </table>
    </div>`;
}

/** Format a [0,1] value as percentage string */
function pct(val) {
  if (val === null || val === undefined) return '—';
  return (Number(val) * 100).toFixed(2) + '%';
}

/** Metric key → human label */
function mlabel(key) {
  return { accuracy:'Accuracy', precision:'Précision',
           recall:'Rappel', f1_score:'F1-Score' }[key] || key;
}

/** Escape HTML special chars */
function esc(str) {
  const d = document.createElement('div');
  d.textContent = String(str);
  return d.innerHTML;
}

/** Format file size */
function fmtBytes(b) {
  if (b < 1024)       return `${b} B`;
  if (b < 1024*1024)  return `${(b/1024).toFixed(1)} KB`;
  return `${(b/1024/1024).toFixed(1)} MB`;
}

/** Toast notification */
function showToast(msg, type = 'info') {
  const existing = document.querySelector('.toast-notif');
  if (existing) existing.remove();
  const colors = { success: 'var(--green)', error: 'var(--red)',
                   info: 'var(--accent-2)' };
  const t = document.createElement('div');
  t.className = 'toast-notif';
  t.style.cssText = `
    position:fixed;bottom:20px;right:20px;z-index:9999;
    background:var(--surface);border:1px solid ${colors[type] || 'var(--border)'};
    color:${colors[type] || 'var(--text)'};
    padding:11px 16px;border-radius:10px;
    font-size:0.84rem;font-weight:500;
    box-shadow:var(--shadow);max-width:320px;
    animation:fadeIn 0.2s ease;
    font-family:var(--font);
  `;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 5000);
}
