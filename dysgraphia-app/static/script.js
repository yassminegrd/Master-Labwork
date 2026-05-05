'use strict';
/* ============================================================
   script.js — DysDetect Frontend
   APM.02 · Master 1 STIC · Dr. NECIBI Khaled · 2025/2026
   ============================================================ */

// ── State
let selFile   = null;
let selModel  = 'cnn';
let dashReady = false;
let evalReady = false;
let chartBar  = null;
let chartHist = null;

const $ = id => document.getElementById(id);

// ──────────────────────────────────────────────────────────────
// NAVIGATION
// ──────────────────────────────────────────────────────────────
function switchPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  const page = $(`page-${name}`);
  if (page) { page.classList.add('active'); page.classList.add('fade-in'); }
  const btn = document.querySelector(`[data-page="${name}"]`);
  if (btn) btn.classList.add('active');
  if (name === 'dashboard' && !dashReady) loadDashboard();
  if (name === 'evaluate'  && !evalReady) loadEvaluation();
}

// ──────────────────────────────────────────────────────────────
// MODEL SELECTOR
// ──────────────────────────────────────────────────────────────
function selectModel(m) {
  selModel = m;
  document.querySelectorAll('.mpill').forEach(p =>
    p.classList.toggle('active', p.dataset.model === m));
}

// ──────────────────────────────────────────────────────────────
// FILE UPLOAD / DRAG & DROP
// ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const zone  = $('dropZone');
  const input = $('fileInput');
  if (!zone) return;

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('keydown', e => { if (e.key === 'Enter') input.click(); });
  input.addEventListener('change', () => { if (input.files[0]) useFile(input.files[0]); });

  ['dragenter','dragover'].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.add('over'); }));
  ['dragleave','drop'].forEach(ev =>
    zone.addEventListener(ev, e => { e.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', e => {
    const f = e.dataTransfer.files[0];
    if (f) useFile(f);
  });
});

function useFile(file) {
  const ok = ['image/png','image/jpeg','image/jpg','image/bmp','image/gif','image/webp'];
  if (!ok.includes(file.type)) { toast('Type non supporté. Utilisez PNG, JPG ou BMP.', 'err'); return; }
  selFile = file;
  showPreview(file);
  clearResult();
  $('analyzeBtn').disabled = false;
}

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const img = $('previewImg');
    img.src = e.target.result;
    img.onload = () => {
      $('previewMeta').textContent =
        `${file.name}  ·  ${img.naturalWidth}×${img.naturalHeight}px  ·  ${fmtSize(file.size)}`;
    };
    $('dropZone').classList.add('hidden');
    $('previewBox').classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  selFile = null;
  $('fileInput').value = '';
  $('previewBox').classList.add('hidden');
  $('dropZone').classList.remove('hidden');
  $('analyzeBtn').disabled = true;
  clearResult();
}

// ──────────────────────────────────────────────────────────────
// PREDICTION
// ──────────────────────────────────────────────────────────────
async function runPrediction() {
  if (!selFile) return;
  setBtnLoading(true);
  const fd = new FormData();
  fd.append('image', selFile);
  fd.append('model', selModel);
  try {
    const res  = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    data.error ? showErr(data.error) : showResult(data);
  } catch(e) {
    showErr('Erreur réseau. Vérifiez que le serveur est actif.');
  } finally {
    setBtnLoading(false);
  }
}

function setBtnLoading(on) {
  $('analyzeBtn').disabled = on;
  $('btnLabel').textContent = on ? 'Analyse en cours…' : "Analyser l'écriture";
  $('btnSpin').classList.toggle('hidden', !on);
}

function showResult(d) {
  const isDys = d.prediction === 1;

  $('resultEmpty').classList.add('hidden');
  $('resultData').classList.remove('hidden');

  // Banner
  const banner = $('verdictBanner');
  banner.className = `verdict-banner ${isDys ? 'is-dysgraphia' : 'is-normal'}`;
  const icon = $('verdictIcon');
  icon.className = `verdict-icon ${isDys ? 'is-dysgraphia' : 'is-normal'}`;
  icon.textContent = isDys ? '⚠' : '✓';
  $('verdictLabel').textContent = d.label;
  $('verdictLabel').style.color = isDys ? 'var(--red)' : 'var(--green)';
  $('verdictConf').textContent  = `Confiance : ${d.confidence.toFixed(1)} %`;
  $('verdictModel').textContent = `Modèle : ${d.model_used}`;

  // Bars (delayed for animation)
  setTimeout(() => {
    $('barNormal').style.width = `${d.probabilities.normal}%`;
    $('barDys').style.width    = `${d.probabilities.dysgraphia}%`;
    $('valNormal').textContent = `${d.probabilities.normal.toFixed(1)} %`;
    $('valDys').textContent    = `${d.probabilities.dysgraphia.toFixed(1)} %`;
  }, 50);

  // Features
  const grid = $('featGrid');
  grid.innerHTML = '';
  (d.features || []).forEach(f => {
    const el = document.createElement('div');
    el.className = 'feat-item';
    el.innerHTML = `<div class="feat-name">${esc(f.name)}</div>
                    <div class="feat-val">${Number(f.value).toFixed(4)}</div>`;
    grid.appendChild(el);
  });

  // Clinic message
  const msg = $('clinicMsg');
  if (isDys) {
    msg.className = 'clinic-msg show warn';
    msg.textContent = '⚠  Signes potentiels de dysgraphie détectés. '
      + "Cet outil est académique — consultez un spécialiste pour un diagnostic clinique.";
  } else {
    msg.className = 'clinic-msg show ok';
    msg.textContent = '✓  Aucun signe significatif de dysgraphie détecté dans cet échantillon.';
  }
}

function showErr(msg) {
  $('resultEmpty').classList.add('hidden');
  $('resultData').classList.remove('hidden');
  $('verdictBanner').className = 'verdict-banner is-dysgraphia';
  $('verdictIcon').className   = 'verdict-icon is-dysgraphia';
  $('verdictIcon').textContent = '✕';
  $('verdictLabel').textContent = 'Erreur d\'analyse';
  $('verdictLabel').style.color = 'var(--red)';
  $('verdictConf').textContent  = msg;
  $('verdictModel').textContent = '';
  $('barNormal').style.width = '0%';
  $('barDys').style.width    = '0%';
  $('valNormal').textContent = '—';
  $('valDys').textContent    = '—';
  $('featGrid').innerHTML    = '';
  $('clinicMsg').className   = 'clinic-msg';
}

function clearResult() {
  $('resultData').classList.add('hidden');
  $('resultEmpty').classList.remove('hidden');
  $('clinicMsg').className = 'clinic-msg';
}

// ──────────────────────────────────────────────────────────────
// DASHBOARD
// ──────────────────────────────────────────────────────────────
async function loadDashboard() {
  $('dashLoad').style.display = 'flex';
  $('dashMain').classList.add('hidden');
  try {
    const r = await fetch('/api/dashboard');
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    renderDashboard(d);
    dashReady = true;
  } catch(e) {
    $('dashLoadMsg').textContent = `Erreur : ${e.message}`;
  }
}

function reloadDash() { dashReady = false; loadDashboard(); }

function renderDashboard(d) {
  const models = d.models;
  const best   = d.best_model;
  const ds     = d.dataset || {};
  const bestM  = models.find(m => m.name === best) || models[0];

  // ── KPIs
  $('kpiRow').innerHTML = [
    { label: 'Meilleur Modèle',  value: best,                sub: 'F1-Score le plus élevé', cls: 'k-purple' },
    { label: 'Accuracy Max',     value: pct(bestM.accuracy), sub: best,                      cls: 'k-green'  },
    { label: 'F1-Score Max',     value: pct(bestM.f1_score), sub: best,                      cls: 'k-blue'   },
    { label: 'Modèles Testés',   value: models.length,       sub: 'CNN + ML classiques',     cls: 'k-amber'  },
    { label: 'Augmentation',     value: `×${ds.augmentation||'—'}`, sub: 'Facteur',          cls: 'k-cyan'   },
  ].map(k => `
    <div class="kpi-card ${k.cls}">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-sub">${k.sub}</div>
    </div>`).join('');

  // ── Bar chart
  const labels   = models.map(m => m.name);
  const colors   = { acc:'rgba(99,102,241,.8)', prec:'rgba(168,85,247,.8)', rec:'rgba(6,182,212,.8)', f1:'rgba(34,197,94,.8)' };
  if (chartBar) chartBar.destroy();
  chartBar = new Chart($('chartBar').getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Accuracy',  data: models.map(m => +(m.accuracy *100).toFixed(2)), backgroundColor: colors.acc },
        { label: 'Précision', data: models.map(m => +(m.precision*100).toFixed(2)), backgroundColor: colors.prec },
        { label: 'Rappel',    data: models.map(m => +(m.recall   *100).toFixed(2)), backgroundColor: colors.rec },
        { label: 'F1-Score',  data: models.map(m => +(m.f1_score *100).toFixed(2)), backgroundColor: colors.f1 },
      ],
    },
    options: chartOpts({ yLabel: '%', min: 0, max: 100 }),
  });

  // ── CNN history chart
  const histCard = $('cnnHistCard');
  if (d.cnn_history) {
    histCard.style.display = '';
    const h = d.cnn_history;
    const epochs = h.accuracy.map((_, i) => i + 1);
    if (chartHist) chartHist.destroy();
    chartHist = new Chart($('chartHist').getContext('2d'), {
      type: 'line',
      data: {
        labels: epochs,
        datasets: [
          { label: 'Train Acc',  data: h.accuracy.map(v => +(v*100).toFixed(2)),    borderColor: '#6366f1', tension: .35, fill: false, pointRadius: 2 },
          { label: 'Val Acc',    data: h.val_accuracy.map(v => +(v*100).toFixed(2)),borderColor: '#22c55e', tension: .35, fill: false, pointRadius: 2 },
          { label: 'Train Loss', data: h.loss.map(v => +v.toFixed(4)),              borderColor: '#ef4444', tension: .35, fill: false, pointRadius: 2, yAxisID: 'y2' },
          { label: 'Val Loss',   data: h.val_loss.map(v => +v.toFixed(4)),           borderColor: '#f59e0b', tension: .35, fill: false, pointRadius: 2, yAxisID: 'y2' },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#8891a8', font: { size: 11 } } } },
        scales: {
          x:  { ticks: { color: '#8891a8' }, grid: { color: '#1e2640' } },
          y:  { position: 'left',  min: 0, max: 100, ticks: { color: '#8891a8', callback: v => v + '%' }, grid: { color: '#1e2640' }, title: { display: true, text: 'Accuracy (%)', color: '#525d74' } },
          y2: { position: 'right', ticks: { color: '#8891a8' }, grid: { drawOnChartArea: false }, title: { display: true, text: 'Loss', color: '#525d74' } },
        },
      },
    });
  } else {
    histCard.innerHTML = `
      <div class="card-head"><h3>Historique CNN</h3><p>Workshop 03 — Plot loss over Epochs</p></div>
      <div style="padding:32px 24px;text-align:center;color:var(--text-3);font-size:.84rem">
        CNN non encore entraîné.<br/>
        <button class="btn-accent" style="margin-top:14px" onclick="trainCNN()">Entraîner le CNN maintenant</button>
      </div>`;
  }

  // ── Comparison table
  $('dashTableBody').innerHTML = models.map(m => {
    const isBest = m.name === best;
    const type   = m.name === 'CNN' ? 'dl' : 'ml';
    return `<tr class="${isBest ? 'best-row' : ''}">
      <td>${esc(m.name)}</td>
      <td><span class="badge-${type}">${m.name==='CNN'?'Deep Learning':'ML Classique'}</span></td>
      <td>${pct(m.accuracy)}</td>
      <td>${pct(m.precision)}</td>
      <td>${pct(m.recall)}</td>
      <td>${pct(m.f1_score)}</td>
      <td>${isBest ? '<span class="badge-best">★ Meilleur</span>' : '<span class="badge-ok">Entraîné</span>'}</td>
    </tr>`;
  }).join('');

  // ── Confusion matrices
  $('dashCmRow').innerHTML = models.map(m => makeCM(m)).join('');

  // ── Dataset info
  $('dashDataInfo').innerHTML = [
    { label: 'Dataset Total', value: ds.size  || '—' },
    { label: 'Train',         value: ds.train || '—' },
    { label: 'Test',          value: ds.test  || '—' },
    { label: 'Features',      value: ds.features || 12 },
    { label: 'Augmentation',  value: `×${ds.augmentation||1}` },
    { label: 'Split',         value: '80 / 20' },
  ].map(x => `<div class="info-item"><span class="info-label">${x.label}</span><span class="info-value">${x.value}</span></div>`).join('');

  $('dashLoad').style.display = 'none';
  $('dashMain').classList.remove('hidden');
}

// ──────────────────────────────────────────────────────────────
// TRAIN CNN
// ──────────────────────────────────────────────────────────────
async function trainCNN() {
  const btn = $('cnnTrainBtn');
  if (btn) { btn.disabled = true; btn.textContent = 'Entraînement…'; }
  toast('Entraînement CNN lancé — cela peut prendre plusieurs minutes.', 'info');
  try {
    const r = await fetch('/api/retrain-cnn', { method: 'POST' });
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    toast(`CNN entraîné ! Accuracy: ${(d.accuracy*100).toFixed(1)}%  F1: ${(d.f1_score*100).toFixed(1)}%`, 'ok');
    dashReady = false;
    loadDashboard();
  } catch(e) {
    toast(`Erreur CNN : ${e.message}`, 'err');
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Entraîner CNN'; }
  }
}

// ──────────────────────────────────────────────────────────────
// EVALUATION PAGE
// ──────────────────────────────────────────────────────────────
async function loadEvaluation() {
  $('evalLoad').style.display = 'flex';
  $('evalMain').classList.add('hidden');
  try {
    const r = await fetch('/api/metrics');
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    renderEvaluation(d);
    evalReady = true;
  } catch(e) {
    $('evalLoad').innerHTML = `<p style="color:var(--red)">Erreur : ${esc(e.message)}</p>`;
  }
}

function reloadEval() { evalReady = false; loadEvaluation(); }

function renderEvaluation(d) {
  const models = d.models;
  const best   = d.best_model;
  const bestM  = models.find(m => m.name === best) || models[0];

  // KPIs
  $('evalKpiRow').innerHTML = [
    { label: 'Meilleur Modèle',    value: best,                 sub: 'F1-Score max',      cls: 'k-purple' },
    { label: 'Meilleure Accuracy', value: pct(bestM.accuracy),  sub: best,                cls: 'k-green'  },
    { label: 'Meilleure Précision',value: pct(bestM.precision), sub: best,                cls: 'k-blue'   },
    { label: 'Dataset (après aug)',value: d.dataset_size,       sub: 'échantillons total',cls: 'k-amber'  },
  ].map(k => `
    <div class="kpi-card ${k.cls}">
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value">${k.value}</div>
      <div class="kpi-sub">${k.sub}</div>
    </div>`).join('');

  // Model cards
  $('modelsRow').innerHTML = models.map(m => `
    <div class="model-card ${m.name === best ? 'is-best' : ''}">
      <div class="mc-header">
        <div class="mc-name">${esc(m.name)}</div>
        ${m.name === best ? '<span class="badge-best">★ Meilleur</span>' : ''}
      </div>
      <div class="mc-metrics">
        ${['accuracy','precision','recall','f1_score'].map(k => `
          <div class="mc-metric">
            <div class="mc-meta">
              <span class="mc-key">${mlabels[k]}</span>
              <span class="mc-val">${pct(m[k])}</span>
            </div>
            <div class="mc-bar">
              <div class="mc-fill" style="width:${(m[k]*100).toFixed(1)}%"></div>
            </div>
          </div>`).join('')}
      </div>
    </div>`).join('');

  // Table
  $('evalTableBody').innerHTML = models.map(m => `
    <tr class="${m.name === best ? 'best-row' : ''}">
      <td>${esc(m.name)}</td>
      <td>${pct(m.accuracy)}</td>
      <td>${pct(m.precision)}</td>
      <td>${pct(m.recall)}</td>
      <td>${pct(m.f1_score)}</td>
      <td>${m.name === best ? '<span class="badge-best">★ Meilleur</span>' : '<span class="badge-ok">Entraîné</span>'}</td>
    </tr>`).join('');

  // Confusion matrices
  $('evalCmRow').innerHTML = models.map(m => makeCM(m)).join('');

  // Classification reports
  $('reportsBlock').innerHTML = models.map(m => `
    <div class="report-block">
      <div class="report-block-title">${esc(m.name)} — Classification Report</div>
      <pre class="report-pre">${esc(m.report || '')}</pre>
    </div>`).join('');

  // Dataset info
  $('evalDataInfo').innerHTML = [
    { label: 'Total Samples',  value: d.dataset_size },
    { label: 'Train (80%)',    value: d.train_size   },
    { label: 'Test  (20%)',    value: d.test_size    },
    { label: 'Features',       value: (d.feature_names||[]).length },
    { label: 'Augmentation',   value: `×${d.augmentation_factor||1}` },
    { label: 'Modèles',        value: models.length  },
  ].map(x => `<div class="info-item"><span class="info-label">${x.label}</span><span class="info-value">${x.value}</span></div>`).join('');

  $('evalLoad').style.display = 'none';
  $('evalMain').classList.remove('hidden');
}

// ──────────────────────────────────────────────────────────────
// HELPERS
// ──────────────────────────────────────────────────────────────

const mlabels = { accuracy:'Accuracy', precision:'Précision', recall:'Rappel', f1_score:'F1-Score' };

function makeCM(m) {
  const cm = m.confusion_matrix;
  if (!cm) return '';
  const [[tn, fp], [fn, tp]] = cm;
  return `
    <div class="cm-card">
      <div class="cm-name">${esc(m.name)}</div>
      <table class="cm-table">
        <thead>
          <tr><th></th><th>Prédit Normal</th><th>Prédit Dys.</th></tr>
        </thead>
        <tbody>
          <tr>
            <td class="row-lbl">Réel Normal</td>
            <td class="cell-tn">${tn}</td>
            <td class="cell-fp">${fp}</td>
          </tr>
          <tr>
            <td class="row-lbl">Réel Dys.</td>
            <td class="cell-fn">${fn}</td>
            <td class="cell-tp">${tp}</td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
        <span style="font-size:.7rem;color:var(--text-3)">
          <span style="color:var(--green);font-weight:700">TP=${tp}</span> ·
          <span style="color:var(--green);font-weight:700">TN=${tn}</span> ·
          <span style="color:var(--red)">FP=${fp}</span> ·
          <span style="color:var(--red)">FN=${fn}</span>
        </span>
      </div>
    </div>`;
}

function chartOpts({ yLabel = '%', min = 0, max = 100 } = {}) {
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#8891a8', font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: '#8891a8' }, grid: { color: '#1e2640' } },
      y: {
        min, max,
        ticks: { color: '#8891a8', callback: v => yLabel === '%' ? v + '%' : v },
        grid:  { color: '#1e2640' },
      },
    },
  };
}

function pct(v) {
  if (v == null) return '—';
  return (Number(v) * 100).toFixed(2) + ' %';
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function fmtSize(b) {
  if (b < 1024)        return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}

function toast(msg, type = 'info') {
  document.querySelectorAll('.toast').forEach(t => t.remove());
  const el = document.createElement('div');
  el.className = `toast t-${type}`;
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 5500);
}
