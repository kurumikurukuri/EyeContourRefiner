/* ── Chart.js global defaults ── */
Chart.defaults.color         = '#888';
Chart.defaults.borderColor   = '#1a2a4a';
Chart.defaults.font.family   = "'Segoe UI', sans-serif";
Chart.defaults.font.size     = 11;

/* ── State ── */
const charts  = {};   // id → Chart instance
const loaded  = new Set();

/* ── Chart helper ── */
function mkBar(id, labels, data, opts = {}) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  charts[id] = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: opts.color || 'rgba(26,120,194,0.75)',
        borderRadius: 2,
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: false },
        title: opts.title
          ? { display: true, text: opts.title, color: '#a8dadc', font: { size: 12 } }
          : { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.formattedValue}${opts.unit || ''}`,
          },
        },
      },
      scales: {
        x: { grid: { color: '#1a2a4a' }, ticks: { maxRotation: 45 } },
        y: { grid: { color: '#1a2a4a' }, beginAtZero: true },
      },
    },
  });
}

function mkGroupedBar(id, ptLabels, datasets) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  charts[id] = new Chart(ctx.getContext('2d'), {
    type: 'bar',
    data: { labels: ptLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: true, position: 'top', labels: { boxWidth: 12, color: '#ccc', padding: 12 } },
      },
      scales: {
        x: { grid: { color: '#1a2a4a' } },
        y: { grid: { color: '#1a2a4a' }, beginAtZero: true },
      },
    },
  });
}

function mkDoughnut(id, labels, data, colors) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  charts[id] = new Chart(ctx.getContext('2d'), {
    type: 'doughnut',
    data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 1, borderColor: '#1a2a4a' }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: { legend: { position: 'right', labels: { boxWidth: 12, padding: 10, color: '#ccc' } } },
    },
  });
}

/* ── API helper ── */
async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
  return res.json();
}

/* ── DOM helpers ── */
function el(id)   { return document.getElementById(id); }
function html(id, h) { el(id).innerHTML = h; }

function statsRowHtml(s) {
  const entries = [
    ['件数',   s.count],
    ['平均',   s.mean],
    ['中央値', s.median],
    ['標準偏差', s.std],
    ['最小',   s.min],
    ['最大',   s.max],
  ];
  return `<div class="stats-row">${entries.map(([k, v]) =>
    `<div class="stats-item"><span class="s-label">${k}</span><span class="s-val">${v}</span></div>`
  ).join('')}</div>`;
}

function spinner() { return '<div class="spinner"></div>'; }

/* ── Tab Switching ── */
function switchTab(tabName) {
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === tabName));
  document.querySelectorAll('.tab-panel').forEach(p =>
    p.classList.toggle('active', p.id === `tab-${tabName}`));

  if (!loaded.has(tabName)) {
    loaded.add(tabName);
    loadTab(tabName);
  }
}

function loadTab(name) {
  switch (name) {
    case 'overview':    return loadOverview();
    case 'tdist':       return loadTDist();
    case 'quality':     return loadQuality();
    case 'experiments': return loadExperiments();
    case 'errors':      return loadErrors();
  }
}

function refreshCurrent() {
  const active = document.querySelector('.tab-btn.active');
  if (!active) return;
  const name = active.dataset.tab;
  loaded.delete(name);
  loaded.add(name);
  loadTab(name);
}

/* ─────────────────────────────────────────────
   OVERVIEW
───────────────────────────────────────────── */
async function loadOverview() {
  html('overview-content', spinner());
  try {
    const d = await api('/api/dashboard/overview');
    if (d.error) { html('overview-content', `<div class="error-msg">${d.error}</div>`); return; }
    renderOverview(d);
  } catch (e) {
    html('overview-content', `<div class="error-msg">読み込みエラー: ${e.message}</div>`);
  }
}

function renderOverview(d) {
  // Stat cards
  const progress = `
    <div class="progress-bar-wrap">
      <div class="progress-bar-fill" style="width:${d.progress_pct}%"></div>
    </div>`;

  const statCards = `
    <div class="card-grid">
      <div class="stat-card">
        <span class="stat-label">総レコード数</span>
        <span class="stat-value">${d.total.toLocaleString()}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">アノテーション済み</span>
        <span class="stat-value">${d.annotated.toLocaleString()}</span>
        <span class="stat-sub">${d.progress_pct}%</span>
        ${progress}
      </div>
      <div class="stat-card">
        <span class="stat-label">未アノテーション</span>
        <span class="stat-value" style="color:${d.unannotated > 0 ? 'var(--orange)' : 'var(--green)'}">${d.unannotated.toLocaleString()}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">ユニーク画像数</span>
        <span class="stat-value">${d.unique_images.toLocaleString()}</span>
      </div>
    </div>`;

  // Charts section
  const chartsHtml = `
    <div class="section-title">分布</div>
    <div class="card-grid-3">
      <div class="card">
        <div class="card-title">Part Name</div>
        <div class="chart-wrap"><canvas id="ov-part"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Sub Key</div>
        <div class="chart-wrap"><canvas id="ov-sub"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Label Source</div>
        <div class="chart-wrap"><canvas id="ov-src"></canvas></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Point Index 分布</div>
      <div class="chart-wrap"><canvas id="ov-pt"></canvas></div>
    </div>`;

  html('overview-content', statCards + chartsHtml);

  // Draw charts
  const partLabels  = Object.keys(d.by_part_name);
  const partData    = Object.values(d.by_part_name);
  mkDoughnut('ov-part', partLabels, partData,
    ['rgba(26,120,194,0.85)', 'rgba(0,230,118,0.85)', 'rgba(255,152,0,0.85)']);

  const subLabels = Object.keys(d.by_sub_key);
  const subData   = Object.values(d.by_sub_key);
  mkDoughnut('ov-sub', subLabels, subData,
    ['rgba(168,218,220,0.85)', 'rgba(156,39,176,0.85)']);

  const srcLabels = Object.keys(d.by_label_source);
  const srcData   = Object.values(d.by_label_source);
  const srcColors = srcLabels.map(l =>
    l === 'manual'       ? 'rgba(27,94,32,0.85)'
    : l === 'unannotated' ? 'rgba(100,100,100,0.6)'
    : l.startsWith('auto_reviewed') ? 'rgba(13,71,161,0.85)'
    : 'rgba(230,81,0,0.85)');
  mkDoughnut('ov-src', srcLabels, srcData, srcColors);

  const ptLabels = Object.keys(d.by_point_index).sort();
  const ptData   = ptLabels.map(k => d.by_point_index[k]);
  mkBar('ov-pt', ptLabels.map(k => `P${k}`), ptData, { color: 'rgba(26,120,194,0.75)' });
}

/* ─────────────────────────────────────────────
   T値分布
───────────────────────────────────────────── */
async function loadTDist() {
  html('tdist-content', spinner());
  try {
    const d = await api('/api/dashboard/t_distribution');
    renderTDist(d);
  } catch (e) {
    html('tdist-content', `<div class="error-msg">読み込みエラー: ${e.message}</div>`);
  }
}

function renderTDist(d) {
  const { overall, by_sub_key, by_part_name, by_sub_pt } = d;

  function histSection(prefix, title, data) {
    if (!data || !data.histogram.bins.length) return '';
    return `
      <div class="card">
        <div class="card-title">${title}</div>
        ${statsRowHtml(data.stats)}
        <div class="chart-wrap"><canvas id="${prefix}-hist"></canvas></div>
      </div>`;
  }

  const subKeys   = Object.keys(by_sub_key);
  const partNames = Object.keys(by_part_name);

  // 14パターン グリッド HTML
  function subPtGrid(sub, colorRgba) {
    const ptMap = (by_sub_pt || {})[sub] || {};
    const pts = Object.keys(ptMap).sort();
    if (!pts.length) return '';
    const cards = pts.map(pt => {
      const data = ptMap[pt];
      const id = `tdist-${sub}-pt${pt}`;
      return `
        <div class="card">
          <div class="card-title" style="font-size:0.82rem">P${pt}
            <span style="color:var(--muted);font-weight:normal;margin-left:6px">
              μ=${data.stats.mean} σ=${data.stats.std}
            </span>
          </div>
          <div class="chart-wrap-sm"><canvas id="${id}"></canvas></div>
        </div>`;
    }).join('');
    return `
      <div class="section-subtitle" style="color:${colorRgba};margin:12px 0 6px">${sub.toUpperCase()}</div>
      <div class="card-grid-7">${cards}</div>`;
  }

  let html_ = `
    <div class="section-title">upper / lower × Point Index 別 t値分布 (14パターン)</div>
    ${subPtGrid('upper', '#90caf9')}
    ${subPtGrid('lower', '#ffcc80')}

    <div class="section-title" style="margin-top:24px">全体 / Sub Key / Part Name</div>
    ${histSection('ov', '全サンプル', overall)}
    <div class="card-grid-2">
      ${subKeys.map(k => histSection(`sk-${k.replace(/\W/g, '_')}`, k, by_sub_key[k])).join('')}
    </div>
    <div class="card-grid-2">
      ${partNames.map(k => histSection(`pn-${k.replace(/\W/g, '_')}`, k, by_part_name[k])).join('')}
    </div>`;

  html('tdist-content', html_);

  // Draw 14-pattern mini histograms (x-axis aligned across all patterns)
  const upperColor = 'rgba(79,195,247,0.8)';
  const lowerColor = 'rgba(255,183,77,0.8)';

  // Collect union of all bins across all 14 patterns
  const allBinSet = new Set();
  ['upper', 'lower'].forEach(sub => {
    const ptMap = (by_sub_pt || {})[sub] || {};
    Object.values(ptMap).forEach(data => {
      if (data && data.histogram.bins.length) {
        data.histogram.bins.forEach(b => allBinSet.add(b.toFixed(2)));
      }
    });
  });
  const globalLabels = [...allBinSet].sort((a, b) => parseFloat(a) - parseFloat(b));

  ['upper', 'lower'].forEach(sub => {
    const ptMap = (by_sub_pt || {})[sub] || {};
    const color = sub === 'upper' ? upperColor : lowerColor;
    Object.keys(ptMap).sort().forEach(pt => {
      const data = ptMap[pt];
      if (!data || !data.histogram.bins.length) return;
      const id = `tdist-${sub}-pt${pt}`;
      // Map counts to global bins (fill 0 for missing bins)
      const binToCount = {};
      data.histogram.bins.forEach((b, i) => { binToCount[b.toFixed(2)] = data.histogram.counts[i]; });
      const alignedCounts = globalLabels.map(label => binToCount[label] ?? 0);
      mkBar(id, globalLabels, alignedCounts, { color });
    });
  });

  // Draw overall / sub_key / part_name histograms
  const colors = ['rgba(26,120,194,0.75)', 'rgba(0,230,118,0.75)', 'rgba(255,152,0,0.75)', 'rgba(156,39,176,0.75)'];
  function drawHist(prefix, data, color) {
    if (!data || !data.histogram.bins.length) return;
    mkBar(`${prefix}-hist`, data.histogram.bins.map(b => b.toFixed(2)), data.histogram.counts, { color });
  }
  drawHist('ov', overall, colors[0]);
  subKeys.forEach((k, i) => drawHist(`sk-${k.replace(/\W/g, '_')}`, by_sub_key[k], colors[i % colors.length]));
  partNames.forEach((k, i) => drawHist(`pn-${k.replace(/\W/g, '_')}`, by_part_name[k], colors[(i + 2) % colors.length]));
}

/* ─────────────────────────────────────────────
   品質分析
───────────────────────────────────────────── */
async function loadQuality() {
  html('quality-content', spinner());
  try {
    const d = await api('/api/dashboard/quality');
    renderQuality(d);
  } catch (e) {
    html('quality-content', `<div class="error-msg">読み込みエラー: ${e.message}</div>`);
  }
}

function renderQuality(d) {
  const { pq_distance, t_outliers, by_sub_pt } = d;

  // PQ Distance section
  const pqStats  = statsRowHtml(pq_distance.stats);
  const pqHist   = `<div class="chart-wrap"><canvas id="pq-hist"></canvas></div>`;
  const pqOutRow = pq_distance.outliers.map(o =>
    `<tr><td>${o.idx}</td><td>${o.part_name || ''}</td><td class="num high">${o.pq_dist}</td>
     <td style="font-size:0.78rem;word-break:break-all">${o.image_path}</td></tr>`
  ).join('');

  // T-value outliers
  const tOutCount = t_outliers.count;
  const iqr = t_outliers.iqr_bounds;
  const tOutRows = t_outliers.samples.map(o =>
    `<tr><td>${o.idx}</td><td>${o.part_name || ''}</td><td>${o.sub_key || ''}</td>
     <td class="num">${o.point_index}</td>
     <td class="num ${Math.abs(o.t - 0.5) > 0.4 ? 'high' : 'mid'}">${o.t}</td>
     <td><a href="/?idx=${o.idx}" target="_blank">Annotatorで確認</a></td></tr>`
  ).join('');

  // upper/lower × pt 別統計 (14パターン)
  const upper = by_sub_pt['upper'] || {};
  const lower = by_sub_pt['lower'] || {};
  const allPts = [...new Set([...Object.keys(upper), ...Object.keys(lower)])].sort();

  const subPtRows = allPts.map(pt => {
    const u = upper[pt] || {};
    const l = lower[pt] || {};
    return `<tr>
      <td><strong>P${pt}</strong></td>
      <td class="num">${u.count ?? '–'}</td>
      <td class="num">${u.mean ?? '–'}</td>
      <td class="num">${u.std ?? '–'}</td>
      <td class="num">${u.min ?? '–'} / ${u.max ?? '–'}</td>
      <td class="num">${l.count ?? '–'}</td>
      <td class="num">${l.mean ?? '–'}</td>
      <td class="num">${l.std ?? '–'}</td>
      <td class="num">${l.min ?? '–'} / ${l.max ?? '–'}</td>
    </tr>`;
  }).join('');

  const h = `
    <div class="section-title">PQ 距離分布</div>
    <div class="card">
      <div class="card-title">PQ距離 (px) ヒストグラム</div>
      ${pqStats}
      ${pqHist}
    </div>
    ${pq_distance.outliers.length > 0 ? `
    <div class="section-title">PQ距離 外れ値 (${pq_distance.outliers.length}件)</div>
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>idx</th><th>part</th><th>PQ距離</th><th>画像パス</th></tr></thead>
        <tbody>${pqOutRow}</tbody>
      </table>
    </div>` : ''}

    <div class="section-title">t値 外れ値 (IQR法) — 計${tOutCount}件</div>
    ${iqr.low !== undefined ? `<p style="font-size:0.82rem;color:var(--muted);margin-bottom:10px">正常範囲: ${iqr.low} ～ ${iqr.high}</p>` : ''}
    ${tOutCount > 0 ? `
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>idx</th><th>part</th><th>sub_key</th><th>pt</th><th>t値</th><th>リンク</th></tr></thead>
        <tbody>${tOutRows}</tbody>
      </table>
    </div>` : '<div class="info-msg">外れ値なし</div>'}

    <div class="section-title">upper/lower × Point Index 別統計 (14パターン)</div>
    <div class="card" style="margin-bottom:16px">
      <div class="card-title">平均 t値 — upper / lower 比較</div>
      <div class="chart-wrap-tall"><canvas id="pt-sub-mean"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">upper/lower × Point Index 別 件数・統計</div>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th rowspan="2">pt</th>
              <th colspan="4" style="text-align:center;color:#90caf9">upper</th>
              <th colspan="4" style="text-align:center;color:#ffcc80">lower</th>
            </tr>
            <tr>
              <th>件数</th><th>平均</th><th>標準偏差</th><th>最小/最大</th>
              <th>件数</th><th>平均</th><th>標準偏差</th><th>最小/最大</th>
            </tr>
          </thead>
          <tbody>${subPtRows}</tbody>
        </table>
      </div>
    </div>`;

  html('quality-content', h);

  // Draw PQ histogram
  const pqLabels = pq_distance.histogram.bins.map(b => b.toFixed(1));
  mkBar('pq-hist', pqLabels, pq_distance.histogram.counts, { color: 'rgba(255,152,0,0.75)' });

  // Draw 14-pattern grouped bar chart
  const ptAxisLabels = allPts.map(k => `P${k}`);
  mkGroupedBar('pt-sub-mean', ptAxisLabels, [
    {
      label: 'upper',
      data: allPts.map(k => upper[k]?.mean ?? null),
      backgroundColor: 'rgba(26,120,194,0.8)',
      borderRadius: 2, borderWidth: 0,
    },
    {
      label: 'lower',
      data: allPts.map(k => lower[k]?.mean ?? null),
      backgroundColor: 'rgba(255,152,0,0.8)',
      borderRadius: 2, borderWidth: 0,
    },
  ]);
}

/* ─────────────────────────────────────────────
   実験比較
───────────────────────────────────────────── */
async function loadExperiments() {
  html('experiments-content', spinner());
  try {
    const d = await api('/api/dashboard/experiments');
    renderExperiments(d);
  } catch (e) {
    html('experiments-content', `<div class="error-msg">読み込みエラー: ${e.message}</div>`);
  }
}

function renderExperiments(d) {
  if (!d.experiments || d.experiments.length === 0) {
    const msg = d.message || 'runs/ に実験データがありません';
    html('experiments-content', `<div class="info-msg">${msg}</div>`);
    return;
  }

  const exps = d.experiments;

  // Build table rows
  const rows = exps.map(e => {
    const mae   = e.best_val_mae != null ? e.best_val_mae.toFixed(5) : '–';
    const folds = e.folds != null ? `${e.folds} fold` : '–';
    const errMsg = e.error ? `<span style="color:var(--muted);font-size:0.75rem">${e.error}</span>` : '';
    return `<tr>
      <td style="font-size:0.8rem;word-break:break-all">${e.name}</td>
      <td class="num">${folds}</td>
      <td class="num ${e.best_val_mae != null ? (e.best_val_mae < 0.05 ? 'high' : '') : ''}">${mae}</td>
      <td>${errMsg}</td>
    </tr>`;
  }).join('');

  // Chart: best_val_mae per experiment
  const withMae = exps.filter(e => e.best_val_mae != null);
  const expNames = withMae.map(e => e.name.replace(/^(kfold|ensemble)-/, '').substring(0, 30));
  const maeVals  = withMae.map(e => e.best_val_mae);

  const h = `
    <div class="section-title">実験一覧 (${exps.length}件)</div>
    <div class="table-wrap" style="margin-bottom:20px">
      <table class="data-table">
        <thead><tr><th>実験名</th><th>Folds</th><th>Best Val MAE</th><th>備考</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${withMae.length > 1 ? `
    <div class="section-title">Best Val MAE 比較</div>
    <div class="card">
      <div class="chart-wrap-tall"><canvas id="exp-mae"></canvas></div>
    </div>` : ''}`;

  html('experiments-content', h);

  if (withMae.length > 1) {
    // Color gradient: lowest MAE = greenest
    const minMae = Math.min(...maeVals);
    const maxMae = Math.max(...maeVals);
    const barColors = maeVals.map(v => {
      const ratio = maxMae > minMae ? (v - minMae) / (maxMae - minMae) : 0;
      const r = Math.round(26  + ratio * (244 - 26));
      const g = Math.round(120 + ratio * (67  - 120));
      const b = Math.round(194 + ratio * (54  - 194));
      return `rgba(${r},${g},${b},0.8)`;
    });

    if (charts['exp-mae']) charts['exp-mae'].destroy();
    const ctx = document.getElementById('exp-mae');
    if (ctx) {
      charts['exp-mae'] = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
          labels: expNames,
          datasets: [{ data: maeVals, backgroundColor: barColors, borderRadius: 2, borderWidth: 0 }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 300 },
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: c => ` MAE: ${c.formattedValue}` } },
          },
          scales: {
            x: { grid: { color: '#1a2a4a' }, ticks: { maxRotation: 45, font: { size: 10 } } },
            y: { grid: { color: '#1a2a4a' }, beginAtZero: false },
          },
        },
      });
    }
  }
}

/* ─────────────────────────────────────────────
   エラー分析
───────────────────────────────────────────── */
async function loadErrors() {
  html('errors-content', spinner());
  try {
    const d = await api('/api/dashboard/errors');
    renderErrors(d);
  } catch (e) {
    html('errors-content', `<div class="error-msg">読み込みエラー: ${e.message}</div>`);
  }
}

function renderErrors(d) {
  if (!d.available) {
    html('errors-content', `<div class="info-msg">${d.message}</div>`);
    return;
  }

  const s = d.summary;
  const b = d.breakdown;

  // Summary cards
  const summaryCards = `
    <div class="card-grid">
      <div class="stat-card"><span class="stat-label">サンプル数</span><span class="stat-value">${s.total.toLocaleString()}</span></div>
      <div class="stat-card"><span class="stat-label">MAE</span><span class="stat-value" style="font-size:1.3rem">${s.mae}</span></div>
      <div class="stat-card"><span class="stat-label">中央値</span><span class="stat-value" style="font-size:1.3rem">${s.median}</span></div>
      <div class="stat-card"><span class="stat-label">P90</span><span class="stat-value" style="font-size:1.3rem;color:var(--orange)">${s.p90}</span></div>
      <div class="stat-card"><span class="stat-label">P95</span><span class="stat-value" style="font-size:1.3rem;color:var(--orange)">${s.p95}</span></div>
      <div class="stat-card"><span class="stat-label">最大誤差</span><span class="stat-value" style="font-size:1.3rem;color:var(--red)">${s.max}</span></div>
    </div>`;

  // Breakdown charts
  const breakdownHtml = `
    <div class="section-title">誤差の内訳 (MAE)</div>
    <div class="card-grid-3">
      <div class="card">
        <div class="card-title">Part Name 別</div>
        <div class="chart-wrap"><canvas id="err-part"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Sub Key 別</div>
        <div class="chart-wrap"><canvas id="err-sub"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Point Index 別 (7パターン)</div>
        <div class="chart-wrap"><canvas id="err-pt"></canvas></div>
      </div>
    </div>
    <div class="card" style="margin-bottom:20px">
      <div class="card-title">upper/lower × Point Index 別 (14パターン)</div>
      <div class="chart-wrap-tall"><canvas id="err-sub-pt"></canvas></div>
    </div>`;

  // Error distribution histogram
  const histHtml = `
    <div class="section-title">誤差分布</div>
    <div class="card">
      <div class="chart-wrap"><canvas id="err-hist"></canvas></div>
    </div>`;

  // Top errors table
  const errRows = d.top_errors.map(r => {
    const cls = r.abs_error >= 0.3 ? 'high' : r.abs_error >= 0.15 ? 'mid' : '';
    return `<tr>
      <td class="num">${r.rank}</td>
      <td class="num ${cls}">${r.abs_error}</td>
      <td class="num">${r.t_true}</td>
      <td class="num">${r.t_pred}</td>
      <td>${r.part_name || ''}</td>
      <td>${r.sub_key || ''}</td>
      <td class="num">${r.point_index}</td>
      <td><a href="/?idx=${r.idx}" target="_blank">${r.idx}</a></td>
      <td><button class="patch-btn" data-idx="${r.idx}" style="font-size:0.75rem;padding:2px 8px">パッチ表示</button></td>
    </tr>
    <tr class="patch-row" id="patch-row-${r.idx}" style="display:none">
      <td colspan="9" style="background:var(--bg-card);padding:8px 16px">
        <img id="patch-img-${r.idx}" class="patch-img" alt="パッチ" />
      </td>
    </tr>`;
  }).join('');

  const topErrHtml = `
    <div class="section-title">誤差上位 ${d.top_errors.length} 件</div>
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr><th>#</th><th>誤差</th><th>t_true</th><th>t_pred</th>
          <th>part</th><th>sub</th><th>pt</th><th>idx</th><th></th></tr>
        </thead>
        <tbody>${errRows}</tbody>
      </table>
    </div>`;

  html('errors-content', summaryCards + breakdownHtml + histHtml + topErrHtml);

  // Draw charts
  const partLabels = Object.keys(b.by_part_name);
  mkBar('err-part', partLabels, partLabels.map(k => b.by_part_name[k]), { color: 'rgba(244,67,54,0.75)' });

  const subLabels = Object.keys(b.by_sub_key);
  mkBar('err-sub', subLabels, subLabels.map(k => b.by_sub_key[k]), { color: 'rgba(255,152,0,0.75)' });

  const ptLabels = Object.keys(b.by_point_index).sort();
  mkBar('err-pt', ptLabels.map(k => `P${k}`), ptLabels.map(k => b.by_point_index[k]), { color: 'rgba(156,39,176,0.75)' });

  // 14-pattern grouped bar
  const subPt = b.by_sub_pt || {};
  const errUpper = subPt['upper'] || {};
  const errLower = subPt['lower'] || {};
  const errAllPts = [...new Set([...Object.keys(errUpper), ...Object.keys(errLower)])].sort();
  mkGroupedBar('err-sub-pt', errAllPts.map(k => `P${k}`), [
    {
      label: 'upper',
      data: errAllPts.map(k => errUpper[k] ?? null),
      backgroundColor: 'rgba(26,120,194,0.8)',
      borderRadius: 2, borderWidth: 0,
    },
    {
      label: 'lower',
      data: errAllPts.map(k => errLower[k] ?? null),
      backgroundColor: 'rgba(255,152,0,0.8)',
      borderRadius: 2, borderWidth: 0,
    },
  ]);

  const histBins = d.error_histogram.bins.map(b => b.toFixed(3));
  mkBar('err-hist', histBins, d.error_histogram.counts, { color: 'rgba(244,67,54,0.65)' });

  // Patch toggle buttons
  document.querySelectorAll('.patch-btn').forEach(btn => {
    btn.addEventListener('click', () => togglePatch(btn.dataset.idx));
  });
}

async function togglePatch(idx) {
  const row = document.getElementById(`patch-row-${idx}`);
  const img = document.getElementById(`patch-img-${idx}`);
  if (!row || !img) return;

  if (row.style.display !== 'none') {
    row.style.display = 'none';
    return;
  }
  row.style.display = '';

  if (img.src) return; // already loaded

  try {
    const res = await fetch(`/api/dashboard/errors/${idx}/patch`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const b64 = await res.text();
    img.src = `data:image/png;base64,${b64}`;
  } catch (e) {
    img.alt = `エラー: ${e.message}`;
  }
}

/* ─────────────────────────────────────────────
   Init
───────────────────────────────────────────── */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

document.getElementById('btn-refresh').addEventListener('click', refreshCurrent);

// Load Overview on start
loadTab('overview');
