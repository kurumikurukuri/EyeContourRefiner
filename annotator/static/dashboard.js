/* ── Chart.js global defaults ── */
Chart.defaults.color         = '#888';
Chart.defaults.borderColor   = '#1a2a4a';
Chart.defaults.font.family   = "'Segoe UI', sans-serif";
Chart.defaults.font.size     = 11;

/* ── State ── */
const charts  = {};   // id → Chart instance
const loaded  = new Set();
let currentExpPath = '';  // prediction.csv分析で参照する実験パス（空 = data/predictions.csv）
let currentSplit = 'test'; // train / test / all
let _qualityCache = null;  // 分布・統計 / 外れ値確認 で共有するキャッシュ

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

function mkBoxplot(id, labels, datasets) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  charts[id] = new Chart(ctx.getContext('2d'), {
    type: 'boxplot',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: true, position: 'top', labels: { boxWidth: 12, color: '#ccc', padding: 12 } },
      },
      scales: {
        x: { grid: { color: '#1a2a4a' } },
        y: { grid: { color: '#1a2a4a' }, beginAtZero: false, ticks: { color: '#888' } },
      },
    },
  });
}

function mkViolin(id, labels, datasets) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  charts[id] = new Chart(ctx.getContext('2d'), {
    type: 'violin',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { display: datasets.length > 1, position: 'top', labels: { boxWidth: 12, color: '#ccc', padding: 12 } },
      },
      scales: {
        x: { grid: { color: '#1a2a4a' } },
        y: { grid: { color: '#1a2a4a' }, beginAtZero: true, ticks: { color: '#888' } },
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

/* scatter: datasets=[{label, data:[{x,y}], color}], refLine: bool */
function mkScatter(id, datasets, opts = {}) {
  if (charts[id]) charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;

  const chartDatasets = datasets.map(ds => ({
    label: ds.label,
    data: ds.data,
    pointRadius: 2.5,
    pointHoverRadius: 4,
    backgroundColor: ds.color || 'rgba(26,120,194,0.5)',
    borderColor: ds.color || 'rgba(26,120,194,0.5)',
    borderWidth: 0,
  }));

  // 参照線は scatter チャート内で showLine: true を使う（type:'line' 上書きは不可）
  function _refLineDataset(label, pts) {
    return {
      label,
      data: pts,
      showLine: true,
      pointRadius: 0,
      pointHoverRadius: 0,
      borderColor: 'rgba(180,180,180,0.55)',
      borderWidth: 1.5,
      borderDash: [5, 5],
      backgroundColor: 'transparent',
    };
  }

  if (opts.refLine) {
    const allX = datasets.flatMap(ds => ds.data.map(p => p.x));
    const minX = allX.length ? Math.min(...allX) : 0;
    const maxX = allX.length ? Math.max(...allX) : 1;
    chartDatasets.push(_refLineDataset('y = x', [{ x: minX, y: minX }, { x: maxX, y: maxX }]));
  }

  if (opts.hline !== undefined) {
    const allX = datasets.flatMap(ds => ds.data.map(p => p.x));
    const minX = allX.length ? Math.min(...allX) : 0;
    const maxX = allX.length ? Math.max(...allX) : 1;
    chartDatasets.push(_refLineDataset('y = 0', [{ x: minX, y: 0 }, { x: maxX, y: 0 }]));
  }

  charts[id] = new Chart(ctx.getContext('2d'), {
    type: 'scatter',
    data: { datasets: chartDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      plugins: {
        legend: {
          display: datasets.length > 1,
          position: 'top',
          labels: { boxWidth: 10, color: '#ccc', padding: 10, usePointStyle: true, pointStyle: 'circle',
                    filter: item => item.text !== 'y = x' && item.text !== 'y = 0' },
        },
        title: opts.title
          ? { display: true, text: opts.title, color: '#a8dadc', font: { size: 12 } }
          : { display: false },
        tooltip: {
          filter: item => item.dataset.showLine !== true,
          callbacks: {
            label: ctx => `(${ctx.parsed.x.toFixed(4)}, ${ctx.parsed.y.toFixed(4)})`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: '#1a2a4a' },
          title: { display: !!opts.xLabel, text: opts.xLabel, color: '#888' },
        },
        y: {
          grid: { color: '#1a2a4a' },
          title: { display: !!opts.yLabel, text: opts.yLabel, color: '#888' },
        },
      },
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
    case 'overview': return loadOverview();
    case 'distrib':  return loadDistrib();
    case 'outliers': return loadOutliers();
    case 'errors':   return loadErrors();
  }
}

function refreshCurrent() {
  const active = document.querySelector('.tab-btn.active');
  if (!active) return;
  const name = active.dataset.tab;
  if (name === 'distrib' || name === 'outliers') _qualityCache = null;
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
        <span class="stat-label">アノテーション済み（非スキップ）</span>
        <span class="stat-value">${d.annotated.toLocaleString()}</span>
        <span class="stat-sub">${d.progress_pct}% / 非スキップ ${d.non_skipped_total.toLocaleString()}件</span>
        ${progress}
      </div>
      <div class="stat-card">
        <span class="stat-label">未アノテーション（非スキップ）</span>
        <span class="stat-value" style="color:${d.unannotated > 0 ? 'var(--orange)' : 'var(--green)'}">${d.unannotated.toLocaleString()}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">ユニーク画像数（全体）</span>
        <span class="stat-value">${d.unique_images.toLocaleString()}</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">ユニーク画像数（非スキップ）</span>
        <span class="stat-value" style="color:var(--green)">${(d.non_skipped_unique_images ?? 0).toLocaleString()}</span>
      </div>
    </div>`;

  // Completion matrix
  const matrixRows = (d.completion_matrix || []).map(m => {
    const color = m.pct >= 100 ? 'var(--green)' : m.pct >= 50 ? 'var(--orange)' : 'var(--red, #f44336)';
    const barColor = m.pct >= 100 ? '#28a745' : m.pct >= 50 ? '#ff9800' : '#f44336';
    return `<tr>
      <td style="font-size:0.8rem">${esc(m.part)}</td>
      <td style="font-size:0.8rem">${esc(m.sub)}</td>
      <td class="num">${m.annotated.toLocaleString()}</td>
      <td class="num">${m.total.toLocaleString()}</td>
      <td style="min-width:120px">
        <div style="display:flex;align-items:center;gap:6px">
          <div style="flex:1;background:#1a2a4a;border-radius:3px;height:8px;overflow:hidden">
            <div style="width:${m.pct}%;background:${barColor};height:100%;border-radius:3px;transition:width 0.3s"></div>
          </div>
          <span style="font-size:0.78rem;color:${color};min-width:36px;text-align:right">${m.pct}%</span>
        </div>
      </td>
    </tr>`;
  }).join('');

  const matrixHtml = `
    <div class="section-title">完了率内訳 (Part × Sub Key)</div>
    <div class="card">
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr><th>Part</th><th>Sub Key</th><th>完了</th><th>合計</th><th>進捗</th></tr></thead>
          <tbody>${matrixRows}</tbody>
        </table>
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
        <div class="card-title">Point Index 分布</div>
        <div class="chart-wrap"><canvas id="ov-pt"></canvas></div>
      </div>`;

  html('overview-content', statCards + matrixHtml + chartsHtml);

  // Draw charts
  const partLabels  = Object.keys(d.by_part_name);
  const partData    = Object.values(d.by_part_name);
  mkDoughnut('ov-part', partLabels, partData,
    ['rgba(26,120,194,0.85)', 'rgba(0,230,118,0.85)', 'rgba(255,152,0,0.85)']);

  const subLabels = Object.keys(d.by_sub_key);
  const subData   = Object.values(d.by_sub_key);
  mkDoughnut('ov-sub', subLabels, subData,
    ['rgba(168,218,220,0.85)', 'rgba(156,39,176,0.85)']);

  const ptLabels = Object.keys(d.by_point_index).sort();
  const ptData   = ptLabels.map(k => d.by_point_index[k]);
  mkBar('ov-pt', ptLabels.map(k => `P${k}`), ptData, { color: 'rgba(26,120,194,0.75)' });
}

/* ─────────────────────────────────────────────
   Quality data cache (分布・統計 / 外れ値確認 で共有)
───────────────────────────────────────────── */
async function fetchQuality() {
  if (!_qualityCache) {
    _qualityCache = await api('/api/dashboard/quality');
  }
  return _qualityCache;
}

/* ─────────────────────────────────────────────
   分布・統計
───────────────────────────────────────────── */
async function loadDistrib() {
  html('distrib-content', spinner());
  try {
    const [td, qd] = await Promise.all([
      api('/api/dashboard/t_distribution'),
      fetchQuality(),
    ]);
    renderDistrib(td, qd);
  } catch (e) {
    html('distrib-content', `<div class="error-msg">読み込みエラー: ${e.message}</div>`);
  }
}

function renderDistrib(td, qd) {
  const { overall, by_sub_key, by_part_name, by_sub_pt: tdSubPt } = td;
  const { pq_distance, by_sub_pt: qdSubPt, by_sub_pt_raw } = qd;

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

  function subPtGrid(sub, colorRgba) {
    const ptMap = (tdSubPt || {})[sub] || {};
    const pts = Object.keys(ptMap).sort();
    if (!pts.length) return '';
    const cards = pts.map(pt => {
      const data = ptMap[pt];
      const id = `distrib-${sub}-pt${pt}`;
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

  // upper/lower × pt 統計テーブル
  const upper = qdSubPt['upper'] || {};
  const lower = qdSubPt['lower'] || {};
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

  const pqHistHtml = `
    <div style="display:flex;justify-content:flex-end;margin-bottom:6px">
      <button id="btn-pq-log-toggle" style="background:#1a3a5c;border:1px solid #4fc3f7;color:#e0e0e0;border-radius:5px;padding:3px 12px;cursor:pointer;font-size:0.78rem">対数表示</button>
    </div>
    <div class="chart-wrap"><canvas id="pq-hist"></canvas></div>`;

  const htmlContent = `
    <div class="section-title">t値分布（全体 / Sub Key / Part Name）</div>
    ${histSection('ov', '全サンプル', overall)}
    <div class="card-grid-2">
      ${subKeys.map(k => histSection(`sk-${k.replace(/\W/g, '_')}`, k, by_sub_key[k])).join('')}
    </div>
    <div class="card-grid-2">
      ${partNames.map(k => histSection(`pn-${k.replace(/\W/g, '_')}`, k, by_part_name[k])).join('')}
    </div>

    <div class="section-title">t値分布（upper / lower × Point Index 14パターン）</div>
    ${subPtGrid('upper', '#90caf9')}
    ${subPtGrid('lower', '#ffcc80')}

    <div class="section-title">PQ 距離分布</div>
    <div class="card">
      <div class="card-title">PQ距離 (px) ヒストグラム</div>
      ${statsRowHtml(pq_distance.stats)}
      ${pqHistHtml}
    </div>

    <div class="section-title">upper/lower × Point Index 分布比較</div>
    <div class="card" style="margin-bottom:16px">
      <div class="card-title">t値分布 — upper / lower 比較 (バイオリンプロット)</div>
      <div class="chart-wrap-tall"><canvas id="pt-sub-box"></canvas></div>
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

  html('distrib-content', htmlContent);

  // 14パターン mini histograms（x軸を全パターン共通に揃える）
  const upperColor = 'rgba(79,195,247,0.8)';
  const lowerColor = 'rgba(255,183,77,0.8)';
  const allBinSet = new Set();
  ['upper', 'lower'].forEach(sub => {
    const ptMap = (tdSubPt || {})[sub] || {};
    Object.values(ptMap).forEach(data => {
      if (data && data.histogram.bins.length) {
        data.histogram.bins.forEach(b => allBinSet.add(b.toFixed(2)));
      }
    });
  });
  const globalLabels = [...allBinSet].sort((a, b) => parseFloat(a) - parseFloat(b));

  ['upper', 'lower'].forEach(sub => {
    const ptMap = (tdSubPt || {})[sub] || {};
    const color = sub === 'upper' ? upperColor : lowerColor;
    Object.keys(ptMap).sort().forEach(pt => {
      const data = ptMap[pt];
      if (!data || !data.histogram.bins.length) return;
      const id = `distrib-${sub}-pt${pt}`;
      const binToCount = {};
      data.histogram.bins.forEach((b, i) => { binToCount[b.toFixed(2)] = data.histogram.counts[i]; });
      const alignedCounts = globalLabels.map(label => binToCount[label] ?? 0);
      mkBar(id, globalLabels, alignedCounts, { color });
    });
  });

  // 全体 / sub / part ヒストグラム
  const colors = ['rgba(26,120,194,0.75)', 'rgba(0,230,118,0.75)', 'rgba(255,152,0,0.75)', 'rgba(156,39,176,0.75)'];
  function drawHist(prefix, data, color) {
    if (!data || !data.histogram.bins.length) return;
    mkBar(`${prefix}-hist`, data.histogram.bins.map(b => b.toFixed(2)), data.histogram.counts, { color });
  }
  drawHist('ov', overall, colors[0]);
  subKeys.forEach((k, i) => drawHist(`sk-${k.replace(/\W/g, '_')}`, by_sub_key[k], colors[i % colors.length]));
  partNames.forEach((k, i) => drawHist(`pn-${k.replace(/\W/g, '_')}`, by_part_name[k], colors[(i + 2) % colors.length]));

  // PQ ヒストグラム
  const pqLabels = pq_distance.histogram.bins.map(b => b.toFixed(1));
  mkBar('pq-hist', pqLabels, pq_distance.histogram.counts, { color: 'rgba(255,152,0,0.75)' });

  const pqLogBtn = document.getElementById('btn-pq-log-toggle');
  if (pqLogBtn) {
    let pqLogScale = false;
    const pqCounts = pq_distance.histogram.counts;
    const logHist  = pq_distance.log_histogram || { bins: [], counts: [] };
    pqLogBtn.addEventListener('click', () => {
      pqLogScale = !pqLogScale;
      if (charts['pq-hist']) charts['pq-hist'].destroy();
      const ctx = document.getElementById('pq-hist');
      if (!ctx) return;
      if (pqLogScale && logHist.bins.length > 0) {
        charts['pq-hist'] = new Chart(ctx.getContext('2d'), {
          type: 'bar',
          data: {
            datasets: [{
              data: logHist.bins.map((b, i) => ({ x: b, y: logHist.counts[i] })),
              backgroundColor: 'rgba(255,152,0,0.75)',
              borderRadius: 2, borderWidth: 0,
              barPercentage: 1.0, categoryPercentage: 0.9,
            }],
          },
          options: {
            responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
            plugins: { legend: { display: false } },
            scales: {
              x: { type: 'logarithmic', grid: { color: '#1a2a4a' }, ticks: { maxRotation: 45 } },
              y: { grid: { color: '#1a2a4a' }, beginAtZero: true },
            },
          },
        });
      } else {
        mkBar('pq-hist', pqLabels, pqCounts, { color: 'rgba(255,152,0,0.75)' });
      }
      pqLogBtn.textContent = pqLogScale ? '線形表示' : '対数表示';
    });
  }

  // バイオリンプロット
  const upperRaw = (by_sub_pt_raw || {})['upper'] || {};
  const lowerRaw = (by_sub_pt_raw || {})['lower'] || {};
  const ptAxisLabels = allPts.map(k => `P${k}`);
  mkViolin('pt-sub-box', ptAxisLabels, [
    {
      label: 'upper',
      data: allPts.map(k => upperRaw[k] || []),
      backgroundColor: 'rgba(79,195,247,0.35)',
      borderColor: 'rgba(79,195,247,0.9)',
      borderWidth: 1.5,
    },
    {
      label: 'lower',
      data: allPts.map(k => lowerRaw[k] || []),
      backgroundColor: 'rgba(255,183,77,0.35)',
      borderColor: 'rgba(255,183,77,0.9)',
      borderWidth: 1.5,
    },
  ]);
}

/* ─────────────────────────────────────────────
   外れ値確認
───────────────────────────────────────────── */
async function loadOutliers() {
  html('outliers-content', spinner());
  try {
    const d = await fetchQuality();
    renderOutliers(d);
  } catch (e) {
    html('outliers-content', `<div class="error-msg">読み込みエラー: ${e.message}</div>`);
  }
}

function renderOutliers(d) {
  const { pq_distance, t_outliers } = d;

  const tOutCount = t_outliers.count;
  const iqr = t_outliers.iqr_bounds;
  const tOutRows = t_outliers.samples.map(o =>
    `<tr><td>${o.idx}</td><td>${o.part_name || ''}</td><td>${o.sub_key || ''}</td>
     <td class="num">${o.point_index}</td>
     <td class="num ${Math.abs(o.t - 0.5) > 0.4 ? 'high' : 'mid'}">${o.t}</td>
     <td><a href="/?idx=${o.idx}" target="_blank">Annotatorで確認</a></td></tr>`
  ).join('');

  const pqOutRow = pq_distance.outliers.map(o =>
    `<tr><td>${o.idx}</td><td>${o.part_name || ''}</td><td class="num high">${o.pq_dist}</td>
     <td style="font-size:0.78rem;word-break:break-all">${esc(o.image_path || '')}</td>
     <td><a href="/?idx=${o.idx}" target="_blank">確認</a></td></tr>`
  ).join('');

  const pqShortRow = (pq_distance.shortest || []).map(o =>
    `<tr><td>${o.idx}</td><td>${o.part_name || ''}</td><td class="num">${o.pq_dist}</td>
     <td style="font-size:0.78rem;word-break:break-all">${esc(o.image_path || '')}</td>
     <td><a href="/?idx=${o.idx}" target="_blank">確認</a></td></tr>`
  ).join('');

  const h = `
    <div class="section-title">t値 外れ値 (IQR法) — 計${tOutCount}件</div>
    ${iqr.low !== undefined ? `<p style="font-size:0.82rem;color:var(--muted);margin-bottom:10px">正常範囲: ${iqr.low} ～ ${iqr.high}</p>` : ''}
    ${tOutCount > 0 ? `
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>idx</th><th>part</th><th>sub_key</th><th>pt</th><th>t値</th><th>リンク</th></tr></thead>
        <tbody>${tOutRows}</tbody>
      </table>
    </div>` : '<div class="info-msg">外れ値なし</div>'}

    ${pq_distance.outliers.length > 0 ? `
    <div class="section-title">PQ距離 外れ値（長い） — ${pq_distance.outliers.length}件</div>
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>idx</th><th>part</th><th>PQ距離</th><th>画像パス</th><th>リンク</th></tr></thead>
        <tbody>${pqOutRow}</tbody>
      </table>
    </div>` : ''}

    ${pqShortRow ? `
    <div class="section-title">PQ距離 外れ値（短い） — ${(pq_distance.shortest || []).length}件</div>
    <div class="table-wrap">
      <table class="data-table">
        <thead><tr><th>idx</th><th>part</th><th>PQ距離</th><th>画像パス</th><th>リンク</th></tr></thead>
        <tbody>${pqShortRow}</tbody>
      </table>
    </div>` : ''}`;

  html('outliers-content', h);
}

function esc(str) {
  return String(str || '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

/* ─────────────────────────────────────────────
   エラー分析
───────────────────────────────────────────── */
async function loadErrors() {
  html('errors-content', spinner());
  try {
    // 実験セレクターのデータを取得
    const expData = await api('/api/dashboard/experiments');
    const expsWithPred = (expData.experiments || []).filter(e => e.has_predictions);

    const params = new URLSearchParams();
    if (currentExpPath) params.set('exp_path', currentExpPath);
    if (currentSplit) params.set('split', currentSplit);
    const qs = params.toString();
    const url = `/api/dashboard/errors${qs ? '?' + qs : ''}`;
    const d = await api(url);

    // 実験セレクターHTML
    const selectorOptions = [
      `<option value="" ${!currentExpPath ? 'selected' : ''}>data/predictions.csv (デフォルト)</option>`,
      ...expsWithPred.map(e =>
        `<option value="${esc(e.exp_dir)}" ${currentExpPath === e.exp_dir ? 'selected' : ''}>${esc(e.exp_name || e.exp_dir)}</option>`
      ),
    ].join('');

    // split セレクターHTML
    const splitOptions = [
      { value: 'test', label: 'test (test_predictions.csv)' },
      { value: 'train', label: 'train (train_predictions.csv)' },
      { value: 'all', label: 'all (train + test)' },
    ].map(o =>
      `<option value="${o.value}" ${currentSplit === o.value ? 'selected' : ''}>${o.label}</option>`
    ).join('');

    const selectorHtml = `
      <div style="margin-bottom:12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <label style="font-size:0.82rem;color:#888">参照実験:</label>
        <select id="err-exp-select" style="background:#0d0d1a;color:#e0e0e0;border:1px solid #1a3460;border-radius:5px;padding:4px 8px;font-size:0.82rem">
          ${selectorOptions}
        </select>
        <label style="font-size:0.82rem;color:#888;margin-left:8px">データ分割:</label>
        <select id="err-split-select" style="background:#0d0d1a;color:#e0e0e0;border:1px solid #1a3460;border-radius:5px;padding:4px 8px;font-size:0.82rem">
          ${splitOptions}
        </select>
      </div>`;

    renderErrors(d, selectorHtml);
  } catch (e) {
    html('errors-content', `<div class="error-msg">読み込みエラー: ${e.message}</div>`);
  }
}

function renderErrors(d, selectorHtml = '') {
  if (!d.available) {
    html('errors-content', (selectorHtml || '') + `<div class="info-msg">${d.message}</div>`);
    const sel = document.getElementById('err-exp-select');
    if (sel) sel.addEventListener('change', onErrExpChange);
    const splitSel = document.getElementById('err-split-select');
    if (splitSel) splitSel.addEventListener('change', onErrSplitChange);
    return;
  }

  const s = d.summary;
  const b = d.breakdown;

  // Summary cards
  const summaryCards = `
    <div class="card-grid">
      <div class="stat-card"><span class="stat-label">サンプル数</span><span class="stat-value">${s.total.toLocaleString()}</span></div>
      <div class="stat-card"><span class="stat-label">MAE (t)</span><span class="stat-value" style="font-size:1.3rem">${s.mae}</span></div>
      <div class="stat-card"><span class="stat-label">MAE (px)</span><span class="stat-value" style="font-size:1.3rem;color:#80cbc4">${s.mae_px}</span></div>
      <div class="stat-card"><span class="stat-label">中央値 (t)</span><span class="stat-value" style="font-size:1.3rem">${s.median}</span></div>
      <div class="stat-card"><span class="stat-label">中央値 (px)</span><span class="stat-value" style="font-size:1.3rem;color:#80cbc4">${s.median_px}</span></div>
      <div class="stat-card"><span class="stat-label">P90 (t)</span><span class="stat-value" style="font-size:1.3rem;color:var(--orange)">${s.p90}</span></div>
      <div class="stat-card"><span class="stat-label">P90 (px)</span><span class="stat-value" style="font-size:1.3rem;color:var(--orange)">${s.p90_px}</span></div>
      <div class="stat-card"><span class="stat-label">最大誤差 (t)</span><span class="stat-value" style="font-size:1.3rem;color:var(--red)">${s.max}</span></div>
      <div class="stat-card"><span class="stat-label">最大誤差 (px)</span><span class="stat-value" style="font-size:1.3rem;color:var(--red)">${s.max_px}</span></div>
    </div>`;

  // Breakdown charts
  const breakdownHtml = `
    <div class="section-title">誤差の分布</div>
    <div class="card-grid-3">
      <div class="card">
        <div class="card-title">Part Name 別 (箱ひげ図)</div>
        <div class="chart-wrap"><canvas id="err-part"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Sub Key 別 (バイオリン図)</div>
        <div class="chart-wrap"><canvas id="err-sub"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Point Index 別 (バイオリン図)</div>
        <div class="chart-wrap"><canvas id="err-pt"></canvas></div>
      </div>
    </div>
    <div class="card" style="margin-bottom:20px">
      <div class="card-title">upper/lower × Point Index 別 (箱ひげ図)</div>
      <div class="chart-wrap-tall"><canvas id="err-sub-pt"></canvas></div>
    </div>`;

  // Scatter / residual analysis section
  const corr        = d.summary.corr_coef     != null ? d.summary.corr_coef     : '–';
  const residMean   = d.summary.resid_mean    != null ? d.summary.resid_mean    : '–';
  const residStd    = d.summary.resid_std     != null ? d.summary.resid_std     : '–';
  const residMeanPx = d.summary.resid_mean_px != null ? d.summary.resid_mean_px : '–';
  const residStdPx  = d.summary.resid_std_px  != null ? d.summary.resid_std_px  : '–';
  const scatterHtml = `
    <div class="section-title">true_t_raw vs pred_t_raw 散布図・残差分析 (t値)</div>
    <div class="card-grid" style="margin-bottom:8px">
      <div class="stat-card"><span class="stat-label">相関係数 r</span><span class="stat-value" style="font-size:1.2rem">${corr}</span></div>
      <div class="stat-card"><span class="stat-label">残差 平均 (t)</span><span class="stat-value" style="font-size:1.2rem">${residMean}</span></div>
      <div class="stat-card"><span class="stat-label">残差 標準偏差 (t)</span><span class="stat-value" style="font-size:1.2rem">${residStd}</span></div>
    </div>
    <div class="card-grid-3" style="margin-bottom:20px">
      <div class="card">
        <div class="card-title">true_t_raw vs pred_t_raw（点線: y=x）</div>
        <div class="chart-wrap"><canvas id="err-scatter-pred"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">残差 (pred_t_raw − true_t_raw) vs true_t_raw</div>
        <div class="chart-wrap"><canvas id="err-scatter-resid"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">残差分布ヒストグラム (t値)</div>
        <div class="chart-wrap"><canvas id="err-resid-hist"></canvas></div>
      </div>
    </div>

    <div class="section-title">残差分析 (ピクセル単位)</div>
    <div class="card-grid" style="margin-bottom:8px">
      <div class="stat-card"><span class="stat-label">残差 平均 (px)</span><span class="stat-value" style="font-size:1.2rem;color:#80cbc4">${residMeanPx}</span></div>
      <div class="stat-card"><span class="stat-label">残差 標準偏差 (px)</span><span class="stat-value" style="font-size:1.2rem;color:#80cbc4">${residStdPx}</span></div>
    </div>
    <div class="card-grid-2" style="margin-bottom:20px">
      <div class="card">
        <div class="card-title">残差 (px) vs true_t_raw</div>
        <div class="chart-wrap"><canvas id="err-scatter-resid-px"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">残差分布ヒストグラム (px)</div>
        <div class="chart-wrap"><canvas id="err-resid-hist-px"></canvas></div>
      </div>
    </div>`;

  // pq_dist vs 誤差 散布図 + ヒートマップ
  const hm = d.heatmap || {};
  const hmParts = Object.keys(hm).sort();
  const hmPts   = [...new Set(hmParts.flatMap(p => Object.keys(hm[p])))].sort((a, b) => +a - +b);
  // カラースケール: 最大値を基準に背景色を計算
  const hmAllVals = hmParts.flatMap(p => Object.values(hm[p]));
  const hmMax = hmAllVals.length ? Math.max(...hmAllVals) : 1;
  const hmRows = hmParts.map(part => {
    const cells = hmPts.map(pt => {
      const v = hm[part]?.[pt];
      if (v == null) return '<td style="background:#0d0d1a;color:#333">–</td>';
      const ratio = Math.min(v / hmMax, 1);
      const r = Math.round(26  + ratio * (244 - 26));
      const g = Math.round(120 + ratio * (67  - 120));
      const b = Math.round(194 + ratio * (54  - 194));
      return `<td style="background:rgba(${r},${g},${b},0.75);color:#fff;font-size:0.78rem;text-align:center">${v.toFixed(2)}</td>`;
    }).join('');
    return `<tr><td style="font-size:0.78rem;white-space:nowrap;color:#a8dadc">${esc(part)}</td>${cells}</tr>`;
  }).join('');
  const hmHtml = hmRows.length ? `
    <div class="section-title">誤差ヒートマップ (Part × Point Index, 単位:px)</div>
    <div class="card" style="margin-bottom:20px;overflow-x:auto">
      <table class="data-table" style="min-width:auto">
        <thead><tr><th>Part</th>${hmPts.map(p => `<th>P${p}</th>`).join('')}</tr></thead>
        <tbody>${hmRows}</tbody>
      </table>
    </div>` : '';

  const extraChartsHtml = `
    ${hmHtml}
    <div class="section-title">pq_dist vs 誤差(px) 散布図</div>
    <div class="card" style="margin-bottom:20px">
      <div class="card-title">PQ距離 vs 絶対誤差(px) — pq_dist が大きいほど誤差(px)が拡大する傾向</div>
      <div class="chart-wrap"><canvas id="err-pqdist-scatter"></canvas></div>
    </div>`;

  // Error distribution histogram
  const histHtml = `
    <div class="section-title">絶対誤差分布</div>
    <div class="card-grid-2">
      <div class="card">
        <div class="card-title">誤差分布 (t値)</div>
        <div class="chart-wrap"><canvas id="err-hist"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">誤差分布 (px)</div>
        <div class="chart-wrap"><canvas id="err-hist-px"></canvas></div>
      </div>
    </div>`;

  // Top errors tables (t-sorted and px-sorted) are built inside errTableHtml()

  function errTableHtml(title, topList) {
    const rows_ = topList.map(r => {
      const cls   = r.abs_error    >= 0.3  ? 'high' : r.abs_error    >= 0.15 ? 'mid' : '';
      const clsPx = r.abs_error_px != null && r.abs_error_px >= 2.0 ? 'high' : r.abs_error_px >= 1.0 ? 'mid' : '';
      return `<tr>
        <td class="num">${r.rank}</td>
        <td class="num ${cls}">${r.abs_error}</td>
        <td class="num ${clsPx}">${r.abs_error_px != null ? r.abs_error_px : '–'}</td>
        <td class="num">${r.true_t_raw}</td>
        <td class="num">${r.pred_t_raw}</td>
        <td class="num">${r.pq_dist != null ? r.pq_dist : '–'}</td>
        <td>${r.part_name || ''}</td>
        <td>${r.sub_key || ''}</td>
        <td class="num">${r.point_index}</td>
        <td><a href="/?idx=${r.idx}" target="_blank">${r.idx}</a></td>
        <td><button class="patch-btn" data-idx="${r.idx}" style="font-size:0.75rem;padding:2px 8px">パッチ表示</button></td>
      </tr>
      <tr class="patch-row" id="patch-row-${r.idx}" style="display:none">
        <td colspan="11" style="background:var(--bg-card);padding:8px 16px">
          <img id="patch-img-${r.idx}" class="patch-img" alt="パッチ" />
        </td>
      </tr>`;
    }).join('');
    return `
      <div class="section-title">${title}</div>
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr><th>#</th><th>誤差(t)</th><th>誤差(px)</th><th>true_t_raw</th><th>pred_t_raw</th>
            <th>pq_dist</th><th>part</th><th>sub</th><th>pt</th><th>idx</th><th></th></tr>
          </thead>
          <tbody>${rows_}</tbody>
        </table>
      </div>`;
  }

  const topErrHtml = errTableHtml(`誤差上位 ${d.top_errors.length} 件（t値順）`, d.top_errors)
    + (d.top_errors_px ? errTableHtml(`誤差上位 ${d.top_errors_px.length} 件（px順）`, d.top_errors_px) : '');

  html('errors-content', (selectorHtml || '') + summaryCards + breakdownHtml + scatterHtml + extraChartsHtml + histHtml + topErrHtml);

  // セレクターイベント
  const sel = document.getElementById('err-exp-select');
  if (sel) sel.addEventListener('change', onErrExpChange);
  const splitSel = document.getElementById('err-split-select');
  if (splitSel) splitSel.addEventListener('change', onErrSplitChange);

  // Draw charts (box/violin plots)
  const raw = b.raw || {};

  // Part Name: boxplot
  const partLabels = Object.keys(raw.by_part_name || b.by_part_name);
  mkBoxplot('err-part', partLabels, [{
    label: '誤差',
    data: partLabels.map(k => (raw.by_part_name || {})[k] || []),
    backgroundColor: 'rgba(244,67,54,0.3)',
    borderColor: 'rgba(244,67,54,0.9)',
    borderWidth: 1.5,
    medianColor: '#ef5350',
  }]);

  // Sub Key: violin
  const rawSub = raw.by_sub_key || {};
  const subLabels = Object.keys(rawSub).length ? Object.keys(rawSub) : Object.keys(b.by_sub_key);
  mkViolin('err-sub', subLabels, [{
    label: '誤差',
    data: subLabels.map(k => rawSub[k] || []),
    backgroundColor: subLabels.map(k => k === 'upper' ? 'rgba(26,120,194,0.4)' : 'rgba(255,152,0,0.4)'),
    borderColor: subLabels.map(k => k === 'upper' ? 'rgba(79,195,247,0.9)' : 'rgba(255,183,77,0.9)'),
    borderWidth: 1.5,
  }]);

  // Point Index: violin
  const rawPt = raw.by_point_index || {};
  const ptLabels = Object.keys(rawPt).length
    ? Object.keys(rawPt).sort()
    : Object.keys(b.by_point_index).sort();
  mkViolin('err-pt', ptLabels.map(k => `P${k}`), [{
    label: '誤差',
    data: ptLabels.map(k => rawPt[k] || []),
    backgroundColor: 'rgba(156,39,176,0.4)',
    borderColor: 'rgba(186,104,200,0.9)',
    borderWidth: 1.5,
  }]);

  // upper/lower × Point Index: grouped boxplot
  const rawSubPt = raw.by_sub_pt || {};
  const errUpper = rawSubPt['upper'] || {};
  const errLower = rawSubPt['lower'] || {};
  const errAllPts = [...new Set([...Object.keys(errUpper), ...Object.keys(errLower)])].sort();
  mkBoxplot('err-sub-pt', errAllPts.map(k => `P${k}`), [
    {
      label: 'upper',
      data: errAllPts.map(k => errUpper[k] || []),
      backgroundColor: 'rgba(26,120,194,0.35)',
      borderColor: 'rgba(79,195,247,0.9)',
      borderWidth: 1.5,
      medianColor: '#4fc3f7',
    },
    {
      label: 'lower',
      data: errAllPts.map(k => errLower[k] || []),
      backgroundColor: 'rgba(255,152,0,0.35)',
      borderColor: 'rgba(255,183,77,0.9)',
      borderWidth: 1.5,
      medianColor: '#ffb74d',
    },
  ]);

  const histBins = d.error_histogram.bins.map(b => b.toFixed(3));
  mkBar('err-hist', histBins, d.error_histogram.counts, { color: 'rgba(244,67,54,0.65)' });

  if (d.error_histogram_px) {
    const histBinsPx = d.error_histogram_px.bins.map(b => b.toFixed(2));
    mkBar('err-hist-px', histBinsPx, d.error_histogram_px.counts, { color: 'rgba(0,188,212,0.65)' });
  }

  // t_true vs t_pred scatter
  const sc = d.scatter_data || {};
  mkScatter('err-scatter-pred', [
    {
      label: 'upper',
      data: sc.upper || [],
      color: 'rgba(79,195,247,0.5)',
    },
    {
      label: 'lower',
      data: sc.lower || [],
      color: 'rgba(255,183,77,0.5)',
    },
  ], { refLine: true, xLabel: 'true_t_raw', yLabel: 'pred_t_raw' });

  // Residual vs t_true scatter
  const rs = d.resid_scatter || {};
  mkScatter('err-scatter-resid', [
    {
      label: 'upper',
      data: rs.upper || [],
      color: 'rgba(79,195,247,0.5)',
    },
    {
      label: 'lower',
      data: rs.lower || [],
      color: 'rgba(255,183,77,0.5)',
    },
  ], { hline: 0, xLabel: 'true_t_raw', yLabel: '残差 (pred_t_raw − true_t_raw)' });

  // Residual histogram (t)
  if (d.residual_histogram) {
    const residBins = d.residual_histogram.bins.map(b => b.toFixed(3));
    mkBar('err-resid-hist', residBins, d.residual_histogram.counts, { color: 'rgba(26,120,194,0.7)' });
  }

  // Residual scatter (px)
  const rsPx = d.resid_scatter_px || {};
  mkScatter('err-scatter-resid-px', [
    { label: 'upper', data: rsPx.upper || [], color: 'rgba(79,195,247,0.5)' },
    { label: 'lower', data: rsPx.lower || [], color: 'rgba(255,183,77,0.5)' },
  ], { hline: 0, xLabel: 'true_t_raw', yLabel: '残差 (px)' });

  // Residual histogram (px)
  if (d.residual_histogram_px) {
    const residBinsPx = d.residual_histogram_px.bins.map(b => b.toFixed(2));
    mkBar('err-resid-hist-px', residBinsPx, d.residual_histogram_px.counts, { color: 'rgba(0,188,212,0.7)' });
  }

  // pq_dist vs 誤差(px) 散布図
  if (d.pq_dist_scatter) {
    mkScatter('err-pqdist-scatter', [
      { label: '誤差(px)', data: d.pq_dist_scatter, color: 'rgba(255,183,77,0.45)' },
    ], { xLabel: 'pq_dist (px)', yLabel: '絶対誤差 (px)' });
  }

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
    const params = new URLSearchParams();
    if (currentExpPath) params.set('exp_path', currentExpPath);
    if (currentSplit) params.set('split', currentSplit);
    const patchUrl = `/api/dashboard/errors/${idx}/patch?${params.toString()}`;
    const res = await fetch(patchUrl);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const b64 = await res.text();
    img.src = `data:image/png;base64,${b64}`;
  } catch (e) {
    img.alt = `エラー: ${e.message}`;
  }
}

function onErrExpChange(e) {
  currentExpPath = e.target.value;
  loaded.delete('errors');
  loadErrors();
}

function onErrSplitChange(e) {
  currentSplit = e.target.value;
  loaded.delete('errors');
  loadErrors();
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
