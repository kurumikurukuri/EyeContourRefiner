// t_pred hints browser フロントエンド（読み取り専用、スキャン機能なし）

const state = {
  limit: 50,
  offset: 0,
};

function qs(id) {
  return document.getElementById(id);
}

async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}`);
  }
  return resp.json();
}

async function loadStats() {
  const stats = await fetchJson("/api/stats");
  qs("stat-total").textContent = stats.total;
  qs("stat-extreme").textContent = stats.extreme;
}

function buildQueryParams() {
  const params = new URLSearchParams();
  const minTMax = qs("f-min-t-max").value;
  const maxTMin = qs("f-max-t-min").value;
  const minNExtreme = qs("f-min-n-extreme").value;
  if (minTMax !== "") params.set("min_t_max", minTMax);
  if (maxTMin !== "") params.set("max_t_min", maxTMin);
  if (minNExtreme !== "") params.set("min_n_extreme", minNExtreme);
  params.set("sort_by", qs("f-sort-by").value);
  params.set("order", qs("f-order").value);
  params.set("limit", state.limit);
  params.set("offset", state.offset);
  return params;
}

function renderRow(row) {
  const tr = document.createElement("tr");

  const tdImage = document.createElement("td");
  tdImage.textContent = row.image_path;
  tr.appendChild(tdImage);

  const tdPart = document.createElement("td");
  tdPart.textContent = row.part_name;
  tr.appendChild(tdPart);

  const tdMin = document.createElement("td");
  tdMin.textContent = row.min_t_pred?.toFixed?.(3) ?? row.min_t_pred;
  tr.appendChild(tdMin);

  const tdMax = document.createElement("td");
  tdMax.textContent = row.max_t_pred?.toFixed?.(3) ?? row.max_t_pred;
  tr.appendChild(tdMax);

  const tdNExt = document.createElement("td");
  tdNExt.textContent = row.n_extreme;
  tr.appendChild(tdNExt);

  const tdNPts = document.createElement("td");
  tdNPts.textContent = row.n_points;
  tr.appendChild(tdNPts);

  const tdPreview = document.createElement("td");
  tdPreview.className = "crop-preview";
  tdPreview.textContent = "読込中...";
  tr.appendChild(tdPreview);

  const tdLink = document.createElement("td");
  const a = document.createElement("a");
  a.href = row.annotator_url;
  a.target = "_blank";
  a.className = "annotator-link";
  a.textContent = "開く";
  tdLink.appendChild(a);
  tr.appendChild(tdLink);

  loadPreview(row, tdPreview);

  return tr;
}

async function loadPreview(row, container) {
  try {
    const params = new URLSearchParams({
      image_path: row.image_path,
      part_name: row.part_name,
    });
    const resp = await fetch(`/api/hints/crop?${params}`);
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const b64 = await resp.text();
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${b64}`;
    container.innerHTML = "";
    container.appendChild(img);
  } catch (err) {
    container.textContent = `読み込み失敗: ${err.message}`;
  }
}

async function loadHints() {
  const params = buildQueryParams();
  const body = await fetchJson(`/api/hints?${params}`);
  const tbody = qs("hints-tbody");
  tbody.innerHTML = "";
  for (const row of body.data) {
    tbody.appendChild(renderRow(row));
  }
  const { total, limit, offset } = body.meta;
  const page = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));
  qs("page-info").textContent = `${page} / ${totalPages} (全 ${total} 件)`;
  qs("btn-prev").disabled = offset <= 0;
  qs("btn-next").disabled = offset + limit >= total;
}

function applyFilters() {
  state.offset = 0;
  loadHints();
}

qs("btn-apply").addEventListener("click", applyFilters);
qs("btn-prev").addEventListener("click", () => {
  state.offset = Math.max(0, state.offset - state.limit);
  loadHints();
});
qs("btn-next").addEventListener("click", () => {
  state.offset += state.limit;
  loadHints();
});

loadStats();
loadHints();
