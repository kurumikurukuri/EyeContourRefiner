"use strict";

// ---------------------------------------------------------------------------
// Constants — annotator の物理 t 座標系 (utils/patch_utils.py T_MIN/T_MAX と同値)
// ---------------------------------------------------------------------------
const ANNOTATOR_T_MIN = -1.0;
const ANNOTATOR_T_MAX =  2.0;

/** t_norm [0,1] → 物理 t 値 (PQ 座標系) */
function toPhysicalT(tNorm) {
  return ANNOTATOR_T_MIN + tNorm * (ANNOTATOR_T_MAX - ANNOTATOR_T_MIN);
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  currentIdx: 0,
  total: 0,
  sampleMeta: [],  // index → {image_path, part_name}
  annotated: 0,
  patchW: 384,   // display width  (updated from API)
  patchH: 192,   // display height (updated from API)
  px: 0, py: 0,
  qx: 0, qy: 0,
  currentT: null,       // null = not yet clicked
  existingT: null,      // value already saved on server
  tPred: null,          // model prediction (null if no model)
  labelSource: null,    // 後方互換用（旧フィールド）
  imageB64: null,
  faceB64: null,
  patchB64: null,
  annotationStatus: [],  // index → true/false (アノテーション済みか)
  skipFlags: [],          // index → true/false (skip_flag)
  skipFlag: false,        // 現在のサンプルの skip_flag
  imageKey: null,          // 現在表示中の画像ファイル名
  // ギャラリーページネーション
  galleryPage: 1,
  galleryPerPage: 80,
  galleryTotalPages: 1,
  galleryTotalImages: 0,
  gallerySkipFilter: "all",  // "all" | "active" | "skipped"
  gallerySortBy: "idx",      // "idx" | "eye_scale_asc" | "eye_scale_desc" | "norm_eye_scale_asc" | "norm_eye_scale_desc"
  sortedIdxList: null,       // null = idx順; ソート時はサンプルidxの配列
};

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------
const fullImg     = document.getElementById("full-img");
const faceImg     = document.getElementById("face-img");
const patchCanvas = document.getElementById("patch-canvas");
const patchCtx    = patchCanvas.getContext("2d");

const elProgress  = document.getElementById("progress");
const elTDisplay  = document.getElementById("t-display");
const elInfoPart  = document.getElementById("info-part");
const elInfoSub   = document.getElementById("info-sub");
const elInfoP     = document.getElementById("info-p");
const elInfoQ     = document.getElementById("info-q");
const elInfoR     = document.getElementById("info-r");
const elInfoPred  = document.getElementById("info-pred");
const elStatus    = document.getElementById("status-msg");
const elBanner    = document.getElementById("complete-banner");

const btnPrevEyes        = document.getElementById("btn-prev-eyes");
const btnNextEyes        = document.getElementById("btn-next-eyes");

const btnPrev            = document.getElementById("btn-prev");
const btnSkip            = document.getElementById("btn-skip");
const btnNext            = document.getElementById("btn-next");
const btnNextUnannotated = document.getElementById("btn-next-unannotated");
const btnSkipImage       = document.getElementById("btn-skip-image");
const inputJump          = document.getElementById("input-jump");
const btnJump            = document.getElementById("btn-jump");
const btnUndo            = document.getElementById("btn-undo");
const btnSave            = document.getElementById("btn-save");
const btnApplyPred       = document.getElementById("btn-apply-pred");
const btnApplyPredEye    = document.getElementById("btn-apply-pred-eye");
const btnResetEye        = document.getElementById("btn-reset-eye");
const btnModel           = document.getElementById("btn-model");
const modelPanel         = document.getElementById("model-panel");
const modelList          = document.getElementById("model-list");
const modelStatus        = document.getElementById("model-status");
const btnModelApply      = document.getElementById("btn-model-apply");

// Eyeliner preview
const eyeTrueImg  = document.getElementById("eye-true-img");
const eyePredImg  = document.getElementById("eye-pred-img");
const eyeTruePh   = document.getElementById("eye-true-ph");
const eyePredPh   = document.getElementById("eye-pred-ph");
const eyeTrueLabel = document.getElementById("eye-true-label");
const eyePredLabel = document.getElementById("eye-pred-label");

const elEyelinerLoading = document.getElementById("eyeliner-loading");

const btnGallery         = document.getElementById("btn-gallery");
const galleryOverlay     = document.getElementById("gallery-overlay");
const galleryGrid        = document.getElementById("gallery-grid");
const btnGalleryClose    = document.getElementById("btn-gallery-close");

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function apiFetch(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Samples summary (annotation status cache)
// ---------------------------------------------------------------------------
async function loadSamplesSummary() {
  try {
    const data = await apiFetch("/api/samples");
    state.annotationStatus = data.samples.map(s => s.t !== null);
    state.skipFlags = data.samples.map(s => s.skip_flag === true);
    state.sampleMeta = data.samples.map(s => ({
      image_path: s.image_path,
      part_name: s.part_name,
    }));
  } catch (err) {
    // サマリー取得失敗は非致命的（未アノテーションジャンプ機能のみ無効化）
  }
}

// ---------------------------------------------------------------------------
// Sort order helpers
// ---------------------------------------------------------------------------

/** ソート済みリスト上での currentIdx の位置を返す。リストがなければ currentIdx そのものを返す。 */
function _sortedPos() {
  if (!state.sortedIdxList) return state.currentIdx;
  const pos = state.sortedIdxList.indexOf(state.currentIdx);
  return pos === -1 ? 0 : pos;
}

/** ソート済みリスト上でのリスト位置 pos から eye group の先頭位置を返す。 */
function _eyeGroupStartInList(pos) {
  const list = state.sortedIdxList;
  const meta = state.sampleMeta;
  const { image_path, part_name } = meta[list[pos]];
  while (pos > 0 &&
         meta[list[pos - 1]].image_path === image_path &&
         meta[list[pos - 1]].part_name === part_name) {
    pos--;
  }
  return pos;
}

/** idx順でのeye group先頭 idx を返す（ソートなし時に使用）。 */
function _eyeGroupStart(idx) {
  const meta = state.sampleMeta;
  if (!meta.length) return idx;
  const { image_path, part_name } = meta[idx];
  while (idx > 0 &&
         meta[idx - 1].image_path === image_path &&
         meta[idx - 1].part_name === part_name) {
    idx--;
  }
  return idx;
}

function jumpToNextEye() {
  const meta = state.sampleMeta;
  if (!meta.length) { advance(14); return; }
  const { image_path, part_name } = meta[state.currentIdx];

  if (state.sortedIdxList) {
    const list = state.sortedIdxList;
    let pos = _sortedPos();
    // 現在のeye groupを末尾まで読み飛ばす
    while (pos < list.length && meta[list[pos]].image_path === image_path && meta[list[pos]].part_name === part_name) pos++;
    // 次のeye groupを探す（skip済みグループはまるごと飛ばす）
    while (pos < list.length) {
      if (!state.skipFlags[list[pos]]) { loadSample(list[pos]); return; }
      // skipグループを末尾まで読み飛ばす
      const sg = meta[list[pos]];
      while (pos < list.length && meta[list[pos]].image_path === sg.image_path && meta[list[pos]].part_name === sg.part_name) pos++;
    }
    setStatus("次の目はありません");
    return;
  }

  for (let i = state.currentIdx + 1; i < state.total; i++) {
    if (meta[i].image_path !== image_path || meta[i].part_name !== part_name) {
      if (state.skipFlags[i]) continue;
      loadSample(i);
      return;
    }
  }
  setStatus("次の目はありません");
}

function jumpToPrevEye() {
  const meta = state.sampleMeta;
  if (!meta.length) { advance(-14); return; }

  if (state.sortedIdxList) {
    const list = state.sortedIdxList;
    let pos = _eyeGroupStartInList(_sortedPos());
    if (pos === 0) { setStatus("前の目はありません"); return; }
    pos--;
    // skip済みグループはまるごと飛ばして前へ
    while (pos >= 0) {
      const groupStart = _eyeGroupStartInList(pos);
      if (!state.skipFlags[list[groupStart]]) { loadSample(list[groupStart]); return; }
      if (groupStart === 0) break;
      pos = groupStart - 1;
    }
    setStatus("前の目はありません");
    return;
  }

  const groupStart = _eyeGroupStart(state.currentIdx);
  if (groupStart === 0) { setStatus("前の目はありません"); return; }
  let candidate = _eyeGroupStart(groupStart - 1);
  while (candidate > 0 && state.skipFlags[candidate]) {
    const prevStart = _eyeGroupStart(candidate - 1);
    if (prevStart === candidate) break;
    candidate = prevStart;
  }
  if (state.skipFlags[candidate]) { setStatus("前の目はありません"); return; }
  loadSample(candidate);
}

function jumpToNextUnannotated() {
  if (state.sortedIdxList) {
    const list = state.sortedIdxList;
    const pos = _sortedPos();
    for (let p = pos + 1; p < list.length; p++) {
      const i = list[p];
      if (!state.annotationStatus[i] && !state.skipFlags[i]) { loadSample(i); return; }
    }
    for (let p = 0; p <= pos; p++) {
      const i = list[p];
      if (!state.annotationStatus[i] && !state.skipFlags[i]) { loadSample(i); return; }
    }
    setStatus("全サンプルのアノテーションが完了しています！");
    return;
  }

  for (let i = state.currentIdx + 1; i < state.total; i++) {
    if (!state.annotationStatus[i] && !state.skipFlags[i]) { loadSample(i); return; }
  }
  for (let i = 0; i < state.currentIdx; i++) {
    if (!state.annotationStatus[i] && !state.skipFlags[i]) { loadSample(i); return; }
  }
  setStatus("全サンプルのアノテーションが完了しています！");
}

// ---------------------------------------------------------------------------
// Load sample
// ---------------------------------------------------------------------------
async function loadSample(idx) {
  setStatus("読み込み中...");
  try {
    const data = await apiFetch(`/api/sample/${idx}`);

    state.currentIdx  = idx;
    state.total       = data.total;
    state.annotated   = data.annotated;
    state.px = data.px; state.py = data.py;
    state.qx = data.qx; state.qy = data.qy;
    state.existingT   = data.existing_t;
    state.currentT    = data.existing_t;  // start with saved value
    state.tPred    = data.t_pred ?? null;
    state.skipFlag = data.skip_flag ?? false;
    state.imageB64    = data.image_b64;
    state.faceB64     = data.face_b64;
    state.patchB64    = data.patch_b64;
    state.patchW      = data.patch_display_w;
    state.patchH      = data.patch_display_h;
    // ファイル名のみ（ギャラリーでの現在画像ハイライト用）
    state.imageKey    = data.image_path ? data.image_path.split(/[\\/]/).pop() : null;

    elInfoPart.textContent = data.part_name;
    elInfoSub.textContent  = `${data.sub_key} [${data.point_index}]`;
    elInfoP.textContent    = `(${data.px.toFixed(1)}, ${data.py.toFixed(1)})`;
    elInfoQ.textContent    = `(${data.qx.toFixed(1)}, ${data.qy.toFixed(1)})`;

    updateProgress();
    updateLabelSourceBadge();
    updateSkipImageButton();
    if (state.imageB64) fullImg.src = "data:image/jpeg;base64," + state.imageB64;
    if (state.faceB64)  faceImg.src = "data:image/jpeg;base64," + state.faceB64;
    drawPatch();
    updateTDisplay();
    updatePredDisplay();
    setStatus("");

    // アイラインプレビューを非同期で更新（重い処理なので分離）
    loadEyelinerPreview(idx);

    if (state.sortedIdxList) {
      const pos = _sortedPos();
      btnPrev.disabled = (pos <= 0);
      btnNext.disabled = (pos >= state.sortedIdxList.length - 1);
      btnPrevEyes.disabled = (pos <= 0);
      btnNextEyes.disabled = (pos >= state.sortedIdxList.length - 1);
    } else {
      btnPrev.disabled = (idx <= 0);
      btnNext.disabled = (idx >= state.total - 1);
      btnPrevEyes.disabled = (idx <= 0);
      btnNextEyes.disabled = (idx >= state.total - 1);
    }

    if (state.annotated >= state.total) {
      elBanner.style.display = "block";
    } else {
      elBanner.style.display = "none";
    }
  } catch (err) {
    elProgress.textContent = "読み込みエラー";
    setStatus("エラー: " + err.message);
  }
}

// ---------------------------------------------------------------------------
// Draw helpers
// ---------------------------------------------------------------------------

function drawPatch() {
  if (!state.patchB64) return;
  patchCanvas.width  = state.patchW;
  patchCanvas.height = state.patchH;

  const img = new Image();
  img.onload = () => {
    patchCtx.drawImage(img, 0, 0);
    drawPredLine();
    drawTLine();
  };
  img.src = "data:image/png;base64," + state.patchB64;
}

function drawPredLine() {
  if (state.tPred === null) return;
  const x = Math.round((1.0 - state.tPred) * (state.patchW - 1));
  patchCtx.save();
  patchCtx.strokeStyle = "#ff9800";
  patchCtx.lineWidth   = 2;
  patchCtx.setLineDash([6, 4]);
  patchCtx.beginPath();
  patchCtx.moveTo(x, 0);
  patchCtx.lineTo(x, state.patchH);
  patchCtx.stroke();
  patchCtx.restore();
}

function drawTLine() {
  if (state.currentT === null) return;
  const x = Math.round((1.0 - state.currentT) * (state.patchW - 1));
  patchCtx.save();
  patchCtx.strokeStyle = "#00e676";
  patchCtx.lineWidth   = 2;
  patchCtx.setLineDash([4, 3]);
  patchCtx.beginPath();
  patchCtx.moveTo(x, 0);
  patchCtx.lineTo(x, state.patchH);
  patchCtx.stroke();
  patchCtx.restore();
}

function updateTDisplay() {
  if (state.currentT === null) {
    elTDisplay.textContent = "t = --";
    elInfoR.textContent = "--";
  } else {
    // state.currentT は t_norm [0,1]。表示は物理 t 値で行う
    const tPhys = toPhysicalT(state.currentT);
    elTDisplay.textContent = `t = ${tPhys.toFixed(4)}`;
    // R 座標: R = t_phys * P + (1 - t_phys) * Q
    const rx = tPhys * state.px + (1 - tPhys) * state.qx;
    const ry = tPhys * state.py + (1 - tPhys) * state.qy;
    elInfoR.textContent = `(${rx.toFixed(1)}, ${ry.toFixed(1)})`;
  }
}

function updatePredDisplay() {
  if (state.tPred === null) {
    elInfoPred.textContent = "-- (モデルなし)";
    btnApplyPred.disabled = true;
    btnApplyPredEye.disabled = true;
  } else {
    // state.tPred は t_norm [0,1]。表示は物理 t 値で行う
    elInfoPred.textContent = toPhysicalT(state.tPred).toFixed(4);
    btnApplyPred.disabled = false;
    btnApplyPredEye.disabled = false;
  }
}

function updateProgress() {
  elProgress.textContent =
    `${state.annotated} / ${state.total}  (現在: ${state.currentIdx})`;
}

function updateLabelSourceBadge() {
  // label_source は削除済み — badge 要素が残っていれば非表示にする
  const el = document.getElementById("label-source-badge");
  if (el) el.style.display = "none";
  const confEl = document.getElementById("label-confidence");
  if (confEl) confEl.style.display = "none";
}

function setStatus(msg) {
  elStatus.textContent = msg;
}

// ---------------------------------------------------------------------------
// Skip image toggle
// ---------------------------------------------------------------------------
function updateSkipImageButton() {
  if (state.skipFlag) {
    btnSkipImage.textContent = "🚫 スキップ解除";
    btnSkipImage.classList.add("skipped");
  } else {
    btnSkipImage.textContent = "🚫 スキップ画像";
    btnSkipImage.classList.remove("skipped");
  }
}

btnSkipImage.addEventListener("click", async () => {
  btnSkipImage.disabled = true;
  try {
    const newSkip = !state.skipFlag;
    const result = await apiFetch(`/api/skip_image/${state.currentIdx}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ skip: newSkip }),
    });
    state.skipFlag = newSkip;
    // 同画像の全サンプルの skipFlags を更新
    const currentMeta = state.sampleMeta[state.currentIdx];
    if (currentMeta) {
      for (let i = 0; i < state.total; i++) {
        if (state.sampleMeta[i] && state.sampleMeta[i].image_path === currentMeta.image_path) {
          state.skipFlags[i] = newSkip;
        }
      }
    }
    updateSkipImageButton();
    setStatus(newSkip
      ? `✅ この画像をスキップ対象にしました (${result.updated} サンプル)`
      : `✅ この画像のスキップを解除しました (${result.updated} サンプル)`
    );
  } catch (err) {
    setStatus("スキップ設定エラー: " + err.message);
  } finally {
    btnSkipImage.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Patch canvas click → compute t
// ---------------------------------------------------------------------------
patchCanvas.addEventListener("click", (e) => {
  const rect = patchCanvas.getBoundingClientRect();
  // getBoundingClientRect reflects CSS size; canvas coordinates are in logical pixels
  const scaleX = patchCanvas.width  / rect.width;
  const clickX = (e.clientX - rect.left) * scaleX;

  const t = Math.round((1.0 - clickX / (state.patchW - 1)) * 1000) / 1000;
  state.currentT = Math.max(0.0, Math.min(1.0, t));

  drawPatch();      // redraw base + lines (drawPredLine/drawTLine は onload 内で実行)
  updateTDisplay();
});

// ---------------------------------------------------------------------------
// Buttons
// ---------------------------------------------------------------------------
btnSave.addEventListener("click", async () => {
  if (state.currentT === null) {
    setStatus("先にアイライン境界をクリックしてください。");
    return;
  }
  await saveCurrent();
  advance(1);
});

btnSkip.addEventListener("click", () => {
  advance(1);
});

btnPrev.addEventListener("click", () => {
  advance(-1);
});

btnNext.addEventListener("click", () => {
  advance(1);
});

btnPrevEyes.addEventListener("click", jumpToPrevEye);
btnNextEyes.addEventListener("click", jumpToNextEye);

btnUndo.addEventListener("click", () => {
  state.currentT = state.existingT;
  drawPatch();
  drawTLine();
  updateTDisplay();
  setStatus("やり直しました。");
});

btnApplyPred.addEventListener("click", () => {
  if (state.tPred === null) return;
  state.currentT = state.tPred;
  drawPatch();
  updateTDisplay();
  setStatus(`予測値を適用しました (t=${toPhysicalT(state.tPred).toFixed(4)})`);
});

btnApplyPredEye.addEventListener("click", async () => {
  if (state.tPred === null) return;
  btnApplyPredEye.disabled = true;
  btnApplyPredEye.textContent = "適用中...";
  try {
    const result = await apiFetch(`/api/apply_pred_eye/${state.currentIdx}`, { method: "POST" });
    if (result.updated === 0) {
      setStatus("未ラベルのサンプルはありません（目全体適用済み）");
    } else {
      setStatus(`✅ ${result.updated} 点に予測を適用しました`);
      await loadSamplesSummary();
      await loadSample(state.currentIdx);
    }
  } catch (err) {
    setStatus("一括適用エラー: " + err.message);
  } finally {
    btnApplyPredEye.disabled = (state.tPred === null);
    btnApplyPredEye.textContent = "目全体に予測を適用";
  }
});

btnResetEye.addEventListener("click", async () => {
  if (!confirm("この目の全アノテーションをNullにリセットしますか？")) return;
  btnResetEye.disabled = true;
  btnResetEye.textContent = "リセット中...";
  try {
    const result = await apiFetch(`/api/reset_eye/${state.currentIdx}`, { method: "POST" });
    setStatus(`🗑 ${result.updated} 点をリセットしました`);
    await loadSamplesSummary();
    await loadSample(state.currentIdx);
  } catch (err) {
    setStatus("リセットエラー: " + err.message);
  } finally {
    btnResetEye.disabled = false;
    btnResetEye.textContent = "目をリセット";
  }
});

btnNextUnannotated.addEventListener("click", jumpToNextUnannotated);

btnJump.addEventListener("click", () => {
  const idx = parseInt(inputJump.value, 10);
  if (!Number.isFinite(idx) || idx < 0 || idx >= state.total) {
    setStatus(`インデックスは 0〜${state.total - 1} の範囲で入力してください。`);
    return;
  }
  loadSample(idx);
  inputJump.value = "";
});

inputJump.addEventListener("keydown", (e) => {
  if (e.key === "Enter") btnJump.click();
});

function advance(delta) {
  if (state.sortedIdxList) {
    const list = state.sortedIdxList;
    let pos = _sortedPos() + delta;
    while (pos >= 0 && pos < list.length && state.skipFlags[list[pos]]) {
      pos += delta;
    }
    if (pos >= 0 && pos < list.length) {
      loadSample(list[pos]);
    }
    return;
  }
  let next = state.currentIdx + delta;
  while (next >= 0 && next < state.total && state.skipFlags[next]) {
    next += delta;
  }
  if (next >= 0 && next < state.total) {
    loadSample(next);
  }
}

async function saveCurrent() {
  try {
    await apiFetch(`/api/annotate/${state.currentIdx}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ t: state.currentT }),
    });
    state.existingT = state.currentT;
    if (!state.annotationStatus[state.currentIdx]) {
      state.annotationStatus[state.currentIdx] = true;
      state.annotated = Math.min(state.annotated + 1, state.total);
    }
    updateProgress();
    setStatus(`✅ 保存しました (idx=${state.currentIdx}, t=${toPhysicalT(state.currentT).toFixed(4)})`);
  } catch (err) {
    setStatus("保存エラー: " + err.message);
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  switch (e.key) {
    case "ArrowRight": advance(1);  break;
    case "ArrowLeft":  advance(-1); break;
    case "s": case "S": advance(1); break;  // skip
    case "n": case "N": jumpToNextUnannotated(); break;  // next unannotated
    case "g": case "G": openGallery(); break;             // gallery
    case "x": case "X": btnSkipImage.click(); break;     // toggle skip image
    case "Enter":
      if (state.currentT !== null) {
        saveCurrent().then(() => advance(1));
      }
      break;
  }
});

// ---------------------------------------------------------------------------
// Model selector (フォルダ選択 UI)
// ---------------------------------------------------------------------------
let _folderData = { folders: [], active_dir: null, loaded_count: 0 };
let _selectedFolder = null;

async function loadModelList() {
  try {
    _folderData = await apiFetch("/api/models/folders");
    _selectedFolder = _folderData.active_dir;
    renderModelPanel(_folderData);
  } catch (err) {
    modelStatus.textContent = "モデル一覧の取得に失敗しました";
  }
}

function renderModelPanel(data) {
  const { folders, active_dir, loaded_count, infer_mode } = data;

  if (loaded_count > 0) {
    const modeLabel = infer_mode === "exp_native" ? " [EXP-native]" : "";
    modelStatus.textContent = `${loaded_count} モデル読み込み中 (アンサンブル)${modeLabel}`;
  } else {
    modelStatus.textContent = "モデルなし（推論補助 無効）";
  }

  if (folders.length === 0) {
    modelList.innerHTML = '<span id="model-none-msg">.pt ファイルが見つかりません</span>';
    return;
  }

  modelList.innerHTML = "";
  for (const f of folders) {
    const isActive = f.path === active_dir;
    const isSelected = f.path === _selectedFolder;

    let badgeText = "";
    if (f.fold_count > 0) {
      badgeText = `fold×${f.fold_count}`;
    } else if (f.has_ensemble) {
      badgeText = "ensemble";
    } else {
      badgeText = `${f.file_count} ファイル`;
    }

    const nativeBadge = f.has_infer_script
      ? '<span style="font-size:10px;background:#1565c0;color:#fff;border-radius:3px;padding:1px 4px;margin-left:4px;">EXP-native</span>'
      : "";

    const item = document.createElement("div");
    item.className = "folder-item" + (isSelected ? " selected" : "");
    item.dataset.path = f.path;
    item.innerHTML = `
      <span class="folder-icon">📁</span>
      <div style="flex:1; min-width:0;">
        <div class="folder-name">${f.name}${nativeBadge}</div>
        <div class="folder-path">${f.path}</div>
      </div>
      <span class="folder-badge${isActive ? " active" : ""}">${badgeText}</span>
    `;
    item.addEventListener("click", () => {
      _selectedFolder = f.path;
      modelList.querySelectorAll(".folder-item").forEach(el => el.classList.remove("selected"));
      item.classList.add("selected");
    });
    modelList.appendChild(item);
  }
}

btnModel.addEventListener("click", (e) => {
  e.stopPropagation();
  modelPanel.classList.toggle("open");
  if (modelPanel.classList.contains("open")) {
    loadModelList();
  }
});

document.addEventListener("click", (e) => {
  if (!modelPanel.contains(e.target) && e.target !== btnModel) {
    modelPanel.classList.remove("open");
  }
});

btnModelApply.addEventListener("click", async () => {
  if (!_selectedFolder) {
    setStatus("フォルダを選択してください");
    return;
  }
  btnModelApply.disabled = true;
  btnModelApply.textContent = "読み込み中...";
  try {
    const result = await apiFetch("/api/models/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ exp_dir: _selectedFolder }),
    });
    const modeLabel = result.infer_mode === "exp_native" ? " [EXP-native]" : "";
    _folderData = { ..._folderData, active_dir: _selectedFolder, loaded_count: result.loaded_count, infer_mode: result.infer_mode };
    renderModelPanel(_folderData);
    setStatus(`モデルを更新しました (${result.loaded_count} 個)${modeLabel}`);
    // 現在サンプルの予測を更新
    await loadSample(state.currentIdx);
  } catch (err) {
    setStatus("モデル切り替えエラー: " + err.message);
  } finally {
    btnModelApply.disabled = false;
    btnModelApply.textContent = "適用";
  }
});

// ---------------------------------------------------------------------------
// Gallery (paginated)
// ---------------------------------------------------------------------------
async function openGallery() {
  galleryOverlay.classList.add("open");
  // 現在の画像が含まれるページを特定
  if (state.imageKey) {
    try {
      const data = await apiFetch(`/api/gallery/find_page?image_key=${encodeURIComponent(state.imageKey)}&per_page=${state.galleryPerPage}`);
      state.galleryPage = data.page;
    } catch {
      state.galleryPage = 1;
    }
  } else {
    state.galleryPage = 1;
  }
  await loadGalleryPage();
}

async function loadGalleryPage() {
  galleryGrid.innerHTML = "";
  const loadingDiv = document.createElement("div");
  loadingDiv.style.cssText = "color:#888; text-align:center; padding:20px; grid-column:1/-1;";
  loadingDiv.textContent = "読み込み中...";
  galleryGrid.appendChild(loadingDiv);

  try {
    const data = await apiFetch(`/api/gallery?page=${state.galleryPage}&per_page=${state.galleryPerPage}&skip_filter=${state.gallerySkipFilter}&sort_by=${state.gallerySortBy}`);
    state.galleryPage = data.page;
    state.galleryTotalPages = data.total_pages;
    state.galleryTotalImages = data.total_images;
    renderGalleryPage(data.items);
    renderPaginationControls();
  } catch (err) {
    const errDiv = document.createElement("div");
    errDiv.style.cssText = "color:#f44336; padding:20px; grid-column:1/-1;";
    errDiv.textContent = `エラー: ${err.message}`;
    galleryGrid.innerHTML = "";
    galleryGrid.appendChild(errDiv);
  }
}

function renderGalleryPage(items) {
  galleryGrid.innerHTML = "";

  if (items.length === 0) {
    const emptyDiv = document.createElement("div");
    emptyDiv.style.cssText = "color:#888; text-align:center; padding:20px; grid-column:1/-1;";
    emptyDiv.textContent = "画像が登録されていません";
    galleryGrid.appendChild(emptyDiv);
    return;
  }

  let currentCard = null;

  for (const item of items) {
    const statusClass =
      item.annotated === item.total ? "done" :
      item.annotated > 0           ? "partial" :
                                     "none";
    const isCurrent = item.image_key === state.imageKey;

    const card = document.createElement("div");
    card.className = `gallery-card ${statusClass}${isCurrent ? " current" : ""}${item.skipped ? " skipped" : ""}`;

    // サムネイル画像（個別API経由）
    const img = document.createElement("img");
    img.src = `/api/thumbnail?image_path=${encodeURIComponent(item.image_path)}`;
    img.alt = item.image_key;
    img.loading = "lazy";

    // 進捗バッジ（画像右上に重ねて表示）
    const badge = document.createElement("div");
    badge.className = "gallery-badge";
    badge.textContent = `${item.annotated}/${item.total}`;

    // 完了マーク（画像左上）
    const doneMark = document.createElement("div");
    doneMark.className = "gallery-done-mark";
    doneMark.textContent = "✓ 完了";

    // スキップバッジ（画像左上、完了マークと重複しないよう下にずらす）
    if (item.skipped) {
      const skipBadge = document.createElement("div");
      skipBadge.className = "gallery-skip-badge";
      skipBadge.textContent = "SKIP";
      card.appendChild(skipBadge);
    }

    // ファイル名ラベル（下部）
    const label = document.createElement("div");
    label.className = "gallery-label";
    label.textContent = item.image_key.replace(/\.[^.]+$/, "");

    // eye_scale バッジ（ソート時に表示）
    if (item.eye_scale != null && state.gallerySortBy !== "idx") {
      const scaleBadge = document.createElement("div");
      scaleBadge.className = "gallery-eye-scale";
      if (state.gallerySortBy.startsWith("norm_eye_scale") && item.norm_eye_scale != null) {
        scaleBadge.textContent = `nes=${item.norm_eye_scale.toFixed(3)}`;
      } else {
        scaleBadge.textContent = `es=${Math.round(item.eye_scale)}px`;
      }
      card.appendChild(scaleBadge);
    }

    card.appendChild(img);
    card.appendChild(badge);
    card.appendChild(doneMark);
    card.appendChild(label);

    card.addEventListener("click", () => {
      galleryOverlay.classList.remove("open");
      loadSample(item.first_idx);
    });
    galleryGrid.appendChild(card);

    if (isCurrent) currentCard = card;
  }

  // 現在の画像カードを表示領域内にスクロール
  if (currentCard) {
    requestAnimationFrame(() => {
      currentCard.scrollIntoView({ block: "center", behavior: "smooth" });
    });
  }
}

function renderPaginationControls() {
  let paginationDiv = document.getElementById("gallery-pagination");
  if (!paginationDiv) {
    paginationDiv = document.createElement("div");
    paginationDiv.id = "gallery-pagination";
    const modal = document.getElementById("gallery-modal");
    const grid = document.getElementById("gallery-grid");
    modal.insertBefore(paginationDiv, grid);
  }

  paginationDiv.innerHTML = "";

  const prevBtn = document.createElement("button");
  prevBtn.textContent = "\u2190 前へ";
  prevBtn.disabled = state.galleryPage <= 1;
  prevBtn.addEventListener("click", () => {
    state.galleryPage--;
    loadGalleryPage();
  });

  const pageInfo = document.createElement("span");
  pageInfo.style.cssText = "color:#a8dadc; font-size:0.85rem;";
  pageInfo.textContent = `${state.galleryPage} / ${state.galleryTotalPages} ページ (${state.galleryTotalImages} 画像)`;

  const nextBtn = document.createElement("button");
  nextBtn.textContent = "次へ \u2192";
  nextBtn.disabled = state.galleryPage >= state.galleryTotalPages;
  nextBtn.addEventListener("click", () => {
    state.galleryPage++;
    loadGalleryPage();
  });

  paginationDiv.appendChild(prevBtn);
  paginationDiv.appendChild(pageInfo);
  paginationDiv.appendChild(nextBtn);
}

btnGallery.addEventListener("click", openGallery);

// ギャラリータブ切替
document.querySelectorAll(".gallery-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".gallery-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    state.gallerySkipFilter = tab.dataset.filter;
    state.galleryPage = 1;
    loadGalleryPage();
  });
});

// ギャラリーソート切替
const gallerySortSelect = document.getElementById("gallery-sort-select");
gallerySortSelect.addEventListener("change", async () => {
  state.gallerySortBy = gallerySortSelect.value;
  state.galleryPage = 1;
  await updateSortedOrder();
  loadGalleryPage();
});

async function updateSortedOrder() {
  if (state.gallerySortBy === "idx") {
    state.sortedIdxList = null;
    return;
  }
  try {
    const data = await apiFetch(`/api/sample_order?sort_by=${state.gallerySortBy}`);
    state.sortedIdxList = data.order;
  } catch (err) {
    state.sortedIdxList = null;
    setStatus("ソート順の取得に失敗しました: " + err.message);
  }
}

btnGalleryClose.addEventListener("click", () => {
  galleryOverlay.classList.remove("open");
});

galleryOverlay.addEventListener("click", (e) => {
  if (e.target === galleryOverlay) {
    galleryOverlay.classList.remove("open");
  }
});

// ---------------------------------------------------------------------------
// Eyeliner preview
// ---------------------------------------------------------------------------

function _setEyelinerImg(img, ph, b64, mimeType) {
  if (b64) {
    img.src = `data:${mimeType};base64,${b64}`;
    img.style.display = "block";
    ph.style.display  = "none";
  } else {
    img.src = "";
    img.style.display = "none";
    ph.textContent   = "--";
    ph.style.display  = "inline";
  }
}

function _resetEyelinerPanels(msg) {
  [eyeTrueImg, eyePredImg].forEach(img => {
    img.src = "";
    img.style.display = "none";
  });
  [eyeTruePh, eyePredPh].forEach(ph => {
    ph.textContent   = msg;
    ph.style.display = "inline";
  });
}

async function loadEyelinerPreview(idx) {
  _resetEyelinerPanels("読み込み中...");
  elEyelinerLoading.textContent = "読み込み中...";
  try {
    const data = await apiFetch(`/api/eyeliner_preview/${idx}`);

    // ラベルに現在の側を反映
    const side = data.side === "left" ? "左目" : data.side === "right" ? "右目" : "";
    eyeTrueLabel.textContent = `${side} t_true`;
    eyePredLabel.textContent = `${side} t_pred`;

    _setEyelinerImg(eyeTrueImg, eyeTruePh, data.eye_true_b64, "image/png");

    if (data.has_model) {
      _setEyelinerImg(eyePredImg, eyePredPh, data.eye_pred_b64, "image/png");
    } else {
      eyePredPh.textContent = "モデルなし";
    }

    elEyelinerLoading.textContent = "";
  } catch (err) {
    _resetEyelinerPanels("エラー");
    elEyelinerLoading.textContent = "";
  }
}

// ---------------------------------------------------------------------------
// Open image in Windows Photos
// ---------------------------------------------------------------------------
fullImg.style.cursor = "pointer";
fullImg.title = "クリックで元画像を写真アプリで開く";
fullImg.addEventListener("click", async () => {
  try {
    await apiFetch(`/api/open_image/${state.currentIdx}`, { method: "POST" });
  } catch (err) {
    setStatus("画像を開けませんでした: " + err.message);
  }
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
loadSamplesSummary().then(async () => {
  const params = new URLSearchParams(window.location.search);
  const idxParam = params.get("idx");
  if (idxParam !== null) {
    const fieldIdx = parseInt(idxParam, 10);
    if (!isNaN(fieldIdx)) {
      try {
        const data = await apiFetch(`/api/resolve_idx/${fieldIdx}`);
        loadSample(data.array_position);
        return;
      } catch {
        // idx フィールドが見つからない場合は配列位置として扱う（後方互換）
        loadSample(fieldIdx);
        return;
      }
    }
  }
  loadSample(0);
});
loadModelList();
