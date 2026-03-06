/**
 * dashboard.js — KoreanStocks 대시보드 전체 인터랙션
 */

// ── 유틸 ────────────────────────────────────────────────────────
async function api(url, opts = {}) {
  const res = await fetch(url, opts);
  if (res.status === 204) return null;
  const ct = res.headers.get("content-type") || "";
  if (!ct.includes("json")) {
    const text = await res.text().catch(() => "");
    throw new Error(`서버 오류 (HTTP ${res.status})${text ? ": " + text.slice(0, 120) : ""}`);
  }
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

function fmt(n, digits = 0) {
  if (n == null || isNaN(n)) return "—";
  return Number(n).toLocaleString("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function chgText(v) {
  if (v == null) return "";
  const pct = (v * 100).toFixed(2);
  return (v >= 0 ? "▲ " : "▼ ") + Math.abs(pct) + "%";
}

function chgClass(v) { return (v >= 0) ? "pos" : "neg"; }

function badgeHtml(action) {
  const a = (action || "HOLD").toUpperCase();
  const cls = { BUY: "badge-buy", SELL: "badge-sell", HOLD: "badge-hold" }[a] || "badge-hold";
  return `<span class="badge ${cls}">${a}</span>`;
}

function mktBadge(market) {
  if (!market) return "";
  const cls = market === "KOSPI" ? "mkt-kospi" : "mkt-kosdaq";
  return `<span class="mkt-badge ${cls}">${market}</span>`;
}

function setStatus(id, msg, isErr = false) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
  el.style.color = isErr ? "var(--sell)" : "var(--muted)";
}

function toggleEl(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("open");
}

// ── 테마 토글 ────────────────────────────────────────────────────
function getChartColors() {
  const s = getComputedStyle(document.documentElement);
  return {
    accent: s.getPropertyValue("--chart-accent").trim(),
    text:   s.getPropertyValue("--chart-text").trim(),
    muted:  s.getPropertyValue("--chart-muted").trim(),
    grid:   s.getPropertyValue("--chart-grid").trim(),
    sell:   s.getPropertyValue("--chart-sell").trim(),
  };
}

function toggleTheme() {
  const root = document.documentElement;
  const next = (root.getAttribute("data-theme") || "dark") === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", next);
  localStorage.setItem("ks-theme", next);
  syncThemeBtn();
}

function syncThemeBtn() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  const isDark = (document.documentElement.getAttribute("data-theme") || "dark") === "dark";
  btn.textContent = isDark ? "☀️" : "🌙";
  btn.title = isDark ? "라이트 모드로 전환" : "다크 모드로 전환";
}

// ── 탭 전환 ─────────────────────────────────────────────────────
let modelHealthLoaded = false;

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const panel = document.getElementById(`tab-${btn.dataset.tab}`);
    if (panel) panel.classList.add("active");

    if (btn.dataset.tab === "model" && !modelHealthLoaded) {
      loadModelHealth();
      modelHealthLoaded = true;
    }
    if (btn.dataset.tab === "value" && !_valueLoaded) {
      loadValueDefaults();
      _valueLoaded = true;
    }
  });
});

// ── 모달 ─────────────────────────────────────────────────────────
function openModal(rec) {
  document.getElementById("modal-body").innerHTML = buildModalHtml(rec);
  document.getElementById("rec-modal").classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeModal(e) {
  if (e && e.target !== document.getElementById("rec-modal")) return;
  document.getElementById("rec-modal").classList.add("hidden");
  document.body.style.overflow = "";
}

function buildModalHtml(rec) {
  const ai   = rec.ai_opinion || {};
  const ind  = rec.indicators || {};
  const stats = rec.stats     || {};
  const si   = rec.sentiment_info || {};
  const action = ai.action || "HOLD";

  const techScore = rec.tech_score ?? "—";
  const mlScore   = rec.ml_score ?? "—";
  const sentRaw   = rec.sentiment_score ?? 0;
  const sentNorm  = Math.min(100, Math.max(0, (sentRaw + 100) / 2));

  const upside = (ai.target_price && rec.current_price)
    ? ((ai.target_price - rec.current_price) / rec.current_price * 100).toFixed(1)
    : null;

  const rsiVal  = ind.rsi != null ? ind.rsi : null;
  const macdDir = (ind.macd != null && ind.macd_sig != null)
    ? (ind.macd > ind.macd_sig ? "▲ 골든크로스" : "▼ 데드크로스")
    : "—";
  const bbPos = ind.bb_pos != null ? ind.bb_pos : null;
  const bbLabel = bbPos == null ? "—" : bbPos < 0.3 ? "하단권" : bbPos > 0.7 ? "상단권" : "중간권";

  const avgVol = stats.avg_vol || 1;
  const volRatio = stats.current_vol ? ((stats.current_vol / avgVol) * 100).toFixed(1) : "—";

  const sentLabel = si.sentiment_label || (sentRaw > 0 ? "긍정" : sentRaw < 0 ? "부정" : "중립");
  const sentColor = sentRaw > 0 ? "var(--buy)" : sentRaw < 0 ? "var(--sell)" : "var(--muted)";

  const articles = si.articles || [];
  const topNews  = si.top_news || "";

  const newsHtml = articles.length
    ? articles.slice(0, 8).map(a => {
        const url   = a.originallink || a.link || "";
        const title = a.title || "제목 없음";
        const age   = a.days_ago ? `<span class="news-age">${a.days_ago}</span>` : "";
        return `<div class="news-item">${age}${url
          ? `<a href="${url}" target="_blank" rel="noopener">${title}</a>`
          : title}</div>`;
      }).join("")
    : topNews
      ? `<div class="news-item">${topNews}</div>`
      : `<span style="color:var(--muted);font-size:.85em">뉴스 정보 없음</span>`;

  return `
    <div class="modal-header">
      <div class="flex-row">
        <span class="modal-name">${rec.name || rec.code}</span>
        <span style="color:var(--muted);font-size:.85em">(${rec.code})</span>
        ${mktBadge(rec.market)}
        ${rec.theme && rec.theme !== "전체"
          ? `<span style="color:var(--muted);font-size:.78em">${rec.theme}</span>` : ""}
      </div>
      <div class="flex-row" style="margin-top:6px">
        <span style="font-size:1.2em;font-weight:700">₩${fmt(rec.current_price)}</span>
        <span class="${chgClass(rec.change_pct)}" style="font-size:.9em">
          ${rec.change_pct != null ? (rec.change_pct >= 0 ? "▲" : "▼") + " " + Math.abs(rec.change_pct).toFixed(2) + "%" : ""}
        </span>
        <div class="spacer"></div>
        ${badgeHtml(action)}
      </div>
    </div>

    <div style="margin:10px 0">
      ${scoreBarHtml("기술점수", techScore)}
      ${scoreBarHtml("ML점수",   mlScore)}
      ${scoreBarHtml("감성점수", sentNorm)}
    </div>

    <hr class="divider">

    <div class="modal-body-grid">
      <!-- 좌측: 지표 + 통계 -->
      <div>
        <div class="modal-section-title">📊 기술적 지표</div>
        <div class="kv-row"><span class="kv-key">RSI(14)</span>
          <span class="kv-val">${rsiVal != null ? rsiVal : "—"}</span></div>
        <div class="kv-row"><span class="kv-key">MACD</span>
          <span class="kv-val">${macdDir}</span></div>
        <div class="kv-row"><span class="kv-key">SMA 20</span>
          <span class="kv-val">${ind.sma_20 ? "₩" + fmt(ind.sma_20) : "—"}</span></div>
        <div class="kv-row"><span class="kv-key">BB 위치</span>
          <span class="kv-val">${bbPos != null ? bbPos + " (" + bbLabel + ")" : "—"}</span></div>

        <div class="modal-section-title" style="margin-top:14px">📈 52주 통계</div>
        <div class="kv-row"><span class="kv-key">52주 최고</span>
          <span class="kv-val">${stats.high_52w ? "₩" + fmt(stats.high_52w) : "—"}</span></div>
        <div class="kv-row"><span class="kv-key">52주 최저</span>
          <span class="kv-val">${stats.low_52w ? "₩" + fmt(stats.low_52w) : "—"}</span></div>
        <div class="kv-row"><span class="kv-key">거래량 (vs 평균)</span>
          <span class="kv-val">${volRatio}%</span></div>

        <div class="modal-section-title" style="margin-top:14px">📰 뉴스 심리</div>
        <div style="font-size:.88em">
          <span style="color:${sentColor};font-weight:700">${sentRaw} · ${sentLabel}</span>
        </div>
        ${si.reason ? `<div style="font-size:.78em;color:var(--muted);margin-top:4px">${si.reason}</div>` : ""}
      </div>

      <!-- 우측: AI 분석 -->
      <div>
        <div class="modal-section-title">🤖 AI 분석 요약</div>
        <div style="background:var(--bg-dark);border-radius:6px;padding:10px 12px;font-size:.88em;line-height:1.7;margin-bottom:12px">
          ${ai.summary || "분석 내용 없음"}
        </div>
        ${ai.strength
          ? `<div style="font-size:.85em;margin-bottom:6px">✅ <strong>강점:</strong> ${ai.strength}</div>` : ""}
        ${ai.weakness
          ? `<div style="font-size:.85em;margin-bottom:10px">⚠️ <strong>약점:</strong> ${ai.weakness}</div>` : ""}

        <div class="modal-section-title">📝 상세 추천 사유</div>
        <div style="font-size:.84em;color:var(--muted);line-height:1.7;margin-bottom:12px">
          ${ai.reasoning || "—"}
        </div>

        ${ai.target_price
          ? `<div style="background:rgba(0,212,170,.1);border:1px solid var(--accent);border-radius:6px;padding:10px 14px;font-size:.9em">
              🎯 <strong>목표가(10거래일): ₩${fmt(ai.target_price)}</strong>
              ${upside != null ? `<span class="${upside >= 0 ? "pos" : "neg"}">(${upside >= 0 ? "+" : ""}${upside}%)</span>` : ""}
              ${ai.target_rationale
                ? `<div style="font-size:.78em;color:var(--muted);margin-top:4px">${ai.target_rationale}</div>` : ""}
             </div>` : ""}
      </div>
    </div>

    ${articles.length || topNews ? `
    <hr class="divider">
    <div class="modal-section-title">📰 관련 뉴스 (${articles.length || 1}건)</div>
    ${newsHtml}` : ""}
  `;
}

function scoreBarHtml(label, val) {
  const v = parseFloat(val) || 0;
  const pct = Math.min(100, Math.max(0, v));
  return `
    <div class="score-bar">
      <span class="score-bar-label">${label}</span>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:${pct}%"></div>
      </div>
      <span class="score-bar-val">${isNaN(v) ? "—" : v.toFixed(0)}</span>
    </div>`;
}

function bucketBadge(bucket, label) {
  if (!bucket || !label) return "";
  return `<span class="bucket-badge bucket-${bucket}">${label}</span>`;
}

// ── 추천 카드 렌더링 ─────────────────────────────────────────────
function buildRecRow(rec) {
  const ai     = rec.ai_opinion || {};
  const action = ai.action || "HOLD";
  const score  = calcComposite(rec);

  return `
    <div class="rec-row" onclick="openModal(${JSON.stringify(rec).replace(/"/g, "&quot;")})">
      <div>
        <div class="rec-row-name">${rec.name || rec.code}</div>
        <div class="rec-row-code">${rec.code} ${mktBadge(rec.market)}${bucketBadge(rec.bucket, rec.bucket_label)}
          ${rec.theme && rec.theme !== "전체"
            ? `<span style="font-size:.72em;color:var(--muted);margin-left:4px">[${rec.theme}]</span>` : ""}</div>
      </div>
      <div class="rec-row-score">
        Tech&nbsp;${rec.tech_score ?? "—"} ·
        ML&nbsp;${rec.ml_score ?? "—"} ·
        News&nbsp;${rec.sentiment_score ?? "—"}&nbsp;&nbsp;
        <span style="color:var(--accent)">종합 ${score}</span>
      </div>
      <div class="spacer"></div>
      <div class="rec-row-price">
        <div style="font-weight:700">₩${fmt(rec.current_price)}</div>
        <div class="${chgClass(rec.change_pct)}" style="font-size:.8em">
          ${rec.change_pct != null ? (rec.change_pct >= 0 ? "▲" : "▼") + " " + Math.abs(rec.change_pct).toFixed(2) + "%" : ""}
        </div>
      </div>
      <div class="rec-row-action">${badgeHtml(action)}</div>
      <div style="font-size:.75em;color:var(--muted)">
        목표가 ₩${fmt((rec.ai_opinion || {}).target_price)}
      </div>
    </div>`;
}

function calcComposite(rec) {
  const t = rec.tech_score ?? 50;
  const m = rec.ml_score  ?? 50;
  const s = Math.min(100, Math.max(0, ((rec.sentiment_score ?? 0) + 100) / 2));
  const hasML = (rec.ml_score != null);
  const score = hasML
    ? t * 0.40 + m * 0.35 + s * 0.25
    : t * 0.65 + s * 0.35;
  return score.toFixed(1);
}

// ── 테마 필터 ────────────────────────────────────────────────────
const THEMES = ["전체", "AI/인공지능", "반도체", "이차전지", "제약/바이오", "로봇/자동화"];

function renderThemeFilter(containerId, onChange) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = THEMES.map(t =>
    `<button class="theme-btn${t === "전체" ? " active" : ""}"
       data-theme="${t}" onclick="selectTheme(event,'${containerId}')">${t}</button>`
  ).join("");
  el._onChange = onChange;
}

function selectTheme(e, containerId) {
  const container = document.getElementById(containerId);
  container.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
  e.target.classList.add("active");
  if (container._onChange) container._onChange(e.target.dataset.theme);
}

function filterByTheme(recs, theme) {
  if (theme === "전체") return recs;
  return recs.filter(r => {
    const text = [r.name, r.sector, r.industry, r.theme].filter(Boolean).join(" ");
    return _themeMatch(text, theme) || r.theme === theme;
  });
}

const _THEME_KW = {
  "AI/인공지능":  ["AI", "인공지능", "소프트웨어", "데이터"],
  "반도체":       ["반도체", "장비", "소재", "부품"],
  "이차전지":     ["배터리", "이차전지", "에너지", "화학"],
  "제약/바이오":  ["제약", "바이오", "의료", "생명"],
  "로봇/자동화":  ["로봇", "자동화", "기계", "장비"],
};

function _themeMatch(text, theme) {
  return (_THEME_KW[theme] || []).some(kw => text.includes(kw));
}

// ═══════════════════════════════════════════════════════
// Tab 1 — Dashboard
// ═══════════════════════════════════════════════════════

async function loadMarketIndices() {
  try {
    const data = await api("/api/market");
    renderIndexCard("idx-kospi",  data.KOSPI);
    renderIndexCard("idx-kosdaq", data.KOSDAQ);
    renderUsdCard("idx-usdkrw", data.USDKRW);
  } catch (e) { console.warn("시장 지수 로드 실패:", e.message); }
}

function renderIndexCard(id, info) {
  const el = document.getElementById(id);
  if (!el || !info) return;
  const chg = info.change || 0;
  const cls = chgClass(chg);
  el.querySelector(".idx-val").textContent = fmt(info.close, 2);
  const chgEl = el.querySelector(".idx-chg");
  chgEl.textContent = chgText(chg);
  chgEl.className = `idx-chg ${cls}`;
}

function renderUsdCard(id, info) {
  const el = document.getElementById(id);
  if (!el || !info) return;
  el.querySelector(".idx-val").textContent = fmt(info.close, 2);
}

async function loadPortfolioSummary() {
  try {
    const wl = await api("/api/watchlist");
    const el = document.getElementById("portfolio-summary");
    if (wl.length) {
      el.innerHTML = `현재 <strong style="color:var(--accent)">${wl.length}개</strong> 종목을 감시 중입니다. ` +
        `<a href="#" onclick="switchTab('watchlist')">Watchlist</a>에서 상세 분석을 실행하세요.`;
    } else {
      el.textContent = "Watchlist에 종목을 추가하여 포트폴리오 관리를 시작하세요.";
    }
  } catch (e) {}
}

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach(b => {
    if (b.dataset.tab === tabName) b.click();
  });
}

// 대시보드 날짜 선택 → 추천 로드
let _dashRecs = [];
let _dashTheme = "전체";

async function loadDashDates() {
  try {
    const dates = await api("/api/recommendations/dates");
    const sel = document.getElementById("dash-date-sel");
    sel.innerHTML = dates.map(d => `<option value="${d}">${d}</option>`).join("");
    if (dates.length) {
      sel.value = dates[0];
      loadDashRecs(dates[0]);
    } else {
      document.getElementById("dash-rec-list").innerHTML =
        `<span style="color:var(--muted)">저장된 추천 데이터가 없습니다. AI 추천 탭에서 분석을 실행하세요.</span>`;
    }
    sel.onchange = () => loadDashRecs(sel.value);
  } catch (e) {}
}

async function loadDashRecs(date) {
  const list = document.getElementById("dash-rec-list");
  list.innerHTML = `<span style="color:var(--muted)">로딩 중…</span>`;
  try {
    const data = await api(`/api/recommendations?date=${date}`);
    _dashRecs = data.recommendations || [];
    renderRecList("dash-rec-list", _dashRecs, _dashTheme);
  } catch (e) {
    list.innerHTML = `<span style="color:var(--sell)">${e.message}</span>`;
  }
}

function renderRecList(containerId, recs, theme) {
  const list = document.getElementById(containerId);
  const filtered = filterByTheme(recs, theme);
  if (!filtered.length) {
    list.innerHTML = `<span style="color:var(--muted)">해당 조건의 추천 종목이 없습니다.</span>`;
    return;
  }
  list.innerHTML = filtered.map(r => buildRecRow(r)).join("");
}

// ── 히트맵 ──────────────────────────────────────────────────────
let _heatmapDays = { dash: 14, rec: 14 };

async function loadHeatmap(containerId, days) {
  const el = document.getElementById(containerId);
  el.innerHTML = `<span style="color:var(--muted);font-size:.85em">로딩 중…</span>`;
  try {
    const history = await api(`/api/recommendations/history?days=${days}`);
    el.innerHTML = buildHeatmapHtml(history);
  } catch (e) {
    el.innerHTML = `<span style="color:var(--sell)">${e.message}</span>`;
  }
}

function buildHeatmapHtml(history) {
  if (!history.length) {
    return `<span style="color:var(--muted);font-size:.85em">히트맵을 그릴 추천 이력이 없습니다. 추천을 여러 날 실행하면 표시됩니다.</span>`;
  }

  // 날짜 목록 (오름차순)
  const dates = [...new Set(history.map(r => r.date))].sort();

  // 종목별 데이터 집계
  const byStock = {};
  history.forEach(r => {
    const key = `${r.name}||${r.code}`;
    if (!byStock[key]) byStock[key] = { name: r.name, code: r.code, days: {} };
    byStock[key].days[r.date] = { score: r.score, action: r.action };
  });

  // 연속 일수 계산
  function streak(days_obj) {
    let cnt = 0;
    for (let i = dates.length - 1; i >= 0; i--) {
      if (days_obj[dates[i]]) cnt++;
      else break;
    }
    return cnt;
  }

  const stocks = Object.values(byStock).sort((a, b) => {
    return streak(b.days) - streak(a.days);
  });

  // 헤더
  const headCols = dates.map(d => `<th>${d.slice(5)}</th>`).join("");

  // 행
  const rows = stocks.map(s => {
    const stk = streak(s.days);
    const nameLabel = `${s.name} (${s.code})${stk >= 2 ? ` <span class="streak-badge">🔥${stk}일</span>` : ""}`;
    const cells = dates.map(d => {
      const entry = s.days[d];
      if (!entry) return `<td class="hm-0" title="미추천">-</td>`;
      const sc = entry.score;
      const cls = sc >= 70 ? "hm-high" : sc >= 40 ? "hm-mid" : "hm-low";
      return `<td class="${cls}" title="${d} | 점수: ${sc?.toFixed(1)} | ${entry.action}">${Math.round(sc)}</td>`;
    }).join("");
    return `<tr><td class="stock-label">${nameLabel}</td>${cells}</tr>`;
  }).join("");

  return `
    <table class="heatmap-table">
      <thead><tr><th>종목</th>${headCols}</tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

// 히트맵 기간 버튼 초기화
function initHeatmapDayBtns(filterId, containerId, stateKey) {
  const container = document.getElementById(filterId);
  if (!container) return;
  container.querySelectorAll(".theme-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      container.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      _heatmapDays[stateKey] = parseInt(btn.dataset.days);
      loadHeatmap(containerId, _heatmapDays[stateKey]);
    });
  });
}

// ═══════════════════════════════════════════════════════
// Tab 2 — Watchlist
// ═══════════════════════════════════════════════════════

async function loadWatchlist() {
  const container = document.getElementById("wl-list");
  try {
    const wl = await api("/api/watchlist");
    if (!wl.length) {
      container.innerHTML = `<span style="color:var(--muted)">등록된 관심 종목이 없습니다.</span>`;
      return;
    }
    container.innerHTML = wl.map(w => buildWlCard(w)).join("");
  } catch (e) {
    container.innerHTML = `<span style="color:var(--sell)">${e.message}</span>`;
  }
}

function buildWlCard(w) {
  return `
    <div class="wl-card" id="wlcard-${w.code}">
      <div class="wl-card-header">
        <div>
          <span class="wl-card-name">⭐ ${w.name || w.code}</span>
          <span class="wl-card-code"> (${w.code})</span>
        </div>
      </div>
      <div class="wl-actions">
        <button class="btn btn-primary btn-sm" onclick="runWlAnalysis('${w.code}','${w.name}')">
          🔍 실시간 심층 분석 실행
        </button>
        <button class="btn btn-secondary btn-sm" onclick="toggleWlHistory('${w.code}')">
          📜 분석 이력
        </button>
        <button class="btn btn-danger btn-sm" onclick="removeWatchlist('${w.code}')">🗑️</button>
        <span class="status-msg" id="wl-status-${w.code}"></span>
      </div>
      <div class="wl-result" id="wl-result-${w.code}"></div>
      <div class="wl-history" id="wl-history-${w.code}"></div>
    </div>`;
}

async function addWatchlist() {
  const code = document.getElementById("wl-code-input").value.trim();
  if (!code) return;
  try {
    const res = await api("/api/watchlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    document.getElementById("wl-code-input").value = "";
    setStatus("wl-add-status", `✅ ${res.name || code} 등록 완료`);
    loadWatchlist();
  } catch (e) {
    setStatus("wl-add-status", e.message, true);
  }
}

async function removeWatchlist(code) {
  try {
    await api(`/api/watchlist/${code}`, { method: "DELETE" });
    loadWatchlist();
    loadPortfolioSummary();
  } catch (e) { console.warn(e); }
}

async function runWlAnalysis(code, name) {
  setStatus(`wl-status-${code}`, "분석 중…");
  const resultEl = document.getElementById(`wl-result-${code}`);
  resultEl.classList.add("open");
  resultEl.innerHTML = `<span style="color:var(--muted);font-size:.85em">AI 분석 중… (최대 60초 소요)</span>`;

  try {
    const res = await api(`/api/analysis/${code}/sync`, { method: "POST" });
    setStatus(`wl-status-${code}`, "✅ 완료");
    resultEl.innerHTML = buildWlResult(res);
  } catch (e) {
    setStatus(`wl-status-${code}`, e.message, true);
    resultEl.innerHTML = `<span style="color:var(--sell)">${e.message}</span>`;
  }
}

function buildWlResult(res) {
  const ai   = res.ai_opinion || {};
  const ind  = res.indicators || {};
  const stats = res.stats || {};
  const si   = res.sentiment_info || {};
  const articles = si.articles || [];
  const topNews  = si.top_news || "";

  const newsHtml = articles.slice(0, 8).map(a => {
    const url   = a.originallink || a.link || "";
    const title = a.title || "제목 없음";
    const age   = a.days_ago ? ` <span class="news-age">${a.days_ago}</span>` : "";
    return `<div class="news-item">${age}${url
      ? `<a href="${url}" target="_blank" rel="noopener">${title}</a>`
      : title}</div>`;
  }).join("");

  return `
    <div style="margin-bottom:10px">
      ${scoreBarHtml("Tech", res.tech_score)}
      ${scoreBarHtml("ML", res.ml_score)}
      ${scoreBarHtml("News", Math.min(100, Math.max(0, ((res.sentiment_score||0)+100)/2)))}
    </div>
    <div style="margin-bottom:8px;font-size:.88em">
      ${badgeHtml(ai.action)} &nbsp;${ai.summary || ""}
    </div>
    ${ai.target_price
      ? `<div style="font-size:.85em;color:var(--accent)">🎯 목표가: ₩${fmt(ai.target_price)}</div>` : ""}
    ${ai.strength  ? `<div style="font-size:.82em;margin-top:6px">✅ <strong>강점:</strong> ${ai.strength}</div>` : ""}
    ${ai.weakness  ? `<div style="font-size:.82em">⚠️ <strong>약점:</strong> ${ai.weakness}</div>` : ""}
    <div style="font-size:.82em;color:var(--muted);margin-top:6px">${ai.reasoning || ""}</div>

    <div style="margin-top:12px">
      <div class="modal-section-title">📊 기술적 지표</div>
      <div class="flex-row" style="font-size:.82em;gap:16px;flex-wrap:wrap">
        ${ind.rsi   != null ? `<span>RSI: ${ind.rsi}</span>` : ""}
        ${ind.macd  != null ? `<span>MACD: ${ind.macd > ind.macd_sig ? "▲ 골든크로스" : "▼ 데드크로스"}</span>` : ""}
        ${ind.sma_20!= null ? `<span>SMA20: ₩${fmt(ind.sma_20)}</span>` : ""}
        ${ind.bb_pos!= null ? `<span>BB: ${ind.bb_pos < 0.3 ? "하단권" : ind.bb_pos > 0.7 ? "상단권" : "중간권"}</span>` : ""}
      </div>
    </div>

    ${stats.high_52w ? `
    <div style="margin-top:10px">
      <div class="modal-section-title">📈 52주 통계</div>
      <div class="flex-row" style="font-size:.82em;gap:16px;flex-wrap:wrap">
        <span>최고: ₩${fmt(stats.high_52w)}</span>
        <span>최저: ₩${fmt(stats.low_52w)}</span>
        <span>거래량: 평균 대비 ${stats.avg_vol ? ((stats.current_vol / stats.avg_vol)*100).toFixed(1) + "%" : "—"}</span>
      </div>
    </div>` : ""}

    ${articles.length || topNews ? `
    <div style="margin-top:10px">
      <div class="modal-section-title">📰 관련 뉴스 (${articles.length || 1}건)</div>
      ${si.reason ? `<div style="font-size:.78em;color:var(--muted);margin-bottom:4px">💬 ${si.reason}</div>` : ""}
      ${newsHtml}
    </div>` : ""}`;
}

async function toggleWlHistory(code) {
  const el = document.getElementById(`wl-history-${code}`);
  if (!el) return;

  if (el.classList.contains("open")) {
    el.classList.remove("open");
    return;
  }

  el.classList.add("open");
  el.innerHTML = `<span style="color:var(--muted);font-size:.82em">이력 조회 중…</span>`;

  try {
    const history = await api(`/api/analysis/${code}/history?limit=5`);
    if (!history.length) {
      el.innerHTML = `<span style="color:var(--muted);font-size:.82em">이전 분석 데이터가 없습니다.</span>`;
      return;
    }

    el.innerHTML = history.map((h, idx) => {
      const detailId  = `hist-detail-${code}-${idx}`;
      const hasDetail = !!h.detail;
      const ts        = h.date ? h.date.replace('T', ' ').substring(0, 16) : '—';
      const techVal   = h.tech_score  != null ? Number(h.tech_score).toFixed(1)  : '—';
      const mlVal     = h.ml_score    != null ? Number(h.ml_score).toFixed(1)    : '—';
      const newsVal   = h.sentiment_score != null
                          ? Number(h.sentiment_score).toFixed(1) : '—';

      return `
        <div class="history-row">
          <div class="flex-row" style="align-items:center;gap:8px">
            <span style="font-size:.85em">📅 <strong>${ts}</strong></span>
            <span>${badgeHtml(h.action)}</span>
            ${hasDetail
              ? `<button class="btn btn-secondary btn-sm hist-toggle-btn"
                         style="margin-left:auto;font-size:.72em;padding:2px 8px"
                         data-detail-id="${detailId}">상세 보기 ▼</button>`
              : ''}
          </div>
          <div style="font-size:.78em;color:var(--muted);margin-top:3px">
            Tech <strong>${techVal}</strong> · ML <strong>${mlVal}</strong> · News <strong>${newsVal}</strong>
          </div>
          <div style="font-size:.82em;margin-top:3px;color:var(--text)">${h.summary || ""}</div>
          ${hasDetail
            ? `<div id="${detailId}" class="hist-detail">
                 ${buildWlResult(h.detail)}
               </div>`
            : ''}
        </div>`;
    }).join("");

    // innerHTML 설정 후 버튼에 이벤트 리스너 직접 부착
    // (inline onclick + this 는 Firefox에서 this 참조가 불안정하므로 사용하지 않음)
    el.querySelectorAll('.hist-toggle-btn').forEach(function(btn) {
      btn.addEventListener('click', function() {
        const detailEl = document.getElementById(btn.dataset.detailId);
        if (!detailEl) return;
        const isOpen = detailEl.classList.contains('open');
        detailEl.classList.toggle('open', !isOpen);
        btn.textContent = isOpen ? '상세 보기 ▼' : '접기 ▲';
      });
    });
  } catch (e) {
    el.innerHTML = `<span style="color:var(--sell)">${e.message}</span>`;
  }
}

// ═══════════════════════════════════════════════════════
// Tab 3 — AI Recommendations
// ═══════════════════════════════════════════════════════

let _recRecs = [];
let _recTheme = "전체";

async function loadRecDates() {
  try {
    const dates = await api("/api/recommendations/dates");
    const sel   = document.getElementById("rec-date-sel");
    if (!dates.length) {
      sel.innerHTML = `<option value="">데이터 없음</option>`;
      return;
    }
    sel.innerHTML = dates.map(d => `<option value="${d}">${d}</option>`).join("");
    sel.value = dates[0];
    loadRecsByDate();
  } catch (e) {}
}

async function loadRecsByDate() {
  const sel  = document.getElementById("rec-date-sel");
  const date = sel?.value;
  if (!date) return;
  const list = document.getElementById("rec-list");
  list.innerHTML = `<span style="color:var(--muted)">로딩 중…</span>`;
  try {
    const data = await api(`/api/recommendations?date=${date}`);
    _recRecs = data.recommendations || [];
    renderRecList("rec-list", _recRecs, _recTheme);
  } catch (e) {
    list.innerHTML = `<span style="color:var(--sell)">${e.message}</span>`;
  }
}

// ── 시장·테마 조합에 따른 종목수 자동 설정 ──────────────────────
const _LIMIT_HINTS = {
  9: { label: "버킷별 3~4개 균형 배분",  reason: "전체 시장: 버킷 풀 충분" },
  5: { label: "테마 풀 최적",             reason: "테마 지정: 소규모 풀 안정" },
};

function autoSetRecLimit() {
  const theme  = (document.getElementById("rec-theme")  || {}).value || "전체";
  const sel    = document.getElementById("rec-limit");
  const hint   = document.getElementById("rec-limit-hint");
  if (!sel) return;

  const optimal = (theme === "전체") ? 9 : 5;
  sel.value = String(optimal);

  if (hint) {
    const h = _LIMIT_HINTS[optimal];
    hint.textContent = `⚡ 자동: ${h.reason} → ${optimal}개 (${h.label})`;
    hint.style.color = "var(--muted)";
  }
}

async function runRecommendations(force = false) {
  const market = document.getElementById("rec-market").value;
  const theme  = document.getElementById("rec-theme").value;
  const limit  = document.getElementById("rec-limit").value;
  setStatus("rec-run-status", "분석 요청 중…");
  try {
    const res = await api(
      `/api/recommendations/run?market=${market}&theme=${encodeURIComponent(theme)}&limit=${limit}&force=${force}`,
      { method: "POST" }
    );
    if (res.status === "cached") {
      const el = document.getElementById("rec-run-status");
      if (el) {
        el.style.color = "var(--muted)";
        el.innerHTML = `✅ ${res.message} <a href="javascript:runRecommendations(true)" style="color:var(--accent);text-decoration:underline">강제 재실행</a>`;
      }
      // 캐시된 결과이므로 날짜 목록 갱신하여 바로 조회
      await loadRecDates();
    } else {
      setStatus("rec-run-status", res.message || "분석 시작됨");
      pollRecStatus();
    }
  } catch (e) {
    setStatus("rec-run-status", e.message, true);
  }
}

function pollRecStatus() {
  const id = setInterval(async () => {
    try {
      const s = await api("/api/recommendations/status");
      if (!s.running) {
        clearInterval(id);
        setStatus("rec-run-status", "✅ 완료. 날짜를 새로고침하면 결과를 볼 수 있습니다.");
        await loadRecDates();
      }
    } catch { clearInterval(id); }
  }, 5000);
}

// ═══════════════════════════════════════════════════════
// Tab 4 — Backtest
// ═══════════════════════════════════════════════════════

let _btChart = null;
let _btStrategy = "RSI";

function initStrategyFilter() {
  document.querySelectorAll("#strategy-filter .theme-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#strategy-filter .theme-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      _btStrategy = btn.dataset.strategy;
    });
  });
}

async function runBacktest() {
  const code    = document.getElementById("bt-code").value.trim();
  const period  = document.getElementById("bt-period").value;
  const capital = document.getElementById("bt-capital").value;

  if (!code) { setStatus("bt-status", "종목 코드를 입력하세요.", true); return; }
  setStatus("bt-status", "백테스트 실행 중…");

  const resultSection = document.getElementById("bt-result");
  resultSection.style.display = "none";

  try {
    const data = await api(`/api/backtest?code=${code}&strategy=${_btStrategy}&period=${period}&initial_capital=${capital}`);
    if (data.error) { setStatus("bt-status", data.error, true); return; }
    setStatus("bt-status", "✅ 완료");
    renderBtResult(data, parseFloat(capital));
    resultSection.style.display = "block";
  } catch (e) {
    setStatus("bt-status", e.message, true);
  }
}

function renderBtResult(data, capital) {
  const total   = data.total_return_pct ?? 0;
  const mdd     = data.mdd_pct         ?? 0;
  const wr      = data.win_rate        ?? 0;
  const final_c = data.final_capital   ?? capital;
  const bnh     = data.bnh_return_pct  ?? 0;
  const profit  = final_c - capital;

  // 판정 배너
  const verdictEl = document.getElementById("bt-verdict");
  if (total >= 10) {
    verdictEl.className = "bt-verdict win";
    verdictEl.innerHTML = `✅ 이 기간 <strong>${_btStrategy} 전략은 수익</strong>을 냈습니다. (총 수익률 ${total}%)`;
  } else if (total >= 0) {
    verdictEl.className = "bt-verdict even";
    verdictEl.innerHTML = `➡️ 이 기간 <strong>소폭 수익 / 본전</strong> 수준이었습니다. (총 수익률 ${total}%)`;
  } else {
    verdictEl.className = "bt-verdict loss";
    verdictEl.innerHTML = `⚠️ 이 기간 <strong>${_btStrategy} 전략은 손실</strong>을 기록했습니다. (총 수익률 ${total}%)`;
  }

  // 지표 카드 4개
  document.getElementById("bt-metrics").innerHTML = `
    <div class="result-card">
      <div class="rc-label">📈 총 수익률</div>
      <div class="rc-val ${total >= 0 ? "pos" : "neg"}">${total}%</div>
      <div class="rc-delta" style="color:var(--muted)">B&H 대비 ${(total - bnh) >= 0 ? "+" : ""}${(total - bnh).toFixed(1)}%p</div>
    </div>
    <div class="result-card">
      <div class="rc-label">📉 최대 낙폭 (MDD)</div>
      <div class="rc-val neg">${mdd}%</div>
      <div class="rc-delta" style="color:var(--muted)">최악 ${Math.round(capital * Math.abs(mdd) / 100).toLocaleString()}원 손실 경험</div>
    </div>
    <div class="result-card">
      <div class="rc-label">🎯 승률</div>
      <div class="rc-val">${wr}%</div>
      <div class="rc-delta" style="color:var(--muted)">${wr >= 60 ? "우수" : wr >= 50 ? "보통" : "낮음"}</div>
    </div>
    <div class="result-card">
      <div class="rc-label">💰 최종 자산</div>
      <div class="rc-val">${final_c.toLocaleString()}원</div>
      <div class="rc-delta ${profit >= 0 ? "pos" : "neg"}">${profit >= 0 ? "+" : ""}${profit.toLocaleString()}원</div>
    </div>`;

  // B&H 비교
  const beatBnh = total >= bnh;
  document.getElementById("bt-bnh-compare").innerHTML =
    `📌 <strong>단순 보유(B&H) 비교:</strong> 같은 기간 보유만 했다면 <strong>${bnh >= 0 ? "+" : ""}${bnh}%</strong> 였습니다. ` +
    (beatBnh ? "이 전략이 단순 보유보다 <strong style='color:var(--buy)'>유리</strong>했습니다. 🟢"
             : "단순 보유가 이 전략보다 <strong style='color:var(--hold)'>유리</strong>했습니다. 🟡");

  // 차트
  if (data.dates && data.cum_returns && window.Chart) {
    if (_btChart) { _btChart.destroy(); _btChart = null; }
    const ctx = document.getElementById("bt-chart").getContext("2d");
    const c = getChartColors();
    _btChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: data.dates,
        datasets: [
          {
            label: `${_btStrategy} 전략`,
            data: data.cum_returns,
            borderColor: c.accent,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.2,
          },
          {
            label: "단순 보유 (B&H)",
            data: data.cum_returns_bnh,
            borderColor: c.muted,
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            tension: 0.2,
          },
          {
            label: "원금선",
            data: data.dates.map(() => 1),
            borderColor: c.sell,
            borderWidth: 1,
            borderDash: [6, 4],
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        interaction: { intersect: false, mode: "index" },
        plugins: {
          legend: { labels: { color: c.text, font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: ctx => `${ctx.dataset.label}: ${((ctx.parsed.y - 1) * 100).toFixed(2)}%`,
            },
          },
        },
        scales: {
          x: {
            ticks: { color: c.muted, maxTicksLimit: 8, font: { size: 10 } },
            grid: { color: c.grid },
          },
          y: {
            ticks: {
              color: c.muted,
              font: { size: 10 },
              callback: v => ((v - 1) * 100).toFixed(1) + "%",
            },
            grid: { color: c.grid },
          },
        },
      },
    });
  }

  // 해석 가이드 등급표
  const sharpe      = data.sharpe_ratio ?? 0;
  const mddGrade    = mdd > -10   ? "안전 ✅" : mdd > -25  ? "주의 🟡" : "위험 🔴";
  const wrGrade     = wr  >= 60   ? "우수 ✅" : wr  >= 50   ? "보통 🟡" : "낮음 🔴";
  const retGrade    = total >= 10 ? "양호 ✅" : total >= 0  ? "보통 🟡" : "손실 🔴";
  const bnhGrade    = beatBnh     ? "전략 우위 ✅"          : "보유 우위 🟡";
  const sharpeGrade = sharpe >= 1.0 ? "우수 ✅" : sharpe >= 0.5 ? "보통 🟡" : "미흡 🔴";

  const sub = t => `<br><small style="color:var(--muted);font-size:.88em">${t}</small>`;
  document.getElementById("bt-grade-tbody").innerHTML = `
    <tr>
      <td>총 수익률</td><td>${total}%</td>
      <td>✅ 10% 이상 양호 &nbsp;🟡 0~10% 보통 &nbsp;🔴 음수 손실${sub("전략 적용 기간의 원금 대비 최종 수익률입니다.")}</td>
      <td>${retGrade}</td>
    </tr>
    <tr>
      <td>최대 낙폭<br><span style="font-size:.85em;color:var(--muted)">(MDD)</span></td><td>${mdd}%</td>
      <td>✅ -10% 이내 안전 &nbsp;🟡 -25% 이내 주의 &nbsp;🔴 -25% 초과 위험${sub("전략 진행 중 고점 대비 최대로 하락한 폭입니다. 실전에서 이 낙폭을 견뎌야 전략을 유지할 수 있습니다. 총 수익이 좋아도 MDD가 크면 중간에 공포로 손절할 위험이 있습니다.")}</td>
      <td>${mddGrade}</td>
    </tr>
    <tr>
      <td>승률</td><td>${wr}%</td>
      <td>✅ 60% 이상 우수 &nbsp;🟡 50~60% 보통 &nbsp;🔴 50% 미만 낮음${sub("매수 포지션 보유일 중 수익이 발생한 날의 비율입니다. 단, 낮은 승률이라도 수익이 날 때 크고 손실이 작으면 전체 수익률은 양호할 수 있습니다.")}</td>
      <td>${wrGrade}</td>
    </tr>
    <tr>
      <td>샤프 지수</td><td>${sharpe.toFixed(2)}</td>
      <td>✅ 1.0 이상 우수 &nbsp;🟡 0.5~1.0 보통 &nbsp;🔴 0.5 미만 미흡${sub("변동성(위험) 1단위당 얻는 초과 수익을 나타냅니다. 수익률이 같아도 샤프 지수가 높은 전략이 더 안정적입니다.")}</td>
      <td>${sharpeGrade}</td>
    </tr>
    <tr>
      <td>단순 보유 대비<br><span style="font-size:.85em;color:var(--muted)">(B&H 비교)</span></td>
      <td>${(total - bnh) >= 0 ? "+" : ""}${(total - bnh).toFixed(1)}%p</td>
      <td>✅ 0%p 이상 전략 유리 &nbsp;🟡 음수면 단순 보유가 유리${sub("같은 기간 처음부터 끝까지 보유만 했을 때(Buy &amp; Hold)와 비교입니다. 복잡한 전략이 단순 보유보다 못한 경우도 많으므로 반드시 확인해야 합니다.")}</td>
      <td>${bnhGrade}</td>
    </tr>`;

  // 최근 10거래일 테이블
  if (data.recent_rows?.length) {
    const keys = Object.keys(data.recent_rows[0]);
    document.getElementById("bt-table-head").innerHTML =
      `<tr>${keys.map(k => `<th>${k}</th>`).join("")}</tr>`;
    document.getElementById("bt-table-body").innerHTML =
      data.recent_rows.map(row =>
        `<tr>${keys.map(k => `<td>${row[k]}</td>`).join("")}</tr>`
      ).join("");
  }
}

// ═══════════════════════════════════════════════════════
// Tab 3 — 추천 성과 추적 (Outcome Tracker)
// ═══════════════════════════════════════════════════════

let _outcomeDays = 90;

async function loadOutcomes(days) {
  const statsEl = document.getElementById("outcome-stats");
  const listEl  = document.getElementById("outcome-list");
  if (!statsEl) return;
  statsEl.className = "result-grid";
  statsEl.innerHTML = `<span style="color:var(--muted);font-size:.85em;grid-column:1/-1">로딩 중…</span>`;
  try {
    const data = await api(`/api/recommendations/outcomes?days=${days}`);
    statsEl.innerHTML = _outcomeStatsHtml(data.stats);
    if (listEl) listEl.innerHTML = _outcomeListHtml(data.outcomes);
  } catch (e) {
    statsEl.className = "";
    statsEl.innerHTML = `<span style="color:var(--sell)">${e.message}</span>`;
  }
}

function _outcomeStatsHtml(stats) {
  if (!stats || stats.total === 0) {
    return `<div style="color:var(--muted);font-size:.88em;grid-column:1/-1;padding:6px 0">
      아직 집계된 성과 데이터가 없습니다.
      추천 후 <strong>5거래일(약 1주일)</strong>이 지나면 자동으로 수집됩니다.
    </div>`;
  }

  function statCard(label, ev, wr, ret) {
    if (!ev) {
      return `<div class="result-card">
        <div class="rc-label">${label}</div>
        <div class="rc-val" style="color:var(--muted)">—</div>
        <div class="rc-delta" style="color:var(--muted)">집계중</div>
      </div>`;
    }
    const wrClass  = wr >= 60 ? "pos" : wr >= 40 ? "" : "neg";
    const retClass = ret > 0  ? "pos" : ret < 0 ? "neg" : "";
    return `<div class="result-card">
      <div class="rc-label">${label}</div>
      <div class="rc-val ${wrClass}">${wr.toFixed(0)}%</div>
      <div class="rc-delta ${retClass}">평균 ${ret >= 0 ? "+" : ""}${ret.toFixed(1)}% · ${ev}건</div>
    </div>`;
  }

  const thr = stats.target_hit_rate;
  const thrCard = thr != null
    ? `<div class="result-card">
        <div class="rc-label">🎯 목표가 달성률</div>
        <div class="rc-val ${thr >= 50 ? "pos" : "neg"}">${thr.toFixed(0)}%</div>
        <div class="rc-delta" style="color:var(--muted)">20거래일 이내</div>
      </div>`
    : `<div class="result-card">
        <div class="rc-label">🎯 목표가 달성률</div>
        <div class="rc-val" style="color:var(--muted)">—</div>
        <div class="rc-delta" style="color:var(--muted)">집계중</div>
      </div>`;

  return statCard(" 5거래일 정답률", stats.evaluated_5d,  stats.win_rate_5d,  stats.avg_return_5d)
       + statCard("10거래일 정답률", stats.evaluated_10d, stats.win_rate_10d, stats.avg_return_10d)
       + statCard("20거래일 정답률", stats.evaluated_20d, stats.win_rate_20d, stats.avg_return_20d)
       + thrCard;
}

function _outcomeListHtml(outcomes) {
  if (!outcomes || !outcomes.length) return "";

  function retCell(o) {
    const ret = o.return_pct;
    if (ret == null) return `<td style="color:var(--muted);text-align:right">집계중</td>`;
    const cls  = ret > 0 ? "pos" : ret < 0 ? "neg" : "";
    const icon = o.correct === 1 ? "✅" : o.correct === 0 ? "❌" : "";
    return `<td class="${cls}" style="text-align:right">${icon} ${ret >= 0 ? "+" : ""}${ret.toFixed(1)}%</td>`;
  }

  const rows = outcomes.map(o => {
    const dateShort = (o.session_date || "").slice(5);
    return `<tr>
      <td style="color:var(--muted)">${dateShort}</td>
      <td><span style="font-weight:500">${o.name || o.code}</span>
          <span style="font-size:.78em;color:var(--muted);margin-left:4px">${o.code}</span></td>
      <td>${badgeHtml(o.action)}</td>
      <td style="text-align:right;color:var(--muted)">₩${fmt(o.entry_price)}</td>
      ${retCell(o.outcome_5d)}
      ${retCell(o.outcome_10d)}
      ${retCell(o.outcome_20d)}
      <td style="text-align:right;color:var(--muted)">${o.target_price ? "₩" + fmt(o.target_price) : "—"}</td>
    </tr>`;
  }).join("");

  return `<table class="bt-data-table">
    <thead><tr>
      <th style="text-align:left">날짜</th>
      <th style="text-align:left">종목</th>
      <th style="text-align:left">액션</th>
      <th>진입가</th>
      <th>5거래일</th>
      <th>10거래일</th>
      <th>20거래일</th>
      <th>AI 목표가</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

function initOutcomeDaysBtns() {
  const container = document.getElementById("outcome-days-filter");
  if (!container) return;
  container.querySelectorAll(".theme-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      container.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      _outcomeDays = parseInt(btn.dataset.days);
      loadOutcomes(_outcomeDays);
    });
  });
}

// ═══════════════════════════════════════════════════════
// 거래일 체크
// ═══════════════════════════════════════════════════════

function _tradingNoticeHtml(date) {
  return `<div style="margin-top:10px;padding:10px 14px;border-radius:8px;
      background:rgba(255,170,0,.12);border:1px solid rgba(255,170,0,.35);
      color:#ffaa00;font-size:.88em;line-height:1.5">
    📅 <strong>${date}</strong>은 한국 증시 <strong>휴장일</strong>입니다.<br>
    분석을 실행해도 시장 데이터가 없어 정확한 결과를 얻기 어렵습니다.
    이전 거래일 추천 결과를 참고하세요.
  </div>`;
}

async function checkTradingDay() {
  try {
    const res = await api("/api/market/trading-day");
    console.log("[trading-day]", res);
    if (!res.is_trading_day) {
      const html = _tradingNoticeHtml(res.date);
      const rec = document.getElementById("rec-trading-notice");
      const settings = document.getElementById("settings-trading-notice");
      if (rec) rec.innerHTML = html;
      if (settings) settings.innerHTML = html;
    }
  } catch (e) {
    console.warn("[trading-day] 확인 실패:", e.message);
  }
}

// ═══════════════════════════════════════════════════════
// Tab 5 — Settings
// ═══════════════════════════════════════════════════════

async function runDailyUpdate() {
  setStatus("settings-status", "실행 요청 중…");
  try {
    const res = await api("/api/recommendations/run?limit=5&market=ALL", { method: "POST" });
    setStatus("settings-status", res.message || "실행 시작됨. 텔레그램 알림을 확인하세요.");
  } catch (e) {
    setStatus("settings-status", e.message, true);
  }
}

async function loadTelegramStatus() {
  // 서버 환경변수 확인 — 대신 /api/market 성공 여부로 서버 상태만 표시
  const el = document.getElementById("telegram-status");
  try {
    await api("/api/market");
    el.innerHTML = `서버 연결 <strong style="color:var(--buy)">정상</strong>.<br>
      텔레그램 설정은 서버의 .env 파일에서 확인하세요 (<code>TELEGRAM_BOT_TOKEN</code>, <code>TELEGRAM_CHAT_ID</code>).`;
  } catch (e) {
    el.innerHTML = `<span style="color:var(--sell)">서버 연결 실패: ${e.message}</span>`;
  }
}

// ═══════════════════════════════════════════════════════
// 데이터 소스 헬스체크
// ═══════════════════════════════════════════════════════

async function runDataSourceCheck() {
  const btn = document.getElementById("btn-datasource-check");
  const el  = document.getElementById("datasource-result");

  btn.disabled = true;
  btn.textContent = "⏳ 확인 중…";
  el.innerHTML = `<div style="color:var(--muted);font-size:.88em;padding:8px 0">
    9개 소스를 병렬로 점검합니다. 최대 15초 소요될 수 있습니다…</div>`;

  try {
    const data = await api("/api/market/data-sources");
    el.innerHTML = _renderDataSources(data);
  } catch (e) {
    el.innerHTML = `<span style="color:var(--sell);font-size:.88em">헬스체크 실패: ${e.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "🔍 헬스체크 실행";
  }
}

function _renderDataSources(data) {
  const sources = data.sources || [];
  const sum = data.summary || {};
  const checkedAt = data.checked_at || "";

  // 요약 배지
  const sumHtml = [
    sum.ok   > 0 ? `<span style="color:var(--buy)">✅ 정상 ${sum.ok}</span>`   : "",
    sum.warn > 0 ? `<span style="color:#ffaa00">⚠️ 경고 ${sum.warn}</span>`   : "",
    sum.error> 0 ? `<span style="color:var(--sell)">❌ 오류 ${sum.error}</span>` : "",
  ].filter(Boolean).join("  ");

  let html = `
    <div style="display:flex;justify-content:space-between;align-items:center;
                margin-bottom:12px;font-size:.82em">
      <div style="display:flex;gap:14px">${sumHtml}</div>
      <div style="color:var(--muted)">확인: ${checkedAt}</div>
    </div>`;

  // 카테고리 순서로 그룹핑
  const categoryOrder = [
    "개별 주가 (OHLCV)", "개별 주가 (대체 소스)",
    "전종목 목록", "전종목 목록 (대체 소스)",
    "전종목 시세 (거래량·등락률)",
    "시장 지수", "뉴스·감성 데이터", "AI / LLM",
    "공시 데이터 (선택)", "로컬 저장소",
  ];
  const grouped = {};
  for (const s of sources) {
    (grouped[s.category] = grouped[s.category] || []).push(s);
  }

  for (const cat of categoryOrder) {
    const items = grouped[cat];
    if (!items) continue;
    for (const s of items) {
      const color   = s.status === "ok" ? "var(--buy)" : s.status === "warn" ? "#ffaa00" : "var(--sell)";
      const icon    = s.status === "ok" ? "✅" : s.status === "warn" ? "⚠️" : "❌";
      const latency = s.latency_ms < 10000 ? `${s.latency_ms}ms` : "—";
      const usedFor = (s.used_for || []).join(" · ");

      html += `
      <div style="border:1px solid var(--border);border-radius:8px;padding:10px 14px;
                  margin-bottom:8px;background:var(--card-bg)">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
          <div>
            <span style="font-weight:600">${icon} ${s.name}</span>
            <span style="margin-left:8px;font-size:.78em;color:var(--muted);
                         background:var(--border);padding:1px 6px;border-radius:4px">${s.category}</span>
          </div>
          <div style="font-size:.8em;color:${color};white-space:nowrap;font-weight:600">
            ${s.status === "ok" ? latency : s.status.toUpperCase()}
          </div>
        </div>
        <div style="font-size:.82em;color:var(--muted);margin-top:5px">${s.description}</div>
        <div style="font-size:.8em;color:var(--muted);margin-top:2px">소스: <code style="font-size:.92em">${s.source}</code></div>
        <div style="font-size:.8em;color:${color};margin-top:4px">${s.detail || ""}</div>
        ${usedFor ? `<div style="font-size:.77em;color:var(--muted);margin-top:5px;
                                  border-top:1px solid var(--border);padding-top:5px">
          용도: ${usedFor}</div>` : ""}
      </div>`;
    }
  }

  return html;
}

// ═══════════════════════════════════════════════════════
// Tab 6 — 모델 신뢰도
// ═══════════════════════════════════════════════════════

function aucColor(v) {
  return v >= 0.57 ? "var(--buy)" : v >= 0.54 ? "var(--hold)" : "var(--sell)";
}
function gapColor(v) {
  return v < 0.07 ? "var(--buy)" : v < 0.10 ? "var(--hold)" : "var(--sell)";
}
function driftColor(level) {
  return level === "LOW" ? "var(--buy)" : level === "MEDIUM" ? "var(--hold)" : "var(--sell)";
}

async function loadModelHealth() {
  const wrap = document.getElementById("model-health-wrap");
  wrap.innerHTML = `<span style="color:var(--muted)">모델 정보 로드 중…</span>`;
  try {
    const data = await api("/api/model_health");
    wrap.innerHTML = "";
    wrap.appendChild(renderEnsembleSummary(data.ensemble));
    wrap.appendChild(renderModelCards(data.models));
    wrap.appendChild(renderFeatureSection(data.models));
    wrap.appendChild(renderComponentReliability(data.ensemble, data.scoring_formula));
  } catch (e) {
    wrap.innerHTML = `<p style="color:var(--sell)">모델 정보를 불러올 수 없습니다: ${e.message}</p>`;
  }
}

function renderEnsembleSummary(ens) {
  const driftC  = driftColor(ens.drift_level);
  const driftLabel = ens.drift_level === "LOW" ? "양호" : ens.drift_level === "MEDIUM" ? "주의" : "위험";
  const daysText = ens.days_since_training >= 0
    ? `${ens.days_since_training}일 전`
    : "알 수 없음";
  const passIcon = ens.all_quality_pass ? "✅" : "❌";

  const factorsHtml = ens.drift_factors.length
    ? `<ul style="margin:8px 0 0;padding-left:18px;font-size:.84em;color:var(--sell)">
        ${ens.drift_factors.map(f => `<li>${f}</li>`).join("")}
       </ul>`
    : `<div style="font-size:.84em;color:var(--buy);margin-top:6px">위험 요인 없음</div>`;

  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `
    <div class="flex-row" style="margin-bottom:12px">
      <div class="card-title" style="margin:0">🧠 앙상블 모델 상태</div>
      <div style="margin-left:auto;padding:4px 12px;border-radius:12px;font-weight:700;font-size:.9em;
                  background:${driftC}22;color:${driftC};border:1px solid ${driftC}55">
        ${ens.drift_level} · ${driftLabel}
      </div>
    </div>
    <div class="result-grid">
      <div class="result-card">
        <div class="rc-label">활성 모델 수</div>
        <div class="rc-val">${ens.active_count} / 5</div>
        <div class="rc-delta" style="color:var(--muted)">RF · GB · LGB · CB · XGBRanker</div>
      </div>
      <div class="result-card">
        <div class="rc-label">평균 Test AUC</div>
        <div class="rc-val" style="color:${aucColor(ens.mean_test_auc)}">${ens.mean_test_auc.toFixed(4)}</div>
        <div class="rc-delta" style="color:var(--muted)">기준 ${ens.min_auc_threshold} 이상</div>
      </div>
      <div class="result-card">
        <div class="rc-label">평균 과적합 갭</div>
        <div class="rc-val" style="color:${gapColor(ens.mean_overfit_gap)}">${ens.mean_overfit_gap.toFixed(4)}</div>
        <div class="rc-delta" style="color:var(--muted)">train_auc − test_auc</div>
      </div>
      <div class="result-card">
        <div class="rc-label">마지막 학습</div>
        <div class="rc-val">${daysText}</div>
        <div class="rc-delta" style="color:${ens.retrain_recommended ? "var(--sell)" : "var(--muted)"}">
          ${ens.retrain_recommended ? "⚠️ 재학습 권장" : "정상"}
        </div>
      </div>
      <div class="result-card">
        <div class="rc-label">품질 기준 통과</div>
        <div class="rc-val">${passIcon} ${ens.all_quality_pass ? "전체 통과" : "미달 존재"}</div>
        <div class="rc-delta" style="color:var(--muted)">AUC > ${ens.min_auc_threshold}</div>
      </div>
      <div class="result-card">
        <div class="rc-label">레짐 갭 (평균)</div>
        <div class="rc-val" style="color:${gapColor(ens.mean_regime_gap)}">${ens.mean_regime_gap.toFixed(4)}</div>
        <div class="rc-delta" style="color:var(--muted)">test_auc − cv_auc</div>
      </div>
    </div>
    ${ens.drift_factors.length ? `<div style="margin-top:12px">
      <div style="font-size:.88em;font-weight:600;color:var(--sell)">주의 요인</div>
      ${factorsHtml}
    </div>` : ""}`;
  return card;
}

function renderModelCards(models) {
  const wrap = document.createElement("div");
  wrap.className = "card";
  wrap.innerHTML = `<div class="card-title">📊 개별 모델 상세</div>`;

  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px";

  models.forEach(m => {
    const passIcon = m.quality_pass ? "✅ 통과" : "❌ 미달";
    const passColor = m.quality_pass ? "var(--buy)" : "var(--sell)";
    const aucPct  = Math.min(100, Math.max(0, ((m.test_auc - 0.5) / 0.15) * 100));
    const gapPct  = Math.min(100, (m.overfit_gap / 0.15) * 100);

    const savedDate = m.saved_at ? m.saved_at.slice(0, 10) : "—";

    const card = document.createElement("div");
    card.style.cssText = "background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px";
    card.innerHTML = `
      <div class="flex-row" style="margin-bottom:10px">
        <div style="font-weight:700;font-size:.95em">${m.label}</div>
        <div style="margin-left:auto;font-size:.78em;color:${passColor}">${passIcon}</div>
      </div>
      <div style="margin-bottom:8px">
        <div style="font-size:.8em;color:var(--muted);margin-bottom:3px">
          Test AUC <span style="float:right;font-weight:700;color:${aucColor(m.test_auc)}">${m.test_auc.toFixed(4)}</span>
        </div>
        <div style="background:var(--bg-dark);border-radius:4px;height:8px;overflow:hidden">
          <div style="width:${aucPct.toFixed(1)}%;height:100%;background:${aucColor(m.test_auc)};border-radius:4px;transition:width .4s"></div>
        </div>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">CV AUC</span>
        <span class="kv-val">${m.cv_auc_mean.toFixed(4)} ± ${m.cv_auc_std.toFixed(4)}</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">Train AUC</span>
        <span class="kv-val" style="color:var(--muted)">${m.train_auc.toFixed(4)}</span>
      </div>
      <div style="margin:8px 0">
        <div style="font-size:.8em;color:var(--muted);margin-bottom:3px">
          과적합 갭 <span style="float:right;font-weight:700;color:${gapColor(m.overfit_gap)}">${m.overfit_gap.toFixed(4)}</span>
        </div>
        <div style="background:var(--bg-dark);border-radius:4px;height:6px;overflow:hidden">
          <div style="width:${gapPct.toFixed(1)}%;height:100%;background:${gapColor(m.overfit_gap)};border-radius:4px;transition:width .4s"></div>
        </div>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">Log Loss</span>
        <span class="kv-val">${m.test_logloss != null ? m.test_logloss.toFixed(4) : "N/A (ranker)"}</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">학습 샘플</span>
        <span class="kv-val">${m.training_samples.toLocaleString()}</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">Purging</span>
        <span class="kv-val">${m.purging_days}거래일</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">학습 시간</span>
        <span class="kv-val">${m.training_duration.toFixed(1)}s</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">저장일</span>
        <span class="kv-val" style="color:var(--muted)">${savedDate} (${m.days_since_training}일 전)</span>
      </div>`;
    grid.appendChild(card);
  });

  wrap.appendChild(grid);
  return wrap;
}

let _activeFeatureModel = "gradient_boosting";

function renderFeatureSection(models) {
  const wrap = document.createElement("div");
  wrap.className = "card";
  wrap.innerHTML = `<div class="card-title">🔍 피처 중요도</div>`;

  // 모델 전환 탭
  const tabRow = document.createElement("div");
  tabRow.className = "theme-filter";
  tabRow.style.marginBottom = "14px";

  models.forEach(m => {
    const btn = document.createElement("button");
    btn.className = "theme-btn" + (m.name === _activeFeatureModel ? " active" : "");
    btn.textContent = m.label;
    btn.addEventListener("click", () => {
      tabRow.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      _activeFeatureModel = m.name;
      renderFeatureChart(chartArea, m);
    });
    tabRow.appendChild(btn);
  });
  wrap.appendChild(tabRow);

  const chartArea = document.createElement("div");
  wrap.appendChild(chartArea);

  const activeModel = models.find(m => m.name === _activeFeatureModel) || models[0];
  if (activeModel) renderFeatureChart(chartArea, activeModel);

  return wrap;
}

function renderFeatureChart(container, model) {
  const features = model.feature_importances || [];
  if (!features.length) {
    container.innerHTML = `<span style="color:var(--muted)">피처 중요도 데이터 없음</span>`;
    return;
  }
  const maxImp = features[0][1];

  container.innerHTML = features.map(([name, imp], idx) => {
    const pct  = maxImp > 0 ? ((imp / maxImp) * 100).toFixed(1) : 0;
    const isTop = idx < 3;
    const barColor = isTop ? "var(--accent)" : "var(--chart-accent)";
    return `
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
        <div style="width:140px;font-size:.8em;color:${isTop ? "var(--text)" : "var(--muted)"};
                    text-align:right;flex-shrink:0;font-weight:${isTop ? 600 : 400}">${name}</div>
        <div style="flex:1;background:var(--bg-dark);border-radius:4px;height:14px;overflow:hidden">
          <div style="width:${pct}%;height:100%;background:${barColor};border-radius:4px;
                      transition:width .4s;opacity:${isTop ? 1 : 0.65}"></div>
        </div>
        <div style="width:52px;font-size:.78em;color:var(--muted);text-align:right">
          ${(imp * 100).toFixed(2)}%
        </div>
      </div>`;
  }).join("");
}

function renderComponentReliability(ens, formula) {
  const wrap = document.createElement("div");
  wrap.className = "card";
  wrap.innerHTML = `<div class="card-title">📋 분석 구성요소 신뢰도</div>`;

  const mlActive = ens.active_count > 0;
  const aucVal   = ens.mean_test_auc;
  const aucStars = aucVal >= 0.57 ? "★★★★☆" : aucVal >= 0.54 ? "★★★☆☆" : "★★☆☆☆";

  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:16px";
  grid.innerHTML = `
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px">
      <div style="font-weight:700;margin-bottom:8px">📈 기술적 점수</div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">방식</span><span class="kv-val">RSI · MACD · BB · ADX 등 13개 지표</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">범위</span><span class="kv-val">0 ~ 100</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">가중치</span><span class="kv-val">${mlActive ? "40%" : "65%"} (ML ${mlActive ? "활성" : "비활성"})</span></div>
      <div style="margin-top:8px;font-size:.8em;color:var(--muted)">
        주의: 과거 패턴 기반. 추세 변환 초기 신호 포착이 강점.
      </div>
      <div style="margin-top:6px;color:#f59e0b;font-size:.9em">★★★★☆</div>
    </div>
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px">
      <div style="font-weight:700;margin-bottom:8px">🤖 ML 예측</div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">방식</span><span class="kv-val">RF · GB · LGB · CB · XGBRanker 앙상블</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">AUC</span>
        <span class="kv-val" style="color:${aucColor(aucVal)}">${aucVal.toFixed(4)}</span>
      </div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">가중치</span><span class="kv-val">${mlActive ? "35%" : "미사용"}</span></div>
      <div style="margin-top:8px;font-size:.8em;color:var(--muted)">
        주의: AUC ~0.58은 무작위 대비 소폭 우위. 과신 금지.
      </div>
      <div style="margin-top:6px;color:#f59e0b;font-size:.9em">${aucStars}</div>
    </div>
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px">
      <div style="font-weight:700;margin-bottom:8px">📰 뉴스 감성</div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">방식</span><span class="kv-val">Naver News + GPT-4o-mini</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">범위</span><span class="kv-val">-100 ~ +100 (정규화 후 0~100)</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">가중치</span><span class="kv-val">25%</span></div>
      <div style="margin-top:8px;font-size:.8em;color:var(--muted)">
        주의: 뉴스 편향 가능. 긍정 편향 수정 적용됨 (v0.3.2).
      </div>
      <div style="margin-top:6px;color:#f59e0b;font-size:.9em">★★★☆☆</div>
    </div>`;
  wrap.appendChild(grid);

  // 종합 점수 공식
  const formulaDiv = document.createElement("div");
  formulaDiv.style.cssText = "background:var(--bg-dark);border-radius:8px;padding:14px 16px;font-size:.88em";
  formulaDiv.innerHTML = `
    <div style="font-weight:700;margin-bottom:8px">종합 점수 산출 공식</div>
    <div style="margin-bottom:6px">
      <span style="color:var(--muted)">ML 모델 활성 시:</span>
      <code style="margin-left:8px;color:var(--accent)">${formula.with_ml}</code>
    </div>
    <div>
      <span style="color:var(--muted)">ML 모델 없을 시:</span>
      <code style="margin-left:8px;color:var(--hold)">${formula.without_ml}</code>
    </div>
    <div style="margin-top:8px;font-size:.8em;color:var(--muted)">
      ※ sentiment_norm = (sentiment_score + 100) / 2  →  0~100 정규화
    </div>`;
  wrap.appendChild(formulaDiv);
  return wrap;
}

// ═══════════════════════════════════════════════════════
// Tab 7 — 가치주 스크리너
// ═══════════════════════════════════════════════════════

let _valueLoaded = false;

async function loadValueDefaults() {
  try {
    const d = await api("/api/value_stocks/filters");
    if (d.per_max  != null) document.getElementById("val-per-max").value  = d.per_max;
    if (d.pbr_max  != null) document.getElementById("val-pbr-max").value  = d.pbr_max;
    if (d.roe_min  != null) document.getElementById("val-roe-min").value  = d.roe_min;
    if (d.debt_max != null) document.getElementById("val-debt-max").value = d.debt_max;
    if (d.f_score_min != null) {
      const sel = document.getElementById("val-fscore-min");
      if (sel) sel.value = String(d.f_score_min);
    }
  } catch (e) { /* 기본값 유지 */ }
}

async function runValueScreener() {
  const market         = document.getElementById("val-market").value;
  const perMax         = document.getElementById("val-per-max").value;
  const pbrMax         = document.getElementById("val-pbr-max").value;
  const roeMin         = document.getElementById("val-roe-min").value;
  const debtMax        = document.getElementById("val-debt-max").value;
  const fMin           = document.getElementById("val-fscore-min").value;
  const candidateLimit = document.getElementById("val-candidate-limit").value;

  setStatus("val-status", `⏳ 스크리닝 중… 시가총액 상위 ${candidateLimit}종목 펀더멘털 수집 + 필터링 (${candidateLimit > 200 ? "2~3분" : "1~2분"} 소요)`);

  const card = document.getElementById("val-result-card");
  const tbody = document.getElementById("val-tbody");
  card.style.display = "none";
  tbody.innerHTML = "";

  const params = new URLSearchParams({
    market, per_max: perMax, pbr_max: pbrMax, roe_min: roeMin,
    debt_max: debtMax, f_score_min: fMin, candidate_limit: candidateLimit,
  });

  try {
    const data = await api(`/api/value_stocks?${params}`);
    const stocks = data.stocks || [];
    setStatus("val-status", "");

    if (!stocks.length) {
      setStatus("val-status", `필터 조건을 통과한 종목이 없습니다. 임계값을 완화해 보세요.`, true);
      return;
    }

    document.getElementById("val-result-meta").textContent =
      `${stocks.length}개 종목 (F-Score · 가치점수 복합 정렬)`;

    tbody.innerHTML = stocks.map(s => {
      const fScore = s.f_score ?? 0;
      const fColor = fScore >= 7 ? "var(--buy)" : fScore >= 4 ? "var(--hold)" : "var(--sell)";
      const vScore = s.value_score ?? 0;
      const vColor = vScore >= 70 ? "var(--buy)" : vScore >= 50 ? "var(--hold)" : "var(--sell)";
      const yoy    = s.op_income_yoy;
      const yoyTxt = yoy == null ? "—" : (yoy >= 0 ? `+${yoy.toFixed(1)}%` : `${yoy.toFixed(1)}%`);
      const yoyCls = yoy == null ? "" : yoy >= 0 ? "pos" : "neg";
      const sector = [s.market, s.sector].filter(Boolean).join(" · ");
      return `<tr>
        <td><strong>${s.name || s.code}</strong>
            <span style="font-size:.76em;color:var(--muted);margin-left:4px">${s.code}</span></td>
        <td style="font-size:.8em;color:var(--muted)">${sector || "—"}</td>
        <td style="text-align:right">${s.per != null ? s.per.toFixed(1) + "x" : "—"}</td>
        <td style="text-align:right">${s.pbr != null ? s.pbr.toFixed(2) + "x" : "—"}</td>
        <td style="text-align:right;color:var(--buy)">${s.roe != null ? s.roe.toFixed(1) + "%" : "—"}</td>
        <td style="text-align:right">${s.debt_ratio != null ? s.debt_ratio.toFixed(1) + "%" : "—"}</td>
        <td class="${yoyCls}" style="text-align:right">${yoyTxt}</td>
        <td style="text-align:right;color:${fColor};font-weight:700">${fScore}/9</td>
        <td style="text-align:right;color:${vColor};font-weight:700">${vScore.toFixed(1)}</td>
      </tr>`;
    }).join("");

    card.style.display = "";
  } catch (e) {
    setStatus("val-status", `오류: ${e.message}`, true);
  }
}

// ═══════════════════════════════════════════════════════
// 초기화
// ═══════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", async () => {
  // 테마 버튼 초기 동기화
  syncThemeBtn();

  // 탭 1 — Dashboard
  loadMarketIndices();
  loadPortfolioSummary();
  loadDashDates();
  loadHeatmap("dash-heatmap", 14);
  initHeatmapDayBtns("heatmap-days-filter", "dash-heatmap", "dash");

  // 테마 필터 (대시보드)
  renderThemeFilter("dash-theme-filter", theme => {
    _dashTheme = theme;
    renderRecList("dash-rec-list", _dashRecs, _dashTheme);
  });

  // 탭 2 — Watchlist
  loadWatchlist();

  // 탭 3 — Recommendations
  loadRecDates();
  loadHeatmap("rec-heatmap", 14);
  initHeatmapDayBtns("rec-heatmap-days", "rec-heatmap", "rec");
  renderThemeFilter("rec-theme-filter", theme => {
    _recTheme = theme;
    renderRecList("rec-list", _recRecs, _recTheme);
  });

  // 시장·테마 변경 시 종목수 자동 설정
  ["rec-theme", "rec-market"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", autoSetRecLimit);
  });
  autoSetRecLimit();  // 초기값 적용

  // 사용자가 직접 종목수를 변경하면 힌트를 "수동 설정"으로 표시
  const limitSel = document.getElementById("rec-limit");
  const limitHint = document.getElementById("rec-limit-hint");
  if (limitSel && limitHint) {
    limitSel.addEventListener("change", () => {
      const theme = (document.getElementById("rec-theme") || {}).value || "전체";
      const optimal = (theme === "전체") ? 9 : 5;
      if (parseInt(limitSel.value) !== optimal) {
        limitHint.textContent = `✏️ 수동 설정 (권장: ${optimal}개)`;
        limitHint.style.color = "var(--hold)";
      } else {
        autoSetRecLimit();
      }
    });
  }
  loadOutcomes(90);
  initOutcomeDaysBtns();

  // 탭 4 — Backtest
  initStrategyFilter();

  // 탭 5 — Settings
  loadTelegramStatus();
  fetch("/api/version").then(r => r.json()).then(d => {
    const el = document.getElementById("app-version");
    if (el) el.textContent = "v" + d.version;
  }).catch(() => {});

  // 거래일 여부 확인 (Tab 3, Tab 5 안내)
  checkTradingDay();
});
