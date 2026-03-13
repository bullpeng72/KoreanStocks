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

// GPT/서버 텍스트를 HTML에 안전하게 삽입 (특수문자 이스케이프)
function esc(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// http(s):// 로 시작하는 URL만 허용 (javascript: URI 방지)
function safeUrl(url) {
  return (typeof url === "string" && /^https?:\/\//i.test(url)) ? url : "";
}

// 뉴스 기사 목록 → HTML (buildModalHtml, buildWlResult 공용)
function buildNewsHtml(articles, topNews) {
  if (articles.length) {
    return articles.slice(0, 8).map(a => {
      const url   = safeUrl(a.originallink || a.link || "");
      const title = esc(a.title || "제목 없음");
      const age   = a.days_ago ? `<span class="news-age">${esc(a.days_ago)}</span>` : "";
      return `<div class="news-item">${age}${url
        ? `<a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a>`
        : title}</div>`;
    }).join("");
  }
  if (topNews) return `<div class="news-item">${esc(topNews)}</div>`;
  return `<span style="color:var(--muted);font-size:.85em">뉴스 정보 없음</span>`;
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

function chgClass(v) { return v > 0 ? "pos" : v < 0 ? "neg" : ""; }

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

// ── 거시 레짐 배너 ───────────────────────────────────────────────
let _macroBannerCache = null; // { data, fetchedAt }

async function loadMacroBanner() {
  // 12시간 캐시 (일별 1회 계산이므로 반일 재사용)
  if (_macroBannerCache && (Date.now() - _macroBannerCache.fetchedAt) < 43200000) {
    _renderMacroBanner(_macroBannerCache.data);
    return;
  }
  try {
    const data = await api("/api/macro_context");
    _macroBannerCache = { data, fetchedAt: Date.now() };
    _renderMacroBanner(data);
  } catch (_e) { /* 실패 시 배너 숨김 */ }
}

function _renderMacroBanner(data) {
  const regime  = data.macro_regime          || "uncertain";
  const label   = data.macro_regime_label    || "불확실";
  const score   = data.macro_sentiment_score ?? 0;
  const summary = data.macro_summary         || "";

  const icon       = regime === "risk_on" ? "🟢" : regime === "risk_off" ? "🔴" : "🟡";
  const scoreColor = score > 0 ? "var(--buy)" : score < 0 ? "var(--sell)" : "var(--muted)";
  const scoreStr   = (score > 0 ? "+" : "") + score;

  const html = `
    <div class="macro-banner ${esc(regime)}">
      <span style="font-size:1.15em;flex-shrink:0">${icon}</span>
      <div>
        <span style="font-weight:700">거시 레짐:</span>
        <span class="regime-badge ${esc(regime)}">${esc(label)}</span>
        <span style="margin-left:10px;color:${scoreColor};font-weight:600">거시감성 ${esc(scoreStr)}</span>
        ${summary
          ? `<span style="margin-left:10px;color:var(--muted);font-size:.93em">${esc(summary)}</span>`
          : ""}
      </div>
    </div>`;

  ["macro-banner-dashboard", "macro-banner-rec"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
  });
}

// ── 탭 전환 ─────────────────────────────────────────────────────
let _modelHealthLoaded = false;
let _valueLoaded       = false;
let _qualityLoaded     = false;

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    const panel = document.getElementById(`tab-${btn.dataset.tab}`);
    if (panel) panel.classList.add("active");

    if (btn.dataset.tab === "model" && !_modelHealthLoaded) {
      loadModelHealth().then(() => { _modelHealthLoaded = true; });
    }
    if (btn.dataset.tab === "value" && !_valueLoaded) {
      loadValueDefaults();
      _valueLoaded = true;
    }
    if (btn.dataset.tab === "quality" && !_qualityLoaded) {
      loadQualityDefaults();
      _qualityLoaded = true;
    }
  });
});

// ── 모달 ─────────────────────────────────────────────────────────
function openModal(rec) {
  const modalBody = document.getElementById("modal-body");
  const recModal  = document.getElementById("rec-modal");
  if (!modalBody || !recModal) return;
  modalBody.innerHTML = buildModalHtml(rec);
  recModal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeModal(e) {
  const recModal = document.getElementById("rec-modal");
  if (!recModal) return;
  if (e && e.target !== recModal) return;
  recModal.classList.add("hidden");
  document.body.style.overflow = "";
}

// ESC 키로 모달 닫기
document.addEventListener("keydown", e => {
  const recModal = document.getElementById("rec-modal");
  if (e.key === "Escape" && recModal && !recModal.classList.contains("hidden")) {
    closeModal();
  }
});

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

  // 거시경제 컨텍스트
  const macroRegime  = rec.macro_regime       || "uncertain";
  const macroLabel   = rec.macro_regime_label || "";
  const macroSentRaw = rec.macro_sentiment    ?? null;
  const macroNorm    = macroSentRaw != null ? Math.min(100, Math.max(0, (macroSentRaw + 100) / 2)) : null;
  const macroSentColor = macroSentRaw == null ? "var(--muted)"
    : macroSentRaw > 0 ? "var(--buy)" : macroSentRaw < 0 ? "var(--sell)" : "var(--muted)";
  const macroSummary = rec.macro_summary || "";

  const newsHtml = buildNewsHtml(articles, topNews);

  return `
    <div class="modal-header">
      <div class="flex-row">
        <span class="modal-name">${esc(rec.name || rec.code)}</span>
        <span style="color:var(--muted);font-size:.85em">(${esc(rec.code)})</span>
        ${mktBadge(rec.market)}
        ${rec.theme && rec.theme !== "전체"
          ? `<span style="color:var(--muted);font-size:.78em">${esc(rec.theme)}</span>` : ""}
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
      ${scoreBarHtml("종목감성", sentNorm)}
      ${macroNorm != null ? scoreBarHtml("거시감성", macroNorm) : ""}
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
        ${si.reason ? `<div style="font-size:.78em;color:var(--muted);margin-top:4px">${esc(si.reason)}</div>` : ""}

        <div class="modal-section-title" style="margin-top:14px">🌐 거시경제 레짐</div>
        <div style="font-size:.88em;display:flex;align-items:center;gap:8px;flex-wrap:wrap">
          ${macroLabel ? `<span class="regime-badge ${esc(macroRegime)}">${esc(macroLabel)}</span>` : "<span style='color:var(--muted)'>—</span>"}
          ${macroSentRaw != null ? `<span style="color:${macroSentColor};font-weight:600">거시감성 ${macroSentRaw > 0 ? "+" : ""}${macroSentRaw}</span>` : ""}
        </div>
        ${macroSummary ? `<div style="font-size:.78em;color:var(--muted);margin-top:4px">${esc(macroSummary)}</div>` : ""}
      </div>

      <!-- 우측: AI 분석 -->
      <div>
        <div class="modal-section-title">🤖 AI 분석 요약</div>
        <div style="background:var(--bg-dark);border-radius:6px;padding:10px 12px;font-size:.88em;line-height:1.7;margin-bottom:12px">
          ${esc(ai.summary) || "분석 내용 없음"}
        </div>
        ${ai.strength
          ? `<div style="font-size:.85em;margin-bottom:6px">✅ <strong>강점:</strong> ${esc(ai.strength)}</div>` : ""}
        ${ai.weakness
          ? `<div style="font-size:.85em;margin-bottom:10px">⚠️ <strong>약점:</strong> ${esc(ai.weakness)}</div>` : ""}

        <div class="modal-section-title">📝 상세 추천 사유</div>
        <div style="font-size:.84em;color:var(--muted);line-height:1.7;margin-bottom:12px">
          ${esc(ai.reasoning) || "—"}
        </div>

        ${ai.target_price
          ? `<div style="background:rgba(0,212,170,.1);border:1px solid var(--accent);border-radius:6px;padding:10px 14px;font-size:.9em">
              🎯 <strong>목표가(10거래일): ₩${fmt(ai.target_price)}</strong>
              ${upside != null ? `<span class="${upside >= 0 ? "pos" : "neg"}">(${upside >= 0 ? "+" : ""}${upside}%)</span>` : ""}
              ${ai.target_rationale
                ? `<div style="font-size:.78em;color:var(--muted);margin-top:4px">${esc(ai.target_rationale)}</div>` : ""}
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
  const v   = val != null ? parseFloat(val) : NaN;
  const pct = Math.min(100, Math.max(0, isNaN(v) ? 0 : v));
  return `
    <div class="score-bar">
      <span class="score-bar-label">${esc(label)}</span>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:${pct}%"></div>
      </div>
      <span class="score-bar-val">${isNaN(v) ? "—" : v.toFixed(0)}</span>
    </div>`;
}

function bucketBadge(bucket, label) {
  if (!bucket || !label) return "";
  return `<span class="bucket-badge bucket-${esc(bucket)}">${esc(label)}</span>`;
}

function regimeBadge(regime, label) {
  if (!regime || regime === "uncertain" || !label) return "";
  return `<span class="regime-badge ${esc(regime)}">${esc(label)}</span>`;
}

// ── 추천 카드 렌더링 ─────────────────────────────────────────────
// 컨테이너별 rec 저장소 — 탭 간 인덱스 충돌 방지
const _recDataStores = {};

function buildRecRow(rec, store) {
  const ai     = rec.ai_opinion || {};
  const action = ai.action || "HOLD";
  const score  = calcComposite(rec);

  const idx = store.length;
  store.push(rec);

  return `
    <div class="rec-row" data-rec-idx="${idx}" style="cursor:pointer">
      <div>
        <div class="rec-row-name">${esc(rec.name || rec.code)}</div>
        <div class="rec-row-code">${esc(rec.code)} ${mktBadge(rec.market)}${bucketBadge(rec.bucket, rec.bucket_label)}${regimeBadge(rec.macro_regime, rec.macro_regime_label)}
          ${rec.theme && rec.theme !== "전체"
            ? `<span style="font-size:.72em;color:var(--muted);margin-left:4px">[${esc(rec.theme)}]</span>` : ""}</div>
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
  // 서버에서 이미 거시감성까지 반영해 계산한 값을 우선 사용
  if (rec.composite_score != null) return Number(rec.composite_score).toFixed(1);
  // 폴백: 클라이언트 재계산 (거시감성 포함)
  const t = rec.tech_score ?? 50;
  const m = rec.ml_score  ?? 50;
  const s = Math.min(100, Math.max(0, ((rec.sentiment_score  ?? 0) + 100) / 2));
  const hasML    = (rec.ml_score != null);
  const macroRaw = rec.macro_sentiment ?? null;
  const macro    = macroRaw != null ? Math.min(100, Math.max(0, (macroRaw + 100) / 2)) : null;
  let score;
  if (hasML && macro != null) {
    score = t * 0.35 + m * 0.35 + s * 0.20 + macro * 0.10;
  } else if (hasML) {
    score = t * 0.40 + m * 0.35 + s * 0.25;
  } else {
    score = t * 0.65 + s * 0.35;
  }
  return score.toFixed(1);
}

// ── 테마 필터 ────────────────────────────────────────────────────
const THEMES = ["전체", "AI/인공지능", "반도체", "이차전지", "제약/바이오", "로봇/자동화"];

function renderThemeFilter(containerId, onChange) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = THEMES.map(t =>
    `<button class="theme-btn${t === "전체" ? " active" : ""}" data-theme="${esc(t)}">${esc(t)}</button>`
  ).join("");
  el.querySelectorAll(".theme-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      el.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      if (onChange) onChange(btn.dataset.theme);
    });
  });
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
  "반도체":       ["반도체", "소재", "부품", "웨이퍼"],
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
  const valEl = el.querySelector(".idx-val");
  if (valEl) valEl.textContent = fmt(info.close, 2);
  const chgEl = el.querySelector(".idx-chg");
  if (chgEl) {
    chgEl.textContent = chgText(chg);
    chgEl.className = `idx-chg ${cls}`;
  }
}

function renderUsdCard(id, info) {
  const el = document.getElementById(id);
  if (!el || !info) return;
  const valEl = el.querySelector(".idx-val");
  if (valEl) valEl.textContent = fmt(info.close, 2);
}

async function loadPortfolioSummary() {
  try {
    const wl = await api("/api/watchlist");
    const el = document.getElementById("portfolio-summary");
    if (!el) return;
    if (wl.length) {
      el.innerHTML = `현재 <strong style="color:var(--accent)">${wl.length}개</strong> 종목을 감시 중입니다. ` +
        `<a href="#" id="portfolio-wl-link">Watchlist</a>에서 상세 분석을 실행하세요.`;
      const wlLink = document.getElementById("portfolio-wl-link");
      if (wlLink) wlLink.addEventListener("click", e => { e.preventDefault(); switchTab("watchlist"); });
    } else {
      el.textContent = "Watchlist에 종목을 추가하여 포트폴리오 관리를 시작하세요.";
    }
  } catch (e) {
    const el = document.getElementById("portfolio-summary");
    if (el) el.textContent = "포트폴리오 정보를 불러올 수 없습니다.";
  }
}

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach(b => {
    if (b.dataset.tab === tabName) b.click();
  });
}

// 대시보드 날짜 선택 → 추천 로드
let _dashRecs = [];
let _dashTheme = "전체";
let _dashRecsAbort = null;

async function loadDashDates() {
  try {
    const dates = await api("/api/recommendations/dates");
    const sel = document.getElementById("dash-date-sel");
    if (!sel) return;
    sel.innerHTML = dates.map(d => `<option value="${esc(d)}">${esc(d)}</option>`).join("");
    if (dates.length) {
      sel.value = dates[0];
      loadDashRecs(dates[0]);
    } else {
      const recList = document.getElementById("dash-rec-list");
      if (recList) recList.innerHTML =
        `<span style="color:var(--muted)">저장된 추천 데이터가 없습니다. AI 추천 탭에서 분석을 실행하세요.</span>`;
    }
    sel.addEventListener("change", () => loadDashRecs(sel.value));
  } catch (e) {
    const errEl = document.getElementById("dash-rec-list");
    if (errEl) errEl.innerHTML = `<span style="color:var(--sell)">날짜 목록 로드 실패: ${esc(e.message)}</span>`;
  }
}

async function loadDashRecs(date) {
  const list = document.getElementById("dash-rec-list");
  if (!list) return;
  if (_dashRecsAbort) { _dashRecsAbort.abort(); }
  _dashRecsAbort = new AbortController();
  list.innerHTML = `<span style="color:var(--muted)">로딩 중…</span>`;
  try {
    const data = await api(`/api/recommendations?date=${encodeURIComponent(date)}`, { signal: _dashRecsAbort.signal });
    _dashRecs = data.recommendations || [];
    renderRecList("dash-rec-list", _dashRecs, _dashTheme);
  } catch (e) {
    if (e.name === "AbortError") return;
    list.innerHTML = `<span style="color:var(--sell)">${esc(e.message)}</span>`;
  }
}

function renderRecList(containerId, recs, theme) {
  const list = document.getElementById(containerId);
  if (!list) return;
  const filtered = filterByTheme(recs, theme);
  if (!filtered.length) {
    list.innerHTML = `<span style="color:var(--muted)">해당 조건의 추천 종목이 없습니다.</span>`;
    return;
  }
  // 컨테이너별 독립 store — 탭 간 인덱스 충돌 방지
  const store = [];
  _recDataStores[containerId] = store;
  list.innerHTML = filtered.map(r => buildRecRow(r, store)).join("");
  // onclick 대신 이벤트 위임 — HTML attribute에 JSON을 삽입하지 않음
  list.querySelectorAll(".rec-row[data-rec-idx]").forEach(el => {
    el.addEventListener("click", () => {
      const rec = store[parseInt(el.dataset.recIdx, 10)];
      if (rec) openModal(rec);
    });
  });
}

// ── 히트맵 ──────────────────────────────────────────────────────
let _heatmapDays = { dash: 14, rec: 14 };
const _heatmapAbort = {};

async function loadHeatmap(containerId, days) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (_heatmapAbort[containerId]) { _heatmapAbort[containerId].abort(); }
  _heatmapAbort[containerId] = new AbortController();
  el.innerHTML = `<span style="color:var(--muted);font-size:.85em">로딩 중…</span>`;
  try {
    const history = await api(`/api/recommendations/history?days=${days}`, { signal: _heatmapAbort[containerId].signal });
    el.innerHTML = buildHeatmapHtml(history);
  } catch (e) {
    if (e.name === "AbortError") return;
    el.innerHTML = `<span style="color:var(--sell)">${esc(e.message)}</span>`;
  }
}

function buildHeatmapHtml(history) {
  if (!history.length) {
    return `<span style="color:var(--muted);font-size:.85em">히트맵을 그릴 추천 이력이 없습니다. 추천을 여러 날 실행하면 표시됩니다.</span>`;
  }

  // 날짜 목록 (오름차순) — 추천이 실제 실행된 세션 날짜 기준
  const dates = [...new Set(history.map(r => r.date))].sort();

  // 지속성 패턴을 보려면 최소 2개 날짜 필요
  if (dates.length < 2) {
    const d = dates[0] || "";
    return `<span style="color:var(--muted);font-size:.85em">지속성 히트맵을 표시하려면 서로 다른 날짜의 추천이 최소 2회 필요합니다${d ? ` (현재: ${esc(d)} 1일분)` : ""}.</span>`;
  }

  // 버킷 메타
  const BUCKET_ICON  = { volume: '📊', momentum: '🚀', rebound: '🔁' };
  const BUCKET_LABEL = { volume: '거래량상위', momentum: '상승모멘텀', rebound: '반등후보' };

  // 종목별 데이터 집계 (bucket 포함)
  const byStock = {};
  history.forEach(r => {
    if (!byStock[r.code]) byStock[r.code] = { name: r.name, code: r.code, days: {} };
    byStock[r.code].days[r.date] = { score: r.score, action: r.action, bucket: r.bucket || null };
  });

  // ── 지표 계산 함수 ─────────────────────────────────────────────

  // 최근 날짜부터 연속 추천 일수
  function calcStreak(days_obj) {
    let cnt = 0;
    for (let i = dates.length - 1; i >= 0; i--) {
      if (days_obj[dates[i]]) cnt++;
      else break;
    }
    return cnt;
  }

  // 기간 전체 최대 연속 일수 (과거 연속 이력 반영)
  function calcMaxStreak(days_obj) {
    let max = 0, cur = 0;
    for (let i = 0; i < dates.length; i++) {
      if (days_obj[dates[i]]) { cur++; max = Math.max(max, cur); }
      else cur = 0;
    }
    return max;
  }

  // 기간 내 총 등장 횟수
  function calcFreq(days_obj) {
    return Object.keys(days_obj).length;
  }

  // 버킷 일관성 분석
  // consistency: 0~1 (1 = 매번 같은 버킷)
  function calcBucketInfo(days_obj) {
    const buckets = dates
      .filter(d => days_obj[d] && days_obj[d].bucket)
      .map(d => days_obj[d].bucket);
    if (buckets.length < 2) return null;
    const counts = {};
    buckets.forEach(b => { counts[b] = (counts[b] || 0) + 1; });
    const [dominant, domCnt] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    return { dominant, consistency: domCnt / buckets.length };
  }

  // SS~D 등급 결정
  // SS: 연속3+ && 이력4+  /  S: 연속3+  /  A: 연속2 && 이력4+
  // B:  연속2  /  Cp: 비연속3+ && 과거최대연속3+  /  C: 비연속3+  /  D: 2회(기간≥3)
  function calcGrade(stk, frq, maxStk) {
    if      (stk >= 3 && frq >= 4)               return "SS";
    else if (stk >= 3)                            return "S";
    else if (stk >= 2 && frq >= 4)               return "A";
    else if (stk >= 2)                            return "B";
    else if (frq >= 3 && maxStk >= 3)             return "Cp"; // 과거 강한 연속 이력 보유
    else if (frq >= 3)                            return "C";
    else if (frq === 2 && dates.length >= 3)      return "D";
    return null;
  }

  // ── 종목별 지표 사전 계산 ──────────────────────────────────────
  const GRADE_ORDER = { SS: 7, S: 6, A: 5, B: 4, Cp: 3, C: 2, D: 1 };

  // 점수 추세: 첫 등장 ~ 최근 등장 간 점수 변화 (5pt 이상만 유의미)
  function calcScoreTrend(days_obj) {
    const scores = dates
      .filter(d => days_obj[d] && days_obj[d].score != null)
      .map(d => days_obj[d].score);
    if (scores.length < 2) return null;
    return scores[scores.length - 1] - scores[0];
  }

  // 가장 최근 등장 날짜의 점수 (행 레이블 표시용)
  function calcLatestScore(days_obj) {
    for (let i = dates.length - 1; i >= 0; i--) {
      const entry = days_obj[dates[i]];
      if (entry && entry.score != null) return entry.score;
    }
    return null;
  }

  const stockMetrics = Object.values(byStock).map(s => {
    const stk         = calcStreak(s.days);
    const maxStk      = calcMaxStreak(s.days);
    const frq         = calcFreq(s.days);
    const grade       = calcGrade(stk, frq, maxStk);
    const bucketInf   = calcBucketInfo(s.days);
    const scoreTrend  = calcScoreTrend(s.days);
    const latestScore = calcLatestScore(s.days);
    return { ...s, stk, maxStk, frq, grade, bucketInf, scoreTrend, latestScore };
  }).filter(s => s.grade !== null); // frq=1 등 지속성 없는 종목 제외

  // 정렬: 등급 → 총 출현 횟수 → 최대연속 → 최신점수(내림차순) → 종목코드 (안정 정렬)
  stockMetrics.sort((a, b) => {
    const go = (GRADE_ORDER[b.grade] || 0) - (GRADE_ORDER[a.grade] || 0);
    if (go !== 0) return go;
    const fq = b.frq - a.frq;
    if (fq !== 0) return fq;
    const ms = b.maxStk - a.maxStk;
    if (ms !== 0) return ms;
    const ls = (b.latestScore ?? -Infinity) - (a.latestScore ?? -Infinity);
    if (ls !== 0) return ls;
    return a.code.localeCompare(b.code);
  });

  if (stockMetrics.length === 0) {
    return `<span style="color:var(--muted);font-size:.85em">아직 지속 추천 종목이 없습니다. 추천이 여러 날 반복되면 패턴이 표시됩니다.</span>`;
  }

  // ── 헤더 ──────────────────────────────────────────────────────
  // 다중 연도 기간이면 연도 변경 시점만 YY-MM-DD, 단일 연도면 항상 MM-DD
  const hasMultiYear = new Set(dates.map(d => d.slice(0, 4))).size > 1;
  const headCols = dates.map((d, i) => {
    const year     = d.slice(0, 4);
    const prevYear = i > 0 ? dates[i - 1].slice(0, 4) : year;
    const label    = hasMultiYear && year !== prevYear ? d.slice(2) : d.slice(5);
    return `<th>${esc(label)}</th>`;
  }).join("");

  // ── 행 렌더링 ─────────────────────────────────────────────────
  const rows = stockMetrics.map(s => {
    const { stk, maxStk, frq, grade, bucketInf, scoreTrend, latestScore } = s;

    // 점수 추세 화살표 (5pt 이상 변화만 표시)
    let trendHtml = "";
    if (scoreTrend != null) {
      if      (scoreTrend >  5) trendHtml = ` <span class="score-trend-up" title="점수 상승 (+${scoreTrend.toFixed(1)}pt)">▲</span>`;
      else if (scoreTrend < -5) trendHtml = ` <span class="score-trend-dn" title="점수 하락 (${scoreTrend.toFixed(1)}pt)">▼</span>`;
    }

    // 등급별 배지 + 행 강조 클래스
    let badgesHtml = "";
    let rowClass   = "";
    switch (grade) {
      case "SS":
        badgesHtml = ` <span class="streak-badge-hot">🔥${stk}일 연속 · 총${frq}회</span>`;
        rowClass   = "hm-row-hot hm-row-strong";
        break;
      case "S":
        badgesHtml = ` <span class="streak-badge-hot">🔥${stk}일 연속</span>`;
        rowClass   = "hm-row-hot";
        break;
      case "A": {
        const aMax = maxStk > stk ? ` (최대${maxStk}일)` : "";
        badgesHtml = ` <span class="streak-badge-em">🔥${stk}일 연속 · 총${frq}회${aMax}</span>`;
        rowClass   = "hm-row-streak hm-row-strong";
        break;
      }
      case "B": {
        const bMax = maxStk > stk ? ` (최대${maxStk}일)` : "";
        badgesHtml = ` <span class="streak-badge">🔥${stk}일 연속${bMax}</span>`;
        rowClass   = "hm-row-streak";
        break;
      }
      case "Cp":
        // 현재는 비연속이지만 과거 최대 N일 연속 이력 보유 (maxStk >= 3)
        badgesHtml = ` <span class="repeat-badge-cp">🔄${frq}회 반복 (최대${maxStk}일)</span>`;
        rowClass   = "hm-row-repeat hm-row-cp hm-row-strong";
        break;
      case "C":
        badgesHtml = ` <span class="repeat-badge">🔄${frq}회 반복</span>`;
        rowClass   = "hm-row-repeat";
        break;
      case "D":
        // stk=1: 최근 세션에 재등장 / stk=0: 과거 기록만
        badgesHtml = stk === 1
          ? ` <span class="repeat-badge">📌재등장 (총2회)</span>`
          : ` <span class="repeat-badge">📌2회</span>`;
        rowClass   = "hm-row-repeat";
        break;
    }

    // 버킷 일관성 배지
    // - 75%+ 일관: 버킷 아이콘 + 레이블 표시
    // - rebound가 연속 2일+ 주도: 반등 미달 경고
    let bucketHtml = "";
    if (bucketInf && bucketInf.consistency >= 0.75) {
      const icon  = BUCKET_ICON[bucketInf.dominant]  || "";
      const label = BUCKET_LABEL[bucketInf.dominant] || bucketInf.dominant;
      if (bucketInf.dominant === "rebound" && (stk >= 2 || frq >= 3)) {
        // 반등 후보 연속 2일+ 또는 비연속 3회+ = 반등 실패 지속 가능성 → 경고
        bucketHtml = ` <span class="bucket-warn-badge">⚠️반등잔류</span>`;
      } else {
        bucketHtml = ` <span class="bucket-consist-badge">${icon}${label}</span>`;
      }
    }

    const scoreHtml = latestScore != null
      ? ` <span class="hm-latest-score" title="마지막 추천 점수">${Math.round(latestScore)}점</span>`
      : "";
    const nameLabel = `${esc(s.name)} (${esc(s.code)})${scoreHtml}${badgesHtml}${bucketHtml}${trendHtml}`;

    // 셀: 점수 + 버킷 정보를 tooltip에 포함
    const cells = dates.map(d => {
      const entry = s.days[d];
      if (!entry) return `<td class="hm-0" title="${esc(d)} | 미추천">-</td>`;
      const sc      = entry.score;
      const cls     = sc != null && sc >= 70 ? "hm-high" : sc != null && sc >= 40 ? "hm-mid" : "hm-low";
      const bLabel  = BUCKET_LABEL[entry.bucket] || entry.bucket || "—";
      const tooltip = `${esc(d)} | 점수: ${sc != null ? sc.toFixed(1) : "—"} | ${esc(entry.action)} | ${esc(bLabel)}`;
      return `<td class="${cls}" title="${tooltip}">${sc != null ? Math.round(sc) : "—"}</td>`;
    }).join("");

    return `<tr class="${rowClass}"><td class="stock-label">${nameLabel}</td>${cells}</tr>`;
  }).join("");

  // ── 범례 ──────────────────────────────────────────────────────
  const legend = `
    <div class="heatmap-legend">
      <span><span class="hm-legend-dot hm-legend-hot"></span> 연속 3일+ (S·SS)</span>
      <span><span class="hm-legend-dot hm-legend-streak"></span> 연속 2일 (A·B)</span>
      <span><span class="hm-legend-dot hm-legend-repeat-cp"></span> 과거 강연속 (Cp)</span>
      <span><span class="hm-legend-dot hm-legend-repeat"></span> 비연속 반복 (C·D)</span>
      <span style="border-left:1px solid var(--border);padding-left:10px;margin-left:2px">
        굵은 보더 = 이력 풍부 (SS·A·Cp) &nbsp;|&nbsp; ▲▼ 점수 추세
      </span>
      <span style="margin-left:8px">
        <span style="background:var(--hm-high-bg);color:var(--hm-high-fg);padding:1px 6px;border-radius:3px">■ ≥70</span>
        <span style="background:var(--hm-mid-bg);color:var(--hm-mid-fg);padding:1px 6px;border-radius:3px;margin-left:4px">■ 40~69</span>
        <span style="background:var(--hm-low-bg);color:var(--hm-low-fg);padding:1px 6px;border-radius:3px;margin-left:4px">■ &lt;40</span>
      </span>
    </div>`;

  return `
    <table class="heatmap-table">
      <thead><tr><th>종목</th>${headCols}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${legend}`;
}

// 히트맵 기간 버튼 초기화
function initHeatmapDayBtns(filterId, containerId, stateKey) {
  const container = document.getElementById(filterId);
  if (!container) return;
  container.querySelectorAll(".theme-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      container.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      _heatmapDays[stateKey] = parseInt(btn.dataset.days, 10);
      loadHeatmap(containerId, _heatmapDays[stateKey]);
    });
  });
}

// ═══════════════════════════════════════════════════════
// Tab 2 — Watchlist
// ═══════════════════════════════════════════════════════

async function loadWatchlist() {
  const container = document.getElementById("wl-list");
  if (!container) return;
  try {
    const wl = await api("/api/watchlist");
    if (!wl.length) {
      container.innerHTML = `<span style="color:var(--muted)">등록된 관심 종목이 없습니다.</span>`;
      return;
    }
    container.innerHTML = wl.map(w => buildWlCard(w)).join("");
    // 분석 버튼: data 속성으로 종목명 전달 (onclick 특수문자 취약점 방지)
    container.querySelectorAll(".wl-analyze-btn").forEach(btn => {
      btn.addEventListener("click", () => runWlAnalysis(btn.dataset.code, btn.dataset.name));
    });
    container.querySelectorAll(".wl-history-btn").forEach(btn => {
      btn.addEventListener("click", () => toggleWlHistory(btn.dataset.code));
    });
    container.querySelectorAll(".wl-remove-btn").forEach(btn => {
      btn.addEventListener("click", () => removeWatchlist(btn.dataset.code));
    });
  } catch (e) {
    container.innerHTML = `<span style="color:var(--sell)">${esc(e.message)}</span>`;
  }
}

function buildWlCard(w) {
  return `
    <div class="wl-card" id="wlcard-${w.code}">
      <div class="wl-card-header">
        <div>
          <span class="wl-card-name">⭐ ${esc(w.name || w.code)}</span>
          <span class="wl-card-code"> (${esc(w.code)})</span>
        </div>
      </div>
      <div class="wl-actions">
        <button class="btn btn-primary btn-sm wl-analyze-btn"
                data-code="${esc(w.code)}" data-name="${esc(w.name || w.code)}">
          🔍 실시간 심층 분석 실행
        </button>
        <button class="btn btn-secondary btn-sm wl-history-btn" data-code="${esc(w.code)}">
          📜 분석 이력
        </button>
        <button class="btn btn-danger btn-sm wl-remove-btn" data-code="${esc(w.code)}">🗑️</button>
        <span class="status-msg" id="wl-status-${w.code}"></span>
      </div>
      <div class="wl-result" id="wl-result-${w.code}"></div>
      <div class="wl-history" id="wl-history-${w.code}"></div>
    </div>`;
}

async function addWatchlist() {
  const codeInputEl = document.getElementById("wl-code-input");
  if (!codeInputEl) return;
  const code = codeInputEl.value.trim();
  if (!code) return;
  if (!/^\d{6}$/.test(code)) {
    setStatus("wl-add-status", "종목 코드는 6자리 숫자여야 합니다. (예: 005930)", true);
    return;
  }
  const btn = document.getElementById("btn-wl-add");
  if (btn) btn.disabled = true;
  try {
    const res = await api("/api/watchlist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    codeInputEl.value = "";
    setStatus("wl-add-status", `✅ ${res.name || code} 등록 완료`);
    loadWatchlist();
  } catch (e) {
    setStatus("wl-add-status", e.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function removeWatchlist(code) {
  if (!confirm(`Watchlist에서 ${code} 종목을 삭제하시겠습니까?`)) return;
  try {
    await api(`/api/watchlist/${encodeURIComponent(code)}`, { method: "DELETE" });
    loadWatchlist();
    loadPortfolioSummary();
  } catch (e) {
    setStatus("wl-add-status", `삭제 실패: ${e.message}`, true);
  }
}

async function runWlAnalysis(code, name) {
  const btn = document.querySelector(`.wl-analyze-btn[data-code="${CSS.escape(code)}"]`);
  if (btn) btn.disabled = true;
  setStatus(`wl-status-${code}`, "분석 중…");
  const resultEl = document.getElementById(`wl-result-${code}`);
  if (!resultEl) return;
  resultEl.classList.add("open");
  resultEl.innerHTML = `<span style="color:var(--muted);font-size:.85em">AI 분석 중… (최대 60초 소요)</span>`;

  const controller = new AbortController();
  const abortTimer = setTimeout(() => controller.abort(), 90000);
  try {
    const res = await api(`/api/analysis/${encodeURIComponent(code)}/sync`, { method: "POST", signal: controller.signal });
    setStatus(`wl-status-${code}`, "✅ 완료");
    resultEl.innerHTML = buildWlResult(res);
  } catch (e) {
    const msg = e.name === "AbortError" ? "응답 없음 (90초 초과)" : e.message;
    setStatus(`wl-status-${code}`, msg, true);
    resultEl.innerHTML = `<span style="color:var(--sell)">${esc(msg)}</span>`;
  } finally {
    clearTimeout(abortTimer);
    if (btn) btn.disabled = false;
  }
}

function buildWlResult(res) {
  const ai   = res.ai_opinion || {};
  const ind  = res.indicators || {};
  const stats = res.stats || {};
  const si   = res.sentiment_info || {};
  const articles = si.articles || [];
  const topNews  = si.top_news || "";

  const newsHtml = articles.length || topNews ? buildNewsHtml(articles, topNews) : "";

  return `
    <div style="margin-bottom:10px">
      ${res.composite_score != null ? scoreBarHtml("종합", res.composite_score) : ""}
      ${scoreBarHtml("Tech", res.tech_score)}
      ${scoreBarHtml("ML", res.ml_score)}
      ${scoreBarHtml("News", Math.min(100, Math.max(0, ((res.sentiment_score||0)+100)/2)))}
    </div>
    <div style="margin-bottom:8px;font-size:.88em">
      ${badgeHtml(ai.action)} &nbsp;${esc(ai.summary)}
    </div>
    ${res.current_price
      ? `<div style="font-size:.85em;color:var(--text);margin-bottom:2px">💰 현재가: ₩${fmt(res.current_price)}${
          res.change_pct != null
            ? ` <span class="${res.change_pct >= 0 ? 'pos' : 'neg'}">(${res.change_pct >= 0 ? '▲' : '▼'}${Math.abs(res.change_pct).toFixed(2)}%)</span>`
            : ''
        }</div>` : ""}
    ${ai.target_price
      ? (() => {
          const upside = res.current_price
            ? ((ai.target_price - res.current_price) / res.current_price * 100).toFixed(1)
            : null;
          const upsideStr = upside != null
            ? ` <span style="color:${upside >= 0 ? 'var(--buy)' : 'var(--sell)'}">
                (${upside >= 0 ? '+' : ''}${upside}%)</span>`
            : '';
          return `<div style="font-size:.85em;color:var(--accent)">🎯 목표가(10거래일): ₩${fmt(ai.target_price)}${upsideStr}</div>`;
        })()
      : ""}
    ${ai.strength  ? `<div style="font-size:.82em;margin-top:6px">✅ <strong>강점:</strong> ${esc(ai.strength)}</div>` : ""}
    ${ai.weakness  ? `<div style="font-size:.82em">⚠️ <strong>약점:</strong> ${esc(ai.weakness)}</div>` : ""}
    <div style="font-size:.82em;color:var(--muted);margin-top:6px">${esc(ai.reasoning)}</div>

    <div style="margin-top:12px">
      <div class="modal-section-title">📊 기술적 지표</div>
      <div class="flex-row" style="font-size:.82em;gap:16px;flex-wrap:wrap">
        ${ind.rsi   != null ? `<span>RSI: ${ind.rsi}</span>` : ""}
        ${(ind.macd != null && ind.macd_sig != null) ? `<span>MACD: ${ind.macd > ind.macd_sig ? "▲ 골든크로스" : "▼ 데드크로스"}</span>` : ""}
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
      ${si.reason ? `<div style="font-size:.78em;color:var(--muted);margin-bottom:4px">💬 ${esc(si.reason)}</div>` : ""}
      ${newsHtml}
    </div>` : ""}`;
}

async function toggleWlHistory(code, limit = 10) {
  const el = document.getElementById(`wl-history-${code}`);
  if (!el) return;

  // limit=10 초기 호출이면 이미 열려 있을 때 닫기, 더 보기 호출이면 그냥 새로 렌더
  if (limit === 10 && el.classList.contains("open")) {
    el.classList.remove("open");
    return;
  }

  el.classList.add("open");
  el.innerHTML = `<span style="color:var(--muted);font-size:.82em">이력 조회 중…</span>`;

  const histBtn = document.querySelector(`.wl-history-btn[data-code="${CSS.escape(code)}"]`);
  if (histBtn) histBtn.disabled = true;
  try {
    const history = await api(`/api/analysis/${encodeURIComponent(code)}/history?limit=${encodeURIComponent(limit)}`);
    if (!history.length) {
      el.innerHTML = `<span style="color:var(--muted);font-size:.82em">이전 분석 데이터가 없습니다.</span>`;
      return;
    }

    el.innerHTML = history.map((h, idx) => {
      const detailId   = `hist-detail-${code}-${idx}`;
      const hasDetail  = !!h.detail;
      const ts         = h.date ? h.date.replace('T', ' ').substring(0, 16) : '—';
      const techVal      = h.tech_score  != null ? Number(h.tech_score).toFixed(1)  : '—';
      const mlVal        = h.ml_score    != null ? Number(h.ml_score).toFixed(1)    : '—';
      const newsVal      = h.sentiment_score != null ? Number(h.sentiment_score).toFixed(1) : '—';
      const compositeVal = h.detail?.composite_score != null ? Number(h.detail.composite_score).toFixed(1) : null;
      const curPrice   = h.detail?.current_price;
      const tgtPrice   = h.detail?.ai_opinion?.target_price;
      const priceStr   = curPrice ? `₩${fmt(curPrice)}` : null;
      const upside     = (curPrice && tgtPrice)
                           ? ((tgtPrice - curPrice) / curPrice * 100).toFixed(1)
                           : null;
      const tgtStr     = tgtPrice
                           ? `→ 🎯 ₩${fmt(tgtPrice)}${upside != null
                               ? ` <span style="color:${upside >= 0 ? 'var(--buy)' : 'var(--sell)'}">(${upside >= 0 ? '+' : ''}${upside}%)</span>`
                               : ''}`
                           : '';

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
            ${compositeVal ? `종합 <strong style="color:var(--accent)">${compositeVal}</strong> · ` : ''}Tech <strong>${techVal}</strong> · ML <strong>${mlVal}</strong> · News <strong>${newsVal}</strong>
            ${priceStr ? `&nbsp;·&nbsp; 💰 <strong style="color:var(--text)">${priceStr}</strong> ${tgtStr}` : ''}
          </div>
          <div style="font-size:.82em;margin-top:3px;color:var(--text)">${esc(h.summary)}</div>
          ${hasDetail
            ? `<div id="${detailId}" class="hist-detail">
                 ${buildWlResult(h.detail)}
               </div>`
            : ''}
        </div>`;
    }).join("");

    // 반환 건수가 요청 limit과 같으면 더 많은 이력이 있을 수 있음 → "더 보기" 버튼 추가
    if (history.length === limit) {
      el.insertAdjacentHTML('beforeend',
        `<div style="text-align:center;margin-top:6px">
           <button class="btn btn-secondary btn-sm wl-hist-more-btn"
                   style="font-size:.75em" data-code="${esc(code)}" data-limit="${limit + 10}">
             더 보기 (${limit + 10}건)
           </button>
         </div>`
      );
      el.querySelector('.wl-hist-more-btn').addEventListener('click', function() {
        toggleWlHistory(this.dataset.code, parseInt(this.dataset.limit, 10));
      });
    }

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
    el.innerHTML = `<span style="color:var(--sell)">${esc(e.message)}</span>`;
  } finally {
    if (histBtn) histBtn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════
// Tab 3 — AI Recommendations
// ═══════════════════════════════════════════════════════

let _recRecs = [];
let _recTheme = "전체";
let _recPollId = null;
let _recsByDateAbort = null;

async function loadRecDates() {
  try {
    const dates = await api("/api/recommendations/dates");
    const sel   = document.getElementById("rec-date-sel");
    if (!sel) return;
    if (!dates.length) {
      sel.innerHTML = `<option value="">데이터 없음</option>`;
      return;
    }
    sel.innerHTML = dates.map(d => `<option value="${esc(d)}">${esc(d)}</option>`).join("");
    sel.value = dates[0];
    loadRecsByDate();
  } catch (e) {
    const list = document.getElementById("rec-list");
    if (list) list.innerHTML = `<span style="color:var(--sell)">날짜 목록 로드 실패: ${esc(e.message)}</span>`;
  }
}

async function loadRecsByDate() {
  const sel  = document.getElementById("rec-date-sel");
  const date = sel?.value;
  if (!date) return;
  const list = document.getElementById("rec-list");
  if (!list) return;
  if (_recsByDateAbort) { _recsByDateAbort.abort(); }
  _recsByDateAbort = new AbortController();
  list.innerHTML = `<span style="color:var(--muted)">로딩 중…</span>`;
  try {
    const data = await api(`/api/recommendations?date=${encodeURIComponent(date)}`, { signal: _recsByDateAbort.signal });
    _recRecs = data.recommendations || [];
    renderRecList("rec-list", _recRecs, _recTheme);
  } catch (e) {
    if (e.name === "AbortError") return;
    list.innerHTML = `<span style="color:var(--sell)">${esc(e.message)}</span>`;
  }
}

// ── 시장·테마 조합에 따른 종목수 자동 설정 ──────────────────────
const _LIMIT_HINTS = {
  9: { label: "버킷별 3~4개 균형 배분",  reason: "전체 시장: 버킷 풀 충분" },
  5: { label: "테마 풀 최적",             reason: "테마 지정: 소규모 풀 안정" },
};

function autoSetRecLimit() {
  const theme  = document.getElementById("rec-theme")?.value ?? "전체";
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
  const btn       = document.getElementById("btn-run-recommendations");
  const marketEl  = document.getElementById("rec-market");
  const themeEl   = document.getElementById("rec-theme");
  const limitEl   = document.getElementById("rec-limit");
  if (!marketEl || !themeEl || !limitEl) return;
  if (btn) btn.disabled = true;
  const market = marketEl.value;
  const theme  = themeEl.value;
  const limit  = limitEl.value;
  setStatus("rec-run-status", "분석 요청 중…");
  try {
    const res = await api(
      `/api/recommendations/run?market=${encodeURIComponent(market)}&theme=${encodeURIComponent(theme)}&limit=${encodeURIComponent(limit)}&force=${encodeURIComponent(force)}`,
      { method: "POST" }
    );
    if (res.status === "cached") {
      const el = document.getElementById("rec-run-status");
      if (el) {
        el.style.color = "var(--muted)";
        el.textContent = `✅ ${res.message} `;
        const link = document.createElement("a");
        link.textContent = "강제 재실행";
        link.href = "#";
        link.style.cssText = "color:var(--accent);text-decoration:underline";
        link.addEventListener("click", e => { e.preventDefault(); runRecommendations(true); });
        el.appendChild(link);
      }
      // 캐시된 결과이므로 날짜 목록 갱신하여 바로 조회
      await loadRecDates();
      if (btn) btn.disabled = false;  // 캐시: 즉시 복원
    } else {
      setStatus("rec-run-status", res.message || "분석 시작됨");
      pollRecStatus(btn);  // 폴링 완료 시점에 버튼 복원
    }
  } catch (e) {
    setStatus("rec-run-status", e.message, true);
    if (btn) btn.disabled = false;  // 오류: 즉시 복원
  }
}

function pollRecStatus(btn) {
  if (_recPollId !== null) {
    clearInterval(_recPollId);
    _recPollId = null;
  }
  let _pollCount = 0;
  const MAX_POLLS = 72;  // 5초 × 72 = 6분
  _recPollId = setInterval(async () => {
    _pollCount++;
    if (_pollCount > MAX_POLLS) {
      clearInterval(_recPollId);
      _recPollId = null;
      setStatus("rec-run-status", "⏰ 분석 응답 없음 (6분 초과). 페이지를 새로고침하세요.", true);
      if (btn) btn.disabled = false;
      return;
    }
    try {
      const s = await api("/api/recommendations/status");
      if (!s.running) {
        clearInterval(_recPollId);
        _recPollId = null;
        setStatus("rec-run-status", "✅ 완료. 날짜를 새로고침하면 결과를 볼 수 있습니다.");
        await loadRecDates();
        if (btn) btn.disabled = false;
      }
    } catch {
      clearInterval(_recPollId);
      _recPollId = null;
      if (btn) btn.disabled = false;
    }
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
  const codeEl    = document.getElementById("bt-code");
  const periodEl  = document.getElementById("bt-period");
  const capitalEl = document.getElementById("bt-capital");
  if (!codeEl || !periodEl || !capitalEl) return;
  const code    = codeEl.value.trim();
  const period  = periodEl.value;
  const capital = capitalEl.value;

  if (!code) { setStatus("bt-status", "종목 코드를 입력하세요.", true); return; }
  if (!/^\d{6}$/.test(code)) { setStatus("bt-status", "종목 코드는 6자리 숫자여야 합니다. (예: 005930)", true); return; }

  const btn = document.getElementById("btn-run-backtest");
  if (btn) btn.disabled = true;
  setStatus("bt-status", "백테스트 실행 중…");

  const resultSection = document.getElementById("bt-result");
  if (resultSection) resultSection.style.display = "none";

  try {
    const data = await api(`/api/backtest?code=${encodeURIComponent(code)}&strategy=${encodeURIComponent(_btStrategy)}&period=${encodeURIComponent(period)}&initial_capital=${encodeURIComponent(capital)}`);
    if (data.error) { setStatus("bt-status", data.error, true); return; }
    setStatus("bt-status", "✅ 완료");
    renderBtResult(data, parseFloat(capital));
    if (resultSection) resultSection.style.display = "block";
  } catch (e) {
    setStatus("bt-status", e.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function _finiteOr(v, fallback) {
  return (v != null && Number.isFinite(v)) ? v : fallback;
}

function renderBtResult(data, capital) {
  const total   = _finiteOr(data.total_return_pct, 0);
  const mdd     = _finiteOr(data.mdd_pct,          0);
  const wr      = _finiteOr(data.win_rate,          0);
  const final_c = _finiteOr(data.final_capital,     capital);
  const bnh     = _finiteOr(data.bnh_return_pct,    0);
  const sharpe  = _finiteOr(data.sharpe_ratio,      0);
  const profit  = final_c - capital;

  // 판정 배너
  const verdictEl = document.getElementById("bt-verdict");
  if (verdictEl) {
    if (total >= 10) {
      verdictEl.className = "bt-verdict win";
      verdictEl.innerHTML = `✅ 이 기간 <strong>${esc(_btStrategy)} 전략은 수익</strong>을 냈습니다. (총 수익률 ${total}%)`;
    } else if (total >= 0) {
      verdictEl.className = "bt-verdict even";
      verdictEl.innerHTML = `➡️ 이 기간 <strong>소폭 수익 / 본전</strong> 수준이었습니다. (총 수익률 ${total}%)`;
    } else {
      verdictEl.className = "bt-verdict loss";
      verdictEl.innerHTML = `⚠️ 이 기간 <strong>${esc(_btStrategy)} 전략은 손실</strong>을 기록했습니다. (총 수익률 ${total}%)`;
    }
  }

  // 지표 카드 4개
  const metricsEl = document.getElementById("bt-metrics");
  if (!metricsEl) return;
  metricsEl.innerHTML = `
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
  const bnhEl = document.getElementById("bt-bnh-compare");
  if (bnhEl) {
    bnhEl.innerHTML =
      `📌 <strong>단순 보유(B&H) 비교:</strong> 같은 기간 보유만 했다면 <strong>${bnh >= 0 ? "+" : ""}${bnh}%</strong> 였습니다. ` +
      (beatBnh ? "이 전략이 단순 보유보다 <strong style='color:var(--buy)'>유리</strong>했습니다. 🟢"
               : "단순 보유가 이 전략보다 <strong style='color:var(--hold)'>유리</strong>했습니다. 🟡");
  }

  // 차트
  if (data.dates && data.cum_returns && window.Chart) {
    if (_btChart) { _btChart.destroy(); _btChart = null; }
    const btChartEl = document.getElementById("bt-chart");
    if (!btChartEl) return;
    const ctx = btChartEl.getContext("2d");
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
  const mddGrade    = mdd > -10   ? "안전 ✅" : mdd > -25  ? "주의 🟡" : "위험 🔴";
  const wrGrade     = wr  >= 60   ? "우수 ✅" : wr  >= 50   ? "보통 🟡" : "낮음 🔴";
  const retGrade    = total >= 10 ? "양호 ✅" : total >= 0  ? "보통 🟡" : "손실 🔴";
  const bnhGrade    = beatBnh     ? "전략 우위 ✅"          : "보유 우위 🟡";
  const sharpeGrade = sharpe >= 1.0 ? "우수 ✅" : sharpe >= 0.5 ? "보통 🟡" : "미흡 🔴";

  const sub = t => `<br><small style="color:var(--muted);font-size:.88em">${t}</small>`;
  const gradeTbody = document.getElementById("bt-grade-tbody");
  if (gradeTbody) gradeTbody.innerHTML = `
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
    const tableHead = document.getElementById("bt-table-head");
    const tableBody = document.getElementById("bt-table-body");
    if (tableHead) tableHead.innerHTML =
      `<tr>${keys.map(k => `<th>${esc(k)}</th>`).join("")}</tr>`;
    if (tableBody) tableBody.innerHTML =
      data.recent_rows.map(row =>
        `<tr>${keys.map(k => `<td>${esc(row[k])}</td>`).join("")}</tr>`
      ).join("");
  }
}

// ═══════════════════════════════════════════════════════
// Tab 3 — 추천 성과 추적 (Outcome Tracker)
// ═══════════════════════════════════════════════════════

let _outcomeDays = 90; // 기본값: 3개월 (dashboard.html active 버튼과 일치)
let _outcomesAbort = null;
let _outcomeRetryTimer = null; // 데이터 없을 때 1회 자동 재시도 타이머

async function loadOutcomes(days, _isRetry = false) {
  if (_outcomeRetryTimer) { clearTimeout(_outcomeRetryTimer); _outcomeRetryTimer = null; }
  if (_outcomesAbort) { _outcomesAbort.abort(); }
  _outcomesAbort = new AbortController();
  const statsEl = document.getElementById("outcome-stats");
  const listEl  = document.getElementById("outcome-list");
  if (!statsEl) return;
  statsEl.className = "result-grid";
  statsEl.innerHTML = `<span style="color:var(--muted);font-size:.85em;grid-column:1/-1">로딩 중…</span>`;
  try {
    const data = await api(`/api/recommendations/outcomes?days=${days}`, { signal: _outcomesAbort.signal });
    statsEl.innerHTML = _outcomeStatsHtml(data.stats, _isRetry);
    if (listEl) listEl.innerHTML = _outcomeListHtml(data.outcomes);
    // 데이터 없으면 백그라운드 수집이 완료될 때까지 1회만 자동 재시도 (8초 후)
    if (!_isRetry && (!data.stats || data.stats.total === 0)) {
      _outcomeRetryTimer = setTimeout(() => {
        _outcomeRetryTimer = null;
        loadOutcomes(days, true);
      }, 8000);
    }
  } catch (e) {
    if (e.name === "AbortError") return;
    statsEl.className = "";
    statsEl.innerHTML = `<span style="color:var(--sell)">${esc(e.message)}</span>`;
    if (listEl) listEl.innerHTML = "";
  }
}

function _outcomeStatsHtml(stats, isRetry = false) {
  if (!stats || stats.total === 0) {
    const retryMsg = isRetry
      ? `<br><span style="color:var(--muted)">아직 수집 중이거나 해당 기간 데이터가 없습니다.</span>`
      : `<br><span style="color:var(--muted)">성과 데이터를 수집하는 중입니다. 잠시 후 자동으로 갱신됩니다…</span>`;
    return `<div style="color:var(--muted);font-size:.88em;grid-column:1/-1;padding:6px 0">
      아직 집계된 성과 데이터가 없습니다.
      추천 후 <strong>5거래일(약 1주일)</strong>이 지나면 자동으로 수집됩니다.
      ${retryMsg}
      <br><button class="btn btn-secondary btn-sm" style="margin-top:8px"
        onclick="loadOutcomes(_outcomeDays)">🔄 새로고침</button>
    </div>`;
  }

  function statCard(label, ev, wr, ret) {
    if (ev == null || wr == null || ret == null) {
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
    ? `<div class="result-card" title="20거래일 이내 장중 고가(BUY)/저가(SELL) 기준으로 목표가 도달 여부를 판정합니다">
        <div class="rc-label">🎯 목표가 달성률</div>
        <div class="rc-val ${thr >= 50 ? "pos" : "neg"}">${thr.toFixed(0)}%</div>
        <div class="rc-delta" style="color:var(--muted)">20거래일 내 장중 기준</div>
      </div>`
    : `<div class="result-card" title="20거래일 이내 장중 고가(BUY)/저가(SELL) 기준으로 목표가 도달 여부를 판정합니다. 추천 후 20거래일(약 4주)이 경과해야 산출됩니다.">
        <div class="rc-label">🎯 목표가 달성률</div>
        <div class="rc-val" style="color:var(--muted)">—</div>
        <div class="rc-delta" style="color:var(--muted)">20거래일 경과 후 산출</div>
      </div>`;

  // 정답률 카드: title에 측정 기준 명시 (statCard 래퍼로 title 전달)
  function statCardTitled(label, ev, wr, ret, title) {
    return statCard(label, ev, wr, ret).replace(
      `<div class="result-card">`,
      `<div class="result-card" title="${esc(title)}">`
    );
  }
  const hint5  = "추천 다음날부터 5번째 거래일 종가 기준. BUY→수익/SELL→손실이면 정답";
  const hint10 = "추천 다음날부터 10번째 거래일 종가 기준. BUY→수익/SELL→손실이면 정답";
  const hint20 = "추천 다음날부터 20번째 거래일 종가 기준. BUY→수익/SELL→손실이면 정답";
  return statCardTitled(" 5거래일 정답률", stats.evaluated_5d,  stats.win_rate_5d,  stats.avg_return_5d,  hint5)
       + statCardTitled("10거래일 정답률", stats.evaluated_10d, stats.win_rate_10d, stats.avg_return_10d, hint10)
       + statCardTitled("20거래일 정답률", stats.evaluated_20d, stats.win_rate_20d, stats.avg_return_20d, hint20)
       + thrCard;
}

function _outcomeListHtml(outcomes) {
  if (!outcomes || !outcomes.length) return "";

  function retCell(o) {
    if (o == null) return `<td style="color:var(--muted);text-align:right">집계중</td>`;
    const ret = o.return_pct;
    if (ret == null || !Number.isFinite(ret)) return `<td style="color:var(--muted);text-align:right">집계중</td>`;
    const cls  = ret > 0 ? "pos" : ret < 0 ? "neg" : "";
    const icon = o.correct === 1 ? "✅" : o.correct === 0 ? "❌" : "";
    return `<td class="${cls}" style="text-align:right">${icon} ${ret >= 0 ? "+" : ""}${ret.toFixed(1)}%</td>`;
  }

  // 동일 종목을 code 기준으로 그룹핑 (outcomes는 session_date DESC 정렬 → items[0]이 최신)
  const groups = new Map();
  for (const o of outcomes) {
    if (!groups.has(o.code)) groups.set(o.code, []);
    groups.get(o.code).push(o);
  }

  const rows = [];
  let gIdx = 0;
  for (const [code, items] of groups) {
    const multi   = items.length > 1;
    const latest  = items[0]; // 가장 최근 추천
    const gid     = `ocg${gIdx++}`;
    const dateShort = (latest.session_date || "").slice(5);

    const countBadge = multi
      ? `<span style="font-size:.72em;background:var(--accent);color:#fff;border-radius:10px;padding:1px 6px;margin-left:5px">${items.length}회 추천</span>`
      : "";
    const arrow = multi
      ? `<span id="oc-arr-${gid}" style="margin-left:5px;font-size:.78em;color:var(--accent)">▼</span>`
      : "";
    const rowClick = multi ? `onclick="_ocToggle('${gid}')" style="cursor:pointer"` : "";

    rows.push(`<tr class="oc-summary-row" ${rowClick}>
      <td style="color:var(--muted)">${esc(dateShort)}${multi ? `<br><span style="font-size:.7em">최근</span>` : ""}</td>
      <td><span style="font-weight:500">${esc(latest.name || latest.code)}</span>
          <span style="font-size:.78em;color:var(--muted);margin-left:4px">${esc(latest.code)}</span>
          ${countBadge}${arrow}</td>
      <td>${badgeHtml(latest.action)}</td>
      <td style="text-align:right;color:var(--muted)">₩${fmt(latest.entry_price)}</td>
      ${retCell(latest.outcome_5d)}
      ${retCell(latest.outcome_10d)}
      ${retCell(latest.outcome_20d)}
      <td style="text-align:right;color:var(--muted)">${latest.target_price ? "₩" + fmt(latest.target_price) : "—"}</td>
    </tr>`);

    if (multi) {
      for (const item of items) {
        const ds = (item.session_date || "").slice(5);
        rows.push(`<tr class="oc-sub-row oc-hidden" data-gid="${gid}">
          <td style="color:var(--muted);padding-left:1.4em">↳ ${esc(ds)}</td>
          <td style="color:var(--muted)"><span style="font-size:.85em">${esc(item.name || item.code)}</span>
              <span style="font-size:.75em;margin-left:3px">${esc(item.code)}</span></td>
          <td>${badgeHtml(item.action)}</td>
          <td style="text-align:right;color:var(--muted)">₩${fmt(item.entry_price)}</td>
          ${retCell(item.outcome_5d)}
          ${retCell(item.outcome_10d)}
          ${retCell(item.outcome_20d)}
          <td style="text-align:right;color:var(--muted)">${item.target_price ? "₩" + fmt(item.target_price) : "—"}</td>
        </tr>`);
      }
    }
  }

  return `<table class="bt-data-table">
    <thead><tr>
      <th style="text-align:left">날짜</th>
      <th style="text-align:left">종목</th>
      <th style="text-align:left">액션</th>
      <th title="추천 당일 종가 (GitHub Actions 장 마감 후 실행 기준)">진입가 (추천일 종가)</th>
      <th title="추천 다음날부터 5번째 거래일 종가 기준 수익률">5거래일 수익률</th>
      <th title="추천 다음날부터 10번째 거래일 종가 기준 수익률">10거래일 수익률</th>
      <th title="추천 다음날부터 20번째 거래일 종가 기준 수익률">20거래일 수익률</th>
      <th title="GPT가 제시한 목표가. 20거래일 내 장중 고가(BUY)/저가(SELL) 도달 시 달성">AI 목표가</th>
    </tr></thead>
    <tbody>${rows.join("")}</tbody>
  </table>`;
}

function _ocToggle(gid) {
  const subRows = document.querySelectorAll(`.oc-sub-row[data-gid="${gid}"]`);
  const isHidden = subRows.length > 0 && subRows[0].classList.contains("oc-hidden");
  subRows.forEach(r => r.classList.toggle("oc-hidden", !isHidden));
  const arr = document.getElementById(`oc-arr-${gid}`);
  if (arr) arr.textContent = isHidden ? "▲" : "▼";
}

function initOutcomeDaysBtns() {
  const container = document.getElementById("outcome-days-filter");
  if (!container) return;
  container.querySelectorAll(".theme-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      container.querySelectorAll(".theme-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      _outcomeDays = parseInt(btn.dataset.days, 10);
      loadOutcomes(_outcomeDays); // _isRetry=false → 자동 재시도 다시 허용
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
    📅 <strong>${esc(date)}</strong>은 한국 증시 <strong>휴장일</strong>입니다.<br>
    분석을 실행해도 시장 데이터가 없어 정확한 결과를 얻기 어렵습니다.
    이전 거래일 추천 결과를 참고하세요.
  </div>`;
}

async function checkTradingDay() {
  try {
    const res = await api("/api/market/trading-day");
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
  const btn = document.getElementById("btn-daily-update");
  if (btn) btn.disabled = true;
  setStatus("settings-status", "실행 요청 중…");
  try {
    const res = await api(`/api/recommendations/run?limit=${encodeURIComponent(9)}&market=${encodeURIComponent("ALL")}`, { method: "POST" });
    setStatus("settings-status", res.message || "실행 시작됨. 텔레그램 알림을 확인하세요.");
  } catch (e) {
    setStatus("settings-status", e.message, true);
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function loadTelegramStatus() {
  // 서버 환경변수 확인 — 대신 /api/market 성공 여부로 서버 상태만 표시
  const el = document.getElementById("telegram-status");
  if (!el) return;
  const controller = new AbortController();
  const abortTimer = setTimeout(() => controller.abort(), 10000);
  try {
    await api("/api/market", { signal: controller.signal });
    el.innerHTML = `서버 연결 <strong style="color:var(--buy)">정상</strong>.<br>
      텔레그램 설정은 서버의 .env 파일에서 확인하세요 (<code>TELEGRAM_BOT_TOKEN</code>, <code>TELEGRAM_CHAT_ID</code>).`;
  } catch (e) {
    const msg = e.name === "AbortError" ? "응답 없음 (10초 초과)" : e.message;
    el.innerHTML = `<span style="color:var(--sell)">서버 연결 실패: ${esc(msg)}</span>`;
  } finally {
    clearTimeout(abortTimer);
  }
}

// ═══════════════════════════════════════════════════════
// 데이터 소스 헬스체크
// ═══════════════════════════════════════════════════════

async function runDataSourceCheck() {
  const btn = document.getElementById("btn-datasource-check");
  const el  = document.getElementById("datasource-result");
  if (!btn || !el) return;

  btn.disabled = true;
  btn.textContent = "⏳ 확인 중…";
  el.innerHTML = `<div style="color:var(--muted);font-size:.88em;padding:8px 0">
    데이터 소스를 병렬로 점검합니다. 최대 15초 소요될 수 있습니다…</div>`;

  const controller = new AbortController();
  const abortTimer = setTimeout(() => controller.abort(), 20000);
  try {
    const data = await api("/api/market/data-sources", { signal: controller.signal });
    el.innerHTML = _renderDataSources(data);
  } catch (e) {
    const msg = e.name === "AbortError" ? "클라이언트 타임아웃 (20초 초과)" : e.message;
    el.innerHTML = `<span style="color:var(--sell);font-size:.88em">헬스체크 실패: ${esc(msg)}</span>`;
  } finally {
    clearTimeout(abortTimer);
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
      <div style="color:var(--muted)">확인: ${esc(checkedAt)}</div>
    </div>`;

  // 카테고리 순서로 그룹핑
  const categoryOrder = [
    "개별 주가 (OHLCV)",
    "전종목 목록", "전종목 목록 (대체 소스)",
    "전종목 시세 (거래량·등락률)",
    "시장 지수", "뉴스·감성 데이터", "AI / LLM",
    "펀더멘털 데이터", "펀더멘털 데이터 (선택)",
    "로컬 저장소",
  ];
  const grouped = {};
  for (const s of sources) {
    (grouped[s.category] = grouped[s.category] || []).push(s);
  }

  // categoryOrder에 없는 카테고리도 마지막에 렌더링 (누락 방지)
  const knownCategories = new Set(categoryOrder);
  const unknownCategories = Object.keys(grouped).filter(c => !knownCategories.has(c));
  const renderOrder = [...categoryOrder, ...unknownCategories];

  for (const cat of renderOrder) {
    const items = grouped[cat];
    if (!items) continue;
    for (const s of items) {
      const color   = s.status === "ok" ? "var(--buy)" : s.status === "warn" ? "#ffaa00" : "var(--sell)";
      const icon    = s.status === "ok" ? "✅" : s.status === "warn" ? "⚠️" : "❌";
      const latency = s.latency_ms != null && s.latency_ms < 10000 ? `${s.latency_ms}ms` : "—";
      const usedFor = (s.used_for || []).join(" · ");

      html += `
      <div style="border:1px solid var(--border);border-radius:8px;padding:10px 14px;
                  margin-bottom:8px;background:var(--card-bg)">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
          <div>
            <span style="font-weight:600">${icon} ${esc(s.name)}</span>
            <span style="margin-left:8px;font-size:.78em;color:var(--muted);
                         background:var(--border);padding:1px 6px;border-radius:4px">${esc(s.category)}</span>
          </div>
          <div style="font-size:.8em;color:${color};white-space:nowrap;font-weight:600">
            ${s.status === "ok" ? latency : s.status === "warn" ? "경고" : "오류"}
          </div>
        </div>
        <div style="font-size:.82em;color:var(--muted);margin-top:5px">${esc(s.description)}</div>
        <div style="font-size:.8em;color:var(--muted);margin-top:2px">소스: <code style="font-size:.92em">${esc(s.source)}</code></div>
        <div style="font-size:.8em;color:${color};margin-top:4px">${esc(s.detail)}</div>
        ${usedFor ? `<div style="font-size:.77em;color:var(--muted);margin-top:5px;
                                  border-top:1px solid var(--border);padding-top:5px">
          용도: ${esc(usedFor)}</div>` : ""}
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
  if (!wrap) return;
  wrap.innerHTML = `<span style="color:var(--muted)">모델 정보 로드 중…</span>`;
  try {
    const data = await api("/api/model_health");
    wrap.innerHTML = "";
    if (!data.ensemble || !data.models) {
      const msg = document.createElement("p");
      msg.style.color = "var(--muted)";
      msg.innerHTML = `모델 정보가 아직 없습니다. 먼저 <code>koreanstocks train</code>을 실행하세요.`;
      wrap.appendChild(msg);
      return;
    }
    wrap.appendChild(renderEnsembleSummary(data.ensemble));
    wrap.appendChild(renderModelCards(data.models, data.ensemble));
    wrap.appendChild(renderFeatureSection(data.models));
    wrap.appendChild(renderComponentReliability(data.ensemble, data.scoring_formula));
    // 안내: 모델 데이터는 재학습 시에만 갱신되므로 페이지 새로고침으로 확인
    const note = document.createElement("p");
    note.style.cssText = "margin-top:16px;font-size:.82em;color:var(--muted)";
    note.textContent = "ℹ️ 모델 재학습(koreanstocks train) 후 페이지를 새로고침하면 최신 정보가 반영됩니다.";
    wrap.appendChild(note);
  } catch (e) {
    wrap.innerHTML = `<p style="color:var(--sell)">모델 정보를 불러올 수 없습니다: ${esc(e.message)}</p>`;
  }
}

function renderEnsembleSummary(ens) {
  const driftC  = driftColor(ens.drift_level);
  const driftLabel = ens.drift_level === "LOW" ? "양호" : ens.drift_level === "MEDIUM" ? "주의" : "위험";
  const daysText = ens.days_since_training != null && ens.days_since_training >= 0
    ? `${ens.days_since_training}일 전`
    : "알 수 없음";
  const passIcon = ens.all_quality_pass ? "✅" : "❌";

  const driftFactors = ens.drift_factors || [];
  const factorsHtml = driftFactors.length
    ? `<ul style="margin:8px 0 0;padding-left:18px;font-size:.84em;color:var(--sell)">
        ${driftFactors.map(f => `<li>${esc(f)}</li>`).join("")}
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
        <div class="rc-val">${ens.active_count} / ${ens.total_model_count || 6}</div>
        <div class="rc-delta" style="color:var(--muted)">RF · GB · LGB · CB · XGBRanker${ens.tcn_active ? " · TCN" : ""}</div>
      </div>
      <div class="result-card">
        <div class="rc-label">평균 Test AUC</div>
        <div class="rc-val" style="color:${aucColor(ens.mean_test_auc)}">${ens.mean_test_auc != null ? ens.mean_test_auc.toFixed(4) : "—"}</div>
        <div class="rc-delta" style="color:var(--muted)">기준 ${ens.min_auc_threshold} 이상</div>
      </div>
      <div class="result-card">
        <div class="rc-label">평균 과적합 갭</div>
        <div class="rc-val" style="color:${gapColor(ens.mean_overfit_gap)}">${ens.mean_overfit_gap != null ? ens.mean_overfit_gap.toFixed(4) : "—"}</div>
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
        <div class="rc-val" style="color:${gapColor(ens.mean_regime_gap)}">${ens.mean_regime_gap != null ? ens.mean_regime_gap.toFixed(4) : "—"}</div>
        <div class="rc-delta" style="color:var(--muted)">test_auc − cv_auc</div>
      </div>
    </div>
    ${driftFactors.length ? `<div style="margin-top:12px">
      <div style="font-size:.88em;font-weight:600;color:var(--sell)">주의 요인</div>
      ${factorsHtml}
    </div>` : ""}`;
  return card;
}

function _buildImprovements(models, ensemble) {
  /* 모델 데이터를 분석해 우선순위별 향상 방안 배열 반환.
     각 항목: { level, title, why, desc, action } */
  const items = [];
  const treeModels = models.filter(m => m.name !== "tcn");

  // ── 1. 재학습 경과일 ──────────────────────────────────────────────────────
  if (ensemble.days_since_training > 30) {
    items.push({
      level: "high",
      title: "재학습 필요",
      why:   "주가·거시 데이터는 매일 갱신되며, 시장 구조는 수주 단위로 변한다. 30일 이상 된 모델은 최신 패턴을 반영하지 못해 예측 신호 품질이 떨어진다.",
      desc:  `마지막 학습 ${ensemble.days_since_training}일 경과 (권장: 30일 이내)`,
      action: "koreanstocks train — 최신 2년치 데이터로 전 모델 재학습",
    });
  } else if (ensemble.days_since_training > 14) {
    items.push({
      level: "medium",
      title: "재학습 권장",
      why:   "14일 이상 경과 시 레짐 갭이 서서히 증가한다. 월 2회 재학습이 성능 유지의 현실적 기준이다.",
      desc:  `마지막 학습 ${ensemble.days_since_training}일 경과 (권장: 14일 이내)`,
      action: "koreanstocks train",
    });
  }

  // ── 2. 트리 모델 과적합 갭 ─────────────────────────────────────────────────
  const rfModel = models.find(m => m.name === "random_forest");
  treeModels.forEach(m => {
    if (m.overfit_gap > 0.10) {
      let actionText;
      if (m.name === "random_forest") {
        actionText = "trainer.py RF 파라미터: max_depth 추가 축소(4→3) 또는 max_samples 0.8→0.7 강화 후 재학습. 효과 미미 시 ExtraTreesClassifier 교체 검토.";
      } else if (m.name === "gradient_boosting") {
        actionText = "trainer.py GB: max_depth 2→1 또는 min_samples_leaf 25→40, subsample 0.7→0.6 강화 후 재학습.";
      } else if (m.name === "catboost") {
        actionText = "trainer.py CB: depth 3→2 또는 l2_leaf_reg 5→10 강화 후 재학습.";
      } else {
        actionText = "해당 모델 정규화 파라미터 강화 후 재학습.";
      }
      items.push({
        level: "high",
        title: `${m.label} 과적합 갭 초과`,
        why:   "훈련 데이터에 과도하게 맞춰져 새 데이터에 일반화가 안 된다는 신호다. 갭이 클수록 실전 성능이 학습 시 측정값보다 낮을 가능성이 높다.",
        desc:  `과적합 갭 ${m.overfit_gap.toFixed(4)} > 임계 0.10 (train AUC ${m.train_auc != null ? m.train_auc.toFixed(4) : "—"} → test AUC ${m.test_auc != null ? m.test_auc.toFixed(4) : "—"})`,
        action: actionText,
      });
    }
  });

  // ── 3. 레짐 갭 (test_auc - cv_auc 편향) ──────────────────────────────────
  if (ensemble.mean_regime_gap > 0.10) {
    items.push({
      level: "high",
      title: "레짐 갭 과다 — 시장 편향 위험",
      why:   "test AUC가 CV AUC 평균보다 크게 높다는 것은 테스트 기간(최근 몇 달)이 모델에 유리한 특정 시장 국면임을 의미한다. 시장이 전환되면 성능이 급락할 수 있다.",
      desc:  `레짐 갭 ${ensemble.mean_regime_gap.toFixed(4)} > 0.10 (test_auc − cv_auc_mean 평균)`,
      action: "① 학습 기간 2y→3y 확장(trainer.py period 파라미터)으로 다양한 레짐 커버 ② 거시 레짐 인식 피처 추가(VIX 레짐 원-핫, 이동평균 크로스오버 상태)",
    });
  } else if (ensemble.mean_regime_gap > 0.07) {
    items.push({
      level: "medium",
      title: "레짐 갭 주의",
      why:   "테스트 기간 시장 구조가 과거 CV 기간과 다를 수 있다. 당장 문제는 아니나 시장 전환 시 성능 하락 리스크가 내재한다.",
      desc:  `레짐 갭 ${ensemble.mean_regime_gap.toFixed(4)} (주의 임계 0.07)`,
      action: "학습 기간 2y→3y 확장으로 다양한 시장 레짐 학습 — 재학습 주기 14일 이내 단축 권장",
    });
  }

  // ── 4. CV AUC 불안정 (std > 0.05, 트리 모델) ─────────────────────────────
  treeModels.forEach(m => {
    if (m.cv_auc_std != null && m.cv_auc_std > 0.05) {
      items.push({
        level: "medium",
        title: `${m.label} CV 성능 불안정`,
        why:   "Walk-Forward fold 간 AUC 편차가 크다는 것은 모델 성능이 학습 기간에 따라 크게 달라짐을 뜻한다. 이는 특정 시장 레짐에만 의존하는 피처가 있거나 정규화가 부족한 신호다.",
        desc:  `CV AUC std ${m.cv_auc_std.toFixed(4)} > 0.05 — fold 간 성능 편차 큼`,
        action: `① ${m.label} min_samples_leaf 증가(현재 대비 +10~20)로 노이즈 피처 의존도 감소 ② 학습 기간 2y→3y 확장으로 fold별 데이터 분포 안정화`,
      });
    }
  });

  // ── 5. 평균 Test AUC ──────────────────────────────────────────────────────
  if (ensemble.mean_test_auc < 0.57) {
    items.push({
      level: "high",
      title: "앙상블 AUC 부족",
      why:   "AUC 0.57 미만은 종목 선별 신호가 랜덤 수준에 가깝다는 뜻이다. 이 상태에서의 추천은 신뢰도가 낮아 실질적 투자 참고 가치가 제한된다.",
      desc:  `평균 Test AUC ${ensemble.mean_test_auc.toFixed(4)} < 0.57`,
      action: "① 피처 중요도 하위 5개 제거 후 재학습 ② rs_vs_mkt_3m·high/low_52w_ratio 등 상대강도 피처 가중치 확인 ③ 타깃 기간 10→15거래일 변경 실험",
    });
  } else if (ensemble.mean_test_auc < 0.58) {
    items.push({
      level: "medium",
      title: "앙상블 AUC 개선 여지",
      why:   "AUC 0.58 미만 구간은 신호가 약하다. 소폭 개선만으로도 추천 적중률이 체감될 수 있다.",
      desc:  `평균 Test AUC ${ensemble.mean_test_auc.toFixed(4)} (권장: 0.58 이상)`,
      action: "피처 중요도 하위 피처 정리 후 재학습 — 레짐별 복합 피처(거시+기술) 추가 검토",
    });
  }

  // ── 6. 활성 모델 수 부족 ──────────────────────────────────────────────────
  if (ensemble.active_count < ensemble.total_model_count) {
    const missing = ensemble.total_model_count - ensemble.active_count;
    const isTcnMissing = !ensemble.tcn_active;
    items.push({
      level: isTcnMissing ? "medium" : "high",
      title: `활성 모델 ${ensemble.active_count}/${ensemble.total_model_count} — 앙상블 불완전`,
      why:   "모델 수가 적을수록 앙상블 다양성이 줄어 특정 시장 상황에서 오류가 집중된다. 특히 TCN 누락 시 시계열 패턴 보완재가 없어진다.",
      desc:  `${missing}개 모델 비활성 (품질 미달 또는 미설치)`,
      action: isTcnMissing
        ? "TCN: pip install 'koreanstocks[dl]' 후 koreanstocks train 재실행"
        : "품질 미달 모델의 params.json 확인 → 해당 모델 정규화 조정 후 재학습",
    });
  }

  // ── 7. 품질 미달 모델 ─────────────────────────────────────────────────────
  if (!ensemble.all_quality_pass) {
    const failed = models.filter(m => !m.quality_pass).map(m => m.label).join(", ");
    items.push({
      level: "high",
      title: "품질 기준 미달 모델 존재",
      why:   "AUC < 0.52 모델은 랜덤 분류기보다 나쁠 수 있다. 앙상블에 포함되면 전체 예측 품질을 끌어내린다.",
      desc:  `품질 미달: ${failed} (AUC < ${ensemble.min_auc_threshold})`,
      action: "해당 모델 하이퍼파라미터 재검토 후 koreanstocks train 재실행 — 지속 미달 시 _MODEL_CONFIGS에서 해당 모델 제외 검토",
    });
  }

  // ── 8. 전체 양호 ─────────────────────────────────────────────────────────
  if (items.length === 0) {
    items.push({
      level: "low",
      title: "앙상블 상태 양호",
      why:   "모든 지표가 권장 범위 이내입니다.",
      desc:  `Test AUC ${ensemble.mean_test_auc.toFixed(4)} · 과적합 갭 ${ensemble.mean_overfit_gap.toFixed(4)} · 학습 ${ensemble.days_since_training}일 전`,
      action: "정기 재학습 주기(14~30일) 유지 — 중기 개선: 학습 기간 3y 확장 및 레짐 피처 보강",
    });
  }

  // 우선순위 정렬: high → medium → low
  const order = { high: 0, medium: 1, low: 2 };
  return items.sort((a, b) => order[a.level] - order[b.level]);
}

function renderImprovementCard(models, ensemble) {
  const items = _buildImprovements(models, ensemble);
  const levelIcon  = { high: "🔴", medium: "🟡", low: "🟢" };
  const levelColor = { high: "var(--sell)", medium: "var(--hold)", low: "var(--buy)" };
  const levelLabel = { high: "즉시 조치", medium: "권장", low: "양호" };

  const card = document.createElement("div");
  // 2칸 너비 (그리드에서 span 2)
  card.style.cssText = "background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px;grid-column:span 2";

  const rows = items.map(it => `
    <div style="margin-bottom:10px;padding:12px;background:var(--bg-dark);border-radius:7px;border-left:3px solid ${levelColor[it.level]}">
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
        <span>${levelIcon[it.level]}</span>
        <span style="font-weight:700;font-size:.87em">${esc(it.title)}</span>
        <span style="margin-left:auto;font-size:.73em;padding:1px 6px;border-radius:3px;background:${levelColor[it.level]}22;color:${levelColor[it.level]};border:1px solid ${levelColor[it.level]}55">${levelLabel[it.level]}</span>
      </div>
      <div style="font-size:.78em;color:var(--muted);margin-bottom:4px;font-style:italic">💡 ${esc(it.why)}</div>
      <div style="font-size:.78em;color:var(--text);margin-bottom:5px">📊 ${esc(it.desc)}</div>
      <div style="font-size:.78em;padding:6px 8px;background:var(--bg-card);border-radius:4px;border:1px solid var(--border)">▶ ${esc(it.action)}</div>
    </div>`).join("");

  const highCount   = items.filter(i => i.level === "high").length;
  const medCount    = items.filter(i => i.level === "medium").length;
  const summaryColor = highCount > 0 ? "var(--sell)" : medCount > 0 ? "var(--hold)" : "var(--buy)";
  const summaryText  = highCount > 0
    ? `즉시 조치 ${highCount}건${medCount > 0 ? ` · 권장 ${medCount}건` : ""}`
    : medCount > 0 ? `권장 ${medCount}건` : "전체 양호";

  card.innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
      <div style="font-weight:700;font-size:.95em">🎯 신뢰도 향상 방안</div>
      <span style="font-size:.78em;padding:2px 8px;border-radius:4px;background:${summaryColor}22;color:${summaryColor};border:1px solid ${summaryColor}55">${summaryText}</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:8px">
      ${rows}
    </div>`;
  return card;
}

function renderModelCards(models, ensemble) {
  const wrap = document.createElement("div");
  wrap.className = "card";
  wrap.innerHTML = `<div class="card-title">📊 개별 모델 상세</div>`;

  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px";

  models.forEach(m => {
    const isTcn   = m.name === "tcn";
    const passIcon = m.quality_pass ? "✅ 통과" : "❌ 미달";
    const passColor = m.quality_pass ? "var(--buy)" : "var(--sell)";
    const aucPct  = m.test_auc != null ? Math.min(100, Math.max(0, ((m.test_auc - 0.5) / 0.15) * 100)) : 0;
    // TCN은 구조적으로 갭이 크므로 바 기준을 0.40으로 완화 (트리 모델은 0.15 기준 유지)
    const gapMax  = isTcn ? 0.40 : 0.15;
    const gapPct  = m.overfit_gap != null ? Math.min(100, (m.overfit_gap / gapMax) * 100) : 0;
    const gapBarColor = isTcn ? "#7c6ff7" : gapColor(m.overfit_gap);  // TCN: 보라색 (설계 범위)

    const savedDate = m.saved_at ? m.saved_at.slice(0, 10) : "—";

    // TCN 과적합 갭 설명 블록
    const tcnGapNote = isTcn ? `
      <div style="margin-top:8px;padding:8px 10px;background:#7c6ff715;border:1px solid #7c6ff755;border-radius:6px;font-size:.78em;line-height:1.6">
        <div style="font-weight:700;color:#7c6ff7;margin-bottom:4px">ℹ️ 설계적 허용 범위</div>
        <div style="color:var(--muted)">
          TCN은 수렴까지 전체 시퀀스를 학습하므로 Train AUC가 구조적으로 높게 나타납니다.
          이 갭은 모델 품질 결함이 아닌 <strong style="color:var(--text)">딥러닝 수렴 특성</strong>입니다.
        </div>
        <div style="margin-top:6px;color:var(--muted)">
          <strong style="color:var(--text)">앙상블 기여 역할</strong><br>
          · 시계열 순서 보존 (Dilated Causal Conv1D)<br>
          · 트리가 포착 못하는 모멘텀 연속성·변동성 레짐 전환 학습<br>
          · Test AUC·CV AUC 기준으로만 품질 판정 (갭 임계 미적용)
        </div>
      </div>` : "";

    const card = document.createElement("div");
    card.style.cssText = "background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px";
    card.innerHTML = `
      <div class="flex-row" style="margin-bottom:10px">
        <div style="font-weight:700;font-size:.95em">${esc(m.label)}</div>
        <div style="margin-left:auto;font-size:.78em;color:${passColor}">${passIcon}</div>
      </div>
      <div style="margin-bottom:8px">
        <div style="font-size:.8em;color:var(--muted);margin-bottom:3px">
          Test AUC <span style="float:right;font-weight:700;color:${aucColor(m.test_auc)}">${m.test_auc != null ? m.test_auc.toFixed(4) : "—"}</span>
        </div>
        <div style="background:var(--bg-dark);border-radius:4px;height:8px;overflow:hidden">
          <div style="width:${aucPct.toFixed(1)}%;height:100%;background:${aucColor(m.test_auc)};border-radius:4px;transition:width .4s"></div>
        </div>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">CV AUC</span>
        <span class="kv-val">${m.cv_auc_mean != null ? m.cv_auc_mean.toFixed(4) : "—"} ± ${m.cv_auc_std != null ? m.cv_auc_std.toFixed(4) : "—"}</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">Train AUC</span>
        <span class="kv-val" style="color:var(--muted)">${m.train_auc != null ? m.train_auc.toFixed(4) : "—"}</span>
      </div>
      <div style="margin:8px 0">
        <div style="font-size:.8em;color:var(--muted);margin-bottom:3px">
          과적합 갭
          ${isTcn ? `<span style="margin-left:4px;font-size:.85em;padding:1px 5px;background:#7c6ff720;color:#7c6ff7;border-radius:3px;border:1px solid #7c6ff755">설계 범위</span>` : ""}
          <span style="float:right;font-weight:700;color:${gapBarColor}">${m.overfit_gap != null ? m.overfit_gap.toFixed(4) : "—"}</span>
        </div>
        <div style="background:var(--bg-dark);border-radius:4px;height:6px;overflow:hidden">
          <div style="width:${gapPct.toFixed(1)}%;height:100%;background:${gapBarColor};border-radius:4px;transition:width .4s"></div>
        </div>
      </div>
      ${tcnGapNote}
      <div class="kv-row" style="font-size:.82em;${isTcn ? "margin-top:8px" : ""}">
        <span class="kv-key">Log Loss</span>
        <span class="kv-val">${m.test_logloss != null ? m.test_logloss.toFixed(4) : (m.logloss_label || "N/A (ranker)")}</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">학습 샘플</span>
        <span class="kv-val">${m.training_samples > 0 ? m.training_samples.toLocaleString() : "—"}</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">Purging</span>
        <span class="kv-val">${m.purging_days != null ? m.purging_days + "거래일" : "—"}</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">학습 시간</span>
        <span class="kv-val">${m.training_duration != null ? m.training_duration.toFixed(1) + "s" : "—"}</span>
      </div>
      <div class="kv-row" style="font-size:.82em">
        <span class="kv-key">저장일</span>
        <span class="kv-val" style="color:var(--muted)">${savedDate} (${m.days_since_training != null ? m.days_since_training + "일 전" : "—"})</span>
      </div>`;
    grid.appendChild(card);
  });

  grid.appendChild(renderImprovementCard(models, ensemble));
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
    // TCN 딥러닝 모델: 피처 중요도 대신 아키텍처 정보 표시
    const isTcn = model.name === "tcn";
    container.innerHTML = isTcn
      ? `<div style="color:var(--muted);font-size:.85em;line-height:1.7">
           <div style="margin-bottom:8px;color:var(--text);font-weight:600">TCN 아키텍처 정보</div>
           <div class="kv-row"><span class="kv-key">구조</span><span class="kv-val">Dilated Causal Conv1D × 3</span></div>
           <div class="kv-row"><span class="kv-key">Dilation</span><span class="kv-val">1 → 2 → 4 (Receptive field: 15 거래일)</span></div>
           <div class="kv-row"><span class="kv-key">입력 시퀀스</span><span class="kv-val">최근 20 거래일 × 20 피처</span></div>
           <div class="kv-row"><span class="kv-key">히든 채널</span><span class="kv-val">32</span></div>
           <div class="kv-row"><span class="kv-key">파라미터 수</span><span class="kv-val">~9,057개 (경량)</span></div>
           <div style="margin-top:10px;padding:8px;background:var(--bg-dark);border-radius:6px;font-size:.8em">
             ℹ️ 딥러닝 모델은 피처별 중요도 대신 시퀀스 전체 패턴을 학습합니다.<br>
             트리 모델이 포착하지 못하는 <strong>모멘텀 연속성·변동성 레짐 전환</strong>을 보완합니다.
           </div>
         </div>`
      : `<span style="color:var(--muted)">피처 중요도 데이터 없음</span>`;
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
                    text-align:right;flex-shrink:0;font-weight:${isTop ? 600 : 400}">${esc(name)}</div>
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
  const aucStars = aucVal == null ? "—" : aucVal >= 0.57 ? "★★★★☆" : aucVal >= 0.54 ? "★★★☆☆" : "★★☆☆☆";

  const grid = document.createElement("div");
  grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:16px";
  grid.innerHTML = `
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px">
      <div style="font-weight:700;margin-bottom:8px">📈 기술적 점수</div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">방식</span><span class="kv-val">RSI · MACD · BB · ADX 등 13개 지표</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">범위</span><span class="kv-val">0 ~ 100</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">가중치</span><span class="kv-val">${mlActive ? "35%" : "65%"} (ML ${mlActive ? "활성" : "비활성"})</span></div>
      <div style="margin-top:8px;font-size:.8em;color:var(--muted)">
        주의: 과거 패턴 기반. 추세 변환 초기 신호 포착이 강점.
      </div>
      <div style="margin-top:6px;color:#f59e0b;font-size:.9em">★★★★☆</div>
    </div>
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px">
      <div style="font-weight:700;margin-bottom:8px">🤖 ML 예측</div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">트리</span><span class="kv-val">RF · GB · LGB · CB · XGBRanker</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">딥러닝</span>
        <span class="kv-val">${ens.tcn_active
          ? `TCN <span style="color:${aucColor(ens.tcn_test_auc)};font-weight:600">AUC ${ens.tcn_test_auc != null ? ens.tcn_test_auc.toFixed(4) : "—"}</span>`
          : `<span style="color:var(--muted)">TCN 미학습</span>`
        }</span>
      </div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">앙상블 AUC</span>
        <span class="kv-val" style="color:${aucVal != null ? aucColor(aucVal) : "var(--muted)"}">${aucVal != null ? aucVal.toFixed(4) : "—"}</span>
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
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">가중치</span><span class="kv-val">${mlActive ? "20%" : "35%"}</span></div>
      <div style="margin-top:8px;font-size:.8em;color:var(--muted)">
        주의: 뉴스 편향 가능. 긍정 편향 수정 적용됨 (v0.3.2).
      </div>
      <div style="margin-top:6px;color:#f59e0b;font-size:.9em">★★★☆☆</div>
    </div>
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:10px;padding:16px">
      <div style="font-weight:700;margin-bottom:8px">🌐 거시감성</div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">방식</span><span class="kv-val">Naver News(거시 키워드 6종) + GPT-4o-mini + 퀀트 레짐</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">범위</span><span class="kv-val">-100 ~ +100 (정규화 후 0~100)</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">레짐 분류</span><span class="kv-val">VIX · 장단기 스프레드 · S&amp;P500 · CSI300 기반</span></div>
      <div class="kv-row" style="font-size:.82em"><span class="kv-key">가중치</span><span class="kv-val">${mlActive ? "10%" : "미사용 (ML 없을 시)"}</span></div>
      <div style="margin-top:8px;font-size:.8em;color:var(--muted)">
        일별 1회 캐시. 레짐 risk_off 시 추천 임계값 57점으로 상향.
      </div>
      <div style="margin-top:6px;color:#f59e0b;font-size:.9em">★★★☆☆</div>
    </div>`;
  wrap.appendChild(grid);

  // 종합 점수 공식
  const formulaDiv = document.createElement("div");
  formulaDiv.style.cssText = "background:var(--bg-dark);border-radius:8px;padding:14px 16px;font-size:.88em";
  formulaDiv.innerHTML = `
    <div style="font-weight:700;margin-bottom:8px">종합 점수 산출 공식</div>
    <div style="margin-bottom:5px">
      <span style="color:var(--muted);font-size:.85em">ML + 거시감성 활성 시:</span><br>
      <code style="color:var(--accent)">${esc(formula.with_ml_macro || formula.with_ml)}</code>
    </div>
    <div style="margin-bottom:5px">
      <span style="color:var(--muted);font-size:.85em">ML 활성 (거시감성 미반영):</span><br>
      <code style="color:var(--accent)">${esc(formula.with_ml)}</code>
    </div>
    <div>
      <span style="color:var(--muted);font-size:.85em">ML 없을 시:</span><br>
      <code style="color:var(--hold)">${esc(formula.without_ml)}</code>
    </div>
    <div style="margin-top:8px;font-size:.8em;color:var(--muted)">
      ※ sentiment_norm = (score + 100) / 2  →  0~100 정규화 (종목감성·거시감성 동일 적용)
    </div>`;
  wrap.appendChild(formulaDiv);
  return wrap;
}

// ═══════════════════════════════════════════════════════
// AI 추천 교차 확인 헬퍼
// ═══════════════════════════════════════════════════════

let _aiRecCodesCache = null; // { map: Set<string>, date: string, fetchedAt: number }

async function fetchAiRecCodes() {
  // 1시간 캐시
  if (_aiRecCodesCache && (Date.now() - _aiRecCodesCache.fetchedAt) < 3600000) {
    return _aiRecCodesCache;
  }
  try {
    const dates = await api("/api/recommendations/dates");
    if (!dates.length) return { map: new Set(), date: null };
    const latest = dates[0];
    const data = await api(`/api/recommendations?date=${encodeURIComponent(latest)}`);
    const recs = data.recommendations || [];
    _aiRecCodesCache = { map: new Set(recs.map(r => r.code)), date: latest, fetchedAt: Date.now() };
    return _aiRecCodesCache;
  } catch (_e) {
    return { map: new Set(), date: null };
  }
}

// ═══════════════════════════════════════════════════════
// Tab 7 — 가치주 스크리너
// ═══════════════════════════════════════════════════════

async function loadValueDefaults() {
  try {
    const d = await api("/api/value_stocks/filters");
    const valPerMax  = document.getElementById("val-per-max");
    const valPbrMax  = document.getElementById("val-pbr-max");
    const valRoeMin  = document.getElementById("val-roe-min");
    const valDebtMax = document.getElementById("val-debt-max");
    const valFscore  = document.getElementById("val-fscore-min");
    if (d.per_max     != null && valPerMax)  valPerMax.value  = d.per_max;
    if (d.pbr_max     != null && valPbrMax)  valPbrMax.value  = d.pbr_max;
    if (d.roe_min     != null && valRoeMin)  valRoeMin.value  = d.roe_min;
    if (d.debt_max    != null && valDebtMax) valDebtMax.value = d.debt_max;
    if (d.f_score_min != null && valFscore)  valFscore.value  = String(d.f_score_min);
  } catch (e) { /* 기본값 유지 */ }
}

async function runValueScreener() {
  const marketEl    = document.getElementById("val-market");
  const perMaxEl    = document.getElementById("val-per-max");
  const pbrMaxEl    = document.getElementById("val-pbr-max");
  const roeMinEl    = document.getElementById("val-roe-min");
  const debtMaxEl   = document.getElementById("val-debt-max");
  const fMinEl      = document.getElementById("val-fscore-min");
  const candLimitEl = document.getElementById("val-candidate-limit");
  if (!marketEl || !perMaxEl || !pbrMaxEl || !roeMinEl || !debtMaxEl || !fMinEl || !candLimitEl) return;
  const market         = marketEl.value;
  const perMax         = perMaxEl.value;
  const pbrMax         = pbrMaxEl.value;
  const roeMin         = roeMinEl.value;
  const debtMax        = debtMaxEl.value;
  const fMin           = fMinEl.value;
  const candidateLimit = parseInt(candLimitEl.value, 10);

  setStatus("val-status", `⏳ 스크리닝 중… 시가총액 상위 ${candidateLimit}종목 펀더멘털 수집 + 필터링 (${candidateLimit > 200 ? "2~3분" : "1~2분"} 소요)`);

  const card = document.getElementById("val-result-card");
  const tbody = document.getElementById("val-tbody");
  if (card) card.style.display = "none";
  if (tbody) tbody.innerHTML = "";

  const params = new URLSearchParams({
    market, per_max: perMax, pbr_max: pbrMax, roe_min: roeMin,
    debt_max: debtMax, f_score_min: fMin, candidate_limit: candidateLimit,
  });

  const btn = document.getElementById("btn-run-value");
  if (btn) btn.disabled = true;
  const controller = new AbortController();
  const abortTimer = setTimeout(() => controller.abort(), 200000);
  try {
    const [data, aiRec] = await Promise.all([
      api(`/api/value_stocks?${params}`, { signal: controller.signal }),
      fetchAiRecCodes(),
    ]);
    const stocks = data.stocks || [];
    const aiMap  = aiRec.map;
    const aiDate = aiRec.date;
    setStatus("val-status", "");

    if (!stocks.length) {
      setStatus("val-status", `필터 조건을 통과한 종목이 없습니다. 임계값을 완화해 보세요.`, true);
      return;
    }

    const aiOverlap = stocks.filter(s => aiMap.has(s.code)).length;
    const aiNote    = aiOverlap > 0 && aiDate ? ` · AI추천(${esc(aiDate)}) ${aiOverlap}종목 포함` : "";
    const valMeta   = document.getElementById("val-result-meta");
    if (valMeta) valMeta.textContent = `${stocks.length}개 종목 (F-Score · 가치점수 복합 정렬)${aiNote}`;
    if (!tbody) return;

    tbody.innerHTML = stocks.map(s => {
      const isAiRec    = aiMap.has(s.code);
      const aiRecBadge = isAiRec ? `<span class="ai-rec-badge">★ AI추천</span>` : "";
      const rowClass   = isAiRec ? ' class="ai-rec-row"' : "";
      const fScore = s.f_score ?? 0;
      const fColor = fScore >= 7 ? "var(--buy)" : fScore >= 4 ? "var(--hold)" : "var(--sell)";
      const vScore = s.value_score ?? 0;
      const vColor = vScore >= 70 ? "var(--buy)" : vScore >= 50 ? "var(--hold)" : "var(--sell)";
      const yoy    = s.op_income_yoy;
      const yoyTxt = yoy == null ? "—" : (yoy >= 0 ? `+${yoy.toFixed(1)}%` : `${yoy.toFixed(1)}%`);
      const yoyCls = yoy == null ? "" : yoy >= 0 ? "pos" : "neg";
      const sector = [s.market, s.sector].filter(Boolean).join(" · ");
      return `<tr${rowClass}>
        <td><strong>${esc(s.name || s.code)}</strong>${aiRecBadge}
            <span style="font-size:.76em;color:var(--muted);margin-left:4px">${esc(s.code)}</span></td>
        <td style="font-size:.8em;color:var(--muted)">${esc(sector) || "—"}</td>
        <td style="text-align:right">${s.per != null ? s.per.toFixed(1) + "x" : "—"}</td>
        <td style="text-align:right">${s.pbr != null ? s.pbr.toFixed(2) + "x" : "—"}</td>
        <td style="text-align:right;color:var(--buy)">${s.roe != null ? s.roe.toFixed(1) + "%" : "—"}</td>
        <td style="text-align:right">${s.debt_ratio != null ? s.debt_ratio.toFixed(1) + "%" : "—"}</td>
        <td class="${yoyCls}" style="text-align:right">${yoyTxt}</td>
        <td style="text-align:right;color:${fColor};font-weight:700">${fScore}/9</td>
        <td style="text-align:right;color:${vColor};font-weight:700">${vScore.toFixed(1)}</td>
      </tr>`;
    }).join("");

    if (card) card.style.display = "";
  } catch (e) {
    const msg = e.name === "AbortError" ? "응답 없음 (200초 초과)" : e.message;
    setStatus("val-status", `오류: ${msg}`, true);
  } finally {
    clearTimeout(abortTimer);
    if (btn) btn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════
// Tab 8 — 우량주 스크리너
// ═══════════════════════════════════════════════════════

async function loadQualityDefaults() {
  try {
    const d = await api("/api/quality_stocks/filters");
    const qualRoeMin  = document.getElementById("qual-roe-min");
    const qualOpMgn   = document.getElementById("qual-op-margin-min");
    const qualYoyMin  = document.getElementById("qual-yoy-min");
    const qualDebtMax = document.getElementById("qual-debt-max");
    const qualPbrMax  = document.getElementById("qual-pbr-max");
    if (d.roe_min       != null && qualRoeMin)  qualRoeMin.value  = d.roe_min;
    if (d.op_margin_min != null && qualOpMgn)   qualOpMgn.value   = d.op_margin_min;
    if (d.yoy_min       != null && qualYoyMin)  qualYoyMin.value  = d.yoy_min;
    if (d.debt_max      != null && qualDebtMax) qualDebtMax.value = d.debt_max;
    if (d.pbr_max       != null && qualPbrMax)  qualPbrMax.value  = d.pbr_max;
  } catch (e) { /* 기본값 유지 */ }
}

async function runQualityScreener() {
  const marketEl    = document.getElementById("qual-market");
  const roeMinEl    = document.getElementById("qual-roe-min");
  const opMarginEl  = document.getElementById("qual-op-margin-min");
  const yoyMinEl    = document.getElementById("qual-yoy-min");
  const debtMaxEl   = document.getElementById("qual-debt-max");
  const pbrMaxEl    = document.getElementById("qual-pbr-max");
  const candLimitEl = document.getElementById("qual-candidate-limit");
  if (!marketEl || !roeMinEl || !opMarginEl || !yoyMinEl || !debtMaxEl || !pbrMaxEl || !candLimitEl) return;
  const market         = marketEl.value;
  const roeMin         = roeMinEl.value;
  const opMarginMin    = opMarginEl.value;
  const yoyMin         = yoyMinEl.value;
  const debtMax        = debtMaxEl.value;
  const pbrMax         = pbrMaxEl.value;
  const candidateLimit = parseInt(candLimitEl.value, 10);

  setStatus("qual-status", `⏳ 스크리닝 중… 시가총액 상위 ${candidateLimit}종목 펀더멘털 수집 + 필터링 (${candidateLimit > 200 ? "2~3분" : "1~2분"} 소요)`);

  const card  = document.getElementById("qual-result-card");
  const tbody = document.getElementById("qual-tbody");
  if (card) card.style.display = "none";
  if (tbody) tbody.innerHTML = "";

  const params = new URLSearchParams({
    market, roe_min: roeMin, op_margin_min: opMarginMin, yoy_min: yoyMin,
    debt_max: debtMax, pbr_max: pbrMax, candidate_limit: candidateLimit,
  });

  const btn = document.getElementById("btn-run-quality");
  if (btn) btn.disabled = true;
  const controller = new AbortController();
  const abortTimer = setTimeout(() => controller.abort(), 200000);
  try {
    const [data, aiRec] = await Promise.all([
      api(`/api/quality_stocks?${params}`, { signal: controller.signal }),
      fetchAiRecCodes(),
    ]);
    const stocks = data.stocks || [];
    const aiMap  = aiRec.map;
    const aiDate = aiRec.date;
    setStatus("qual-status", "");

    if (!stocks.length) {
      setStatus("qual-status", "필터 조건을 통과한 종목이 없습니다. 임계값을 완화해 보세요.", true);
      return;
    }

    const aiOverlap = stocks.filter(s => aiMap.has(s.code)).length;
    const aiNote    = aiOverlap > 0 && aiDate ? ` · AI추천(${esc(aiDate)}) ${aiOverlap}종목 포함` : "";
    const qualMeta  = document.getElementById("qual-result-meta");
    if (qualMeta) qualMeta.textContent = `${stocks.length}개 종목 (우량점수 내림차순)${aiNote}`;
    if (!tbody) return;

    tbody.innerHTML = stocks.map(s => {
      const isAiRec    = aiMap.has(s.code);
      const aiRecBadge = isAiRec ? `<span class="ai-rec-badge">★ AI추천</span>` : "";
      const rowClass   = isAiRec ? ' class="ai-rec-row"' : "";
      const qScore  = s.quality_score ?? 0;
      const qColor  = qScore >= 70 ? "var(--buy)" : qScore >= 50 ? "var(--hold)" : "var(--sell)";
      const yoy     = s.op_income_yoy;
      const yoyTxt  = yoy == null ? "—" : (yoy >= 0 ? `+${yoy.toFixed(1)}%` : `${yoy.toFixed(1)}%`);
      const yoyCls  = yoy == null ? "" : yoy >= 0 ? "pos" : "neg";
      const sector  = [s.market, s.sector].filter(Boolean).join(" · ");
      const margin  = s.op_margin;
      const div     = s.dividend_yield;
      return `<tr${rowClass}>
        <td><strong>${esc(s.name || s.code)}</strong>${aiRecBadge}
            <span style="font-size:.76em;color:var(--muted);margin-left:4px">${esc(s.code)}</span></td>
        <td style="font-size:.8em;color:var(--muted)">${esc(sector) || "—"}</td>
        <td style="text-align:right;color:var(--buy)">${s.roe != null ? s.roe.toFixed(1) + "%" : "—"}</td>
        <td style="text-align:right">${margin != null ? margin.toFixed(1) + "%" : "—"}</td>
        <td class="${yoyCls}" style="text-align:right">${yoyTxt}</td>
        <td style="text-align:right">${s.debt_ratio != null ? s.debt_ratio.toFixed(1) + "%" : "—"}</td>
        <td style="text-align:right">${s.pbr != null ? s.pbr.toFixed(2) + "x" : "—"}</td>
        <td style="text-align:right">${div != null ? div.toFixed(2) + "%" : "—"}</td>
        <td style="text-align:right;color:${qColor};font-weight:700">${qScore.toFixed(1)}</td>
      </tr>`;
    }).join("");

    if (card) card.style.display = "";
  } catch (e) {
    const msg = e.name === "AbortError" ? "응답 없음 (200초 초과)" : e.message;
    setStatus("qual-status", `오류: ${msg}`, true);
  } finally {
    clearTimeout(abortTimer);
    if (btn) btn.disabled = false;
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
  loadMacroBanner();     // 거시 레짐 배너 (탭1·탭3 공용)
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
      const theme = document.getElementById("rec-theme")?.value ?? "전체";
      const optimal = (theme === "전체") ? 9 : 5;
      if (parseInt(limitSel.value, 10) !== optimal) {
        limitHint.textContent = `✏️ 수동 설정 (권장: ${optimal}개)`;
        limitHint.style.color = "var(--hold)";
      } else {
        autoSetRecLimit();
      }
    });
  }
  loadOutcomes(_outcomeDays);
  initOutcomeDaysBtns();

  // 탭 4 — Backtest
  initStrategyFilter();

  // 탭 5 — Settings
  loadTelegramStatus();
  api("/api/version").then(d => {
    const el = document.getElementById("app-version");
    if (el) el.textContent = "v" + d.version;
  }).catch(() => {});

  // 거래일 여부 확인 (Tab 3, Tab 5 안내)
  checkTradingDay();

  // 페이지 이탈 시 폴링 interval 정리
  window.addEventListener("beforeunload", () => {
    if (_recPollId !== null) { clearInterval(_recPollId); _recPollId = null; }
  });
});
