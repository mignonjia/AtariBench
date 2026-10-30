const state = {
  data: null,
  sortKey: "rank",
  sortDirection: "asc",
  modelFilter: "",
  promptFilter: "all",
};

const numberFormat = new Intl.NumberFormat("en-US");
const compactFormat = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

const BASELINE_KEYS = new Set(["rl", "random"]);

function isBaseline(row) {
  return BASELINE_KEYS.has(row.model_key);
}

function modelLabel(row) {
  return row.prompt_mode === "append_only" && !isBaseline(row) ? "AO" : "";
}

function normalizeId(value) {
  return String(value).replaceAll("_", " ");
}

async function loadData() {
  const response = await fetch("data/leaderboard.json?v=20261012-videos-2");
  if (!response.ok) {
    throw new Error(`Failed to load leaderboard data: ${response.status}`);
  }
  state.data = await response.json();
  initializeControls();
  renderPage();
}

function initializeControls() {
  const promptSelect = document.querySelector("#prompt-filter");

  document.querySelector("#model-filter").addEventListener("input", (event) => {
    state.modelFilter = event.target.value.trim().toLowerCase();
    renderLeaderboard();
  });
  promptSelect.addEventListener("change", (event) => {
    state.promptFilter = event.target.value;
    renderLeaderboard();
  });

  document.querySelector(".leaderboard-table").addEventListener("click", (event) => {
    const header = event.target.closest("th[data-sort]");
    if (!header) {
      return;
    }
    const key = header.dataset.sort;
    if (state.sortKey === key) {
      state.sortDirection = state.sortDirection === "asc" ? "desc" : "asc";
    } else {
      state.sortKey = key;
      state.sortDirection = key === "rank" || key === "model" || key === "think" ? "asc" : "desc";
    }
    renderLeaderboard();
  });
}

function renderPage() {
  renderStats();
  renderMetric();
  renderMontage();
  renderLeaderboard();
  renderVideos();
}

function renderStats() {
  document.querySelector("#stat-models").textContent = numberFormat.format(state.data.leaderboard.length);
  document.querySelector("#stat-games").textContent = numberFormat.format(state.data.games.length);
}

function renderMetric() {
  document.querySelector("#metric-note").textContent = state.data.metric;
}

function videoModelName(video) {
  const row = state.data.leaderboard.find((entry) => entry.model_key === video.model);
  return row ? row.model : video.model;
}

function renderMontage() {
  const montage = document.querySelector("#hero-montage");
  montage.replaceChildren(
    ...state.data.videos.slice(0, 2).map((video) => {
      const tile = document.createElement("div");
      tile.className = "montage-tile";
      const src = video.hero_src || video.src;
      tile.innerHTML = `
        <div class="montage-video">
          <video src="${src}" autoplay muted loop playsinline preload="metadata"></video>
        </div>
        <div class="montage-label">${normalizeId(video.game)} · ${videoModelName(video)}</div>
      `;
      return tile;
    }),
  );
}

function filteredRows() {
  return state.data.leaderboard
    .filter((row) => !state.modelFilter || row.model.toLowerCase().includes(state.modelFilter))
    .filter((row) => state.promptFilter === "all" || isBaseline(row) || row.prompt_mode === state.promptFilter)
    .sort((a, b) => compareRows(a, b));
}

function compareRows(a, b) {
  const direction = state.sortDirection === "asc" ? 1 : -1;
  const keyMap = {
    rank: "rank",
    model: "model",
    think: "think",
    score: "score",
    avgRank: "avg_rank",
    games: "games",
    totalRuns: "total_runs",
    avgReward: "avg_reward",
  };
  const key = keyMap[state.sortKey];
  const aValue = a[key];
  const bValue = b[key];
  if (typeof aValue === "string") {
    return aValue.localeCompare(bValue) * direction;
  }
  return (Number(aValue) - Number(bValue)) * direction;
}

function renderLeaderboard() {
  const head = document.querySelector("#leaderboard-head");
  const body = document.querySelector("#leaderboard-body");
  head.innerHTML = `
    <tr>
      <th data-sort="rank" class="rank-cell sticky-col sticky-rank">Rank</th>
      <th data-sort="model" class="sticky-col sticky-model">Model</th>
      <th data-sort="think" class="sticky-col sticky-think">Think</th>
      <th data-sort="score" class="sticky-col sticky-score">Score</th>
      <th data-sort="avgRank" class="sticky-col sticky-avg-rank">Avg Rank</th>
      ${state.data.games.map((game) => `<th class="game-head">${normalizeId(game)}</th>`).join("")}
    </tr>
  `;
  body.replaceChildren(
    ...filteredRows().map((row) => {
      const tr = document.createElement("tr");
      const gameCells = state.data.games
        .map((game) => {
          const cellData = row.per_game[game];
          if (!cellData) {
            return `<td class="heat-cell missing-cell">-</td>`;
          }
          const score = Math.round(cellData.normalized_score);
          return `
            <td
              class="heat-cell"
              style="background: ${heatColor(score)}; color: ${score > 62 ? "#10201d" : "#20312d"}"
              title="${normalizeId(game)} · rank ${cellData.rank} · reward ${cellData.reward}"
            >${score}</td>
          `;
        })
        .join("");
      tr.innerHTML = `
        <td class="rank-cell sticky-col sticky-rank">${row.rank}</td>
        <td class="sticky-col sticky-model">
          <div class="model-name">
            <strong>${row.model}</strong>
            <span class="mode-tag" title="Append-only prompting">${modelLabel(row)}</span>
          </div>
        </td>
        <td class="sticky-col sticky-think">${isBaseline(row) ? "-" : row.think}</td>
        <td class="sticky-col sticky-score"><span class="score-pill">${formatNumber(row.score, 1)}</span></td>
        <td class="sticky-col sticky-avg-rank">${formatNumber(row.avg_rank, 2)}</td>
        ${gameCells}
      `;
      return tr;
    }),
  );
}

function heatColor(score) {
  const clamped = Math.max(0, Math.min(100, Number(score)));
  const hue = 15 + clamped * 1.35;
  const lightness = 92 - clamped * 0.34;
  const saturation = 72;
  return `hsl(${hue} ${saturation}% ${lightness}%)`;
}

function renderVideos() {
  const grid = document.querySelector("#video-grid");
  grid.replaceChildren(
    ...state.data.videos.map((video) => {
      const card = document.createElement("article");
      card.className = "video-card";
      card.innerHTML = `
        <div class="video-frame">
          <video src="${video.src}" muted loop playsinline controls preload="metadata"></video>
        </div>
        <div class="video-meta">
          <strong>${normalizeId(video.game)}</strong>
          <span>${videoModelName(video)} · thinking ${video.thinking_mode}</span>
        </div>
      `;
      return card;
    }),
  );
}

function initCarousel(root) {
  const track = root.querySelector(".carousel-track");
  const slides = [...track.children];
  const tabs = root.querySelector(".carousel-tabs");
  const counter = root.querySelector(".carousel-counter");
  let current = 0;

  const goTo = (index) => {
    const slide = slides[(index + slides.length) % slides.length];
    track.scrollTo({ left: slide.offsetLeft - (track.clientWidth - slide.offsetWidth) / 2 });
  };

  tabs.replaceChildren(
    ...slides.map((slide, index) => {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.setAttribute("role", "tab");
      tab.innerHTML = `<span class="tab-index">${index + 1}</span>${slide.dataset.title}`;
      tab.addEventListener("click", () => goTo(index));
      return tab;
    }),
  );

  const markCurrent = () => {
    const center = track.scrollLeft + track.clientWidth / 2;
    const distances = slides.map((slide) => Math.abs(slide.offsetLeft + slide.offsetWidth / 2 - center));
    current = distances.indexOf(Math.min(...distances));
    slides.forEach((slide, index) => slide.classList.toggle("is-active", index === current));
    [...tabs.children].forEach((tab, index) => tab.setAttribute("aria-selected", String(index === current)));
    counter.textContent = `${current + 1} / ${slides.length}`;
  };

  slides.forEach((slide, index) =>
    slide.addEventListener("click", () => {
      if (index !== current) goTo(index);
    }),
  );
  root.querySelector(".carousel-arrow.prev").addEventListener("click", () => goTo(current - 1));
  root.querySelector(".carousel-arrow.next").addEventListener("click", () => goTo(current + 1));
  track.addEventListener("scroll", markCurrent, { passive: true });
  root.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") goTo(current - 1);
    if (event.key === "ArrowRight") goTo(current + 1);
  });
  markCurrent();
}

document.querySelectorAll(".carousel").forEach(initCarousel);

loadData().catch((error) => {
  document.body.innerHTML = `<main class="section"><h1>AtariBench</h1><p>${error.message}</p></main>`;
});
