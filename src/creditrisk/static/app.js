const STORAGE_KEY = "creditrisk.onboarding.dismissed.v1";
const THEME_KEY = "creditrisk.theme";

let lastDecisionId = null;

function initOnboarding() {
  const backdrop = document.getElementById("onboarding-backdrop");
  const dismissBtn = document.getElementById("onboarding-dismiss");
  let dismissed = false;
  try {
    dismissed = !!window.localStorage.getItem(STORAGE_KEY);
  } catch {
    dismissed = false;
  }
  if (!dismissed) backdrop.hidden = false;

  dismissBtn.addEventListener("click", () => {
    backdrop.hidden = true;
    try {
      window.localStorage.setItem(STORAGE_KEY, "1");
    } catch {
      // ignore
    }
  });
}

function initTheme() {
  let theme = "dark";
  try {
    theme = window.localStorage.getItem(THEME_KEY) || "dark";
  } catch {
    // ignore
  }
  document.documentElement.setAttribute("data-theme", theme);
  const btn = document.getElementById("theme-toggle");
  btn.textContent = theme === "dark" ? "☀︎" : "☾";
  btn.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    btn.textContent = next === "dark" ? "☀︎" : "☾";
    try {
      window.localStorage.setItem(THEME_KEY, next);
    } catch {
      // ignore
    }
  });
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${url} failed: ${res.status}`);
  }
  return res.json();
}

async function loadHealth() {
  const health = await fetchJson("/healthz");
  const strip = document.getElementById("stat-strip");
  strip.innerHTML = "";
  const items = [
    ["modello in produzione", health.production_model_version || "—"],
    ["stadio", health.production_model_stage || "—"],
    ["decisioni registrate", health.inference_log_size],
  ];
  for (const [label, value] of items) {
    const div = document.createElement("div");
    div.className = "stat-item";
    div.innerHTML = `<span class="stat-value">${value}</span><span class="stat-label">${label}</span>`;
    strip.appendChild(div);
  }
}

function formToJson(form) {
  const data = Object.fromEntries(new FormData(form).entries());
  const out = {};
  for (const [key, value] of Object.entries(data)) {
    out[key] = form.elements[key].type === "number" ? Number(value) : value;
  }
  return out;
}

async function submitScore(event) {
  event.preventDefault();
  const form = event.target;
  const body = formToJson(form);
  body.application_id = `console-${Date.now()}`;

  const box = document.getElementById("score-result");
  box.textContent = "Valutazione in corso...";
  try {
    const result = await fetchJson("/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    lastDecisionId = result.decision_id;
    box.innerHTML = `
      <span class="decision-pill ${result.decision}">${result.decision.toUpperCase()}</span>
      Probabilità di default: ${(result.probability_of_default * 100).toFixed(1)}%
      ID decisione: ${result.decision_id}
    `;
    document.querySelector('#override-form input[name="decision_id"]').value = result.decision_id;
    renderReasonCodes(result.reason_codes);
    loadHealth();
  } catch (err) {
    box.textContent = `Errore: ${err.message}`;
  }
}

function renderReasonCodes(codes) {
  const container = document.getElementById("reason-codes");
  if (!codes.length) {
    container.innerHTML = '<p class="muted">Nessun reason code.</p>';
    return;
  }
  const maxAbs = Math.max(...codes.map((c) => Math.abs(c.contribution)), 0.001);
  container.innerHTML = codes
    .map((c) => {
      const pct = (Math.abs(c.contribution) / maxAbs) * 50;
      const cls = c.direction === "increases_risk" ? "positive" : "negative";
      return `
        <div class="reason-row">
          <span class="reason-name">${c.feature_name}</span>
          <div class="reason-bar-track"><div class="reason-bar-fill ${cls}" style="width:${pct}%"></div></div>
          <span class="reason-value">${c.contribution.toFixed(3)}</span>
        </div>
      `;
    })
    .join("");
}

async function loadFairness() {
  const report = await fetchJson("/fairness");
  document.getElementById("fairness-summary").textContent =
    `parità demografica: gap ${(report.demographic_parity_difference * 100).toFixed(1)}pp · pari opportunità: gap ${(report.equal_opportunity_difference * 100).toFixed(1)}pp`;
  const list = document.getElementById("fairness-list");
  list.innerHTML = report.groups
    .map(
      (g) => `
      <div class="fairness-item">
        <div class="fairness-head"><span>${g.group}</span><span>n=${g.n}</span></div>
        <div class="fairness-metric"><span>tasso approvazione</span><span>${(g.approval_rate * 100).toFixed(1)}%</span></div>
        <div class="fairness-metric"><span>vero-positivo</span><span>${(g.true_positive_rate * 100).toFixed(1)}%</span></div>
      </div>
    `
    )
    .join("");
}

async function submitOverride(event) {
  event.preventDefault();
  const form = event.target;
  const body = formToJson(form);
  const decisionId = body.decision_id;
  delete body.decision_id;

  const box = document.getElementById("override-result");
  box.textContent = "Applicazione in corso...";
  try {
    const result = await fetchJson(`/override/${encodeURIComponent(decisionId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    box.textContent =
      `Decisione originale: ${result.original_decision}\n` +
      `Decisione corretta: ${result.overridden_decision}\n` +
      `Motivazione: ${result.reason}\n` +
      `Revisore: ${result.overridden_by}`;
  } catch (err) {
    box.textContent = `Errore: ${err.message}`;
  }
}

async function loadDossier() {
  const pre = document.getElementById("dossier");
  pre.textContent = "Caricamento...";
  try {
    const res = await fetch("/dossier");
    pre.textContent = await res.text();
  } catch {
    pre.textContent = "Impossibile caricare il dossier.";
  }
}

async function main() {
  initOnboarding();
  initTheme();
  document.getElementById("score-form").addEventListener("submit", submitScore);
  document.getElementById("override-form").addEventListener("submit", submitOverride);
  document.getElementById("refresh-dossier").addEventListener("click", loadDossier);
  await Promise.all([loadHealth(), loadFairness(), loadDossier()]);
}

main();
