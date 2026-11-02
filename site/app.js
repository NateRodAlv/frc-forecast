/* FRC Forecast front end: static, reads ./data/*.json produced by backend/run.py. No frameworks, no build step. */
const S = { meta: null, teams: [], byNum: {}, events: [], matches: {}, sort: { key: "rank", dir: 1 } };
const app = document.getElementById("app");

// ---------- tiny safe DOM helper (never uses innerHTML, so TBA strings can't inject markup)
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) el.append(k.nodeType ? k : document.createTextNode(String(k)));
  return el;
}
const svg = (tag, attrs, ...kids) => {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  for (const k of kids.flat()) if (k != null) el.append(k.nodeType ? k : document.createTextNode(String(k)));
  return el;
};
const pct = (p, d = 1) => (p == null ? "–" : (p * 100).toFixed(p > 0 && p < 0.001 ? 2 : d) + "%");
const fx = (v, d = 2) => (v == null ? "–" : (v > 0 ? "+" : "") + v.toFixed(d));
const link = (href, text) => h("a", { href }, text);
const teamLink = (n) => link(`#/team/${n}`, String(n));
const tname = (n) => (S.byNum[n] && S.byNum[n].name) || "";
function bar(p) { return h("span", {}, pct(p), h("span", { class: "barbg" }, h("span", { class: "bar", style: `width:${Math.min(100, p * 100)}%` }))); }

async function load() {
  const get = (f) => fetch(`data/${f}`).then((r) => { if (!r.ok) throw new Error(f + " " + r.status); return r.json(); });
  [S.meta, S.teams, S.events, S.matches] = await Promise.all(["meta.json", "teams.json", "events.json", "matches.json"].map(get));
  S.teams.forEach((t) => (S.byNum[t.team] = t));
  document.getElementById("gen").textContent = " Updated " + new Date(S.meta.generated).toLocaleString() + ".";
  const warn = [...(S.meta.warnings || [])];
  if (warn.length) { const b = document.getElementById("banner"); b.hidden = false; b.textContent = warn.join("  •  "); }
}

// ---------- views
function table(cols, rows, opts = {}) {
  const sortState = opts.sort;
  const head = h("tr", {}, cols.map((c) => h("th", { class: c.left ? "l" : "", title: c.tip || "", onclick: sortState ? () => { sortState.dir = sortState.key === c.key ? -sortState.dir : (c.defDir || -1); sortState.key = c.key; opts.rerender(); } : null },
    c.label + (sortState && sortState.key === c.key ? (sortState.dir > 0 ? " ▲" : " ▼") : ""))));
  const body = rows.map((r) => h("tr", {}, cols.map((c) => h("td", { class: (c.left ? "l " : "") + (c.cls || "") }, c.render ? c.render(r) : r[c.key]))));
  return h("div", { class: "tablewrap" }, h("table", {}, h("thead", {}, head), h("tbody", {}, body)));
}

function flagPills(flags) {
  return (flags || []).map((f) => h("span", { class: "pill " + (/upside|rebound|breakout/.test(f) ? "up" : /downside|decline|drawdown/.test(f) ? "down" : "") }, f));
}

function viewRankings() {
  const q = (location.hash.split("?q=")[1] || "").toLowerCase();
  const rerender = () => draw();
  let rows = S.teams.filter((t) => !q || String(t.team).includes(q) || (t.name || "").toLowerCase().includes(q));
  const keyf = { rank: (t) => t.rank, team: (t) => t.team, mean: (t) => t.forecast.mean, now: (t) => t.z_now ?? -9,
    delta: (t) => t.forecast.mean - (t.z_now ?? t.forecast.mean), dcmp: (t) => (t.adv.p_dcmp ?? t.adv.p_direct ?? -1), cmp: (t) => t.adv.p_cmp ?? -1 }[S.sort.key] || ((t) => t.rank);
  function draw() {
    rows = rows.slice().sort((a, b) => (keyf(a) - keyf(b)) * S.sort.dir);
    const cols = [
      { key: "rank", label: "#", defDir: 1, render: (t) => t.rank },
      { key: "team", label: "Team", left: true, defDir: 1, render: (t) => h("span", {}, teamLink(t.team), " ", h("span", { class: "muted small" }, t.name || "")) },
      { key: "mean", label: `${S.meta.target_year} forecast`, tip: "Predicted strength in standard deviations above the field average", render: (t) => `${fx(t.forecast.mean)} ± ${t.forecast.sd.toFixed(2)}` },
      { key: "now", label: "Last season", render: (t) => fx(t.z_now) },
      { key: "delta", label: "Δ", tip: "Forecast minus last season", render: (t) => { const d = t.forecast.mean - (t.z_now ?? t.forecast.mean); return h("span", { class: d > 0.05 ? "pos" : d < -0.05 ? "neg" : "" }, fx(d)); } },
      { key: "flags", label: "Trend signals", left: true, cls: "flags", render: (t) => flagPills(t.forecast.flags) },
      { key: "dcmp", label: "District Champ / Direct", tip: "P(qualify for district championship) or P(direct Worlds spot at a regional)", render: (t) => pct(t.adv.p_dcmp ?? t.adv.p_direct) },
      { key: "cmp", label: "Worlds", render: (t) => (t.adv.p_cmp != null ? bar(t.adv.p_cmp) : "–") },
    ];
    app.replaceChildren(h("h1", {}, `Team rankings — ${S.meta.target_year} forecast`),
      h("p", { class: "muted" }, "Strength is an end-of-season rating standardised within each year (0 = average team, +1 = one standard deviation better). ",
        "The forecast comes from the team's own history plus how teams with similar trajectories actually did."),
      table(cols.slice(0, 8).filter((c) => c.key !== "flags" || true), rows.slice(0, 400), { sort: S.sort, rerender }),
      rows.length > 400 ? h("p", { class: "muted small" }, `Showing 400 of ${rows.length}. Use search to find a team.`) : null);
  }
  draw();
}

function trajectoryChart(t) {
  const hist = t.forecast.history || [], f = t.forecast, W = 520, H = 190, P = 28;
  const yrs = hist.map((x) => x[0]).concat([S.meta.target_year]);
  const vals = hist.map((x) => x[1]).concat([f.q10 ?? f.mean - f.sd, f.q90 ?? f.mean + f.sd, f.mean]);
  const lo = Math.min(-2, ...vals) - 0.2, hi = Math.max(2, ...vals) + 0.2;
  const x = (i) => P + (i * (W - 2 * P)) / Math.max(1, yrs.length - 1), y = (v) => H - P - ((v - lo) / (hi - lo)) * (H - 2 * P);
  const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Strength history and forecast" });
  g.append(svg("line", { x1: P, x2: W - P, y1: y(0), y2: y(0), stroke: "var(--line)" }));
  hist.forEach((p, i) => g.append(svg("text", { x: x(i), y: H - 8, "text-anchor": "middle" }, p[0])));
  g.append(svg("text", { x: x(yrs.length - 1), y: H - 8, "text-anchor": "middle" }, S.meta.target_year + " (fc)"));
  if (hist.length) {
    const pts = hist.map((p, i) => `${x(i)},${y(p[1])}`).join(" ");
    g.append(svg("polyline", { points: pts, fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
    hist.forEach((p, i) => g.append(svg("circle", { cx: x(i), cy: y(p[1]), r: 3, fill: "var(--accent)" })));
    g.append(svg("line", { x1: x(hist.length - 1), y1: y(hist[hist.length - 1][1]), x2: x(yrs.length - 1), y2: y(f.mean), stroke: "var(--accent)", "stroke-dasharray": "4 3" }));
  }
  const xf = x(yrs.length - 1);
  g.append(svg("line", { x1: xf, x2: xf, y1: y(f.q90 ?? f.mean + f.sd), y2: y(f.q10 ?? f.mean - f.sd), stroke: "var(--accent)", "stroke-width": 8, opacity: 0.2, "stroke-linecap": "round" }));
  g.append(svg("circle", { cx: xf, cy: y(f.mean), r: 5, fill: "var(--panel)", stroke: "var(--accent)", "stroke-width": 2 }));
  return g;
}

function miniPath(path, next) {
  const all = path.concat([next]), W = 110, H = 28, lo = Math.min(...all, -1.5), hi = Math.max(...all, 1.5);
  const pts = all.map((v, i) => `${(i * (W - 8)) / (all.length - 1) + 4},${H - 3 - ((v - lo) / (hi - lo)) * (H - 6)}`);
  return svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H },
    svg("polyline", { points: pts.slice(0, -1).join(" "), fill: "none", stroke: "var(--muted)", "stroke-width": 1.5 }),
    svg("line", { x1: pts[pts.length - 2].split(",")[0], y1: pts[pts.length - 2].split(",")[1], x2: pts[pts.length - 1].split(",")[0], y2: pts[pts.length - 1].split(",")[1], stroke: next >= path[path.length - 1] ? "var(--up)" : "var(--down)", "stroke-width": 2 }));
}

function viewTeam(n) {
  const t = S.byNum[n];
  if (!t) return app.replaceChildren(h("h1", {}, `Team ${n}`), h("p", { class: "muted" }, "Not in this model's data."));
  const f = t.forecast, a = t.adv || {};
  const evs = S.events.filter((e) => e.sim && e.sim.teams[n]);
  app.replaceChildren(
    h("h1", {}, `Team ${n} ${t.name ? "· " + t.name : ""}`),
    h("div", { class: "row" }, flagPills(f.flags), h("span", { class: "muted small" }, `Rank #${t.rank} of ${S.teams.length} · rookie year ${t.rookie_year ?? "?"}`)),
    h("div", { class: "grid" },
      h("div", { class: "card stat" }, "Forecast strength", h("b", {}, fx(f.mean)), h("span", { class: "muted small" }, f.q10 != null ? `80% range ${fx(f.q10)} to ${fx(f.q90)}` : `± ${f.sd.toFixed(2)}`)),
      a.p_dcmp != null ? h("div", { class: "card stat" }, "District championship", h("b", {}, pct(a.p_dcmp)), h("span", { class: "muted small" }, `Exp. district pts ${a.exp_district_pts}`)) : null,
      a.p_direct != null ? h("div", { class: "card stat" }, "Direct Worlds (event win-type)", h("b", {}, pct(a.p_direct)), h("span", { class: "muted small" }, `Exp. regional pts ${a.exp_regional_pts}`)) : null,
      a.p_cmp != null ? h("div", { class: "card stat" }, "FIRST Championship", h("b", {}, pct(a.p_cmp)), h("span", { class: "muted small" }, a.district ? `${S.meta.districts[a.district]?.name || a.district} model` : "Regional model")) : null),
    h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "Trajectory"), trajectoryChart(t),
      h("p", { class: "muted small" }, "Solid: past seasons. Dashed + band: forecast and 10–90% range.",
        f.analog_edge != null ? ` Analog teams pull the forecast ${fx(f.analog_edge)} vs. a plain trend model.` : "")),
    f.analogs && f.analogs.length ? h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "Teams that looked like this before"),
      h("p", { class: "muted small" }, "Past team-seasons with the closest trajectory shape, and what they did the next year (green = improved)."),
      table([{ label: "Team", left: true, render: (r) => teamLink(r.team) }, { label: "Season", render: (r) => r.year },
        { label: "Path", render: (r) => miniPath(r.path, r.next) }, { label: "Next season", render: (r) => fx(r.next) }], f.analogs)) : null,
    evs.length ? h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, `${S.meta.target_year} events`),
      table([{ label: "Event", left: true, render: (e) => link(`#/event/${e.key}`, e.name) }, { label: "Date", render: (e) => e.start },
        { label: "Win", render: (e) => pct(e.sim.teams[n].p_win) }, { label: "Finalist", render: (e) => pct(e.sim.teams[n].p_finalist) },
        { label: "Exp. rank", render: (e) => e.sim.teams[n].exp_rank }], evs)) : null);
}

function viewEvents() {
  const types = [...new Set(S.events.map((e) => e.type))];
  const state = { type: "", q: "", year: String(S.meta.target_year) };
  const years = [...new Set(S.events.map((e) => e.year))].sort().reverse();
  const out = h("div", {});
  function draw() {
    const rows = S.events.filter((e) => (!state.type || e.type === state.type) && (!state.year || String(e.year) === state.year) && (!state.q || e.name.toLowerCase().includes(state.q) || (e.district || "").includes(state.q))).sort((a, b) => a.start.localeCompare(b.start));
    out.replaceChildren(table([{ label: "Event", left: true, render: (e) => link(`#/event/${e.key}`, e.name) }, { label: "Type", left: true, render: (e) => e.type + (e.district ? " · " + e.district : "") },
      { label: "Start", render: (e) => e.start }, { label: "Teams", render: (e) => e.n_teams || "–" }, { label: "Schedule", render: (e) => (e.has_schedule ? "published" : "–") },
      { label: "Predicted", render: (e) => (e.sim ? "yes" : "–") }], rows));
  }
  app.replaceChildren(h("h1", {}, "Events"),
    h("div", { class: "row" },
      h("select", { onchange: (ev) => { state.year = ev.target.value; draw(); } }, years.map((y) => h("option", { value: y, selected: String(y) === state.year }, y))),
      h("select", { onchange: (ev) => { state.type = ev.target.value; draw(); } }, h("option", { value: "" }, "All types"), types.map((t) => h("option", { value: t }, t))),
      h("input", { type: "search", placeholder: "Filter events", oninput: (ev) => { state.q = ev.target.value.toLowerCase(); draw(); } })), out);
  draw();
}

function viewEvent(key) {
  const e = S.events.find((x) => x.key === key);
  if (!e) return app.replaceChildren(h("h1", {}, "Event not found"));
  const ms = S.matches[key] || [];
  const parts = [h("h1", {}, e.name), h("p", { class: "muted" }, `${e.type}${e.district ? " · " + e.district : ""} · ${e.start} to ${e.end} · ${e.n_teams} teams`)];
  if (e.sim) {
    const rows = Object.entries(e.sim.teams).map(([t, v]) => ({ team: +t, ...v })).sort((a, b) => b.p_win - a.p_win);
    parts.push(h("h2", {}, "Event prediction"), h("p", { class: "muted small" }, e.sim.scheduled ? "Uses the published qualification schedule." : "Qualification schedule not published yet: uses random balanced schedules.",
      " Assumes the standard 8-alliance double-elimination bracket."),
      table([{ label: "Team", left: true, render: (r) => h("span", {}, teamLink(r.team), " ", h("span", { class: "muted small" }, tname(r.team))) },
        { label: "Win event", render: (r) => bar(r.p_win) }, { label: "Finalist", render: (r) => pct(r.p_finalist) }, { label: "Top 8 seed", render: (r) => pct(r.p_top8) },
        { label: "Exp. rank", render: (r) => r.exp_rank }, { label: "Exp. points", render: (r) => r.exp_pts }], rows));
  }
  if (ms.length) {
    parts.push(h("h2", {}, "Matches"), table([{ label: "Match", left: true, render: (m) => `${m.lvl.toUpperCase()}${m.lvl === "qm" ? "" : " " + m.set + "-"}${m.num}` },
      { label: "Red", left: true, render: (m) => h("span", { class: "match-red" }, m.red.map((t, i) => [i ? ", " : "", teamLink(t)])) },
      { label: "Blue", left: true, render: (m) => h("span", { class: "match-blue" }, m.blue.map((t, i) => [i ? ", " : "", teamLink(t)])) },
      { label: "P(red wins)", render: (m) => pct(m.p_red, 0) },
      { label: "Result", render: (m) => (m.rs == null ? "–" : `${m.rs}–${m.bs} ${m.rs === m.bs ? "tie" : (m.rs > m.bs) === (m.p_red >= 0.5) ? "✓" : "✗"}`) }], ms.slice(0, 300)));
  }
  app.replaceChildren(...parts);
}

function viewAdvancement() {
  const dists = Object.entries(S.meta.districts || {});
  const state = { d: dists.length ? dists[0][0] : "" };
  const out = h("div", {});
  function draw() {
    const d = S.meta.districts[state.d];
    if (!d) { out.replaceChildren(h("p", { class: "muted" }, "No district/regional events with rosters yet for this year.")); return; }
    const isReg = state.d === "_regional";
    const rows = S.teams.filter((t) => t.adv && (isReg ? t.adv.model === "regional" : t.adv.district === state.d)).sort((a, b) => b.adv.p_cmp - a.adv.p_cmp);
    const summary = isReg ? `${d.teams} teams · ${d.direct_slots} direct slots at events · ${d.pool_slots} Regional Pool slots`
      : `${d.teams} teams · District Championship capacity ${d.dcmp_capacity} · ${d.cmp_slots} Worlds slots`;
    out.replaceChildren(h("p", { class: "muted" }, summary),
      table(isReg ? [{ label: "Team", left: true, render: (t) => h("span", {}, teamLink(t.team), " ", h("span", { class: "muted small" }, t.name || "")) },
        { label: "Direct (event)", render: (t) => pct(t.adv.p_direct) }, { label: "Worlds (any route)", render: (t) => bar(t.adv.p_cmp) }, { label: "Exp. pts", render: (t) => t.adv.exp_regional_pts }]
        : [{ label: "Team", left: true, render: (t) => h("span", {}, teamLink(t.team), " ", h("span", { class: "muted small" }, t.name || "")) },
          { label: "→ District Champ", render: (t) => bar(t.adv.p_dcmp) }, { label: "→ Worlds", render: (t) => bar(t.adv.p_cmp) },
          { label: "Exp. district pts", render: (t) => t.adv.exp_district_pts }], rows.slice(0, 300)));
  }
  app.replaceChildren(h("h1", {}, `Advancement odds — ${S.meta.target_year}`),
    h("p", { class: "muted" }, "District teams: first two district events → district championship (points ×3 there) → Worlds. Regional teams: top finishers at each event, then the Regional Pool. Rules come from the 2026 manual; the 2027 manual publishes January 9, 2027."),
    h("div", { class: "row" }, h("select", { onchange: (ev) => { state.d = ev.target.value; draw(); } },
      dists.map(([k, v]) => h("option", { value: k }, k === "_regional" ? "Regionals" : v.name)))), out);
  draw();
}

function viewPredict() {
  const mk = (cls, id) => h("div", { class: cls }, h("b", {}, id === "r" ? "Red alliance" : "Blue alliance"), h("div", { class: "alliance" }, [0, 1, 2].map((i) => h("input", { type: "number", id: `${id}${i}`, placeholder: "team #" }))));
  const out = h("div", { class: "card" }, h("span", { class: "muted" }, "Enter six teams."));
  function go() {
    const get = (id) => [0, 1, 2].map((i) => +document.getElementById(id + i).value).filter(Boolean);
    const R = get("r"), B = get("b");
    if (R.length + B.length < 2 || !R.length || !B.length) return;
    const st = (t) => (S.byNum[t] ? [S.byNum[t].forecast.mean, S.byNum[t].forecast.sd] : [-0.3, 0.9]);
    const agg = (ts) => ts.reduce((a, t) => { const [m, s] = st(t); return [a[0] + m, a[1] + s * s]; }, [0, 0]);
    const [mr, vr] = agg(R), [mb, vb] = agg(B), beta = S.meta.beta, be = beta / Math.sqrt(1 + (Math.PI / 8) * beta * beta * (vr + vb));
    const p = 1 / (1 + Math.exp(-be * (mr - mb)));
    out.replaceChildren(h("div", { class: "prob" }, `Red ${pct(p, 0)} · Blue ${pct(1 - p, 0)}`),
      h("p", { class: "muted small" }, `Alliance strength: red ${fx(mr)}, blue ${fx(mb)}. Uses the ${S.meta.target_year} forecast for each team, including its uncertainty.`));
  }
  app.replaceChildren(h("h1", {}, "Match predictor"), h("div", { class: "grid" }, mk("red card", "r"), mk("blue card", "b")), h("div", { class: "row" }, h("button", { onclick: go }, "Predict")), out);
}

function viewModel() {
  const m = S.meta, fb = m.forecast_backtest;
  app.replaceChildren(h("h1", {}, "Model & validation"),
    h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "1. Which rating model ranks teams best?"),
      h("p", { class: "muted small" }, `Walk-forward backtest (every match predicted before the model sees its result), tuned on seasons before ${m.test_years[0]}, scored on ${m.test_years.join(", ")}. Lower log loss is better. Winner: `, h("b", {}, m.best_rating_model), "."),
      table([{ label: "Model", left: true, render: (r) => r.model }, { label: "Log loss", render: (r) => r.test_logloss.toFixed(4) }, { label: "Brier", render: (r) => r.test_brier.toFixed(4) },
        { label: "Accuracy", render: (r) => pct(r.test_acc) }, { label: "Matches", render: (r) => r.test_matches }], m.rating_leaderboard)),
    fb ? h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "2. Trajectory forecaster vs. baselines"),
      h("p", { class: "muted small" }, `Predicting each team's next-season strength, tested on ${fb.years.join(", ")} using only earlier seasons. RMSE in standard deviations; lower is better.`),
      table([{ label: "Method", left: true, render: (r) => r[0] }, { label: "RMSE", render: (r) => r[1].toFixed(3) }, { label: "MAE", render: (r) => r[2] }],
        [["Persistence (same as last year)", fb.persist.rmse, fb.persist.mae.toFixed(3)], ["Ridge (mean reversion + momentum)", fb.ridge.rmse, fb.ridge.mae.toFixed(3)],
         ["Analog teams (trajectory matching)", fb.analog.rmse, fb.analog.mae.toFixed(3)], [`Blend (analog weight ${m.blend_analog_weight})`, fb.blend.rmse, "–"]]),
      h("p", { class: "muted small" }, `10–90% analog interval coverage: ${pct(fb.analog_interval_coverage_10_90, 0)} (ideal 80%). The forecaster only leans on analogs as much as the backtest justifies.`)) : null,
    h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "3. Assumptions"),
      h("ul", {}, h("li", {}, m.rules_note), h("li", {}, m.game_note), h("li", {}, `${m.nsims} Monte Carlo simulations per run; each draws one strength per team and reuses it across all that team's events.`),
        h("li", {}, "Playoffs assume 8 alliances, double elimination (matches 1–13) then best-of-3 finals. Offseason events vary and are shown for strength only."))));
}

// ---------- router
function route() {
  const [path] = location.hash.slice(1).split("?");
  const seg = path.split("/").filter(Boolean);
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("on", a.dataset.nav === (seg[0] === "team" || !seg[0] ? "rankings" : seg[0] === "event" ? "events" : seg[0])));
  window.scrollTo(0, 0);
  try {
    if (!seg[0]) viewRankings();
    else if (seg[0] === "team") viewTeam(+seg[1]);
    else if (seg[0] === "events") viewEvents();
    else if (seg[0] === "event") viewEvent(seg[1]);
    else if (seg[0] === "advancement") viewAdvancement();
    else if (seg[0] === "predict") viewPredict();
    else if (seg[0] === "model") viewModel();
    else viewRankings();
  } catch (err) { app.replaceChildren(h("p", { class: "neg" }, "Something went wrong: " + err.message)); console.error(err); }
}
document.getElementById("search").addEventListener("input", (e) => {
  const v = e.target.value.trim();
  if (/^\d+$/.test(v) && S.byNum[+v]) location.hash = `#/team/${v}`; else location.hash = `#/?q=${encodeURIComponent(v)}`;
});
window.addEventListener("hashchange", route);
load().then(route).catch((e) => app.replaceChildren(h("p", { class: "neg" }, "Could not load data. Run backend/run.py first. " + e.message)));
