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

const UP_RE = /improve|jump|rebound|did better/, DOWN_RE = /regress|drop|below|did worse/;
const pill = (f) => h("span", { class: "pill " + (UP_RE.test(f) ? "up" : DOWN_RE.test(f) ? "down" : "") }, f);
// compact (table): last-season tags, an arrow, then outlook tags
function flagPills(fc) {
  const shape = fc.shape_flags || fc.flags || [], out = fc.outlook_flags || [];
  return [...shape.map(pill), out.length ? h("span", { class: "sep" }, shape.length ? "→" : "outlook:") : null, ...out.map(pill)];
}
// labelled (team page)
function flagGroups(fc) {
  const shape = fc.shape_flags || fc.flags || [], out = fc.outlook_flags || [];
  return [shape.length ? h("span", { class: "muted small" }, "Last season: ") : null, ...shape.map(pill),
    out.length ? h("span", { class: "muted small", style: "margin-left:8px" }, "Outlook: ") : null, ...out.map(pill)];
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
      { key: "flags", label: "Trend signals", left: true, cls: "flags", render: (t) => flagPills(t.forecast) },
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

const AGE_ORDER = ["1", "2", "3", "4", "5-7", "8-12", "13+", "?"];
const byAge = (o) => Object.entries(o).sort((x, y) => AGE_ORDER.indexOf(x[0]) - AGE_ORDER.indexOf(y[0]));
const ordinal = (n) => { n = Math.round(n); const s = ["th", "st", "nd", "rd"], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };

function trajectoryChart(t) {
  const f = t.forecast, W = 560, H = 215, P = 30;
  // hist rows: [year, rating (sd above season average), percentile, matches, raw model rating | null]
  const hist = t.hist && t.hist.length ? t.hist : (f.history || []).map((x) => [x[0], x[1], null, null, null]);
  const yrs = hist.map((x) => x[0]).concat([S.meta.target_year]);
  const vals = hist.map((x) => x[1]).concat([f.q10 ?? f.mean - f.sd, f.q90 ?? f.mean + f.sd, f.mean]);
  const lo = Math.min(-2, ...vals) - 0.2, hi = Math.max(2, ...vals) + 0.2;
  const x = (i) => P + (i * (W - 2 * P)) / Math.max(1, yrs.length - 1), y = (v) => H - P - ((v - lo) / (hi - lo)) * (H - 2 * P);
  const g = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Season rating history and forecast" });
  g.append(svg("line", { x1: P, x2: W - P, y1: y(0), y2: y(0), stroke: "var(--line)" }));
  g.append(svg("text", { x: 4, y: y(0) + 4 }, "avg"));
  hist.forEach((p, i) => g.append(svg("text", { x: x(i), y: H - 8, "text-anchor": "middle" }, p[0])));
  g.append(svg("text", { x: x(yrs.length - 1), y: H - 8, "text-anchor": "middle" }, S.meta.target_year + " (fc)"));
  if (hist.length) {
    g.append(svg("polyline", { points: hist.map((p, i) => `${x(i)},${y(p[1])}`).join(" "), fill: "none", stroke: "var(--accent)", "stroke-width": 2 }));
    hist.forEach((p, i) => g.append(svg("circle", { cx: x(i), cy: y(p[1]), r: 3.5, fill: "var(--accent)" })));
    g.append(svg("line", { x1: x(hist.length - 1), y1: y(hist[hist.length - 1][1]), x2: x(yrs.length - 1), y2: y(f.mean), stroke: "var(--accent)", "stroke-dasharray": "4 3" }));
  }
  const xf = x(yrs.length - 1);
  g.append(svg("line", { x1: xf, x2: xf, y1: y(f.q90 ?? f.mean + f.sd), y2: y(f.q10 ?? f.mean - f.sd), stroke: "var(--accent)", "stroke-width": 8, opacity: 0.2, "stroke-linecap": "round" }));
  g.append(svg("circle", { cx: xf, cy: y(f.mean), r: 5, fill: "var(--panel)", stroke: "var(--accent)", "stroke-width": 2 }));

  // hover: one invisible column per season; shows that season's rating on the model, percentile, matches
  const tip = h("div", { class: "tip", hidden: true });
  const wrap = h("div", { class: "chartwrap" }, g, tip);
  const guide = svg("line", { y1: P / 2, y2: H - P, stroke: "var(--muted)", "stroke-dasharray": "2 3", opacity: 0 });
  g.append(guide);
  const colW = (W - 2 * P) / Math.max(1, yrs.length - 1);
  const lines = (i) => {
    if (i === yrs.length - 1) {
      return [h("b", {}, `${S.meta.target_year} forecast`), h("div", {}, `Rating: ${fx(f.mean)} sd`),
        h("div", { class: "muted" }, f.q10 != null ? `80% range ${fx(f.q10)} to ${fx(f.q90)}` : `± ${f.sd.toFixed(2)}`)];
    }
    const p = hist[i];
    return [h("b", {}, String(p[0])), h("div", {}, `Rating: ${fx(p[1])} sd above the ${p[0]} average`),
      p[2] != null ? h("div", {}, `${ordinal(p[2])} percentile`) : null,
      p[3] != null ? h("div", { class: "muted" }, `${p[3]} matches${p[3] < 8 ? " (few: less reliable)" : ""}`) : null,
      p[4] != null ? h("div", { class: "muted" }, `${S.meta.best_rating_model} raw: ${p[4]}`) : null];
  };
  yrs.forEach((_, i) => {
    const r = svg("rect", { x: x(i) - colW / 2, y: 0, width: colW, height: H, fill: "transparent", style: "cursor:crosshair", tabindex: 0 });
    const show = (ev) => {
      tip.replaceChildren(...lines(i).filter(Boolean)); tip.hidden = false;
      guide.setAttribute("x1", x(i)); guide.setAttribute("x2", x(i)); guide.setAttribute("opacity", 0.6);
      const b = wrap.getBoundingClientRect(), cx = ev.clientX != null && ev.clientX ? ev.clientX - b.left : (x(i) / W) * b.width;
      tip.style.left = Math.min(Math.max(0, cx + 12), b.width - 190) + "px"; tip.style.top = "6px";
    };
    r.addEventListener("pointerenter", show); r.addEventListener("pointermove", show); r.addEventListener("focus", show);
    const hide = () => { tip.hidden = true; guide.setAttribute("opacity", 0); };
    r.addEventListener("pointerleave", hide); r.addEventListener("blur", hide);
    g.append(r);
  });
  return wrap;
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
    h("div", { class: "row" }, flagGroups(f), h("span", { class: "muted small" }, `Rank #${t.rank} of ${S.teams.length} · rookie year ${t.rookie_year ?? "?"}`)),
    h("div", { class: "grid" },
      h("div", { class: "card stat" }, "Forecast strength", h("b", {}, fx(f.mean)), h("span", { class: "muted small" }, f.q10 != null ? `80% range ${fx(f.q10)} to ${fx(f.q90)}` : `± ${f.sd.toFixed(2)}`)),
      a.p_dcmp != null ? h("div", { class: "card stat" }, "District championship", h("b", {}, pct(a.p_dcmp)), h("span", { class: "muted small" }, `Exp. district pts ${a.exp_district_pts}`)) : null,
      a.p_direct != null ? h("div", { class: "card stat" }, "Direct Worlds (event win-type)", h("b", {}, pct(a.p_direct)), h("span", { class: "muted small" }, `Exp. regional pts ${a.exp_regional_pts}`)) : null,
      a.p_cmp != null ? h("div", { class: "card stat" }, "FIRST Championship", h("b", {}, pct(a.p_cmp)), h("span", { class: "muted small" }, a.district ? `${S.meta.districts[a.district]?.name || a.district} model` : "Regional model")) : null),
    h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "Trajectory"), trajectoryChart(t),
      h("p", { class: "muted small" }, "Rating = the model's strength for that season, standardised against every team that year (0 = season average). Hover a season for details. Dashed + band: forecast and 10–90% range.",
        f.analog_edge != null ? ` Analog teams pull the forecast ${fx(f.analog_edge)} vs. a plain trend model.` : "")),
    f.analogs && f.analogs.length ? h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "Teams that looked like this before"),
      h("p", { class: "muted small" }, f.analog_pool ? `Compared only with past team-seasons at the same age (${f.analog_pool.age_band}) and the same amount of history (${f.analog_pool.history} season${f.analog_pool.history === "1" ? "" : "s"}): ${f.analog_pool.n} candidates${f.analog_pool.level > 0 ? ", pool too small so the match was widened" : ""}. ` : "", "Closest trajectories and what they did the next year (green = improved)."),
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
      ` Assumes the standard 8-alliance bracket (${e.year >= (S.meta.double_elim_from || 2023) ? "double elimination" : "best-of-3 single elimination, the format before 2023"}).`),
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
  const trendUsed = m.trend_model_used || "trend", sdUsed = m.sd_model_used || "analog";
  const label = (name, isNew, used) => h("span", {}, isNew ? h("span", { class: "badge" }, "NEW") : null, used ? h("span", { class: "badge used" }, "IN USE") : null, name);
  const T2 = fb && fb.trend2;
  app.replaceChildren(h("h1", {}, "Model & validation"),
    h("p", { class: "muted small" }, h("span", { class: "badge" }, "NEW"), "marks an experimental model added alongside the original, which stays in every table so you can compare them. A NEW model only drives the forecasts when it beats the original in the backtest (", h("span", { class: "badge used" }, "IN USE"), " shows which one does)."),
    h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "1. Which rating model ranks teams best?"),
      h("p", { class: "muted small" }, `Walk-forward backtest (every match predicted before the model sees its result), tuned on seasons before ${m.test_years[0]}, scored on ${m.test_years.join(", ")}. Lower log loss is better. A stacked ensemble of the best few is used only if it beats every single model. In use: `, h("b", {}, m.best_rating_model), "."),
      m.seeded_rating_models && m.seeded_rating_models.test_logloss_seeded != null ? h("p", { class: "muted small" }, h("span", { class: "badge" }, "NEW"), `*_seeded models start every season from a preseason forecast (made only from earlier seasons) instead of just shrinking last season's rating. They are separate models listed next to the originals, and they drive the forecasts only if the best selection with them beats the best without them (${m.seeded_rating_models.test_logloss_unseeded.toFixed(4)} without → ${m.seeded_rating_models.test_logloss_seeded.toFixed(4)} with: `, h("b", {}, m.seeded_rating_models.used ? "in use" : "not in use"), ").") : null,
      table([{ label: "Model", left: true, render: (r) => h("span", {}, r.new ? h("span", { class: "badge" }, "NEW") : null, r.model) }, { label: "Log loss", render: (r) => r.test_logloss.toFixed(4) }, { label: "Brier", render: (r) => r.test_brier.toFixed(4) },
        { label: "Accuracy", render: (r) => pct(r.test_acc) }, { label: "Matches", render: (r) => r.test_matches }, { label: "Used", render: (r) => (r.selected ? "✓" : "") }], m.rating_leaderboard)),
    fb ? h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "2. Trajectory forecaster vs. baselines"),
      h("p", { class: "muted small" }, `Predicting each team's next-season strength, tested on ${fb.years.join(", ")} using only earlier seasons. RMSE in standard deviations; lower is better.`),
      table([{ label: "Method", left: true, render: (r) => label(r[0], r[1], r[2]) }, { label: "RMSE", render: (r) => r[3].toFixed(3) }, { label: "MAE", render: (r) => (r[4] == null ? "–" : r[4].toFixed(3)) }],
        [["Persistence (same as last year)", false, false, fb.persist.rmse, fb.persist.mae],
         ["Trend model (ridge + boosted trees)", false, trendUsed === "trend", fb.ridge.rmse, fb.ridge.mae],
         ...(T2 ? [["Trend v2: also sees how many matches last season's rating rests on, and the team's long-run level", true, trendUsed === "trend2", T2.rmse, T2.mae]] : []),
         ["Analog teams (same age band + history length)", false, false, fb.analog.rmse, fb.analog.mae],
         ...(fb.analog_nomatch ? [["Analog teams, no age matching", false, false, fb.analog_nomatch.rmse, fb.analog_nomatch.mae]] : []),
         [`Blend: Trend + analogs (analog weight ${fb.blend.analog_weight})`, false, trendUsed === "trend", fb.blend.rmse, null],
         ...(T2 && fb.blend2 ? [[`Blend: Trend v2 + analogs (analog weight ${fb.blend2.analog_weight})`, true, trendUsed === "trend2", fb.blend2.rmse, null]] : [])]),
      h("p", { class: "muted small" }, `10–90% analog interval coverage: ${pct(fb.analog_interval_coverage_10_90, 0)} (ideal 80%). The forecaster only leans on analogs as much as the backtest justifies.`),
      fb.by_age ? h("div", {}, h("p", { class: "muted small" }, "Error by team age (RMSE, lower is better):"),
        table([{ label: "Age", left: true, render: (r) => r[0] }, { label: "Team-seasons", render: (r) => r[1].n }, { label: "Persistence", render: (r) => r[1].persist.toFixed(3) },
          { label: "Trend model (ridge + boosted trees)", render: (r) => r[1].ridge.toFixed(3) },
          ...(T2 ? [{ label: "Trend v2 (NEW)", render: (r) => (r[1].trend2 == null ? "–" : r[1].trend2.toFixed(3)) }] : []),
          { label: "Analog", render: (r) => r[1].analog.toFixed(3) }, { label: "Blend", render: (r) => r[1].blend.toFixed(3) }], byAge(fb.by_age))) : null) : null,
    fb && fb.sd ? h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "2b. How wide is each team's forecast range?"),
      h("p", { class: "muted small" }, "The simulator draws every team's strength from its forecast and this spread, so a spread that is too wide flattens the odds (favourites look less likely to win than they are). A well-calibrated 80% range contains the real outcome 80% of the time. Same test seasons and same forecast mean for both rows, so only the spread differs."),
      table([{ label: "Method", left: true, render: (r) => label(r[0], r[1], r[2]) }, { label: "Log score", tip: "negative log likelihood per team-season; lower is better", render: (r) => r[3].nll.toFixed(3) },
        { label: "80% range hit rate", tip: "ideal 80%", render: (r) => pct(r[3].cov80, 0) }, { label: "50% range hit rate", tip: "ideal 50%", render: (r) => pct(r[3].cov50, 0) }, { label: "Average spread", render: (r) => r[3].mean_sd.toFixed(2) }],
        [["Spread of similar teams' outcomes (original)", false, sdUsed === "analog", fb.sd.analog], ["Variance model v2: predicts the error size from the team's features and the season", true, sdUsed === "sd2", fb.sd.sd2]])) : null,
    h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "3. Candidate models not included"),
      h("ul", {}, (m.not_included || []).map((x) => h("li", {}, h("b", {}, x.model + ": "), x.why)))),
    h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "4. Assumptions"),
      h("ul", {}, h("li", {}, m.normalization), h("li", {}, m.rules_note), h("li", {}, m.game_note), h("li", {}, `${m.nsims} Monte Carlo simulations per run; each draws one strength per team and reuses it across all that team's events.`),
        h("li", {}, "Playoffs assume 8 alliances: best-of-3 single elimination through 2022, double elimination (matches 1–13) then best-of-3 finals from 2023. Offseason events vary and are shown for strength only."))));
}


const stat = (label, val, sub) => h("div", { class: "card stat" }, label, h("b", {}, val), sub ? h("span", { class: "muted small" }, sub) : null);
const h2 = (t) => h("h2", { style: "margin-top:0" }, t);

function probCard(title, pm, blurb, why) {
  const head = h2(title);
  if (!pm) return h("div", { class: "card" }, head, blurb ? h("p", { class: "muted small" }, blurb) : null, h("p", { class: "muted" }, why || "Not enough data to score this."));
  return h("div", { class: "card" }, head, blurb ? h("p", { class: "muted small" }, blurb) : null,
    h("div", { class: "grid" }, stat("Cases scored", pm.n.toLocaleString()), stat("Brier score", pm.brier.toFixed(4), `guess-the-base-rate: ${pm.brier_base.toFixed(4)}`),
      stat("Skill vs. base rate", pm.skill == null ? "–" : (pm.skill * 100).toFixed(1) + "%", "above 0% = better than the baseline")),
    h("p", { class: "muted small" }, "Calibration: of the teams given a probability in each range, how many actually did it. Bars that match the average prediction mean the odds are honest."),
    table([{ label: "Predicted range", left: true, render: (r) => r.range }, { label: "Cases", render: (r) => r.n }, { label: "Avg predicted", render: (r) => pct(r.pred) },
      { label: "Actually happened", render: (r) => bar(r.actual) }], pm.bins));
}

function strengthCard(s, blurb) {
  return h("div", { class: "card" }, h2("Team strength forecasts"), h("p", { class: "muted small" }, blurb),
    h("div", { class: "grid" }, stat("Our forecast", s.rmse_forecast.toFixed(3), "RMSE"), stat("Same as last year", s.rmse_persistence.toFixed(3), "RMSE"), stat("Always average", s.rmse_always_average.toFixed(3), "RMSE"),
      stat("80% range hit rate", pct(s.coverage_10_90, 0), "similar-teams range; ideal: 80%"),
      s.coverage_sd_80 != null ? stat("80% range hit rate (simulator's spread)", pct(s.coverage_sd_80, 0), "forecast ± 1.28 × spread; ideal: 80%") : null,
      stat("Rank correlation", s.spearman.toFixed(2), `${s.n.toLocaleString()} team-seasons`)),
    Object.keys(s.by_age).length ? table([{ label: "Team age", left: true, render: (x) => x[0] }, { label: "Teams", render: (x) => x[1].n }, { label: "Our forecast", render: (x) => x[1].forecast.toFixed(3) },
      { label: "Same as last year", render: (x) => x[1].persistence.toFixed(3) }], byAge(s.by_age)) : null);
}

function matchCard(m, blurb) {
  return h("div", { class: "card" }, h2("Match predictions from preseason strengths only"), blurb ? h("p", { class: "muted small" }, blurb) : null,
    h("div", { class: "grid" }, stat("Accuracy", pct(m.accuracy), `${m.n.toLocaleString()} qualification matches`), stat("Log loss", m.logloss.toFixed(4), `coin flip: ${m.logloss_coinflip.toFixed(4)}`),
      m.inseason_logloss != null ? stat("With in-season ratings", m.inseason_logloss.toFixed(4), `accuracy ${pct(m.inseason_accuracy)}`) : null));
}

const yrs = (a) => (!a || !a.length ? "none" : a.length === 1 ? String(a[0]) : a.slice(0, -1).join(", ") + " and " + a[a.length - 1]);
const whyGroups = (sk) => { const g = {}; sk.forEach((s) => (g[/^only/.test(s.reason) ? "too little earlier history" : s.reason] ||= []).push(s.year)); return Object.entries(g).map(([r, y]) => `${yrs(y)} (${r})`).join("; "); };
const TUNING = { in_sample: ["in-sample", "Settings were tuned using this season, so its score is flattering."], partly_held_out: ["partial", "Rating settings never saw this season, but the match-sharpness and playoff factors were chosen by looking at it."], held_out: ["held out", "No settings were tuned using this season."] };

function replayView(r) {
  const ns = r.not_scored || {};
  const parts = [h("p", { class: "muted" }, `Run as if it were the ${r.year} preseason: trained only on seasons through ${r.trained_through} (${r.history_seasons ?? "?"} seasons; model: ${r.rating_model}; analog weight ${r.analog_weight}), no ${r.year} match results, random qualification schedules.`)];
  if (r.strength) parts.push(strengthCard(r.strength, `Forecast ${r.year} rating vs. the rating each team actually earned (teams with 8+ matches; rating = sd above the season average). Lower error is better.`));
  if (r.matches) parts.push(matchCard(r.matches));
  const adv = r.advancement || {};
  if (r.events) parts.push(probCard("Event winners", r.events.win, `${r.events.events} events with a decided final${r.events.skipped_format ? ` (${r.events.skipped_format} more left out: a different bracket shape)` : ""}. Each team's chance of being on the winning alliance.`), probCard("Event finalists", r.events.finalist, "Each team's chance of reaching the finals."));
  else parts.push(h("div", { class: "card" }, h2("Event winners and finalists"), h("p", { class: "muted" }, `Not scored: ${ns.events || "no data"}.`)));
  if (r.advancement) {
    const note = "No Championship / district championship rosters were found for this season, so this can't be scored.";
    parts.push(probCard("Making the District Championship", adv.dcmp, null, note), probCard("Making Worlds (district teams)", adv.cmp_district, null, note), probCard("Making Worlds (regional teams)", adv.cmp_regional, null, note));
  } else parts.push(h("div", { class: "card" }, h2("District Championship and Worlds odds"), h("p", { class: "muted" }, `Not scored: ${ns.advancement || "no data"}.`)));
  return parts;
}

const skill = (pm) => (pm && pm.skill != null ? (pm.skill * 100).toFixed(1) + "%" : "–");

function yearTable(replays) {
  const rows = [...replays].sort((a, b) => b.year - a.year);
  return table([
    { label: "Season", left: true, render: (r) => r.year },
    { label: "History", tip: "Earlier played seasons the model learned from", render: (r) => r.history_seasons ?? "–" },
    { label: "Strength error", tip: "Forecast RMSE vs. 'same as last year' RMSE (lower is better)", render: (r) => (r.strength ? `${r.strength.rmse_forecast.toFixed(3)} vs ${r.strength.rmse_persistence.toFixed(3)}` : "–") },
    { label: "Match acc.", tip: "Qualification matches called correctly from preseason strengths", render: (r) => (r.matches ? pct(r.matches.accuracy) : "–") },
    { label: "Log loss", tip: "Match log loss: lower is better; a coin flip scores 0.693", render: (r) => (r.matches ? r.matches.logloss.toFixed(4) : "–") },
    { label: "Event win", tip: "Brier skill vs. guessing the base rate", render: (r) => skill(r.events && r.events.win) },
    { label: "Finalist", tip: "Brier skill vs. guessing the base rate", render: (r) => skill(r.events && r.events.finalist) },
    { label: "DCMP", tip: "Brier skill: making the District Championship", render: (r) => skill(r.advancement && r.advancement.dcmp) },
    { label: "Worlds", tip: "Brier skill: district teams making Worlds", render: (r) => skill(r.advancement && r.advancement.cmp_district) },
    { label: "Tuning", tip: "Whether this season influenced any tuned setting", render: (r) => h("span", { title: (TUNING[r.tuning] || [])[1] || "" }, (TUNING[r.tuning] || ["–"])[0]) },
  ], rows);
}

function pooledView(p) {
  const out = [h("h2", {}, `All replayed seasons combined (${yrs(p.seasons)})`),
    h("p", { class: "muted" }, "Every season below, pooled into one score. Seasons are weighted by how many matches / teams / events they contain.")];
  if (p.strength) out.push(strengthCard(p.strength, "Each season's forecast rating vs. the rating each team actually earned that season (teams with 8+ matches). Lower error is better."));
  if (p.matches) out.push(matchCard(p.matches, "Preseason strengths only; no results from the season being predicted."));
  if (p.events) out.push(probCard("Event winners", p.events.win, `${p.events.events} events across ${yrs(p.events.seasons)}. Each team's chance of being on the winning alliance.`),
    probCard("Event finalists", p.events.finalist, "Each team's chance of reaching the finals."));
  if (p.advancement) {
    const b = `Seasons scored: ${yrs(p.advancement.seasons)}.`;
    out.push(probCard("Making the District Championship", p.advancement.dcmp, b), probCard("Making Worlds (district teams)", p.advancement.cmp_district, b), probCard("Making Worlds (regional teams)", p.advancement.cmp_regional, b));
  }
  return out;
}

async function viewScorecard() {
  app.replaceChildren(h("h1", {}, "Scorecard"), h("p", { class: "muted" }, "Loading…"));
  let sc;
  try { sc = await fetch("data/scorecard.json").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }); } catch (e) { return app.replaceChildren(h("h1", {}, "Scorecard"), h("p", { class: "muted" }, "No scorecard data yet.")); }
  const n = sc.notes || {}, replays = sc.replays || [];
  const parts = [h("h1", {}, "Scorecard: how accurate are the predictions?"),
    h("p", { class: "muted" }, "Predictions are only useful if they hold up. This page replays every past season that has enough history as if it were still to come, trains only on the seasons before it, and scores the forecasts against what really happened. Once the new season starts, it also scores the live forecast as events finish.")];
  if (sc.live) {
    const l = sc.live;
    parts.push(h("h2", {}, `Live: ${l.year} season so far`), l.matches ? h("div", { class: "card" }, h("div", { class: "grid" }, stat("Match accuracy", pct(l.matches.accuracy), `${l.matches.n.toLocaleString()} matches`), stat("Log loss", l.matches.logloss.toFixed(4), `coin flip: ${l.matches.logloss_coinflip.toFixed(4)}`))) : null,
      l.events && l.events.events ? probCard("Event winners so far", l.events.win, `${l.events.events} completed events.`) : null);
  }
  if (!replays.length) parts.push(h("p", { class: "muted" }, `No replay available (a season needs at least ${n.min_seasons ?? 4} earlier seasons of data to be replayed).`));
  else {
    if (sc.pooled) parts.push(...pooledView(sc.pooled));
    parts.push(h("h2", {}, "Year by year"), h("p", { class: "muted small" }, "Strength error is forecast vs. 'same as last year' (lower is better). Event win, Finalist, DCMP and Worlds are skill vs. guessing the base rate (higher is better; '–' means not scored, see the notes below). Hover a heading for details."), h("div", { class: "card" }, yearTable(replays)));
    [...replays].sort((a, b) => b.year - a.year).forEach((r) => parts.push(h("details", { class: "card" }, h("summary", {}, `${r.year}: full detail`), ...replayView(r))));
    const notes = [
      h("li", {}, `A season is replayed only if at least ${n.min_seasons ?? 4} earlier seasons exist to learn from.` + ((sc.skipped || []).length ? " Not replayed: " + whyGroups(sc.skipped) + "." : "")),
      n.tuned_through != null ? h("li", {}, `Rating settings were tuned using seasons through ${n.tuned_through}, so earlier rows are partly in-sample and flatter the model. The match-sharpness and playoff factors were chosen by looking at ${yrs(n.scale_factors_chosen_with)}. The "Tuning" column marks which rows are clean.`) : null,
      n.double_elim_from != null ? h("li", {}, `Event winners / finalists are scored for every season, using the playoff format that season really used (best-of-3 single elimination before ${n.double_elim_from}, double elimination from then on). An event is scored only if its recorded bracket has that shape; events with a different bracket are left out, and each season's detail says how many.`) : null,
      n.advancement_years ? h("li", {}, `District Championship / Worlds odds are scored only for seasons whose advancement rules are in config/rules.json (${yrs(n.advancement_years)}); replaying a season under another year's rules would measure the rules mismatch, not the model. Attendance stands in for qualification, so declines and waitlist moves are ignored.`) : null,
      h("li", {}, "Seasons with six or fewer earlier seasons can't backtest how much weight the analog-team model deserves, so they use a default blend weight of 0.4 (shown in each season's detail)."),
      h("li", {}, "Replays use each season's real event rosters, which are only approximately known in a real preseason.")];
    parts.push(h("div", { class: "card" }, h2("Read this before quoting the numbers"), h("ul", {}, notes)));
  }
  app.replaceChildren(...parts.filter(Boolean));
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
    else if (seg[0] === "scorecard") viewScorecard();
    else viewRankings();
  } catch (err) { app.replaceChildren(h("p", { class: "neg" }, "Something went wrong: " + err.message)); console.error(err); }
}
document.getElementById("search").addEventListener("input", (e) => {
  const v = e.target.value.trim();
  if (/^\d+$/.test(v) && S.byNum[+v]) location.hash = `#/team/${v}`; else location.hash = `#/?q=${encodeURIComponent(v)}`;
});
window.addEventListener("hashchange", route);
load().then(route).catch((e) => app.replaceChildren(h("p", { class: "neg" }, "Could not load data. Run backend/run.py first. " + e.message)));
