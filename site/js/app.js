// ShuttleDestiny website: interactive court, live predictions, and charts.
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var SERVES = ["short service", "long service"];
  // Court in metres. x: + is the hitter's right. y: distance from the net,
  // hitter's half negative. Every match in ShuttleSet is singles, so the
  // court uses singles boundaries: the side alleys are out, and the back
  // boundary line is also the singles long service line.
  var COURT = { halfWidth: 3.05, halfLength: 6.7, singles: 2.59, shortService: 1.98, longServiceDoubles: 5.94 };
  var S = COURT.singles, L = COURT.halfLength, SS = COURT.shortService;
  var BODY = 0.25;        // keep a player's centre this far inside their box
  var BEHIND = 0.5;       // players may stand a little behind the back line

  // Which service court the server uses: right on an even score, left on odd.
  function serveSide() {
    return (Number($("score-own").value) || 0) % 2 === 0 ? 1 : -1;
  }
  function isServe() { return SERVES.indexOf($("shot").value) >= 0; }

  // Where each marker may go under singles rules.
  function limitsFor(key) {
    if (isServe()) {
      var side = serveSide();            // +1 right court, -1 left court
      var serverX = side > 0 ? [BODY, S - BODY] : [-(S - BODY), -BODY];
      var targetX = side > 0 ? [-S, 0] : [0, S]; // diagonally opposite box
      var receiverX = side > 0 ? [-(S - BODY), -BODY] : [BODY, S - BODY];
      return {
        hitter: { x: serverX, y: [-(L - BODY), -(SS + BODY)] },
        shuttle: { x: targetX, y: [SS, L] },
        receiver: { x: receiverX, y: [SS + BODY, L - BODY] }
      }[key];
    }
    return {
      shuttle: { x: [-S, S], y: [0.05, L] },
      receiver: { x: [-S, S], y: [0.2, L + BEHIND] },
      hitter: { x: [-S, S], y: [-(L + BEHIND), -0.2] }
    }[key];
  }
  var state = {
    shuttle: { x: 1.9, y: 5.9 },
    receiver: { x: 0.1, y: 3.6 },
    hitter: { x: -0.4, y: -3.9 }
  };
  var model = null;
  var analytics = null;
  var pending = false;

  var $ = function (id) { return document.getElementById(id); };

  function cap(text) { return text.charAt(0).toUpperCase() + text.slice(1); }
  function playerLabel(name) {
    return name.split(" ").map(function (w) {
      return /^[A-Z][A-Z.-]+$/.test(w) ? w.charAt(0) + w.slice(1).toLowerCase() : w;
    }).join(" ");
  }
  function pct(p, digits) { return (p * 100).toFixed(digits === undefined ? 0 : digits) + "%"; }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function el(tag, attrs, parent) {
    var node = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (parent) parent.appendChild(node);
    return node;
  }
  function option(select, value, label) {
    var o = document.createElement("option");
    o.value = value; o.textContent = label;
    select.appendChild(o);
  }

  /* ---------- Court drawing ---------- */
  // Court metres -> SVG: receiver's half drawn at the top, so screen y = -y.
  function drawCourt() {
    var svg = $("court");
    var g = svg.querySelector(".lines");
    var W = COURT.halfWidth, L = COURT.halfLength;
    el("rect", { class: "surface", x: -3.6, y: -7.25, width: 7.2, height: 14.5, rx: 0.12 }, g);
    el("rect", { class: "half-receiver", x: -W, y: -L, width: 2 * W, height: L }, g);
    el("g", { class: "zones", id: "zones" }, g); // shading sits under the lines
    el("rect", { class: "outline", x: -W, y: -L, width: 2 * W, height: 2 * L }, g);
    [-COURT.singles, COURT.singles].forEach(function (x) { el("line", { x1: x, y1: -L, x2: x, y2: L }, g); });
    [COURT.shortService, -COURT.shortService, COURT.longServiceDoubles, -COURT.longServiceDoubles].forEach(function (y) {
      el("line", { x1: -W, y1: y, x2: W, y2: y }, g);
    });
    el("line", { x1: 0, y1: -L, x2: 0, y2: -COURT.shortService }, g);
    el("line", { x1: 0, y1: COURT.shortService, x2: 0, y2: L }, g);
    el("line", { class: "net", x1: -3.35, y1: 0, x2: 3.35, y2: 0 }, g);
    var t1 = el("text", { class: "half-label", x: 0, y: -6.86, "text-anchor": "middle" }, g);
    t1.textContent = "Receiver's half";
    var t2 = el("text", { class: "half-label", x: 0, y: 7.12, "text-anchor": "middle" }, g);
    t2.textContent = "Hitter's half";

    var m = svg.querySelector(".markers");
    el("line", { class: "path", id: "shot-path" }, m);
    el("line", { class: "reach", id: "reach-line" }, m);

    var hitter = el("g", { class: "marker", id: "m-hitter", tabindex: 0, role: "button",
      "aria-label": "Hitter position. Drag, or use arrow keys to move." }, m);
    el("circle", { class: "ring", r: 0.36 }, hitter);
    el("circle", { r: 0.22, fill: "none", stroke: "#FFFFFF", "stroke-width": 0.08 }, hitter);

    var receiver = el("g", { class: "marker", id: "m-receiver", tabindex: 0, role: "button",
      "aria-label": "Receiver position. Drag, or use arrow keys to move." }, m);
    el("circle", { class: "ring", r: 0.36 }, receiver);
    el("circle", { r: 0.24, fill: "#F3C623" }, receiver);

    var shuttle = el("g", { class: "marker", id: "m-shuttle", tabindex: 0, role: "button",
      "aria-label": "Where the shuttle lands. Drag, or use arrow keys to move." }, m);
    el("circle", { class: "ring", r: 0.33 }, shuttle);
    el("rect", { x: -0.15, y: -0.15, width: 0.3, height: 0.3, rx: 0.03, fill: "#FFFFFF", transform: "rotate(45)" }, shuttle);
  }

  function renderMarkers() {
    ["shuttle", "receiver", "hitter"].forEach(function (key) {
      var p = state[key];
      $("m-" + key).setAttribute("transform", "translate(" + p.x.toFixed(3) + " " + (-p.y).toFixed(3) + ")");
    });
    var path = $("shot-path"), reach = $("reach-line");
    path.setAttribute("x1", state.hitter.x); path.setAttribute("y1", -state.hitter.y);
    path.setAttribute("x2", state.shuttle.x); path.setAttribute("y2", -state.shuttle.y);
    reach.setAttribute("x1", state.receiver.x); reach.setAttribute("y1", -state.receiver.y);
    reach.setAttribute("x2", state.shuttle.x); reach.setAttribute("y2", -state.shuttle.y);
  }

  function setMarker(key, x, y) {
    var lim = limitsFor(key);
    state[key] = { x: clamp(x, lim.x[0], lim.x[1]), y: clamp(y, lim.y[0], lim.y[1]) };
    renderMarkers();
    schedule();
  }

  // After the rules change (serve on/off, server's side), move each marker
  // to the nearest legal spot, mirroring it across the centre line if it is
  // on the wrong side rather than piling it up on the line.
  function applyCourtRules() {
    ["hitter", "shuttle", "receiver"].forEach(function (key) {
      var lim = limitsFor(key), p = state[key], x = p.x;
      if (x < lim.x[0] || x > lim.x[1]) {
        if (-x >= lim.x[0] && -x <= lim.x[1]) x = -x;
      }
      state[key] = { x: clamp(x, lim.x[0], lim.x[1]), y: clamp(p.y, lim.y[0], lim.y[1]) };
    });
    drawZones();
    renderMarkers();
  }

  // Shade what is out under the current rules and mark the service courts.
  function rect(g, cls, x0, x1, y0, y1) {
    // court coordinates -> SVG (screen y = -y)
    return el("rect", { class: cls, x: x0, y: -y1, width: x1 - x0, height: y1 - y0 }, g);
  }
  function label(g, text, x, y, rotate) {
    var t = el("text", { class: "zone-label", x: x, y: -y, "text-anchor": "middle", "dominant-baseline": "middle",
      transform: rotate ? "rotate(-90 " + x + " " + (-y) + ")" : "" }, g);
    t.textContent = text;
  }
  function drawZones() {
    var g = $("zones");
    while (g.firstChild) g.removeChild(g.firstChild);
    var W = COURT.halfWidth;
    rect(g, "out", -W, -S, -L, L);
    rect(g, "out", S, W, -L, L);
    label(g, "Out", -(W + S) / 2, 4.3, true);
    label(g, "Out", (W + S) / 2, 4.3, true);
    var note;
    if (isServe()) {
      var side = serveSide();
      rect(g, "out", -S, S, 0, SS);                      // in front of the short service line
      if (side > 0) rect(g, "out", 0, S, SS, L); else rect(g, "out", -S, 0, SS, L); // the other box
      label(g, "Out", 0, SS / 2, false);
      rect(g, "box-target", side > 0 ? -S : 0, side > 0 ? 0 : S, SS, L);
      rect(g, "box-serve", side > 0 ? 0 : -S, side > 0 ? S : 0, -L, -SS);
      var score = Number($("score-own").value) || 0;
      note = "Serve: the server's score is " + score + " (" + (side > 0 ? "even" : "odd") +
        "), so they serve from the " + (side > 0 ? "right" : "left") +
        " service court into the diagonal box. Change the score under Contact and score.";
    } else {
      note = "Singles court: the side alleys are out, and the back line is the boundary.";
    }
    $("court-note").textContent = note;
  }

  function toCourt(svg, evt) {
    var pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    var local = pt.matrixTransform(svg.getScreenCTM().inverse());
    return { x: local.x, y: -local.y };
  }

  function wireCourt() {
    var svg = $("court");
    var dragging = null;
    svg.addEventListener("pointerdown", function (evt) {
      var target = evt.target.closest(".marker");
      var p = toCourt(svg, evt);
      if (target) dragging = target.id.replace("m-", "");
      else dragging = p.y >= 0 ? "shuttle" : "hitter"; // tap a half to move its marker
      svg.setPointerCapture(evt.pointerId);
      setMarker(dragging, p.x, p.y);
      evt.preventDefault();
    });
    svg.addEventListener("pointermove", function (evt) {
      if (!dragging) return;
      var p = toCourt(svg, evt);
      setMarker(dragging, p.x, p.y);
    });
    var stop = function () { dragging = null; };
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);

    ["shuttle", "receiver", "hitter"].forEach(function (key) {
      $("m-" + key).addEventListener("keydown", function (evt) {
        var step = evt.shiftKey ? 0.5 : 0.1;
        var d = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[evt.key];
        if (!d) return;
        evt.preventDefault();
        setMarker(key, state[key].x + d[0], state[key].y + d[1]);
      });
    });
  }

  /* ---------- Form ---------- */
  function fillForm() {
    var counts = analytics.shot_counts;
    var shots = Object.keys(counts).filter(function (s) { return counts[s] > 0; })
      .sort(function (a, b) { return counts[b] - counts[a]; });
    [$("shot"), $("prev"), $("prev2")].forEach(function (sel, i) {
      if (i > 0) option(sel, "none", "Nothing (start of rally)");
      shots.forEach(function (s) { option(sel, s, cap(s)); });
    });
    $("shot").value = "clear";
    $("prev").value = "lob";
    $("prev2").value = "net shot";

    option($("receiver"), "", "Any player");
    model.spec.players.slice().sort(function (a, b) { return playerLabel(a).localeCompare(playerLabel(b)); })
      .forEach(function (p) { option($("receiver"), p, playerLabel(p)); });

    $("setup").addEventListener("input", onFormChange);
    $("setup").addEventListener("change", onFormChange);
    $("setup").addEventListener("submit", function (e) { e.preventDefault(); });
    syncRallyRules();
  }

  // Keep the rally legal: a rally starts with a serve (shot 1), shot 2 is the
  // reply to that serve, and no later shot can be a serve.
  function allowOptions(select, allowed, fallback) {
    Array.prototype.forEach.call(select.options, function (o) {
      var ok = allowed(o.value);
      o.disabled = !ok;
      o.hidden = !ok;
    });
    if (!allowed(select.value)) select.value = fallback;
  }
  function syncRallyRules(changed) {
    var serve = isServe();
    var round = $("round");
    var n = parseInt(round.value, 10);
    if (serve) n = 1;
    else if (!(n >= 2)) n = 2;
    n = clamp(n, 1, 60);
    round.value = n;
    round.min = serve ? 1 : 2;
    round.disabled = serve;

    var isServeName = function (v) { return SERVES.indexOf(v) >= 0; };
    var notServe = function (v) { return v !== "none" && !isServeName(v); };
    var prev = $("prev"), prev2 = $("prev2");
    if (n === 1) allowOptions(prev, function (v) { return v === "none"; }, "none");
    else if (n === 2) allowOptions(prev, isServeName, "short service");
    else allowOptions(prev, notServe, "lob");
    if (n <= 2) allowOptions(prev2, function (v) { return v === "none"; }, "none");
    else if (n === 3) allowOptions(prev2, isServeName, "short service");
    else allowOptions(prev2, notServe, "net shot");
    prev.disabled = n < 2;
    prev2.disabled = n < 3;

    // Serves must be hit below the net (and below the waist) in badminton.
    var high = $("high");
    if (serve) {
      if (!high.disabled) high.dataset.before = high.checked ? "1" : "0";
      high.checked = false;
      high.disabled = true;
    } else if (high.disabled) {
      high.disabled = false;
      high.checked = high.dataset.before !== "0";
    }
    applyCourtRules();
  }

  function onFormChange(evt) {
    var id = evt && evt.target ? evt.target.id : "";
    if (id === "shot" || id === "round") syncRallyRules(id);
    else if (id === "score-own") applyCourtRules();
    schedule();
  }

  function currentInputs() {
    var own = Number($("score-own").value) || 0;
    var opp = Number($("score-opp").value) || 0;
    var s = state.shuttle, r = state.receiver, h = state.hitter;
    return {
      shot: $("shot").value,
      prev_shot: $("prev").value,
      prev2_shot: $("prev2").value,
      receiver: $("receiver").value || null,
      ball_round: Number($("round").value) || 1,
      hit_high: $("high").checked ? 1 : 0,
      backhand: $("backhand").checked ? 1 : 0,
      around_head: $("aroundhead").checked ? 1 : 0,
      score_diff: own - opp,
      points_total: own + opp,
      land_x: s.x, land_y: s.y, land_depth: s.y,
      hitter_x: h.x, hitter_y: h.y,
      receiver_x: r.x, receiver_y: r.y,
      receiver_to_land: Math.hypot(s.x - r.x, s.y - r.y)
    };
  }

  /* ---------- Prediction ---------- */
  function schedule() {
    if (pending || !model) return;
    pending = true;
    requestAnimationFrame(function () { pending = false; predict(); });
  }

  function predict() {
    var ranked = model.predict(currentInputs());
    var top = ranked[0], second = ranked[1];
    var headline = $("headline");
    if (top.p - second.p < 0.04) {
      headline.innerHTML = "Close call between <strong>" + cap(top.shot) + "</strong> and <strong>" +
        cap(second.shot) + "</strong>.";
    } else {
      headline.innerHTML = "Most likely reply: <strong>" + cap(top.shot) + "</strong>, " + pct(top.p) + ".";
    }
    var inp = currentInputs();
    var side = inp.land_x > 0.6 ? "right" : inp.land_x < -0.6 ? "left" : "middle";
    var depth = inp.land_y > 4.5 ? "deep" : inp.land_y < 2.2 ? "short" : "mid-court";
    $("context").textContent = "Replying to a " + inp.shot + " landing " + depth + " on the " + side +
      " side. The receiver is " + inp.receiver_to_land.toFixed(1) + " m away from it.";
    var list = $("bars");
    var rows = ranked.slice(0, 6);
    while (list.children.length < rows.length) {
      var li = document.createElement("li");
      li.innerHTML = '<span class="name"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>';
      list.appendChild(li);
    }
    rows.forEach(function (row, i) {
      var li = list.children[i];
      li.querySelector(".name").textContent = cap(row.shot);
      li.querySelector(".fill").style.width = (row.p * 100).toFixed(1) + "%";
      li.querySelector(".pct").textContent = pct(row.p, 1);
    });
  }

  /* ---------- Charts ---------- */
  function hbar(list, label, value, max, display, cls) {
    var li = document.createElement("li");
    li.innerHTML = '<span class="name"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>';
    li.querySelector(".name").textContent = label;
    var fill = li.querySelector(".fill");
    fill.style.width = (Math.max(0, value) / max * 100).toFixed(1) + "%";
    if (cls) fill.classList.add(cls);
    li.querySelector(".pct").textContent = display;
    list.appendChild(li);
  }

  function renderAccuracy(metrics) {
    var list = $("accuracy-bars");
    var rows = [
      ["Always guess the most common shot", metrics.baseline_accuracy, "dim"],
      ["Version 1: Random Forest, 3 inputs", metrics.v1_random_forest.accuracy, "dim"],
      ["Version 2, any player", metrics.new_any_player.accuracy, ""],
      ["Version 2, receiving player chosen", metrics.new_known_player.accuracy, "highlight"]
    ];
    rows.forEach(function (r) {
      var li = document.createElement("li");
      li.innerHTML = '<span class="row"><span></span><b></b></span><span class="track"><span class="fill"></span></span>';
      li.querySelector(".row span").textContent = r[0];
      li.querySelector(".row b").textContent = pct(r[1], 1);
      var fill = li.querySelector(".fill");
      fill.style.width = (r[1] * 100).toFixed(1) + "%";
      if (r[2]) fill.classList.add(r[2]);
      list.appendChild(li);
    });

    var k = metrics.new_known_player, v1 = metrics.v1_random_forest;
    document.querySelectorAll("[data-metric=v1]").forEach(function (n) { n.textContent = pct(v1.accuracy, 1); });
    document.querySelectorAll("[data-metric=v2]").forEach(function (n) { n.textContent = pct(k.accuracy, 1); });
    var callouts = [
      [pct(k.top3), "of the time, the real reply is among the model's top three guesses."],
      [(k.accuracy / metrics.baseline_accuracy).toFixed(1) + "×", "as accurate as always guessing the most common shot."],
      ["+" + Math.round((k.macro_f1 / v1.macro_f1 - 1) * 100) + "%", "better at rarer shots than version 1, measured by average F1 across reply types."]
    ];
    var box = $("callouts");
    callouts.forEach(function (c) {
      var d = document.createElement("div");
      d.className = "callout";
      d.innerHTML = '<p class="big"></p><p class="what"></p>';
      d.querySelector(".big").textContent = c[0];
      d.querySelector(".what").textContent = c[1];
      box.appendChild(d);
    });

    var imp = metrics.importance;
    var keys = Object.keys(imp);
    var maxImp = Math.max.apply(null, keys.map(function (x) { return imp[x]; }));
    keys.forEach(function (key) {
      hbar($("importance-bars"), key, imp[key], maxImp, (imp[key] >= 0 ? "−" : "+") + Math.abs(imp[key] * 100).toFixed(1) + " pts");
    });

    var f1 = metrics.per_shot_f1;
    Object.keys(f1).sort(function (a, b) { return f1[b] - f1[a]; }).forEach(function (s) {
      hbar($("f1-bars"), cap(s), f1[s], 1, f1[s].toFixed(2));
    });
  }

  function renderPatterns() {
    var fm = analytics.follow_matrix;
    var labels = fm.labels;
    var colTotals = labels.map(function (_, j) {
      return fm.matrix.reduce(function (s, row) { return s + row[j]; }, 0);
    });
    var grand = colTotals.reduce(function (a, b) { return a + b; }, 0);
    var average = {};
    labels.forEach(function (l, j) { average[l] = colTotals[j] / grand; });

    var shotSel = $("pattern-shot");
    labels.forEach(function (l, i) {
      var total = fm.matrix[i].reduce(function (a, b) { return a + b; }, 0);
      if (total >= 50) option(shotSel, l, l === "none" ? l : cap(l) + " (" + total.toLocaleString() + ")");
    });
    shotSel.value = "smash";
    function drawFollow() {
      var i = labels.indexOf(shotSel.value);
      var row = fm.matrix[i];
      var total = row.reduce(function (a, b) { return a + b; }, 0);
      var list = $("follow-bars");
      list.innerHTML = "";
      labels.map(function (l, j) { return [l, row[j] / total]; })
        .filter(function (x) { return x[1] >= 0.005; })
        .sort(function (a, b) { return b[1] - a[1]; })
        .slice(0, 8)
        .forEach(function (x, n) { hbar(list, cap(x[0]), x[1], 1, pct(x[1]), n === 0 ? "highlight" : ""); });
    }
    shotSel.addEventListener("change", drawFollow);
    drawFollow();

    var players = Object.keys(analytics.by_player)
      .sort(function (a, b) { return playerLabel(a).localeCompare(playerLabel(b)); });
    var playerSel = $("pattern-player");
    players.forEach(function (p) {
      option(playerSel, p, playerLabel(p) + " (" + analytics.by_player[p].replies.toLocaleString() + " replies)");
    });
    playerSel.value = players.indexOf("Viktor AXELSEN") >= 0 ? "Viktor AXELSEN" : players[0];
    function drawPlayer() {
      var mix = analytics.by_player[playerSel.value].mix;
      var list = $("player-bars");
      list.innerHTML = "";
      Object.keys(mix).filter(function (s) { return mix[s] > 0; })
        .sort(function (a, b) { return mix[b] - mix[a]; })
        .slice(0, 8)
        .forEach(function (s) {
          var li = document.createElement("li");
          li.innerHTML = '<span class="name"></span><span class="track"><span class="fill player"></span><span class="fill avg"></span></span><span class="pct"></span>';
          li.querySelector(".name").textContent = cap(s);
          var scale = 0.45; // widest share shown at full width
          li.querySelector(".player").style.width = Math.min(100, mix[s] / scale * 100).toFixed(1) + "%";
          li.querySelector(".avg").style.width = Math.min(100, (average[s] || 0) / scale * 100).toFixed(1) + "%";
          li.querySelector(".pct").innerHTML = pct(mix[s]) + " <span>/ " + pct(average[s] || 0) + "</span>";
          list.appendChild(li);
        });
    }
    playerSel.addEventListener("change", drawPlayer);
    drawPlayer();
  }

  /* ---------- Start ---------- */
  function load(path) {
    return fetch(path).then(function (r) {
      if (!r.ok) throw new Error(path + " returned " + r.status);
      return r.json();
    });
  }

  drawCourt();
  drawZones();
  renderMarkers();
  wireCourt();

  Promise.all([load("data/model.json"), load("data/analytics.json"), load("data/metrics.json")])
    .then(function (res) {
      model = new window.ShuttleModel(res[0]);
      analytics = res[1];
      fillForm();
      predict();
      renderAccuracy(res[2]);
      renderPatterns();
    })
    .catch(function (err) {
      $("headline").textContent = "The model files didn't load. Open this page from a web server, not directly from disk (" + err.message + ").";
    });
})();
