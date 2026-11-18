/* DeepTrace console — vanilla JS, no build step, no external deps. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const MAX_BYTES = 200 * 1024 * 1024;

  const SAMPLES = [
    { file: "authentic_call.mp4", label: "authentic_call.mp4", dot: "real", note: "no swap" },
    { file: "oneshot_swap.mp4", label: "oneshot_swap.mp4", dot: "fake", note: "medium swap" },
    { file: "swap_compressed.mp4", label: "swap_compressed.mp4", dot: "fake", note: "swap + heavy re-encode" },
    { file: "subtle_swap.mp4", label: "subtle_swap.mp4", dot: "unknown", note: "well-blended swap" },
    { file: "two_faces_swap.mp4", label: "two_faces_swap.mp4", dot: "fake", note: "two people, one swapped" },
    { file: "authentic_still.jpg", label: "authentic_still.jpg", dot: "real", note: "image" },

  ];

  const el = {
    dropzone: $("#dropzone"),
    fileInput: $("#fileInput"),
    sampleRow: $("#sampleRow"),
    frames: $("#framesRange"),
    framesOut: $("#framesOut"),
    pdf: $("#pdfCheck"),
    progress: $("#progress"),
    progressFill: $("#progressFill"),
    progressLabel: $("#progressLabel"),
    results: $("#results"),
    calibPill: $("#calibPill"),
  };

  let busy = false;

  /* ------------------------------ helpers ------------------------------ */
  function toast(msg, isErr = false) {
    const t = document.createElement("div");
    t.className = "toast" + (isErr ? " err" : "");
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), isErr ? 6500 : 3200);
  }

  function fmtBytes(b) {
    if (!b) return "—";
    const u = ["B", "KB", "MB", "GB"];
    let i = 0, v = b;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
  }

  function setProgress(pct, label) {
    el.progress.hidden = false;
    el.progressFill.style.width = `${Math.max(4, Math.min(100, pct))}%`;
    if (label) el.progressLabel.textContent = label;
  }

  function hideProgress() {
    setTimeout(() => { el.progress.hidden = true; el.progressFill.style.width = "4%"; }, 700);
  }

  function verdictClass(v) {
    return v === "FAKE" ? "is-fake" : v === "REAL" ? "is-real" : "is-inconclusive";
  }

  /* ------------------------------ upload ------------------------------ */
  el.fileInput.addEventListener("change", () => {
    if (el.fileInput.files && el.fileInput.files[0]) analyzeFile(el.fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    el.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove("drag");
    })
  );
  el.dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) analyzeFile(f);
  });
  el.dropzone.addEventListener("click", () => el.fileInput.click());
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.fileInput.click(); }
  });

  el.frames.addEventListener("input", () => { el.framesOut.textContent = el.frames.value; });

  function buildSamples() {
    el.sampleRow.innerHTML = "";
    SAMPLES.forEach((s) => {
      const b = document.createElement("button");
      b.className = "sample";
      b.type = "button";
      b.innerHTML = `<i class="dot dot-${s.dot}"></i>${s.label}<span class="muted">&nbsp;· ${s.note}</span>`;
      b.addEventListener("click", () => analyzeSample(s));
      el.sampleRow.appendChild(b);
    });
  }

  async function analyzeSample(sample) {
    if (busy) return;
    try {
      setProgress(6, `fetching ${sample.file}…`);
      const res = await fetch(`/samples/${sample.file}`);
      if (!res.ok) throw new Error(`${sample.file} not found on the server`);
      const blob = await res.blob();
      const file = new File([blob], sample.file, { type: blob.type || "video/mp4" });
      analyzeFile(file);
    } catch (err) {
      hideProgress();
      toast(err.message, true);
    }
  }

  async function analyzeFile(file) {
    if (busy) return;
    if (file.size > MAX_BYTES) { toast("File is larger than the 200 MB limit.", true); return; }
    busy = true;
    setProgress(12, `uploading ${file.name} (${fmtBytes(file.size)})…`);

    const fd = new FormData();
    fd.append("file", file, file.name);
    fd.append("max_frames", el.frames.value);
    fd.append("make_pdf", el.pdf.checked ? "true" : "false");

    let tick = 14;
    const timer = setInterval(() => {
      tick = Math.min(tick + 3, 88);
      setProgress(tick, "sampling frames · aligning faces · scoring 3 evidence streams…");
    }, 900);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: fd });
      clearInterval(timer);
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail || `analysis failed (HTTP ${res.status})`);
      }
      const result = await res.json();
      setProgress(100, "done");
      render(result);
      hideProgress();
      $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
      loadHistory();
    } catch (err) {
      clearInterval(timer);
      hideProgress();
      toast(err.message || "analysis failed", true);
    } finally {
      busy = false;
    }
  }

  /* ------------------------------ rendering ------------------------------ */
  let activeTrack = null;
  let lastResult = null;

  function render(r) {
    lastResult = r;
    el.results.hidden = false;
    const v = r.verdict;
    const faceCount = (r.faces && r.faces.count) || 1;
    const analysed = (r.faces && r.faces.analysed) || faceCount;
    // default selection = the face the clip verdict came from
    const tracks = (r.faces && r.faces.tracks) || [];
    activeTrack = v.track_id != null
      ? v.track_id
      : (tracks[0] ? tracks[0].track_id : null);

    // header meta
    $("#resultsMeta").innerHTML = [
      `<span class="tag">${r.input.filename}</span>`,
      `<span class="tag">${r.input.kind} · ${r.input.resolution} · ${r.input.duration_s}s</span>`,
      `<span class="tag">${r.input.frames_sampled} frames sampled</span>`,
      `<span class="tag">${faceCount} face${faceCount === 1 ? "" : "s"} · ${analysed} scored</span>`,
      `<span class="tag">face ${Math.round(r.faces.detection_rate * 100)}% · ${r.faces.method}</span>`,
      `<span class="tag">run ${r.run_id}</span>`,
    ].join("");

    $("#pdfLink").href = r.artifacts.pdf || "#";
    $("#pdfLink").style.display = r.artifacts.pdf ? "" : "none";
    $("#jsonLink").href = r.artifacts.json || "#";
    $("#jsonLink").style.display = r.artifacts.json ? "" : "none";

    // verdict card
    const card = $("#verdictCard");
    card.className = "verdict-card " + verdictClass(v.verdict);
    $("#verdictLabel").textContent = v.verdict;
    $("#verdictLabel").style.color =
      v.verdict === "FAKE" ? "var(--red)" : v.verdict === "REAL" ? "var(--green)" : "var(--amber)";
    $("#verdictP").textContent = `p(fake) ${v.p_fake.toFixed(3)}`;
    $("#verdictConf").textContent = `confidence ${v.confidence.toFixed(2)}`;

    const arc = $("#gaugeArc");
    const circumference = 2 * Math.PI * 82;
    arc.setAttribute("stroke-dasharray", String(circumference));
    arc.style.strokeDashoffset = String(circumference * (1 - Math.max(0, Math.min(1, v.p_fake))));
    arc.style.stroke = v.verdict === "FAKE" ? "var(--red)" : v.verdict === "REAL" ? "var(--green)" : "var(--amber)";

    $("#verdictSummary").textContent = v.summary || "";

    const chips = [];
    chips.push(`<span class="chip">calibration ${r.verdict.calibration_version}</span>`);
    chips.push(`<span class="chip">leading: ${v.leading_stream || "—"}</span>`);
    chips.push(`<span class="chip">spread ${v.stream_spread.toFixed(2)}</span>`);
    if (v.mixed_evidence) chips.push(`<span class="chip warn">streams disagree</span>`);
    if (r.warnings) r.warnings.forEach((w) => chips.push(`<span class="chip warn">${w}</span>`));
    $("#verdictChips").innerHTML = chips.join("");

    // band marker
    const marker = $("#bandMarker");
    const t = v.thresholds;
    marker.style.left = `${Math.min(96, Math.max(4, v.p_fake * 100))}%`;
    const note = v.abstention_note ? `  ${v.abstention_note}.` : "";
    $("#verdictBand").title =
      `thresholds: fake ≥ ${t.fake}, real ≤ ${t.real}. Marker = p(fake).${note}`;

    renderFaces(r);
    paint(r, activeTrack);

    // artifacts
    const labels = {
      faces: "Per-face verdicts (annotated on the frame)",
      tracks: "Face identity across the sampled timeline",
      overlay: "Fused evidence heat-map (left: analysed crop, right: evidence)",
      timeline: "Per-frame blending-seam energy",
      spectrum: "Centred log-power spectrum",
      streams: "Stream scores",
      faces: "Per-face verdicts (annotated on the frame)",
      tracks: "Face identity across the sampled timeline",
    };
    $("#artifactGrid").innerHTML = Object.entries(r.artifacts)
      .filter(([k]) => labels[k])
      .map(([k, url]) => `
        <figure class="artifact" data-full="${url}">
          <img src="${url}" alt="${labels[k]}" loading="lazy" />
          <figcaption>${labels[k]}</figcaption>
        </figure>`).join("");

    $$(".artifact").forEach((fig) =>
      fig.addEventListener("click", () => {
        const box = document.createElement("div");
        box.className = "lightbox";
        box.innerHTML = `<img src="${fig.dataset.full}" alt="" />`;
        box.addEventListener("click", () => box.remove());
        document.body.appendChild(box);
      })
    );

    $("#disclaimerPanel").innerHTML = `<b>Reading this result.</b> ${r.disclaimer}
      Thresholds in force: fake ≥ ${t.fake}, real ≤ ${t.real}; anything between is the
      review band and is reported as INCONCLUSIVE on purpose.`;
  }

  /* ---------------------------- faces + evidence ---------------------------- */
  function facePayload(r, trackId) {
    const tracks = (r.faces && r.faces.tracks) || [];
    return tracks.find((t) => t.track_id === trackId) || null;
  }

  function renderFaces(r) {
    const wrap = $("#faceCards");
    const note = $("#faceNote");
    const panel = $("#facesPanel");
    const tracks = (r.faces && r.faces.tracks) || [];

    if (!tracks.length) {
      panel.hidden = true;
      return;
    }
    panel.hidden = false;

    wrap.innerHTML = tracks.map((t) => {
      const scored = t.analysed !== false && t.verdict;
      const v = scored ? t.verdict : null;
      const cls = scored ? verdictClass(v.verdict) : "is-unscored";
      const thumb = r.artifacts[`face_${t.track_id}`];
      const tags = [];
      if (t.role === "primary") tags.push(`<span class="face-tag primary">primary face</span>`);
      if (t.verdict_source && tracks.length > 1) tags.push(`<span class="face-tag source">verdict source</span>`);
      if (!scored) tags.push(`<span class="face-tag">${t.reason || "not scored"}</span>`);
      return `
        <button class="face-card ${cls}${t.track_id === activeTrack ? " is-active" : ""}"
                type="button" data-track="${t.track_id}" ${scored ? "" : "disabled"}>
          ${thumb
            ? `<img class="face-thumb" src="${thumb}" alt="face #${t.track_id}" loading="lazy" />`
            : `<div class="face-thumb empty">no crop</div>`}
          <div class="face-body">
            <div class="face-head"><span class="face-id">face #${t.track_id}</span>
              <span>${t.frames}/${t.frames_total} frames</span></div>
            <div class="face-verdict ${cls}">${scored ? v.verdict : "not scored"}</div>
            <div class="face-meta">${scored ? `p ${v.p_fake.toFixed(3)} · conf ${v.confidence.toFixed(2)}` : ""}</div>
            <div class="face-meta">continuity ${(t.continuity ?? 0).toFixed(2)}${t.gaps ? ` · ${t.gaps} gap${t.gaps === 1 ? "" : "s"}` : ""}</div>
            ${scored ? `<div class="face-bar"><i style="width:${Math.round(v.p_fake * 100)}%"></i></div>` : ""}
            ${tags.join(" ")}
          </div>
        </button>`;
    }).join("");

    $$(".face-card", wrap).forEach((card) =>
      card.addEventListener("click", () => {
        if (card.disabled) return;
        activeTrack = Number(card.dataset.track);
        $$(".face-card", wrap).forEach((c) => c.classList.toggle("is-active", c === card));
        paint(lastResult, activeTrack);
      })
    );

    const agg = r.faces.aggregate || {};
    if (tracks.length > 1) {
      note.innerHTML = `Multi-face rule: <b>worst-case face</b> wins — the clip is
        ${r.verdict.verdict} because face #${agg.track_id} scored p=${(agg.p_fake ?? 0).toFixed(3)}
        while the others scored ${agg.p_fake_min != null ? agg.p_fake_min.toFixed(3) : "—"}.
        ${agg.agreement ? `Agreement: ${agg.agreement}.` : ""}`;
    } else {
      note.textContent = r.faces.max_simultaneous > 1
        ? "Only one face carried enough frames to be scored."
        : "One face in this clip — per-face and clip verdicts are the same.";
    }
  }

  function paint(r, trackId) {
    const t = facePayload(r, trackId);
    const scored = t && t.analysed !== false && t.verdict;
    const view = scored
      ? { verdict: t.verdict, streams: t.streams, temporal: t.temporal, label: `face #${t.track_id}` }
      : { verdict: r.verdict, streams: r.streams, temporal: r.temporal, label: "clip" };
    const v = view.verdict;
    const t_th = v.thresholds || r.verdict.thresholds;

    const names = { spatial: "Spatial domain", frequency: "Frequency domain", temporal: "Temporal domain" };
    const cards = Object.entries(view.streams).map(([key, s]) => {
      const pct = Math.round(s.score * 100);
      const cls = s.score >= t_th.fake ? "hi" : s.score <= t_th.real ? "" : "mid";
      const top = (s.evidence || [])[0] || {};
      return `
        <div class="stream-card">
          <div class="stream-top">
            <span class="stream-name">${names[key] || key}</span>
            <span class="stream-score">${s.score.toFixed(2)}</span>
            <span class="stream-rel">w ${(s.reliability ?? 1).toFixed(2)}</span>
          </div>
          <div class="stream-bar"><i class="${cls}" style="width:${pct}%"></i></div>
          <p class="stream-ev">${top.label || "—"}
            <b class="${(top.logit_contribution || 0) > 0 ? "pos" : "neg"}">
              ${(top.logit_contribution || 0) > 0 ? "+" : ""}${(top.logit_contribution || 0).toFixed(2)}
            </b></p>
          ${s.reliability_note ? `<p class="stream-note">${s.reliability_note}</p>` : ""}
        </div>`;
    });
    $("#streamCards").innerHTML = cards.join("");

    // ledger
    const rows = [];
    Object.entries(view.streams).forEach(([key, s]) => {
      (s.evidence || []).slice(0, 5).forEach((e) => {
        const contrib = e.logit_contribution;
        rows.push(`<tr>
          <td>${key}</td>
          <td class="sig">${e.label}</td>
          <td>${(s.features[e.feature] ?? 0).toFixed(4)}</td>
          <td>${(s.z_scores[e.feature] ?? 0).toFixed(2)}</td>
          <td class="${contrib > 0 ? "pos" : "neg"}">${contrib > 0 ? "+" : ""}${contrib.toFixed(3)}</td>
          <td>${e.direction}</td>
        </tr>`);
      });
    });
    $("#ledger tbody").innerHTML = rows.join("");

    // fusion
    const frows = Object.entries(v.stream_contributions).map(([key, c]) => {
      const w = v.effective_weights[key];
      return `<div class="fusion-row">
        <span>${key}</span>
        <span class="w">w=${w.toFixed(2)}</span>
        <span class="muted">·(2s−1)</span>
        <span class="c ${c > 0 ? "pos" : "neg"}">${c > 0 ? "+" : ""}${c.toFixed(3)}</span>
      </div>`;
    });
    frows.push(`<div class="fusion-row" style="border-top:1px solid var(--line-soft);padding-top:8px;margin-top:4px">
        <span>bias ${v.logit >= 0 ? "+" : ""}${(v.logit - Object.values(v.stream_contributions).reduce((a, b) => a + b, 0)).toFixed(3)}</span>
        <span class="c">logit ${v.logit >= 0 ? "+" : ""}${v.logit.toFixed(3)} → p ${v.p_fake.toFixed(3)}</span>
      </div>`);
    $("#fusionBody").innerHTML = frows.join("");

    // temporal diagnostics
    const tf = view.streams.temporal.features;
    const ts = view.temporal || {};
    $("#temporalBody").innerHTML = `
      <span class="k">frames used</span><span class="v">${ts.frames_used ?? "—"}</span>
      <span class="k">temporal support</span><span class="v">${(ts.temporal_support ?? 0).toFixed(2)}</span>
      <span class="k">flicker</span><span class="v">${tf.flicker.toFixed(3)}</span>
      <span class="k">identity jitter</span><span class="v">${tf.identity_jitter.toFixed(3)}</span>
      <span class="k">pulse strength</span><span class="v">${tf.pulse_strength.toFixed(3)}</span>
      <span class="k">pulse estimate</span><span class="v">${tf.pulse_bpm ? tf.pulse_bpm.toFixed(0) + " bpm" : "not recoverable"}</span>`;

    // which face the panels describe
    const scope = $("#evidenceScope");
    if (scope) {
      scope.innerHTML = scored
        ? `Evidence panels below describe <b>face #${t.track_id}</b> (${t.role === "primary" ? "primary face" : "secondary face"}).
           ${v.verdict} at p=${v.p_fake.toFixed(3)}.`
        : `Evidence panels describe the clip verdict.`;
    }
  }

  /* ------------------------------ boot ------------------------------ */
  async function loadHealth() {
    try {
      const res = await fetch("/api/health");
      const h = await res.json();
      el.calibPill.textContent = `${h.calibration.version} · ${h.calibration.kind}`;
      el.calibPill.title =
        `calibration ${h.calibration.version} (${h.calibration.kind}, source: ${h.calibration.source}); ` +
        `thresholds fake≥${h.thresholds.fake}, real≤${h.thresholds.real}`;
    } catch {
      el.calibPill.textContent = "engine offline";
    }
  }

  async function loadEval() {
    const box = $("#evalKv");
    try {
      const res = await fetch("/api/eval");
      if (!res.ok) throw new Error("no eval file");
      const e = await res.json();
      const rows = [];
      const push = (k, v) => rows.push(`<span class="k">${k}</span><span class="v">${v}</span>`);
      if (e.cross_corpus) {
        push("cross-corpus AUC", `${e.cross_corpus.fused.auc.toFixed(3)} ${e.cross_corpus.fused.auc_ci95}`);
        push("cross-corpus EER", e.cross_corpus.fused.eer.toFixed(3));
      }
      if (e.in_corpus) push("held-out in-corpus AUC", Number(e.in_corpus).toFixed(3));
      if (e.trained) push("calibration", e.trained);
      if (e.samples) push("training clips", e.samples);
      box.innerHTML = rows.join("");
    } catch {
      box.innerHTML = `<span class="k">evaluation</span><span class="v">run train/eval_protocol.py</span>`;
    }
  }

  async function loadHistory() {
    const wrap = $("#historyList");
    try {
      const res = await fetch("/api/runs?limit=8");
      const data = await res.json();
      if (!data.runs.length) { wrap.innerHTML = `<span class="muted">No runs yet.</span>`; return; }
      wrap.innerHTML = data.runs.map((r) => `
        <div class="history-row" data-run="${r.run_id}">
          <span class="fname">${r.filename}</span>
          <span class="v v-${r.verdict}">${r.verdict}</span>
          <span class="v muted">p=${r.p_fake.toFixed(3)}</span>
        </div>`).join("");
      $$(".history-row", wrap).forEach((row) =>
        row.addEventListener("click", async () => {
          const res2 = await fetch(`/api/runs/${row.dataset.run}`);
          if (!res2.ok) { toast("run not found", true); return; }
          render(await res2.json());
          $("#results").scrollIntoView({ behavior: "smooth" });
        })
      );
    } catch {
      wrap.innerHTML = `<span class="muted">history unavailable</span>`;
    }
  }

  buildSamples();
  loadHealth();
  loadEval();
  loadHistory();
})();
