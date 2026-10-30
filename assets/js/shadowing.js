/* Shadowing player.
 * captions -> sentence chunks -> YouTube segment loop.
 *
 * Captions come from, in order: localStorage, assets/captions/ committed to
 * the repo, the local caption server (bin/caption-server.py), or text the user
 * pastes or uploads. Everything fetched is written back to localStorage, so a
 * video is only fetched once per browser.
 *
 * localStorage is per-origin AND per-browser-profile, so it syncs nothing; the
 * committed files are what make a video openable from a phone or a second
 * Chrome profile.
 */
(function () {
  "use strict";

  var CAP_KEY = function (id) { return "shadow:cap:" + id; };
  var USER_KEY = function (id) { return "shadow:user:" + id; };
  var RECENT_KEY = "shadow:recent";
  var HIDDEN_KEY = "shadow:hidden";

  var MAX_DUR = 18;   // a chunk never runs longer than this
  var MIN_DUR = 1.0;  // ...only long enough to not be a sliver. Short sentences
                      // ("That's nothing.") are prime shadowing material, so
                      // this stays low on purpose.
  var GAP = 0.7;      // silence this long is a sentence boundary in auto-captions

  var S = {
    mode: "youtube",
    videoId: null,
    title: "",
    lang: "en",
    kind: "",       // "manual" | "auto" | "" (uploaded by hand)
    cues: [],
    segments: [],
    view: [],        // indices into segments, after the search filter
    cur: -1,
    perChunk: 2,
    offset: 0,
    rate: 1,
    repeatTarget: 3, // 0 = infinite
    repeatDone: 0,
    autoAdvance: true,
    gap: 1,
    filter: "",        // sentence search
    savedFilter: "",   // saved-video search
  };

  var player = null;
  var mountedId = null;
  var playerReady = false;
  var pendingSeek = null;
  var timer = null;
  var seekGuard = 0;
  var audioUrl = null;
  var audioFile = null;
  var decodedAudio = null;
  var bgUrl = null;

  var $ = function (id) { return document.getElementById(id); };

  /* ---------- storage ---------- */

  function load(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) {
      return fallback;
    }
  }

  function save(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (e) {
      // Quota exceeded or storage blocked (private window). Not fatal.
      return false;
    }
  }

  function saveUser() {
    if (!S.videoId) return;
    save(USER_KEY(S.videoId), {
      offset: S.offset,
      perChunk: S.perChunk,
      rate: S.rate,
      repeatTarget: S.repeatTarget,
      autoAdvance: S.autoAdvance,
      gap: S.gap,
    });
  }

  function applyUser(u) {
    if (!u) return;
    if (typeof u.offset === "number") S.offset = u.offset;
    if (typeof u.perChunk === "number") S.perChunk = u.perChunk;
    if (typeof u.rate === "number") S.rate = u.rate;
    if (typeof u.repeatTarget === "number") S.repeatTarget = u.repeatTarget;
    if (typeof u.autoAdvance === "boolean") S.autoAdvance = u.autoAdvance;
    if (typeof u.gap === "number") S.gap = u.gap;
  }

  function touchRecent(id, title, count) {
    // Opening a previously hidden shared video explicitly restores it.
    save(HIDDEN_KEY, load(HIDDEN_KEY, []).filter(function (hiddenId) { return hiddenId !== id; }));
    var list = load(RECENT_KEY, []);
    var prev = null;
    list = list.filter(function (r) {
      if (r.id === id) { prev = r; return false; }
      return true;
    });
    list.unshift({
      id: id,
      // The title arrives from the player a moment after the captions do, so
      // never let a later call overwrite a real title with the bare video id.
      title: title || (prev && prev.title) || id,
      n: count,
      kind: S.kind || (prev && prev.kind) || "",
      lang: S.lang,
      ts: Date.now(),
    });
    save(RECENT_KEY, list.slice(0, 200));
    renderRecent();
  }

  // Drop a video from the library, along with its captions and its per-video
  // settings -- otherwise "삭제" would leave the heavy part behind.
  function forget(id) {
    save(RECENT_KEY, load(RECENT_KEY, []).filter(function (r) { return r.id !== id; }));
    if (repoEntry(id)) {
      var hidden = load(HIDDEN_KEY, []);
      if (hidden.indexOf(id) === -1) hidden.push(id);
      save(HIDDEN_KEY, hidden);
    }
    try {
      localStorage.removeItem(CAP_KEY(id));
      localStorage.removeItem(USER_KEY(id));
    } catch (e) {}
    renderRecent();
  }

  function forgetAll() {
    var list = load(RECENT_KEY, []);
    for (var i = 0; i < list.length; i++) {
      try {
        localStorage.removeItem(CAP_KEY(list[i].id));
        localStorage.removeItem(USER_KEY(list[i].id));
      } catch (e) {}
    }
    save(RECENT_KEY, []);
    save(HIDDEN_KEY, REPO_INDEX.map(function (r) { return r.id; }));
    renderRecent();
  }

  /* ---------- captions committed to the repo ---------- */

  // localStorage is per-origin AND per-browser-profile, so a phone and two
  // Chrome profiles share nothing. Captions committed under assets/captions/
  // are served by GitHub Pages instead, which every device can read with no
  // login, no token and no local server.
  var REPO_INDEX = [];

  function repoDir() {
    var el = $("sh-app");
    return (el && el.getAttribute("data-captions")) || "/assets/captions";
  }

  function repoEntry(id) {
    for (var i = 0; i < REPO_INDEX.length; i++) {
      if (REPO_INDEX[i].id === id) return REPO_INDEX[i];
    }
    return null;
  }

  function loadRepoIndex(done) {
    if (typeof fetch !== "function") { done(); return; }
    fetch(repoDir() + "/index.json")
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (list) { if (Array.isArray(list)) REPO_INDEX = list; done(); })
      .catch(function () { done(); });   // no index published yet -- not an error
  }

  function fetchFromRepo(id, cb) {
    var entry = repoEntry(id);
    if (!entry || typeof fetch !== "function") { cb(new Error("not in the repo")); return; }
    fetch(repoDir() + "/" + id + "." + entry.lang + ".json")
      .then(function (r) { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then(function (d) {
        if (d && d.cues && d.cues.length) cb(null, d);
        else cb(new Error("empty caption file"));
      })
      .catch(function () { cb(new Error("레포의 자막 파일을 읽지 못했습니다.")); });
  }

  /* ---------- caption server ---------- */

  // Set once via the "자막 서버 설정" box, or baked into the page as data-server.
  function serverBase() {
    var stored = load("shadow:server", "");
    if (!stored) {
      var el = $("sh-app");
      stored = (el && el.getAttribute("data-server")) || "";
    }
    return String(stored).trim().replace(/\/+$/, "");
  }

  // Reaching 127.0.0.1 from the deployed https page fails in ways the browser
  // will not explain, so say the likely causes out loud.
  function unreachable() {
    return new Error(
      "자막 서버(" + serverBase() + ")에 연결하지 못했습니다. " +
      "서버가 켜져 있는지, Safari가 아닌지, 그리고 이 사이트에서 주소를 저장했는지 확인하세요."
    );
  }

  function pingServer(cb) {
    var base = serverBase();
    if (!base || typeof fetch !== "function") { cb(new Error("주소가 없습니다.")); return; }
    fetch(base + "/search?q=test")
      .then(function (r) {
        if (r.ok) { cb(null, true); return; }
        // Reached it, but it is an older build without /search.
        cb(null, false);
      })
      .catch(function () { cb(unreachable()); });
  }

  function fetchFromServer(id, opts, cb) {
    var base = serverBase();
    if (!base) { cb(new Error("자막 서버 주소가 설정되지 않았습니다.")); return; }
    if (typeof fetch !== "function") { cb(new Error("이 브라우저는 fetch를 지원하지 않습니다.")); return; }

    var url =
      base + (base.indexOf("?") === -1 ? "?" : "&") +
      "v=" + encodeURIComponent(id) + "&lang=" + encodeURIComponent(S.lang) +
      (opts && opts.refresh ? "&refresh=1" : "");

    fetch(url)
      .then(function (r) {
        return r.text().then(function (body) {
          var parsed = null;
          try { parsed = JSON.parse(body); } catch (e) {}
          return { ok: r.ok, status: r.status, body: parsed };
        });
      })
      .then(function (res) {
        if (!res.body) {
          cb(new Error("자막 서버 응답을 읽지 못했습니다 (HTTP " + res.status + ")."));
          return;
        }
        if (!res.ok) {
          var err = new Error(res.body.error || "자막 서버 오류 (HTTP " + res.status + ")");
          err.detail = res.body.detail;
          cb(err);
          return;
        }
        if (!res.body.cues || !res.body.cues.length) {
          cb(new Error("이 영상에는 " + S.lang + " 자막이 없습니다."));
          return;
        }
        cb(null, res.body);
      })
      .catch(function () {
        // Network-level failure: wrong URL, server not running, or CORS refusal.
        cb(unreachable());
      });
  }

  // yt-dlp can search YouTube without an API key, so the box accepts words as
  // well as links -- no trip to the YouTube app to copy a URL back.
  function searchServer(query, cb) {
    var base = serverBase();
    if (!base) { cb(new Error("검색하려면 자막 서버가 필요합니다.")); return; }
    if (typeof fetch !== "function") { cb(new Error("이 브라우저는 fetch를 지원하지 않습니다.")); return; }
    fetch(base + "/search?q=" + encodeURIComponent(query))
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) {
        if (!res.ok || !res.body.results) { cb(new Error(res.body.error || "검색에 실패했습니다.")); return; }
        cb(null, res.body.results);
      })
      .catch(function () { cb(unreachable()); });
  }

  function hhmm(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    var m = Math.floor(sec / 60), ss = sec % 60;
    return m + ":" + (ss < 10 ? "0" : "") + ss;
  }

  function renderResults(list) {
    var box = $("sh-results");
    if (!list) { box.hidden = true; box.innerHTML = ""; return; }
    if (!list.length) {
      box.hidden = false;
      box.innerHTML = '<p class="sh-empty">검색 결과가 없습니다.</p>';
      return;
    }
    var html = "";
    for (var i = 0; i < list.length; i++) {
      var r = list[i];
      html +=
        '<button type="button" class="sh-result" data-id="' + r.id + '">' +
          '<img class="sh-thumb" src="https://i.ytimg.com/vi/' + r.id + '/mqdefault.jpg" alt="" loading="lazy">' +
          '<span class="sh-result-body">' +
            '<span class="sh-result-title">' + escapeHTML(r.title) + "</span>" +
            '<span class="sh-result-meta">' + escapeHTML(r.channel) + " · " + hhmm(r.duration) + "</span>" +
          "</span>" +
        "</button>";
    }
    box.hidden = false;
    box.innerHTML = html;
  }

  function runSearch(query) {
    setBusy(true);
    status("유튜브에서 검색 중…");
    searchServer(query, function (err, list) {
      setBusy(false);
      if (err) { renderResults(null); status(err.message, "err"); return; }
      renderResults(list);
      status(list.length + "개 찾았습니다. 영상을 고르세요.", "ok");
    });
  }

  /* ---------- parsing ---------- */

  function parseVideoId(input) {
    var s = (input || "").trim();
    if (/^[\w-]{11}$/.test(s)) return s;
    var m = s.match(/(?:youtu\.be\/|[?&]v=|\/embed\/|\/shorts\/|\/live\/)([\w-]{11})/);
    return m ? m[1] : null;
  }

  function parseTime(s) {
    var m = String(s).trim().match(/(?:(\d+):)?(\d{1,2}):(\d{1,2})[.,](\d{1,3})/);
    if (!m) return null;
    var ms = (m[4] + "00").slice(0, 3);
    return (+(m[1] || 0)) * 3600 + (+m[2]) * 60 + (+m[3]) + (+ms) / 1000;
  }

  function stripTags(line) {
    return line
      .replace(/<[^>]*>/g, "")       // <c>, <00:00:13.000>, <i> ...
      .replace(/\{\\[^}]*\}/g, "")   // ASS-style overrides
      .replace(/&nbsp;/g, " ")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&#39;/g, "'")
      .replace(/&quot;/g, '"')
      .trim();
  }

  function parseCues(raw) {
    var text = String(raw).replace(/^﻿/, "").replace(/\r\n?/g, "\n");
    var blocks = text.split(/\n{2,}/);
    var cues = [];
    var prev = [];

    for (var i = 0; i < blocks.length; i++) {
      var lines = blocks[i].split("\n").filter(function (l) { return l.trim() !== ""; });
      if (!lines.length) continue;
      if (/^WEBVTT/i.test(lines[0]) || /^NOTE\b/i.test(lines[0])) continue;

      var ti = -1;
      for (var j = 0; j < lines.length; j++) {
        if (lines[j].indexOf("-->") !== -1) { ti = j; break; }
      }
      if (ti === -1) continue;

      var parts = lines[ti].split("-->");
      var start = parseTime(parts[0]);
      var end = parseTime(parts[1]);
      if (start === null || end === null || end <= start) continue;

      var body = lines.slice(ti + 1).map(stripTags).filter(Boolean);
      if (!body.length) continue;

      // YouTube auto-captions roll: each cue repeats the previous cue's last line.
      var fresh = body.filter(function (l) { return prev.indexOf(l) === -1; });
      prev = body;
      if (!fresh.length) continue;

      cues.push({
        start: start,
        end: end,
        text: fresh.join(" ").replace(/\s+/g, " ").trim(),
      });
    }
    return cues;
  }

  // YouTube's own "Show transcript" panel, copied to the clipboard. No install
  // needed, which makes it the path anyone can use. The shapes seen in the wild:
  //   "0:12\nWell, the thing is..."   (timestamp on its own line)
  //   "0:12\tWell, the thing is..."   (tab separated)
  //   "1:02:05 Well, the thing..."    (hours, space separated)
  // A cue ends where the next one starts -- the panel gives no durations.
  function parseTranscript(raw) {
    var lines = String(raw).replace(/\r\n?/g, "\n").split("\n");
    var stamp = /^(?:(\d{1,2}):)?(\d{1,3}):([0-5]\d)(?:[\s\t]+(.*))?$/;
    var out = [];
    var start = null;
    var buf = [];

    function flush(end) {
      if (start === null) return;
      var text = buf.join(" ").replace(/\s+/g, " ").trim();
      buf = [];
      if (text) out.push({ start: start, end: Math.max(end, start + 0.5), text: text });
    }

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i].trim();
      if (!line) continue;
      var m = line.match(stamp);
      if (m) {
        var t = (+(m[1] || 0)) * 3600 + (+m[2]) * 60 + (+m[3]);
        flush(t);
        start = t;
        if (m[4] && m[4].trim()) buf.push(m[4].trim());
      } else if (start !== null) {
        buf.push(line);
      }
    }
    flush(start === null ? 0 : start + 4);
    return out;
  }

  // SRT/VTT if it looks like one, otherwise a pasted transcript.
  function parseAny(raw) {
    var cues = parseCues(raw);
    return cues.length ? cues : parseTranscript(raw);
  }

  /* ---------- chunking ---------- */

  function hasPunctuation(cues) {
    if (!cues.length) return false;
    var n = 0;
    for (var i = 0; i < cues.length; i++) {
      if (/[.!?]["')\]]?$/.test(cues[i].text.trim())) n++;
    }
    return n / cues.length > 0.12;
  }

  function mkSegment(buf) {
    return {
      start: buf[0].start,
      end: buf[buf.length - 1].end,
      text: buf.map(function (c) { return c.text; }).join(" ").replace(/\s+/g, " ").trim(),
    };
  }

  function byPunctuation(cues, perChunk) {
    var out = [], buf = [];
    for (var i = 0; i < cues.length; i++) {
      // Close the chunk *before* a cue that would overrun the cap, otherwise a
      // single long cue drags the segment well past MAX_DUR.
      if (buf.length) {
        var held = buf[buf.length - 1].end - buf[0].start;
        if (cues[i].end - buf[0].start > MAX_DUR && held >= MIN_DUR) {
          out.push(mkSegment(buf));
          buf = [];
        }
      }
      buf.push(cues[i]);
      var dur = buf[buf.length - 1].end - buf[0].start;
      var joined = buf.map(function (c) { return c.text; }).join(" ");
      var enders = (joined.match(/[.!?]["')\]]?(\s|$)/g) || []).length;
      var clean = /[.!?]["')\]]?$/.test(cues[i].text.trim());
      if ((clean && enders >= perChunk && dur >= MIN_DUR) || dur >= MAX_DUR) {
        out.push(mkSegment(buf));
        buf = [];
      }
    }
    if (buf.length) out.push(mkSegment(buf));
    return out;
  }

  // Auto-captions have no punctuation, so "sentences" are a fiction here and
  // perChunk just scales the target length. Their cues also run back-to-back,
  // with gaps of ~0, so a purely gap-driven rule almost never fires and every
  // chunk ends up pinned at MAX_DUR. Length drives the split; a real pause is
  // only allowed to cut earlier.
  function byGap(cues, perChunk) {
    var target = 4 * perChunk;                       // 4 / 8 / 12s
    var hardMax = Math.min(MAX_DUR, target * 1.6);
    var out = [], buf = [];
    for (var i = 0; i < cues.length; i++) {
      if (buf.length) {
        var dur = buf[buf.length - 1].end - buf[0].start;
        var gap = cues[i].start - buf[buf.length - 1].end;
        var grown = cues[i].end - buf[0].start;
        if (dur >= target ||
            (dur >= MIN_DUR && gap >= 1.2) ||         // an audible pause
            (grown > hardMax && dur >= MIN_DUR)) {    // this cue would overrun
          out.push(mkSegment(buf));
          buf = [];
        }
      }
      buf.push(cues[i]);
    }
    if (buf.length) out.push(mkSegment(buf));
    return out;
  }

  function rebuild() {
    S.segments = S.cues.length
      ? (hasPunctuation(S.cues) ? byPunctuation(S.cues, S.perChunk) : byGap(S.cues, S.perChunk))
      : [];
    S.cur = -1;
    applyFilter();
  }

  /* ---------- player ---------- */

  function usingAudio() { return S.mode === "audio"; }

  function mediaReady() {
    return usingAudio() ? !!audioFile && $("sh-audio").readyState >= 1 : playerReady;
  }

  function mediaTime() {
    if (usingAudio()) return $("sh-audio").currentTime || 0;
    return player.getCurrentTime();
  }

  function mediaSeek(t) {
    if (usingAudio()) $("sh-audio").currentTime = Math.max(0, t);
    else player.seekTo(t, true);
  }

  function mediaPlay() {
    if (usingAudio()) {
      var promise = $("sh-audio").play();
      if (promise && promise.catch) promise.catch(function () {});
    } else player.playVideo();
  }

  function mediaPause() {
    if (usingAudio()) $("sh-audio").pause();
    else player.pauseVideo();
  }

  function mediaSetRate(rate) {
    if (usingAudio()) $("sh-audio").playbackRate = rate;
    else if (playerReady) player.setPlaybackRate(rate);
  }

  function mediaIsPlaying() {
    if (usingAudio()) return !$("sh-audio").paused;
    return playerReady && player.getPlayerState() === YT.PlayerState.PLAYING;
  }

  function loadYouTubeAPI(cb) {
    if (window.YT && window.YT.Player) { cb(); return; }
    var prev = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = function () {
      if (typeof prev === "function") prev();
      cb();
    };
    if (!document.getElementById("sh-yt-api")) {
      var s = document.createElement("script");
      s.id = "sh-yt-api";
      s.src = "https://www.youtube.com/iframe_api";
      document.head.appendChild(s);
    }
  }

  function mountPlayer(videoId) {
    // Called from both the load handler and openVideo. Re-mounting the video
    // already on screen used to clear playerReady, and since onReady fires only
    // once per player object it never came back -- which silently killed the
    // segment loop and every seek.
    // Guard on mountedId alone, not on `player`: the API loads asynchronously,
    // so during the first load `player` is still null and a second call would
    // slip through and load the video twice.
    if (mountedId === videoId) return;
    mountedId = videoId;
    loadYouTubeAPI(function () {
      if (player) {
        // A different video in the same player: the player object stays ready,
        // only the media changes. Clearing playerReady here would strand it.
        pendingSeek = null;
        player.loadVideoById(videoId);
        return;
      }
      player = new YT.Player("sh-yt", {
        videoId: videoId,
        playerVars: { rel: 0, modestbranding: 1, playsinline: 1, cc_load_policy: 0 },
        events: {
          onReady: onPlayerReady,
          onStateChange: onPlayerState,
        },
      });
    });
  }

  function onPlayerReady() {
    playerReady = true;
    try { player.setPlaybackRate(S.rate); } catch (e) {}
    try {
      var d = player.getVideoData();
      if (d && d.title) {
        S.title = d.title;
        $("sh-title").textContent = d.title;
        touchRecent(S.videoId, d.title, S.segments.length);
      }
    } catch (e) {}
    if (pendingSeek !== null) {
      var i = pendingSeek;
      pendingSeek = null;
      goTo(i, true);
    }
  }

  function onPlayerState(e) {
    if (e.data === YT.PlayerState.PLAYING) {
      try { player.setPlaybackRate(S.rate); } catch (err) {}
      startTimer();
      $("sh-play").textContent = "⏸";
    } else {
      if (e.data !== YT.PlayerState.BUFFERING) stopTimer();
      $("sh-play").textContent = "▶";
    }
  }

  function startTimer() {
    if (timer) return;
    timer = setInterval(tick, 100);
  }

  function stopTimer() {
    if (!timer) return;
    clearInterval(timer);
    timer = null;
  }

  function tick() {
    if (!mediaReady() || S.cur < 0) return;
    var seg = S.segments[S.cur];
    if (!seg) return;
    if (performance.now() - seekGuard < 300) return; // let the seek settle

    var t;
    try { t = mediaTime(); } catch (e) { return; }
    var start = seg.start + S.offset;
    var end = seg.end + S.offset;

    if (t >= end - 0.04) {
      S.repeatDone++;
      var more = S.repeatTarget === 0 || S.repeatDone < S.repeatTarget;
      if (more) {
        seekGuard = performance.now();
        mediaSeek(start);
      } else if (S.autoAdvance && S.cur < S.segments.length - 1) {
        goTo(S.cur + 1, true);
      } else {
        mediaPause();
        S.repeatDone = 0;
      }
      renderRepeat();
    } else if (t < start - 0.6) {
      // User scrubbed backwards out of the segment; follow them.
      seekGuard = performance.now();
      mediaSeek(start);
    }
  }

  function goTo(i, play) {
    if (!S.segments.length) return;
    i = Math.max(0, Math.min(i, S.segments.length - 1));
    S.cur = i;
    S.repeatDone = 0;
    renderActive();
    renderRepeat();
    if (!mediaReady()) { pendingSeek = i; return; }
    seekGuard = performance.now();
    mediaSeek(S.segments[i].start + S.offset);
    if (play !== false) mediaPlay();
  }

  /* ---------- local audio & background practice track ---------- */

  function setMode(mode) {
    S.mode = mode === "audio" ? "audio" : "youtube";
    var isAudio = usingAudio();
    $("sh-app").classList.toggle("is-audio", isAudio);
    $("sh-mode-youtube").classList.toggle("is-on", !isAudio);
    $("sh-mode-audio").classList.toggle("is-on", isAudio);
    $("sh-mode-youtube").setAttribute("aria-pressed", isAudio ? "false" : "true");
    $("sh-mode-audio").setAttribute("aria-pressed", isAudio ? "true" : "false");
    $("sh-youtube-load").hidden = isAudio;
    $("sh-audio-load").hidden = !isAudio;
    $("sh-background").hidden = !isAudio;
    stopTimer();
    if (isAudio) {
      if (playerReady) player.pauseVideo();
      if (audioFile) {
        $("sh-app").classList.add("is-loaded");
        mediaSetRate(S.rate);
      }
    } else {
      $("sh-audio").pause();
      if (S.videoId) $("sh-app").classList.add("is-loaded");
    }
    if (S.cur >= 0 && mediaReady()) goTo(S.cur, false);
  }

  function openAudioFile(file) {
    if (!file) return;
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    if (bgUrl) { URL.revokeObjectURL(bgUrl); bgUrl = null; }
    audioFile = file;
    decodedAudio = null;
    audioUrl = URL.createObjectURL(file);
    $("sh-audio").src = audioUrl;
    $("sh-audio-name").textContent = file.name + " · " + (file.size / 1024 / 1024).toFixed(1) + " MB";
    $("sh-title").textContent = file.name;
    $("sh-app").classList.add("is-loaded");
    mediaSetRate(S.rate);
    status(S.segments.length
      ? "오디오를 불러왔습니다. 자막 싱크를 맞춘 뒤 문장을 선택하세요."
      : "오디오를 불러왔습니다. SRT/VTT 파일을 열거나 자막을 붙여넣으세요.", "ok");
    if (S.segments.length) goTo(Math.max(0, S.cur), false);
  }

  function decodeLocalAudio() {
    if (decodedAudio) return Promise.resolve(decodedAudio);
    if (!audioFile) return Promise.reject(new Error("먼저 기기에서 오디오를 선택하세요."));
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return Promise.reject(new Error("이 브라우저는 오디오 변환을 지원하지 않습니다."));
    return audioFile.arrayBuffer().then(function (data) {
      var ctx = new AC();
      return ctx.decodeAudioData(data.slice(0)).then(function (buffer) {
        decodedAudio = buffer;
        if (ctx.close) ctx.close();
        return buffer;
      });
    });
  }

  function practiceParts(buffer) {
    if (!S.segments.length) throw new Error("먼저 자막을 불러오세요.");
    var first = S.cur < 0 ? 0 : S.cur;
    var last = S.autoAdvance ? S.segments.length - 1 : first;
    var repeats = S.repeatTarget === 0 ? 1 : Math.max(1, S.repeatTarget);
    var parts = [];
    for (var i = first; i <= last; i++) {
      var start = Math.max(0, S.segments[i].start + S.offset);
      var end = Math.min(buffer.duration, S.segments[i].end + S.offset);
      if (end <= start) continue;
      for (var r = 0; r < repeats; r++) parts.push({ start: start, end: end });
    }
    if (!parts.length) throw new Error("자막 구간이 오디오 재생 시간 안에 없습니다. 싱크 보정을 확인하세요.");
    return parts;
  }

  function writeAscii(view, at, value) {
    for (var i = 0; i < value.length; i++) view.setUint8(at + i, value.charCodeAt(i));
  }

  function renderPracticeWav(buffer) {
    var parts = practiceParts(buffer);
    var sampleRate = buffer.sampleRate;
    var rate = Math.max(.5, Math.min(2, S.rate));
    var gapFrames = Math.round(Math.max(0, S.gap) * sampleRate);
    var frames = 0;
    for (var i = 0; i < parts.length; i++) {
      frames += Math.max(1, Math.round((parts[i].end - parts[i].start) * sampleRate / rate));
      if (i < parts.length - 1) frames += gapFrames;
    }
    if (S.repeatTarget === 0) frames += gapFrames;
    if (frames / sampleRate > 12 * 60) {
      throw new Error("연습 트랙이 12분을 넘습니다. 현재 문장을 뒤쪽으로 옮기거나 자동 이동을 꺼주세요.");
    }

    // Mono keeps a mobile-generated practice track small and preserves speech clearly.
    var bytes = 44 + frames * 2;
    var out = new ArrayBuffer(bytes);
    var view = new DataView(out);
    writeAscii(view, 0, "RIFF"); view.setUint32(4, bytes - 8, true);
    writeAscii(view, 8, "WAVE"); writeAscii(view, 12, "fmt ");
    view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); writeAscii(view, 36, "data");
    view.setUint32(40, frames * 2, true);

    var channels = [];
    for (var c = 0; c < buffer.numberOfChannels; c++) channels.push(buffer.getChannelData(c));
    var pos = 44;
    for (var p = 0; p < parts.length; p++) {
      var sourceStart = parts[p].start * sampleRate;
      var count = Math.max(1, Math.round((parts[p].end - parts[p].start) * sampleRate / rate));
      for (var f = 0; f < count; f++) {
        var sourceIndex = Math.min(buffer.length - 1, Math.floor(sourceStart + f * rate));
        var sample = 0;
        for (var ch = 0; ch < channels.length; ch++) sample += channels[ch][sourceIndex] || 0;
        sample = Math.max(-1, Math.min(1, sample / channels.length));
        view.setInt16(pos, sample < 0 ? sample * 32768 : sample * 32767, true);
        pos += 2;
      }
      if (p < parts.length - 1) pos += gapFrames * 2;
    }
    return new Blob([out], { type: "audio/wav" });
  }

  function startBackgroundPractice() {
    if (!audioFile) { status("먼저 기기에서 오디오를 선택하세요.", "err"); return; }
    if (!S.segments.length) { status("먼저 자막을 불러오세요.", "err"); return; }
    var btn = $("sh-bg-start");
    btn.disabled = true;
    btn.textContent = "연습 트랙 만드는 중…";
    $("sh-bg-note").textContent = "오디오는 서버로 전송되지 않으며 이 기기 안에서만 처리됩니다.";
    decodeLocalAudio().then(function (buffer) {
      var blob = renderPracticeWav(buffer);
      if (bgUrl) URL.revokeObjectURL(bgUrl);
      bgUrl = URL.createObjectURL(blob);
      var bg = $("sh-bg-audio");
      bg.src = bgUrl;
      bg.loop = S.repeatTarget === 0;
      bg.hidden = false;
      if ("mediaSession" in navigator) {
        try {
          navigator.mediaSession.metadata = new MediaMetadata({
            title: S.repeatTarget === 0 ? "현재 문장 반복" : "Shadowing 연습 트랙",
            artist: audioFile.name,
            album: "Shadowing",
          });
        } catch (e) {}
      }
      $("sh-audio").pause();
      return bg.play().catch(function (err) {
        if (err && err.name === "NotAllowedError") {
          $("sh-bg-note").textContent = "연습 트랙이 준비됐습니다. 아래 재생 버튼을 한 번 눌러 시작하세요.";
          status("모바일 브라우저가 자동 재생을 막았습니다. 아래 오디오의 재생 버튼을 누르면 백그라운드 연습이 시작됩니다.", "warn");
          return "manual";
        }
        throw err;
      });
    }).then(function (playState) {
      if (playState === "manual") return;
      status("백그라운드 연습을 시작했습니다. 화면을 끄기 전에 잠금 화면 재생 컨트롤이 나타나는지 확인하세요.", "ok");
      $("sh-bg-note").textContent = S.repeatTarget === 0
        ? "현재 구간을 계속 반복합니다."
        : "현재 문장부터 설정한 횟수·공백으로 만든 연습 트랙입니다.";
    }).catch(function (err) {
      status(err && err.message ? err.message : "연습 트랙을 만들지 못했습니다.", "err");
      $("sh-bg-note").textContent = "긴 파일은 모바일 메모리 제한 때문에 처리하지 못할 수 있습니다.";
    }).then(function () {
      btn.disabled = false;
      btn.textContent = "백그라운드 연습 시작";
    });
  }

  /* ---------- rendering ---------- */

  function fmt(t) {
    t = Math.max(0, Math.floor(t));
    var m = Math.floor(t / 60), s = t % 60;
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function escapeHTML(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function highlight(text, term) {
    var safe = escapeHTML(text);
    if (!term) return safe;
    var re = new RegExp("(" + term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "gi");
    return safe.replace(re, "<mark>$1</mark>");
  }

  function applyFilter() {
    var term = S.filter.trim().toLowerCase();
    S.view = [];
    for (var i = 0; i < S.segments.length; i++) {
      if (!term || S.segments[i].text.toLowerCase().indexOf(term) !== -1) S.view.push(i);
    }
    renderList();
    renderCount();
  }

  function renderList() {
    var box = $("sh-list");
    if (!S.segments.length) {
      box.innerHTML = '<p class="sh-empty">자막을 불러오면 문장이 여기에 나옵니다.</p>';
      return;
    }
    if (!S.view.length) {
      box.innerHTML = '<p class="sh-empty">검색어와 맞는 문장이 없습니다.</p>';
      return;
    }
    var html = "";
    for (var k = 0; k < S.view.length; k++) {
      var i = S.view[k];
      var seg = S.segments[i];
      html +=
        '<button type="button" class="sh-item" data-i="' + i + '">' +
        '<span class="sh-time">' + fmt(seg.start + S.offset) + "</span>" +
        '<span class="sh-text">' + highlight(seg.text, S.filter.trim()) + "</span>" +
        "</button>";
    }
    box.innerHTML = html;
    renderActive();
  }

  function renderActive() {
    var items = document.querySelectorAll(".sh-item");
    for (var i = 0; i < items.length; i++) {
      var on = +items[i].dataset.i === S.cur;
      items[i].classList.toggle("is-active", on);
      if (on) items[i].scrollIntoView({ block: "nearest" });
    }
    var seg = S.segments[S.cur];
    $("sh-caption").textContent = seg ? seg.text : "";
    $("sh-pos").textContent = S.segments.length
      ? (S.cur < 0 ? "—" : S.cur + 1) + " / " + S.segments.length
      : "—";
  }

  function renderRepeat() {
    $("sh-repeat-now").textContent = S.repeatTarget === 0
      ? S.repeatDone + "회"
      : Math.min(S.repeatDone + 1, S.repeatTarget) + " / " + S.repeatTarget;
  }

  function renderCount() {
    var n = S.segments.length;
    var shown = S.view.length;
    $("sh-count").textContent = n
      ? (S.filter.trim() ? shown + " / " + n + " 문장" : n + " 문장")
      : "";
  }

  function renderOffset() {
    $("sh-offset-val").textContent = (S.offset >= 0 ? "+" : "") + S.offset.toFixed(1) + "s";
  }

  function savedMeta(r) {
    var bits = [];
    if (r.n) bits.push(r.n + "구간");
    if (r.kind) bits.push(r.kind === "auto" ? "자동 자막" : "수동 자막");
    if (r.lang) bits.push(String(r.lang).toUpperCase());
    if (r.ts) {
      var d = new Date(r.ts);
      bits.push((d.getMonth() + 1) + "월 " + d.getDate() + "일");
    }
    return bits.join(" · ");
  }

  // What the library shows: everything saved in this browser, plus everything
  // committed to the repo. A repo entry is visible on a device that has never
  // opened it, which is the whole point of publishing them.
  function library() {
    var seen = {};
    var hidden = load(HIDDEN_KEY, []);
    var out = load(RECENT_KEY, []).map(function (r) {
      seen[r.id] = true;
      return { r: r, local: true, repo: !!repoEntry(r.id) };
    });
    for (var i = 0; i < REPO_INDEX.length; i++) {
      if (!seen[REPO_INDEX[i].id] && hidden.indexOf(REPO_INDEX[i].id) === -1) {
        out.push({ r: REPO_INDEX[i], local: false, repo: true });
      }
    }
    out.sort(function (a, b) { return (b.r.ts || 0) - (a.r.ts || 0); });
    return out;
  }

  function renderRecent() {
    var list = library();
    var wrap = $("sh-saved");
    wrap.hidden = !list.length;
    if (!list.length) return;


    var q = S.savedFilter.trim().toLowerCase();
    var shown = list.filter(function (e) {
      return !q || String(e.r.title || e.r.id).toLowerCase().indexOf(q) !== -1;
    });
    $("sh-saved-count").textContent = q ? shown.length + " / " + list.length : String(list.length);

    var box = $("sh-saved-list");
    if (!shown.length) {
      box.innerHTML = '<p class="sh-empty">제목이 맞는 영상이 없습니다.</p>';
      return;
    }
    var html = "";
    for (var i = 0; i < shown.length; i++) {
      var e = shown[i], r = e.r;
      var title = escapeHTML(String(r.title || r.id));
      html +=
        '<div class="sh-card">' +
          '<button type="button" class="sh-card-open" data-id="' + r.id + '" title="' + title + '">' +
            '<span class="sh-card-title">' + title + "</span>" +
            '<span class="sh-card-meta">' + escapeHTML(savedMeta(r)) +
              (e.repo ? ' <span class="sh-tag">사이트 자막</span>' : "") + "</span>" +
          "</button>" +
          '<button type="button" class="sh-card-del" data-del="' + r.id + '" title="' +
            (e.repo ? "이 브라우저의 목록에서 숨기기" : "목록과 저장된 자막에서 삭제") +
            '" aria-label="' + title + ' 삭제">삭제</button>' +
        "</div>";
    }
    box.innerHTML = html;
  }

  function renderServerState() {
    var on = !!serverBase();
    $("sh-settings-toggle").textContent = on ? "자막 서버 ✓" : "자막 서버 설정";
    $("sh-refetch").hidden = !on;
  }

  function status(msg, kind) {
    var el = $("sh-status");
    el.textContent = msg || "";
    el.className = "sh-status" + (kind ? " is-" + kind : "");
  }

  /* ---------- flow ---------- */

  var ORIGIN_LABEL = {
    cache: "브라우저에 저장된 자막",
    repo: "공유 자막",
    server: "자막 서버에서 가져옴",
    file: "직접 올린 파일",
  };

  function openVideo(id, cues, meta) {
    meta = meta || {};
    S.videoId = id;
    S.cues = cues;
    S.kind = meta.kind || "";
    if (meta.title) S.title = meta.title;
    applyUser(load(USER_KEY(id), null));
    syncControls();
    rebuild();
    renderOffset();
    mountPlayer(id);
    $("sh-app").classList.add("is-loaded");
    if (S.title) $("sh-title").textContent = S.title;

    // Which splitter ran matters to the user: auto-captions carry no
    // punctuation, so chunks are cut on silence and land less cleanly.
    var how = hasPunctuation(cues) ? "문장부호 기준" : "무음 구간 기준";
    var kindLabel = S.kind === "auto" ? "자동 자막" : S.kind === "manual" ? "수동 자막" : "";
    status(
      [ORIGIN_LABEL[meta.from] || "", kindLabel, cues.length + "개 자막 → " +
        S.segments.length + "개 구간 (" + how + ")"]
        .filter(Boolean).join(" · "),
      "ok"
    );
    touchRecent(id, S.title || id, S.segments.length);
    if (S.segments.length) goTo(0, false);
  }

  function cacheCaptions(id, data) {
    return save(CAP_KEY(id), {
      videoId: id,
      cues: data.cues,
      kind: data.kind || "",
      lang: data.lang || S.lang,
      title: data.title || "",
      savedAt: Date.now(),
    });
  }

  function loadFromCache(id) {
    var cached = load(CAP_KEY(id), null);
    if (!cached || !cached.cues || !cached.cues.length) return false;
    openVideo(id, cached.cues, { from: "cache", kind: cached.kind, title: cached.title });
    return true;
  }

  function ingestCaptions(raw) {
    var id = parseVideoId($("sh-url").value);
    if (!id && !usingAudio()) { status("먼저 유튜브 주소를 입력하세요.", "err"); return; }
    var cues = parseAny(raw);
    if (!cues.length) {
      status("자막을 읽지 못했습니다. SRT/VTT 파일이거나, 유튜브 '스크립트 표시'에서 복사한 내용이어야 합니다.", "err");
      return;
    }
    if (usingAudio()) {
      S.cues = cues;
      S.kind = "";
      rebuild();
      syncControls();
      renderOffset();
      $("sh-app").classList.add("is-loaded");
      status(cues.length + "개 자막 → " + S.segments.length + "개 구간 · 내 오디오", "ok");
      if (S.segments.length) goTo(0, false);
      return;
    }
    var ok = cacheCaptions(id, { cues: cues });
    openVideo(id, cues, { from: "file" });
    if (!ok) status(S.segments.length + "개 구간 (브라우저 저장 공간이 꽉 차 캐시하지 못함)", "err");
  }

  function setBusy(on) {
    var btn = $("sh-go");
    btn.disabled = on;
    btn.textContent = on ? "가져오는 중…" : "불러오기";
  }

  function serverFailed(err) {
    var msg = err.message;
    if (err.detail && err.detail.length) {
      // The server reports what each yt-dlp player client said; the first line
      // is usually enough to tell "no captions" from "YouTube refused".
      msg += " — " + err.detail[0];
    }
    status(msg + " SRT/VTT 파일을 직접 올려도 됩니다.", "err");
  }

  // The single entry point for "show me this video". Both the 불러오기 button
  // and the library cards go through here, so they can never drift apart.
  function openVideoId(id) {
    if (loadFromCache(id)) return;

    S.videoId = id;
    mountPlayer(id);
    $("sh-app").classList.add("is-loaded");

    if (repoEntry(id)) { pullFromRepo(id); return; }
    if (serverBase()) { pullCaptions(id, false); return; }
    status("이 영상의 자막이 아직 없습니다. 유튜브 스크립트를 붙여넣거나 SRT/VTT 파일을 올리세요.", "warn");
  }

  function pullFromRepo(id) {
    setBusy(true);
    status("공유 자막을 가져오는 중…");
    fetchFromRepo(id, function (err, data) {
      setBusy(false);
      if (err) {
        // Fall through rather than dead-end: the server may still have it.
        if (serverBase()) { pullCaptions(id, false); return; }
        status(err.message + " SRT/VTT 파일을 직접 올려도 됩니다.", "err");
        return;
      }
      cacheCaptions(id, data);
      openVideo(id, data.cues, { from: "repo", kind: data.kind, title: data.title });
    });
  }

  function pullCaptions(id, refresh) {
    setBusy(true);
    status("자막을 가져오는 중…");
    fetchFromServer(id, { refresh: refresh }, function (err, data) {
      setBusy(false);
      if (err) { serverFailed(err); return; }
      var ok = cacheCaptions(id, data);
      openVideo(id, data.cues, { from: "server", kind: data.kind, title: data.title });
      if (!ok) status(S.segments.length + "개 구간 (브라우저 저장 공간이 꽉 차 캐시하지 못함)", "err");
    });
  }

  function syncControls() {
    $("sh-perchunk").value = String(S.perChunk);
    $("sh-repeat").value = String(S.repeatTarget);
    $("sh-advance").checked = S.autoAdvance;
    $("sh-gap").value = String(S.gap);
    var rates = document.querySelectorAll(".sh-rate");
    for (var i = 0; i < rates.length; i++) {
      rates[i].classList.toggle("is-on", parseFloat(rates[i].dataset.rate) === S.rate);
    }
  }

  /* ---------- wiring ---------- */

  function init() {
    if (!$("sh-app")) return;

    S.lang = load("shadow:lang", "en") || "en";
    S.gap = load("shadow:audioGap", 1);
    $("sh-lang").value = S.lang;

    if (load("shadow:savedClosed", false)) {
      $("sh-saved").classList.add("is-closed");
      $("sh-saved-toggle").setAttribute("aria-expanded", "false");
    }
    loadRepoIndex(function () {
      renderRecent();
      // ?v=<id> lets a bookmark, an iOS Shortcut or a share sheet open a video
      // directly; the repo index has to be in before we can resolve it.
      var m = String(location.search).match(/[?&]v=([\w-]{11})/);
      if (m) { $("sh-url").value = "https://youtu.be/" + m[1]; openVideoId(m[1]); }
    });
    renderRecent();
    renderList();
    renderOffset();
    renderServerState();
    syncControls();

    $("sh-mode-youtube").addEventListener("click", function () { setMode("youtube"); });
    $("sh-mode-audio").addEventListener("click", function () { setMode("audio"); });

    $("sh-audio-file").addEventListener("change", function (e) {
      var f = e.target.files && e.target.files[0];
      if (f) openAudioFile(f);
      e.target.value = "";
    });

    $("sh-audio").addEventListener("play", function () {
      startTimer();
      $("sh-play").textContent = "⏸";
    });
    $("sh-audio").addEventListener("pause", function () {
      stopTimer();
      $("sh-play").textContent = "▶";
    });
    $("sh-audio").addEventListener("loadedmetadata", function () {
      if (pendingSeek !== null) {
        var i = pendingSeek;
        pendingSeek = null;
        goTo(i, false);
      }
    });

    $("sh-gap").addEventListener("change", function (e) {
      var value = parseFloat(e.target.value);
      S.gap = isFinite(value) ? Math.max(0, Math.min(10, value)) : 1;
      e.target.value = String(S.gap);
      save("shadow:audioGap", S.gap);
      saveUser();
    });
    $("sh-bg-start").addEventListener("click", startBackgroundPractice);

    $("sh-go").addEventListener("click", function () {
      setMode("youtube");
      var raw = $("sh-url").value.trim();
      if (!raw) return;
      var id = parseVideoId(raw);
      if (!id) {
        if (serverBase()) { runSearch(raw); return; }
        // Telling someone to go press a button is worse than pressing it.
        $("sh-settings").hidden = false;
        $("sh-server-url").value = serverBase();
        $("sh-server-url").focus();
        status("키워드로 검색하려면 자막 서버 주소가 필요합니다. 이 사이트에는 아직 저장돼 있지 않습니다.", "warn");
        return;
      }
      renderResults(null);
      // ① browser cache -> ② repo -> ③ caption server -> ask for a file
      openVideoId(id);
    });

    $("sh-refetch").addEventListener("click", function () {
      if (!S.videoId) { status("먼저 영상을 불러오세요.", "err"); return; }
      if (!serverBase()) { status("자막 서버 주소가 설정되지 않았습니다.", "err"); return; }
      pullCaptions(S.videoId, true);
    });

    $("sh-settings-toggle").addEventListener("click", function () {
      var box = $("sh-settings");
      box.hidden = !box.hidden;
      if (!box.hidden) { $("sh-server-url").value = serverBase(); $("sh-server-url").focus(); }
    });

    $("sh-server-save").addEventListener("click", function () {
      var v = $("sh-server-url").value.trim();
      save("shadow:server", v);
      renderServerState();
      if (!v) { $("sh-settings").hidden = true; status("자막 서버 주소를 지웠습니다.", "ok"); return; }
      status("연결 확인 중…");
      pingServer(function (err, hasSearch) {
        if (err) { status(err.message, "err"); return; }
        $("sh-settings").hidden = true;
        status(hasSearch
          ? "자막 서버에 연결했습니다. 이제 키워드로 검색할 수 있습니다."
          : "연결은 됐지만 검색이 없는 예전 서버입니다. bin/caption-server.py --install 로 갱신하세요.",
          hasSearch ? "ok" : "warn");
      });
    });

    $("sh-lang").addEventListener("change", function (e) {
      S.lang = e.target.value;
      save("shadow:lang", S.lang);
    });

    $("sh-url").addEventListener("keydown", function (e) {
      if (e.key === "Enter") { e.preventDefault(); $("sh-go").click(); }
    });

    $("sh-file").addEventListener("change", function (e) {
      var f = e.target.files && e.target.files[0];
      if (!f) return;
      var r = new FileReader();
      r.onload = function () { ingestCaptions(r.result); };
      r.readAsText(f);
      e.target.value = "";
    });

    $("sh-paste-toggle").addEventListener("click", function () {
      var box = $("sh-paste-wrap");
      box.hidden = !box.hidden;
      if (!box.hidden) $("sh-paste").focus();
    });

    $("sh-paste-apply").addEventListener("click", function () {
      ingestCaptions($("sh-paste").value);
      $("sh-paste-wrap").hidden = true;
    });

    $("sh-saved-list").addEventListener("click", function (e) {
      var del = e.target.closest(".sh-card-del");
      if (del) {
        forget(del.dataset.del);
        status("저장된 영상 목록에서 삭제했습니다.", "ok");
        return;
      }
      var open = e.target.closest(".sh-card-open");
      if (!open) return;
      setMode("youtube");
      $("sh-url").value = "https://youtu.be/" + open.dataset.id;
      openVideoId(open.dataset.id);
    });

    $("sh-results").addEventListener("click", function (e) {
      var hit = e.target.closest(".sh-result");
      if (!hit) return;
      setMode("youtube");
      $("sh-url").value = "https://youtu.be/" + hit.dataset.id;
      renderResults(null);
      openVideoId(hit.dataset.id);
    });

    $("sh-saved-search").addEventListener("input", function (e) {
      S.savedFilter = e.target.value;
      renderRecent();
    });

    $("sh-saved-clear").addEventListener("click", function () {
      var n = library().length;
      if (!n) { status("저장된 영상 목록이 비어 있습니다.", "warn"); return; }
      if (!window.confirm(n + "개 영상을 이 브라우저의 목록에서 지웁니다.\n직접 저장한 자막과 설정도 함께 삭제합니다. 계속할까요?")) return;
      forgetAll();
      status("저장된 영상 목록을 모두 비웠습니다.", "ok");
    });

    $("sh-saved-toggle").addEventListener("click", function () {
      var closed = $("sh-saved").classList.toggle("is-closed");
      $("sh-saved-toggle").setAttribute("aria-expanded", closed ? "false" : "true");
      save("shadow:savedClosed", closed);
    });

    $("sh-list").addEventListener("click", function (e) {
      var item = e.target.closest(".sh-item");
      if (item) goTo(+item.dataset.i, true);
    });

    $("sh-prev").addEventListener("click", function () { goTo(S.cur - 1, true); });
    $("sh-next").addEventListener("click", function () { goTo(S.cur + 1, true); });

    $("sh-play").addEventListener("click", function () {
      if (!mediaReady()) return;
      if (mediaIsPlaying()) mediaPause();
      else if (S.cur < 0) goTo(0, true);
      else mediaPlay();
    });

    $("sh-replay").addEventListener("click", function () {
      if (S.cur >= 0) { S.repeatDone = 0; goTo(S.cur, true); }
    });

    var rates = document.querySelectorAll(".sh-rate");
    for (var i = 0; i < rates.length; i++) {
      rates[i].addEventListener("click", function (e) {
        S.rate = parseFloat(e.currentTarget.dataset.rate);
        if (mediaReady()) mediaSetRate(S.rate);
        syncControls();
        saveUser();
      });
    }

    $("sh-offset-minus").addEventListener("click", function () {
      S.offset = Math.round((S.offset - 0.2) * 10) / 10;
      renderOffset(); renderList(); saveUser();
      if (S.cur >= 0) goTo(S.cur, false);
    });

    $("sh-offset-plus").addEventListener("click", function () {
      S.offset = Math.round((S.offset + 0.2) * 10) / 10;
      renderOffset(); renderList(); saveUser();
      if (S.cur >= 0) goTo(S.cur, false);
    });

    $("sh-offset-reset").addEventListener("click", function () {
      S.offset = 0;
      renderOffset(); renderList(); saveUser();
      if (S.cur >= 0) goTo(S.cur, false);
    });

    $("sh-perchunk").addEventListener("change", function (e) {
      S.perChunk = parseInt(e.target.value, 10);
      rebuild(); saveUser();
      if (S.segments.length) goTo(0, false);
      status(S.segments.length + "개 구간으로 다시 나눴습니다.", "ok");
    });

    $("sh-repeat").addEventListener("change", function (e) {
      S.repeatTarget = parseInt(e.target.value, 10);
      S.repeatDone = 0;
      renderRepeat(); saveUser();
    });

    $("sh-advance").addEventListener("change", function (e) {
      S.autoAdvance = e.target.checked;
      saveUser();
    });

    $("sh-search").addEventListener("input", function (e) {
      S.filter = e.target.value;
      applyFilter();
    });

    document.addEventListener("keydown", function (e) {
      if (!S.segments.length) return;
      var tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || e.metaKey || e.ctrlKey) return;
      if (e.key === "ArrowLeft") { e.preventDefault(); goTo(S.cur - 1, true); }
      else if (e.key === "ArrowRight") { e.preventDefault(); goTo(S.cur + 1, true); }
      else if (e.key === " ") { e.preventDefault(); $("sh-play").click(); }
      else if (e.key.toLowerCase() === "r") { e.preventDefault(); $("sh-replay").click(); }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
