---
layout: page
title: Shadowing
permalink: /shadowing/
description: 유튜브 영상을 문장 단위로 끊어 반복 재생하는 영어 쉐도잉 도구
nav: true
nav_order: 4.6
_styles: |
  .sh-app { margin-top: 1rem; }

  .sh-load { display: flex; gap: .5rem; flex-wrap: wrap; }
  .sh-load input {
    flex: 1 1 18rem; min-width: 0; padding: .55rem .75rem;
    border: 1px solid var(--global-divider-color); border-radius: .4rem;
    background: var(--global-bg-color); color: var(--global-text-color); font-size: .95rem;
  }
  .sh-load input:focus { outline: 2px solid var(--global-theme-color); outline-offset: -1px; }

  .sh-btn {
    padding: .55rem .9rem; border: 1px solid var(--global-divider-color);
    border-radius: .4rem; background: var(--global-card-bg-color);
    color: var(--global-text-color); font-size: .9rem; cursor: pointer; white-space: nowrap;
  }
  .sh-btn:hover { border-color: var(--global-theme-color); color: var(--global-theme-color); }
  .sh-btn.is-primary { background: var(--global-theme-color); color: #fff; border-color: var(--global-theme-color); }
  .sh-btn.is-primary:hover { color: #fff; opacity: .88; }
  .sh-btn.is-on { background: var(--global-theme-color); color: #fff; border-color: var(--global-theme-color); }

  .sh-sources { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; margin-top: .6rem; }
  .sh-sources select {
    padding: .5rem .6rem; font-size: .88rem; border-radius: .4rem;
    border: 1px solid var(--global-divider-color);
    background: var(--global-bg-color); color: var(--global-text-color);
  }
  .sh-sep { flex: 1 1 0; min-width: 0; }

  .sh-settings {
    margin-top: .6rem; padding: .8rem .9rem;
    border: 1px solid var(--global-divider-color); border-radius: .5rem;
    background: var(--global-card-bg-color);
  }
  .sh-settings p { font-size: .82rem; color: var(--global-text-color-light); margin: 0 0 .6rem; }
  .sh-settings pre {
    margin: 0 0 .7rem; padding: .55rem .7rem; font-size: .78rem; line-height: 1.6;
    border-radius: .35rem; background: var(--global-code-bg-color); overflow-x: auto;
  }
  .sh-settings pre code { background: none; padding: 0; }
  .sh-settings .sh-row { flex-wrap: nowrap; }
  .sh-settings input {
    flex: 1 1 auto; min-width: 0; padding: .5rem .6rem; font-size: .88rem;
    border: 1px solid var(--global-divider-color); border-radius: .4rem;
    background: var(--global-bg-color); color: var(--global-text-color);
  }
  #sh-go[disabled] { opacity: .6; cursor: progress; }
  .sh-paste-wrap { margin-top: .6rem; }
  .sh-paste-wrap textarea {
    width: 100%; min-height: 9rem; padding: .6rem; font-family: monospace; font-size: .8rem;
    border: 1px solid var(--global-divider-color); border-radius: .4rem;
    background: var(--global-bg-color); color: var(--global-text-color);
  }

  .sh-status { margin-top: .6rem; font-size: .85rem; color: var(--global-text-color-light); min-height: 1.2em; }
  .sh-status.is-ok { color: var(--global-theme-color); }
  .sh-status.is-err { color: var(--global-danger-block); }
  .sh-status.is-warn { color: var(--global-warning-block); }

  /* ---- search results ---- */
  .sh-results {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(17rem, 1fr));
    gap: .5rem; margin-top: .7rem; max-height: 26rem; overflow-y: auto;
  }
  .sh-result {
    display: flex; gap: .6rem; align-items: stretch; min-width: 0; padding: 0;
    text-align: left; cursor: pointer; overflow: hidden;
    border: 1px solid var(--global-divider-color); border-radius: .5rem;
    background: transparent; color: var(--global-text-color);
  }
  .sh-result:hover { border-color: var(--global-theme-color); background: var(--global-card-bg-color); }
  .sh-thumb { flex: 0 0 6.4rem; width: 6.4rem; height: 3.6rem; object-fit: cover; background: #000; }
  .sh-result-body {
    flex: 1 1 auto; min-width: 0;
    display: flex; flex-direction: column; justify-content: center; gap: .2rem;
    padding: .4rem .6rem .4rem 0;
  }
  .sh-result-title {
    font-size: .84rem; font-weight: 500; line-height: 1.3;
    overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  }
  .sh-result-meta { font-size: .73rem; color: var(--global-text-color-light); }

  /* ---- saved video library ---- */
  .sh-saved { margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid var(--global-divider-color); }
  .sh-saved-head { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; margin-bottom: .75rem; }
  .sh-saved-toggle {
    display: inline-flex; align-items: center; gap: .4rem;
    border: 0; background: none; padding: 0; cursor: pointer;
    color: var(--global-text-color); font-size: .98rem; font-weight: 600;
  }
  .sh-saved-toggle:hover { color: var(--global-theme-color); }
  .sh-caret { display: inline-block; transition: transform .15s ease; font-size: .8em; }
  .sh-saved.is-closed .sh-caret { transform: rotate(-90deg); }
  .sh-saved.is-closed .sh-saved-list,
  .sh-saved.is-closed #sh-saved-search,
  .sh-saved.is-closed #sh-saved-clear { display: none; }
  .sh-badge {
    font-size: .76rem; font-weight: 500; padding: .1rem .45rem; border-radius: .8rem;
    background: var(--global-card-bg-color); color: var(--global-text-color-light);
    font-variant-numeric: tabular-nums;
  }
  .sh-saved-head input {
    margin-left: auto; width: 11rem; padding: .42rem .6rem; font-size: .86rem;
    border: 1px solid var(--global-divider-color); border-radius: .4rem;
    background: var(--global-bg-color); color: var(--global-text-color);
  }

  .sh-saved-list {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr));
    gap: .55rem; max-height: 24rem; overflow-y: auto;
  }
  .sh-card {
    display: flex; align-items: stretch; min-width: 0;
    border: 1px solid var(--global-divider-color); border-radius: .5rem; overflow: hidden;
  }
  .sh-card:hover { border-color: var(--global-theme-color); }
  .sh-card-open {
    flex: 1 1 auto; min-width: 0;
    display: flex; flex-direction: column; align-items: flex-start; gap: .25rem;
    padding: .6rem .7rem; text-align: left; cursor: pointer;
    border: 0; background: transparent; color: var(--global-text-color);
  }
  .sh-card-open:hover { background: var(--global-card-bg-color); }
  .sh-card-title {
    width: 100%; font-size: .9rem; font-weight: 500; line-height: 1.35;
    overflow: hidden; text-overflow: ellipsis;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
  }
  .sh-card-meta { font-size: .76rem; color: var(--global-text-color-light); font-variant-numeric: tabular-nums; }
  .sh-tag {
    display: inline-block; margin-left: .15rem; padding: 0 .3rem; border-radius: .25rem;
    font-size: .68rem; letter-spacing: .02em;
    background: var(--global-theme-color); color: #fff; opacity: .85;
  }
  .sh-card-del {
    flex: 0 0 auto; width: 2.1rem; cursor: pointer; font-size: .8rem;
    border: 0; border-left: 1px solid var(--global-divider-color);
    background: transparent; color: var(--global-text-color-light);
  }
  .sh-card-del:hover { background: var(--global-card-bg-color); color: var(--global-danger-block); }

  .sh-main {
    display: none; grid-template-columns: minmax(0,1fr) 22rem;
    gap: 1.25rem; margin-top: 1.5rem; align-items: start;
  }
  .sh-app.is-loaded .sh-main { display: grid; }

  .sh-video { position: relative; width: 100%; aspect-ratio: 16/9; background: #000; border-radius: .5rem; overflow: hidden; }
  .sh-video iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; }

  .sh-vid-title { font-size: .85rem; color: var(--global-text-color-light); margin: .5rem 0 0; }

  .sh-caption {
    margin-top: .9rem; padding: 1.1rem 1rem; min-height: 5.5rem;
    display: flex; align-items: center; justify-content: center; text-align: center;
    font-size: 1.35rem; line-height: 1.55; font-weight: 500;
    border: 1px solid var(--global-divider-color); border-radius: .5rem;
    background: var(--global-card-bg-color);
  }

  .sh-controls { margin-top: .9rem; display: flex; flex-direction: column; gap: .6rem; }
  .sh-row { display: flex; gap: .4rem; align-items: center; flex-wrap: wrap; }

  /* transport: one bar, play weighted, counters right-aligned */
  .sh-transport {
    display: flex; align-items: center; gap: .35rem;
    padding: .45rem .55rem;
    border: 1px solid var(--global-divider-color); border-radius: .55rem;
    background: var(--global-card-bg-color);
  }
  .sh-tbtn {
    flex: 0 0 auto; width: 3rem; height: 2.35rem;
    display: inline-flex; align-items: center; justify-content: center;
    border: 1px solid transparent; border-radius: .4rem;
    background: transparent; color: var(--global-text-color);
    font-size: 1rem; cursor: pointer;
  }
  .sh-tbtn:hover { background: var(--global-bg-color); border-color: var(--global-divider-color); }
  .sh-tbtn.is-play {
    width: 4.25rem; background: var(--global-theme-color);
    color: #fff; border-color: var(--global-theme-color);
  }
  .sh-tbtn.is-play:hover { opacity: .88; color: #fff; }
  .sh-readout { margin-left: auto; display: flex; align-items: baseline; gap: .7rem; padding-right: .35rem; }
  .sh-pos { font-size: .92rem; font-weight: 600; font-variant-numeric: tabular-nums; }
  .sh-repeat-now { font-size: .78rem; color: var(--global-text-color-light); font-variant-numeric: tabular-nums; }

  /* options: labels share one gutter so every row starts at the same x */
  .sh-opts {
    display: grid; grid-template-columns: auto minmax(0, 1fr);
    gap: .5rem .9rem; align-items: center;
    padding: .8rem .9rem;
    border: 1px solid var(--global-divider-color); border-radius: .55rem;
  }
  .sh-opt { display: contents; }
  .sh-label {
    font-size: .78rem; color: var(--global-text-color-light);
    text-align: right; white-space: nowrap;
  }
  .sh-opt-body { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }

  .sh-opts select {
    padding: .38rem .5rem; font-size: .85rem; border-radius: .4rem;
    border: 1px solid var(--global-divider-color);
    background: var(--global-bg-color); color: var(--global-text-color);
  }
  .sh-check { display: flex; align-items: center; gap: .35rem; font-size: .85rem; cursor: pointer; }

  /* segmented control (배속) */
  .sh-seg { display: inline-flex; border: 1px solid var(--global-divider-color); border-radius: .4rem; overflow: hidden; }
  .sh-seg button {
    border: 0; border-left: 1px solid var(--global-divider-color);
    padding: .38rem .72rem; font-size: .82rem; cursor: pointer;
    background: var(--global-bg-color); color: var(--global-text-color);
    font-variant-numeric: tabular-nums;
  }
  .sh-seg button:first-child { border-left: 0; }
  .sh-seg button:hover { background: var(--global-card-bg-color); color: var(--global-theme-color); }
  .sh-seg button.is-on { background: var(--global-theme-color); color: #fff; }

  /* stepper (싱크 보정) -- the value belongs inside the control, not beside it */
  .sh-stepper {
    display: inline-flex; align-items: stretch;
    border: 1px solid var(--global-divider-color); border-radius: .4rem; overflow: hidden;
  }
  .sh-stepper button {
    border: 0; padding: .38rem .8rem; font-size: 1rem; line-height: 1.2; cursor: pointer;
    background: var(--global-bg-color); color: var(--global-text-color);
  }
  .sh-stepper button:hover { background: var(--global-card-bg-color); color: var(--global-theme-color); }
  .sh-stepper span {
    min-width: 3.8rem; padding: .38rem .25rem; text-align: center;
    font-size: .82rem; font-variant-numeric: tabular-nums;
    border-left: 1px solid var(--global-divider-color);
    border-right: 1px solid var(--global-divider-color);
  }
  .sh-btn.is-quiet { padding: .35rem .65rem; font-size: .8rem; background: transparent; }

  /* Stretch to the video column's height instead of stopping short of it, and
     stay put while the page scrolls. */
  .sh-side {
    display: flex; flex-direction: column; min-height: 0;
    position: sticky; top: 5rem; max-height: calc(100vh - 7rem);
  }
  .sh-side-head { display: flex; gap: .5rem; align-items: center; margin-bottom: .5rem; }
  .sh-side-head input {
    flex: 1 1 auto; min-width: 0; padding: .45rem .6rem; font-size: .88rem;
    border: 1px solid var(--global-divider-color); border-radius: .4rem;
    background: var(--global-bg-color); color: var(--global-text-color);
  }
  .sh-count { font-size: .75rem; color: var(--global-text-color-light); white-space: nowrap; }

  .sh-list {
    flex: 1 1 auto; min-height: 14rem; overflow-y: auto;
    border: 1px solid var(--global-divider-color); border-radius: .5rem;
  }
  .sh-empty { padding: 1.25rem; font-size: .85rem; color: var(--global-text-color-light); text-align: center; margin: 0; }
  .sh-item {
    display: flex; gap: .6rem; width: 100%; text-align: left; padding: .55rem .7rem;
    border: 0; border-bottom: 1px solid var(--global-divider-color);
    background: transparent; color: var(--global-text-color);
    font-size: .88rem; line-height: 1.45; cursor: pointer;
  }
  .sh-item:last-child { border-bottom: 0; }
  .sh-item:hover { background: var(--global-card-bg-color); }
  .sh-item.is-active { background: var(--global-card-bg-color); box-shadow: inset 3px 0 0 var(--global-theme-color); }
  .sh-item.is-active .sh-text { font-weight: 600; }
  .sh-time { flex: 0 0 auto; font-size: .75rem; color: var(--global-text-color-light); font-variant-numeric: tabular-nums; padding-top: .15rem; }
  .sh-item mark { background: var(--global-theme-color); color: #fff; padding: 0 .1em; border-radius: .15em; }

  .sh-help { margin-top: 2rem; font-size: .85rem; color: var(--global-text-color-light); }
  .sh-help kbd {
    padding: .1rem .35rem; font-size: .78rem; border-radius: .25rem;
    border: 1px solid var(--global-divider-color); background: var(--global-card-bg-color);
  }

  @media (max-width: 900px) {
    .sh-main { grid-template-columns: minmax(0,1fr); }
    .sh-caption { font-size: 1.15rem; }
    /* Sticky only makes sense beside the video, not stacked under it. */
    .sh-side { position: static; max-height: none; }
    .sh-list { flex: 0 0 auto; max-height: 22rem; }
    .sh-opts { grid-template-columns: minmax(0, 1fr); gap: .3rem; }
    .sh-label { text-align: left; }
    .sh-opt + .sh-opt .sh-label { margin-top: .5rem; }
    .sh-tbtn { width: 2.6rem; }
    .sh-tbtn.is-play { width: 3.4rem; }
    .sh-saved-head input { margin-left: 0; width: 100%; order: 3; }
    .sh-saved-list { grid-template-columns: minmax(0, 1fr); max-height: 18rem; }
    .sh-results { grid-template-columns: minmax(0, 1fr); max-height: 20rem; }
  }
---

유튜브 영상을 문장 2~3개 단위로 끊어 반복 재생합니다. 한 번 불러온 자막은 브라우저에 저장되어, 다음에 같은 영상을 열면 바로 뜹니다.

자막을 넣는 방법은 세 가지입니다 — 유튜브 스크립트를 복사해 붙여넣기, 자막 파일 올리기, 자막 서버로 자동 수집. 가장 간단한 건 첫 번째이고 아무것도 설치할 필요가 없습니다. [아래 사용법](#사용법)을 보세요.

<div class="sh-app" id="sh-app" data-captions="{{ '/assets/captions' | relative_url }}">

  <div class="sh-load">
    <input type="text" id="sh-url" placeholder="유튜브 주소를 붙여넣거나, 검색어를 입력하세요" autocomplete="off" spellcheck="false">
    <button type="button" class="sh-btn is-primary" id="sh-go">불러오기</button>
  </div>

  <div class="sh-sources">
    <select id="sh-lang" title="가져올 자막 언어">
      <option value="en">English</option>
      <option value="ko">한국어</option>
      <option value="ja">日本語</option>
      <option value="es">Español</option>
      <option value="fr">Français</option>
    </select>
    <button type="button" class="sh-btn" id="sh-refetch" hidden>자막 다시 가져오기</button>
    <span class="sh-sep"></span>
    <label class="sh-btn" for="sh-file">SRT / VTT 파일 열기</label>
    <input type="file" id="sh-file" accept=".srt,.vtt,.txt" hidden>
    <button type="button" class="sh-btn" id="sh-paste-toggle">자막 붙여넣기</button>
    <button type="button" class="sh-btn" id="sh-settings-toggle">자막 서버 설정</button>
  </div>

  <div class="sh-settings" id="sh-settings" hidden>
    <p>자막을 자동으로 가져오고 키워드로 검색하려면 자막 서버를 한 번 설치하세요. 로그인할 때 알아서 뜨므로 그 뒤로는 터미널을 열 일이 없습니다.
    비워두면 SRT/VTT 파일을 직접 올리는 방식으로만 동작합니다.</p>
    <p><strong>이 주소는 사이트마다 따로 저장됩니다.</strong> <code>localhost:4000</code>에서 넣었더라도 배포된 사이트에서 한 번 더 넣어야 합니다. Safari에서는 동작하지 않습니다.</p>
    <pre><code>python3 -m pip install -U yt-dlp
bin/caption-server.py --install</code></pre>
    <p>확인은 <code>--status</code>, 제거는 <code>--uninstall</code>.</p>
    <div class="sh-row">
      <input type="text" id="sh-server-url" placeholder="http://127.0.0.1:8787" autocomplete="off" spellcheck="false">
      <button type="button" class="sh-btn is-primary" id="sh-server-save">저장</button>
    </div>
  </div>

  <div class="sh-paste-wrap">
    <textarea id="sh-paste" hidden placeholder="유튜브 &#39;스크립트 표시&#39;에서 복사한 내용을 그대로 붙여넣으세요. SRT/VTT 내용도 됩니다."></textarea>
    <button type="button" class="sh-btn" id="sh-paste-apply" style="margin-top:.4rem">적용</button>
  </div>

  <p class="sh-status" id="sh-status"></p>

  <div class="sh-results" id="sh-results" hidden></div>
  <section class="sh-saved" id="sh-saved" hidden>
    <div class="sh-saved-head">
      <button type="button" class="sh-saved-toggle" id="sh-saved-toggle" aria-expanded="true">
        <span class="sh-caret" aria-hidden="true">▾</span>저장된 영상
        <span class="sh-badge" id="sh-saved-count">0</span>
      </button>
      <input type="text" id="sh-saved-search" placeholder="제목으로 찾기" autocomplete="off">
      <button type="button" class="sh-btn is-quiet" id="sh-saved-clear">모두 지우기</button>
    </div>
    <div class="sh-saved-list" id="sh-saved-list"></div>
  </section>

  <div class="sh-main">

    <div class="sh-left">
      <div class="sh-video"><div id="sh-yt"></div></div>
      <p class="sh-vid-title" id="sh-title"></p>

      <div class="sh-caption" id="sh-caption"></div>

      <div class="sh-controls">
        <div class="sh-transport">
          <button type="button" class="sh-tbtn" id="sh-prev" title="이전 문장 (←)">⏮</button>
          <button type="button" class="sh-tbtn is-play" id="sh-play" title="재생 / 일시정지 (Space)">▶</button>
          <button type="button" class="sh-tbtn" id="sh-replay" title="이 문장 다시 (R)">↻</button>
          <button type="button" class="sh-tbtn" id="sh-next" title="다음 문장 (→)">⏭</button>
          <div class="sh-readout">
            <span class="sh-repeat-now" id="sh-repeat-now"></span>
            <span class="sh-pos" id="sh-pos">—</span>
          </div>
        </div>

        <div class="sh-opts">
          <div class="sh-opt">
            <span class="sh-label">배속</span>
            <div class="sh-opt-body">
              <div class="sh-seg">
                <button type="button" class="sh-rate" data-rate="0.5">0.5×</button>
                <button type="button" class="sh-rate" data-rate="0.75">0.75×</button>
                <button type="button" class="sh-rate" data-rate="1">1.0×</button>
                <button type="button" class="sh-rate" data-rate="1.25">1.25×</button>
              </div>
            </div>
          </div>

          <div class="sh-opt">
            <span class="sh-label">반복</span>
            <div class="sh-opt-body">
              <select id="sh-repeat">
                <option value="1">1회</option>
                <option value="3">3회</option>
                <option value="5">5회</option>
                <option value="0">무한</option>
              </select>
              <label class="sh-check"><input type="checkbox" id="sh-advance"> 끝나면 다음 문장으로</label>
            </div>
          </div>

          <div class="sh-opt">
            <span class="sh-label">구간 길이</span>
            <div class="sh-opt-body">
              <select id="sh-perchunk">
                <option value="1">문장 1개씩</option>
                <option value="2">문장 2개씩</option>
                <option value="3">문장 3개씩</option>
              </select>
            </div>
          </div>

          <div class="sh-opt">
            <span class="sh-label">싱크 보정</span>
            <div class="sh-opt-body">
              <div class="sh-stepper">
                <button type="button" id="sh-offset-minus" title="0.2초 앞당기기">−</button>
                <span id="sh-offset-val">+0.0s</span>
                <button type="button" id="sh-offset-plus" title="0.2초 늦추기">+</button>
              </div>
              <button type="button" class="sh-btn is-quiet" id="sh-offset-reset">초기화</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <aside class="sh-side">
      <div class="sh-side-head">
        <input type="text" id="sh-search" placeholder="단어로 문장 찾기" autocomplete="off">
        <span class="sh-count" id="sh-count"></span>
      </div>
      <div class="sh-list" id="sh-list"></div>
    </aside>

  </div>
</div>

<div class="sh-help">
  <p><strong>단축키</strong> — <kbd>←</kbd> 이전 문장 · <kbd>→</kbd> 다음 문장 · <kbd>Space</kbd> 재생/정지 · <kbd>R</kbd> 이 문장 다시</p>
  <p><strong>자막이 실제 발화보다 빠르거나 늦게 시작하면</strong> 싱크 보정 버튼으로 맞추세요. 보정값은 영상별로 저장됩니다.</p>
  <p><strong>수동 자막이 자동 자막보다 훨씬 낫습니다.</strong> 자동 자막에는 문장부호가 없어서 무음 구간으로 끊을 수밖에 없고, 구간이 문장 중간에서 잘리기 쉽습니다. 서버는 수동 자막이 있으면 항상 그쪽을 먼저 고릅니다.</p>
</div>

<script src="{{ '/assets/js/shadowing.js' | relative_url }}"></script>

## 사용법

자막을 이 페이지에 넣는 방법이 세 가지입니다. 위로 갈수록 준비가 간단합니다.

### 1. 유튜브 스크립트 복사 — 설치 없음

1. 유튜브에서 영상을 열고, 설명란의 **더보기** → **스크립트 표시**
2. 오른쪽에 뜬 스크립트를 전체 드래그해서 복사
3. 이 페이지에 영상 주소를 넣고 → **자막 붙여넣기** → 붙여넣은 뒤 **적용**

맥·윈도우·아이패드 어디서든 되고 설치할 것이 없습니다. 타임스탬프가 줄바꿈으로 붙어 있든 탭으로 붙어 있든 알아서 읽습니다. 영상에 자막 자체가 없으면 **스크립트 표시** 메뉴도 나타나지 않습니다.

### 2. 자막 파일 올리기

`.srt` / `.vtt` 파일이 이미 있으면 **SRT / VTT 파일 열기**로 바로 올리면 됩니다. 직접 뽑으려면:

```bash
yt-dlp --write-auto-sub --sub-lang en --skip-download --convert-subs srt "영상주소"
```

수동 자막이 있는 영상이면 `--write-auto-sub` 대신 `--write-sub`를 쓰세요. 문장부호가 살아 있어서 구간이 훨씬 깔끔하게 나뉩니다.

### 3. 자막 서버 — 주소만 넣으면 자동

영상 주소만 붙여넣으면 자막이 알아서 따라옵니다. 매번 스크립트를 복사할 필요가 없어서, 자주 쓴다면 이쪽이 편합니다. 대신 한 번 설치가 필요합니다.

```bash
python3 -m pip install -U yt-dlp
python3 bin/caption-server.py --install
```

그다음 **자막 서버 설정**에 `http://127.0.0.1:8787`을 넣으면 끝입니다. 확인은 `--status`, 제거는 `--uninstall`.

서버를 켜두면 **입력창이 검색창으로도 동작합니다.** 주소를 붙여넣으면 그 영상을 열고, 그냥 단어를 치면 유튜브를 검색해서 결과를 보여줍니다. 유튜브 앱에 들어가 링크를 복사해 올 필요가 없습니다.

알아둘 것:

- `bin/caption-server.py` **파일 하나만** 있으면 됩니다. 레포 전체를 clone 하지 않아도 돼요.
- `--install`은 launchd를 쓰므로 **macOS 전용**입니다. 윈도우·리눅스에서는 `python3 caption-server.py`로 직접 띄워 두고 쓰시면 똑같이 동작합니다.
- **Safari에서는 동작하지 않습니다.** https 페이지가 `http://localhost`를 호출하는 것을 Safari가 차단합니다. Chrome이나 Firefox를 쓰세요.
- 서버는 내 컴퓨터에서만 돕니다(`127.0.0.1`). 다른 사람 컴퓨터에서는 각자 설치해야 합니다.

### 폰에서 영상 하나만 바로 열기

주소에 `?v=` 를 붙이면 그 영상이 바로 열립니다.

```
https://daye-lee18.github.io/shadowing/?v=8S0FDjFBj8o
```

iOS 단축어나 안드로이드 공유 메뉴에서 유튜브 링크를 받아 이 주소로 넘기게 해두면, 유튜브 앱에서 공유 한 번으로 바로 넘어옵니다.

### 여러 기기에서 같은 영상 보기

브라우저 저장소는 **기기별로, 그리고 브라우저 프로필별로** 완전히 분리됩니다. 폰과 회사 계정 크롬과 개인 계정 크롬은 서로의 저장 내용을 볼 수 없습니다. 그래서 한 번 받은 자막은 레포에 커밋해서 공유합니다.

레포 안에서 자막 서버를 실행하면 받은 자막이 `assets/captions/`에 바로 쌓입니다. 커밋해서 올리기만 하면 됩니다.

```bash
git add assets/captions && git commit -m "captions" && git push
```

배포가 끝나면(약 1분) 폰이든 어느 브라우저든 **저장된 영상** 목록에 `공유` 표시와 함께 나타나고, 로그인이나 설정 없이 바로 열립니다. 다른 사람에게도 똑같이 보입니다.

`--captions-dir`로 위치를 바꿀 수 있고, 레포 밖에서 실행하면 `~/.cache/shadowing-captions/`에 저장됩니다.

### 자막은 어디에 저장되나

| 저장 위치 | 범위 |
|---|---|
| `localStorage` | 그 브라우저 프로필에서만 |
| `assets/captions/` (커밋됨) | 모든 기기·모든 사람 |
| `~/.cache/shadowing-captions/` | 그 컴퓨터에서만 (레포 밖에서 실행했을 때) |

브라우저에 저장된 자막과 개인 설정(배속·싱크 보정)은 어디로도 전송되지 않습니다. 커밋한 자막은 공개 레포에 올라가므로 어떤 영상을 공부하는지 드러납니다.
