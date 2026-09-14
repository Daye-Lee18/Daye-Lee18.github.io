#!/usr/bin/env python3
"""Local caption server for the /shadowing/ page.

YouTube now requires a PO token on timedtext requests, so a Cloudflare Worker
(or any plain HTTP client) gets HTTP 200 with an empty body. yt-dlp knows how
to work around that, but it has to run somewhere with a real IP -- i.e. here.

    bin/caption-server.py --install      # run at login, no terminal needed
    bin/caption-server.py                # or just run it in the foreground
    bin/caption-server.py --status       # is it up?
    bin/caption-server.py --uninstall

Then open /shadowing/, click "자막 서버 설정", and paste http://127.0.0.1:8787.

API
    GET /?v=<videoId>&lang=en[&refresh=1]
    -> {"videoId", "lang", "kind", "title", "cues": [{"start","end","text"}]}
"""

import argparse
import errno
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

CACHE_DIR = os.path.expanduser("~/.cache/shadowing-captions")
VIDEO_ID = re.compile(r"^[\w-]{11}$")

# "android" is the one that currently returns caption data; the rest are kept
# as fallbacks for when that changes.
PLAYER_CLIENTS = ["android", "default", "ios", "web"]

YTDLP = None


def find_ytdlp():
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    for path in sorted(glob.glob(os.path.expanduser("~/Library/Python/*/bin/yt-dlp"))):
        if os.access(path, os.X_OK):
            return [path]
    try:
        subprocess.run([sys.executable, "-m", "yt_dlp", "--version"],
                       capture_output=True, check=True, timeout=30)
        return [sys.executable, "-m", "yt_dlp"]
    except Exception:
        return None


def run_ytdlp(args, timeout=150):
    proc = subprocess.run(YTDLP + args, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def probe(video_id):
    """Return (metadata, client) for the first player client that answers."""
    problems = []
    for client in PLAYER_CLIENTS:
        code, out, err = run_ytdlp([
            "-J", "--skip-download", "--no-warnings",
            "--extractor-args", "youtube:player_client=" + client,
            "https://www.youtube.com/watch?v=" + video_id,
        ])
        if code == 0 and out.strip():
            try:
                return json.loads(out), client
            except json.JSONDecodeError:
                problems.append(client + ": unparseable metadata")
                continue
        line = next((l for l in err.splitlines() if l.startswith("ERROR")), err.strip()[:120])
        problems.append(client + ": " + line)
    raise RuntimeError("yt-dlp could not read the video. " + " | ".join(problems))


def pick_lang(meta, want):
    """Prefer a human-written track: it has the punctuation the splitter needs."""
    manual = meta.get("subtitles") or {}
    auto = meta.get("automatic_captions") or {}
    want = want.lower()
    base = want[:2]
    for kind, table in (("manual", manual), ("auto", auto)):
        keys = [k for k in table if k.lower() == want]
        if not keys:
            keys = [k for k in table if k.lower().startswith(base)]
        if keys:
            return keys[0], kind
    return None, None


def cues_from_json3(doc):
    out = []
    last = None
    for ev in doc.get("events") or []:
        segs = ev.get("segs")
        if not segs:
            continue
        text = " ".join("".join(s.get("utf8", "") for s in segs).split())
        if not text or text == last:
            continue
        start = ev.get("tStartMs", 0) / 1000.0
        end = start + ev.get("dDurationMs", 0) / 1000.0
        if end <= start:
            continue
        last = text
        out.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return out


def download_cues(video_id, lang, kind, client):
    with tempfile.TemporaryDirectory() as tmp:
        flag = "--write-sub" if kind == "manual" else "--write-auto-sub"
        code, _, err = run_ytdlp([
            "--skip-download", flag, "--sub-lang", lang, "--sub-format", "json3",
            "--no-warnings", "--extractor-args", "youtube:player_client=" + client,
            "-o", os.path.join(tmp, "cap"),
            "https://www.youtube.com/watch?v=" + video_id,
        ])
        files = glob.glob(os.path.join(tmp, "*.json3"))
        if not files:
            line = next((l for l in err.splitlines() if l.startswith("ERROR")), err.strip()[:160])
            raise RuntimeError("subtitle download failed: " + (line or "no file produced"))
        with open(files[0], encoding="utf-8") as fh:
            return cues_from_json3(json.load(fh))


def build(video_id, lang):
    meta, client = probe(video_id)
    chosen, kind = pick_lang(meta, lang)
    if not chosen:
        have = sorted(set(list((meta.get("subtitles") or {}).keys()) +
                          list((meta.get("automatic_captions") or {}).keys())))
        raise RuntimeError("no '%s' captions. available: %s"
                           % (lang, ", ".join(have[:20]) or "none"))
    cues = download_cues(video_id, chosen, kind, client)
    if not cues:
        raise RuntimeError("caption track was empty")
    return {
        "videoId": video_id,
        "lang": chosen,
        "kind": kind,
        "title": meta.get("title") or "",
        "via": client,
        "cues": cues,
    }


def cache_path(video_id, lang):
    return os.path.join(CACHE_DIR, "%s.%s.json" % (video_id, lang))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Bound to loopback only, so any local origin may call it -- this is
        # what lets the page work from both localhost:4000 and github.io.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        video_id = (q.get("v") or [""])[0]
        lang = (q.get("lang") or ["en"])[0].lower()
        refresh = (q.get("refresh") or [""])[0] == "1"

        if not VIDEO_ID.match(video_id):
            self._send(400, {"error": "bad or missing ?v=<videoId>"})
            return

        path = cache_path(video_id, lang)
        if not refresh and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    payload = json.load(fh)
                sys.stderr.write("  cache hit  %s (%s)\n" % (video_id, lang))
                self._send(200, payload)
                return
            except Exception:
                pass

        sys.stderr.write("  fetching   %s (%s)\n" % (video_id, lang))
        try:
            payload = build(video_id, lang)
        except subprocess.TimeoutExpired:
            self._send(504, {"error": "yt-dlp timed out"})
            return
        except Exception as exc:
            sys.stderr.write("  FAILED     %s\n" % exc)
            self._send(502, {"error": str(exc)})
            return

        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        sys.stderr.write("  ok         %s cues, %s, via %s\n"
                         % (len(payload["cues"]), payload["kind"], payload["via"]))
        self._send(200, payload)


# ---------------------------------------------------------------- launchd --
# Having to open a terminal before studying defeats the point, so --install
# registers a LaunchAgent: it starts at login, restarts if it dies, and is
# invisible the rest of the time.

LABEL = "com.shadowing.captions"
PLIST = os.path.expanduser("~/Library/LaunchAgents/%s.plist" % LABEL)
LOG = os.path.expanduser("~/Library/Logs/shadowing-captions.log")

# macOS TCC denies launchd jobs access to ~/Documents, ~/Desktop and ~/Downloads,
# so a job pointed at a checkout living there dies with "Operation not permitted".
# Install a copy somewhere unprotected instead -- which also means moving or
# deleting the repo does not break the installed service.
INSTALL_DIR = os.path.expanduser("~/Library/Application Support/shadowing-captions")
INSTALLED = os.path.join(INSTALL_DIR, "caption-server.py")


def launchctl(*args):
    return subprocess.run(["launchctl"] + list(args), capture_output=True, text=True)


def ping(host, port, tries=25):
    """True once the server answers. It replies 400 to a bogus id -- proof of
    life without spending a yt-dlp run."""
    import time
    import urllib.error
    import urllib.request
    url = "http://%s:%d/?v=nope" % (host, port)
    for _ in range(tries):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except urllib.error.HTTPError:
            return True          # answered, just not 200
        except Exception:
            time.sleep(0.2)
    return False


def install(host, port):
    import plistlib
    if sys.platform != "darwin":
        sys.exit(
            "--install registers a launchd agent, which is macOS only.\n"
            "Elsewhere, run the server directly and leave it open:\n"
            "    python3 %s\n"
            "or wrap that command in a systemd user unit (Linux) or Task\n"
            "Scheduler entry (Windows)." % os.path.abspath(__file__)
        )
    ytdlp = find_ytdlp()
    if not ytdlp:
        sys.exit("yt-dlp not found. Install it first:  python3 -m pip install -U yt-dlp")

    # Re-copy every time, so re-running --install picks up edits to the repo copy.
    os.makedirs(INSTALL_DIR, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), INSTALLED)
    os.chmod(INSTALLED, 0o755)

    # launchd hands the job a bare PATH, so point it at the yt-dlp we resolved.
    bindir = os.path.dirname(ytdlp[0] if len(ytdlp) == 1 else sys.executable)
    os.makedirs(os.path.dirname(PLIST), exist_ok=True)
    with open(PLIST, "wb") as fh:
        plistlib.dump({
            "Label": LABEL,
            "ProgramArguments": [sys.executable, INSTALLED,
                                 "--host", host, "--port", str(port)],
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Background",
            "StandardOutPath": LOG,
            "StandardErrorPath": LOG,
            "EnvironmentVariables": {
                "PATH": bindir + ":/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
            },
        }, fh)

    target = "gui/%d" % os.getuid()
    launchctl("bootout", target, PLIST)          # clear any previous copy
    res = launchctl("bootstrap", target, PLIST)
    if res.returncode != 0:
        res = launchctl("load", "-w", PLIST)     # older macOS
        if res.returncode != 0:
            sys.exit("launchctl failed: " + (res.stderr or res.stdout).strip())
    launchctl("kickstart", "-k", "%s/%s" % (target, LABEL))

    url = "http://%s:%d" % (host, port)
    ok = ping(host, port)
    print("installed   %s" % PLIST)
    print("service     %s" % INSTALLED)
    print("yt-dlp      %s" % " ".join(ytdlp))
    print("log         %s" % LOG)
    print("status      %s" % ("running at " + url if ok else "NOT responding -- check the log"))
    if ok:
        print('\nPaste %s into "자막 서버 설정" once, then never think about it again.' % url)
        print("The server now starts on login. Remove it with:  %s --uninstall"
              % os.path.abspath(__file__))
    sys.exit(0 if ok else 1)


def uninstall():
    target = "gui/%d" % os.getuid()
    launchctl("bootout", target, PLIST)
    launchctl("unload", "-w", PLIST)
    removed = False
    for path in (PLIST, INSTALLED):
        if os.path.exists(path):
            os.remove(path)
            print("removed  %s" % path)
            removed = True
    if os.path.isdir(INSTALL_DIR) and not os.listdir(INSTALL_DIR):
        os.rmdir(INSTALL_DIR)
    if not removed:
        print("nothing installed at %s" % PLIST)
    print("the cache in %s was left alone" % CACHE_DIR)
    sys.exit(0)


def status(host, port):
    installed = os.path.exists(PLIST)
    listed = launchctl("list", LABEL).returncode == 0
    alive = ping(host, port, tries=3)
    print("plist       %s" % (PLIST if installed else "not installed"))
    print("service     %s" % (INSTALLED if os.path.exists(INSTALLED) else "-"))
    print("launchd     %s" % ("loaded" if listed else "not loaded"))
    print("responding  %s" % ("yes, http://%s:%d" % (host, port) if alive else "no"))
    if installed and not alive:
        print("\nlog tail (%s):" % LOG)
        if os.path.exists(LOG):
            with open(LOG, errors="replace") as fh:
                for line in fh.readlines()[-12:]:
                    print("  " + line.rstrip())
    sys.exit(0 if alive else 1)


def main():
    global YTDLP
    # Under launchd stdout is a log file, so it would otherwise stay block-
    # buffered and the log would look empty for a long while.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--install", action="store_true",
                    help="run at login via launchd, so no terminal is needed")
    ap.add_argument("--uninstall", action="store_true", help="remove the LaunchAgent")
    ap.add_argument("--status", action="store_true", help="is it installed and answering?")
    args = ap.parse_args()

    if args.uninstall:
        uninstall()
    if args.status:
        status(args.host, args.port)
    if args.install:
        install(args.host, args.port)

    YTDLP = find_ytdlp()
    if not YTDLP:
        sys.exit("yt-dlp not found. Install it with:  python3 -m pip install -U yt-dlp")

    os.makedirs(CACHE_DIR, exist_ok=True)
    url = "http://%s:%d" % (args.host, args.port)

    # Bind before printing anything: the usual reason this fails is that the
    # installed LaunchAgent already holds the port, which is good news, not an
    # error worth a stack trace.
    try:
        srv = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        if ping(args.host, args.port, tries=1):
            print("Already serving on %s -- that will be the installed service." % url)
            print("Nothing to do here. Check it with:  %s --status"
                  % os.path.abspath(__file__))
            sys.exit(0)
        sys.exit("Port %d is taken by something else. Try --port <other>." % args.port)

    print("caption server  %s" % url)
    print("yt-dlp          %s" % " ".join(YTDLP))
    print("cache           %s" % CACHE_DIR)
    print('\nPaste %s into the server box on the /shadowing/ page.' % url)
    print("Tip: %s --install  runs this at login instead.\n" % os.path.abspath(__file__))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
