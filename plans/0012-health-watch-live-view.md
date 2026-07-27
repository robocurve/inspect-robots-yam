# 0012: `inspect-robots-yam-health --watch` — live browser view (issue #75)

Revision 1.

## Motivation

Adjusting cameras during rig setup needs a live view of what each camera
sees. The one-shot health check (#72, plans/0011) proves cameras deliver
frames but exits immediately. `--watch` keeps the same camera plumbing open
and serves a browser-viewable live stream until interrupted, with no display
requirement on the host (operators watch from a laptop on the tailnet or a
monitor at the rig).

## CLI

`--watch` is a new flag on the existing `inspect-robots-yam-health`:

```
inspect-robots-yam-health --watch [--port 8807] \
  --top-cam ... --left-cam ... --right-cam ...   # or -E *_cam_device=
```

- `--watch` is cameras-only: it implies no motor connection at all (watch is
  for aiming cameras; connecting torque controllers to idly stream video is
  risk with no benefit). Combining `--watch` with `--skip-cameras` or
  `--skip-motors` is a usage error (exit 2): the skips describe the one-shot
  gate, and `--watch --skip-cameras` would stream nothing.
- `--watch` requires configured camera devices (flags or `-E`); without them
  it is a usage error (exit 2), unlike the one-shot path's auto-skip.
- `--port` (default 8807) binds the HTTP server on all interfaces; only
  meaningful with `--watch`, and a usage error without it.
- `--out`, `--settle-s`, `--joint-epsilon`, `--json` are one-shot options; the
  implementation ignores none of them silently: `--json` with `--watch` is a
  usage error (there is no report to emit), the others keep their defaults
  harmlessly and need no guard.
- On start, print to stderr one line per URL form: `http://<hostname>:<port>/`
  and `http://<ip>:<port>/` when resolvable, so the operator can click
  straight from the terminal. Serving loop runs until SIGINT; a clean Ctrl-C
  exits 0.

## Design

New module `src/inspect_robots_yam/watch.py` (keeps `health.py` one-shot
logic untouched; `health.main` dispatches to `watch.serve` when `--watch` is
set). Stdlib only: `http.server.ThreadingHTTPServer` +
`multipart/x-mixed-replace` MJPEG. No new dependencies; JPEG encoding via
`cv2.imencode` behind the existing `_import_cv2` seam.

### Camera plumbing

Reuses the per-camera readers from plans/0011: one `_OpenCVCameraReader` per
device (their daemon drain threads already hold the newest frame, so
concurrent HTTP pulls are cheap and never block on capture). Readers are
constructed once at server start via the same injectable
`(name, device) -> reader` factory and closed on shutdown in a `finally`.
A reader whose read raises serves a placeholder tile (same black-tile idea as
the montage) with the error text rendered into it, and keeps being polled:
during setup, cables get unplugged and replugged, and the stream must recover
without restarting the server. The reader is NOT reconstructed on error;
`_OpenCVCameraReader`'s drain thread already retries reads, and a latched
"stopped reading" fault is surfaced in the tile text telling the operator to
restart watch.

### HTTP surface

- `GET /` — a minimal static HTML page: three `<img>` tags (one per camera,
  labeled, `DEFAULT_CAMERAS` order) pointing at the stream endpoints, plus
  the rig hostname in `<title>`. Inline CSS only, dark background; no JS.
- `GET /stream/<name>` — `multipart/x-mixed-replace; boundary=frame` MJPEG:
  loop { read latest frame from that camera's reader, BGR-encode via
  `imencode('.jpg')`, write part, sleep `1/fps` } until the client
  disconnects (`BrokenPipeError`/`ConnectionResetError` swallowed per
  connection).
- Unknown paths: 404. Frame rate capped at 10 fps per connection (setup use
  case; keeps three streams cheap on the rig CPU).

### Injection and coverage (100% gate)

`watch.py` follows the same seam discipline:

```python
def serve(
    cfg: YamConfig,
    *,
    port: int,
    reader_factory: ReaderFactory = health._default_reader_factory,
    cv2_module: Any | None = None,          # encoder; default via _import_cv2
    server_factory: ... = ThreadingHTTPServer,  # injectable for tests
    clock/sleep_fn: ...                     # paces the stream loop in tests
) -> int: ...
```

Tests drive the handler class directly against fake readers and a fake cv2
(`imencode` returns canned bytes) plus an in-memory socketless request shim
(stdlib `BaseHTTPRequestHandler` accepts injected rfile/wfile), so: index
HTML contains the three names; stream endpoint writes multipart boundaries
and JPEG payloads; client disconnect mid-stream closes that connection only;
faulted reader yields placeholder frames and the loop continues; unknown
path 404s; readers closed on shutdown. The real `ThreadingHTTPServer.serve_forever`
call and the SIGINT `KeyboardInterrupt` translation live in one thin function
whose body is exercised with an injected server fake; nothing new is
pragma'd except any direct `import cv2` (already covered by `_import_cv2`).

`main` changes in `health.py`: the `--watch`/`--port`/`--json` flag wiring and
usage-error rules above, plus dispatch; all covered by CLI-level tests with
an injected `serve`.

## Docs

- README: extend the health-check section with a "Live view: aim the cameras"
  subsection (URL example, Ctrl-C to stop, note that watch never touches the
  motors). Style rules apply (no em dashes, minimal bold).
- `src/inspect_robots_yam/CLAUDE.md`: add the `watch.py` row.
- No `uv lock` (stdlib only), no new CI jobs, nothing added to `__all__`.

## Out of scope

Motor state in the live view, recording/snapshots from the stream, TLS/auth
(tailnet-internal tooling), camera controls (exposure/focus), and the
one-shot report semantics (unchanged).
