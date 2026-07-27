# 0012: `inspect-robots-yam-health --watch` — live browser view (issue #75)

Revision 5: round 1 replaced the broken replug-recovery story with supervised
reader reconstruction, fixed the 224x224 default resolution, serialized first
open behind the supervisor lock, pinned the handler test shim, the dispatch
point, the URL line, the BGR flip, `--bind`, and the docstring updates.
Round 2 pinned non-blocking server shutdown, a terminal supervisor closed
state, `--port`/`--bind` None sentinels, per-key resolution substitution in
`main`, the typed server subclass, and the degraded-camera stall behavior.
Round 3 refuted round 2's shutdown-hang premise against CPython source
(stock ThreadingHTTPServer never joins daemon handler threads), added port
range validation, the last_error API, the prime-after-bind ordering, and
closed four coverage/typing gaps. Round 4 scoped resolution substitution to
the watch branch (it would otherwise silently change the one-shot montage),
added the explicit --port/--bind happy-path test, and fixed shim/count
inconsistencies.

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
inspect-robots-yam-health --watch [--port 8807] [--bind 0.0.0.0] \
  --top-cam ... --left-cam ... --right-cam ...   # or -E *_cam_device=
```

- `--watch` is cameras-only: it implies no motor connection at all (watch is
  for aiming cameras; connecting torque controllers to idly stream video is
  risk with no benefit). Combining `--watch` with `--skip-cameras`,
  `--skip-motors`, or `--json` is a usage error (exit 2): the skips and the
  report format describe the one-shot gate, which watch is not.
- `--watch` requires configured camera devices (flags or `-E`); without them
  it is a usage error (exit 2), unlike the one-shot path's auto-skip.
- `--port` and `--bind` set the listen address; both are usage errors
  without `--watch`. Their argparse defaults are `None` sentinels (a real
  default of 8807 would make `--port 8807` without `--watch` undetectable);
  after validation, `None` resolves to 8807 and `0.0.0.0`, written as
  conditional EXPRESSIONS (`port = args.port if args.port is not None else
  8807`), not if/else statement blocks whose explicit arms the branch gate
  would demand separate tests for — and one CLI test passes
  `--watch --port N --bind ADDR` and asserts the monkeypatched `watch.serve`
  received that port, that bind, and `bind_was_explicit=True` (without it, a
  swapped argument or inverted flag would pass the whole suite). The resolved
  default binds all interfaces — any LAN the rig touches, not just the
  tailnet — and the stream is unauthenticated; the README says so and shows
  `--bind <tailscale-ip>` for narrowing. `main` validates the port range
  (1-65535; out-of-range raises `OverflowError` from `socket.bind`, not
  `OSError`, so range checking must happen up front as a usage error). A
  bind failure (`EADDRINUSE`, bad address) is operator input, not a bug:
  `serve()` catches `OSError` from server construction, prints the reason
  to stderr, and returns 2.
- `--out`, `--settle-s`, `--joint-epsilon` keep their one-shot defaults and
  are simply unused by watch; no guard needed.
- On start, print one stderr line: `serving http://<host>:<port>/` where
  `<host>` is the explicit `--bind` address when one was given and otherwise
  the hostname from an injectable `hostname_fn` (default
  `socket.gethostname`) — no fallible IP resolution, no uncoverable branch.
  Serving runs until SIGINT; a clean Ctrl-C exits 0 (shutdown mechanics
  below).

### Dispatch point in `health.main`

The watch branch runs after argument validation and config construction but
BEFORE the zero-checks gate, the cameras-auto-skip note, and the motor torque
warning: `motors_will_run` is irrelevant to watch, and reaching the existing
warning would print a false, alarming torque line for a cameras-only server.
Concretely: validate watch's usage rules in the same block as the existing
exit-2 rules, build `cfg`, then `if args.watch: return watch.serve(...)`
(lazy `from inspect_robots_yam import watch` inside `main` — see imports
below). The one-shot path below the branch is unchanged.

`health.py`'s module docstring enumerates exit-2 causes exhaustively; extend
it with all five new ones (the four `parser.error` causes plus serve's
bind-failure 2). The argparse description ("Check configured cameras and
both YAM arms once.") must also change to cover both modes.

## Design

New module `src/inspect_robots_yam/watch.py`. No new dependencies: HTTP via
stdlib (`http.server.ThreadingHTTPServer` + `multipart/x-mixed-replace`
MJPEG); JPEG encoding via the already-depended-on cv2 (`imencode`) behind
the existing `_import_cv2` seam.

Imports: `health` imports `watch` lazily inside `main()`; because of that,
`watch` importing `health` at module top level is NOT a cycle in either
import order, and `watch` does exactly that to reuse the shared types —
`ReaderFactory`, `HealthCameraReader`, `Image`, and
`_default_reader_factory` all stay defined in `health.py` and are imported
by `watch`, not duplicated. (CLI-level tests exercise the dispatch by
monkeypatching `watch.serve`; `main` gains no serve parameter.)

### Capture resolution

Readers resize frames to `cfg.cam_width x cfg.cam_height`, which defaults to
224x224 — thumbnails, useless for judging focus and framing. The
substitution happens in `health.main` INSIDE the `if args.watch:` branch,
immediately before calling `serve` (so CLI tests observing the cfg passed to
a monkeypatched `watch.serve` cover it) and NOWHERE else: applied
unconditionally it would silently change every one-shot run's capture and
montage resolution, and no existing test inspects `cam_width`/`cam_height`.
A one-shot CLI test asserting `run` still receives 224x224 locks this down.
Per key independently,
any of `cam_width`/`cam_height` not explicitly provided via `-E` is replaced
with the native capture negotiation size, written as one loop over
`(("cam_width", 640), ("cam_height", 480))` so there is a single shared
branch (two independent `if` statements would leave the height-explicit arm
uncovered by the listed tests) via `dataclasses.replace` (sound on the
frozen config). `-E cam_width=1280` alone therefore yields 1280x480.
`serve()` receives the final cfg and does no substitution of its own.

### Camera supervision (replaces naive reader reuse)

`_OpenCVCameraReader` cannot recover from unplug/replug on its own: a raising
read latches a permanent "stopped reading" fault (drain thread exits), and an
unplugged V4L2 device typically returns `ok=False` reads forever on the dead
fd — the drain loop then spins without sleeping and the existing
`VideoCapture` never reopens the new device node. So watch wraps each camera
in a `_CameraSupervisor` owning the current reader behind a `threading.Lock`:

- `frame() -> Image | None`: under the lock, call the current reader; on
  success return the frame. On exception: record the message, and if at
  least `RECOVER_INTERVAL_S` (2.0) has passed since the last rebuild,
  `close()` the old reader (joins its drain threads, bounding the dead-fd
  spin) and construct a fresh one via the factory — a new reader opens a
  new `VideoCapture`, which is what actually picks up a replugged device
  node. The rebuilt reader is NOT retried within the same `frame()` call
  (bounding the lock hold); this call returns `None` and the next one reads
  from the fresh reader.
- `last_error() -> str`: the recorded message, read under the lock — this is
  the API the stream loop renders into placeholder tiles (without it the
  handler has no way to get the text).
- The lock also serializes the reader's lazy first open (`__call__` only
  opens devices when its cap map is empty, and that path is unsynchronized
  in the reader itself), closing the double-open race between two handler
  threads hitting the same camera's first frame.
- Ordering: `serve()` constructs (binds) the server FIRST; the `OSError` →
  exit-2 path therefore runs before any V4L2 device opens. Supervisors are
  constructed and primed (one single-threaded `frame()` each) after the
  bind, inside the `try` whose `finally` closes them — so a good device
  streams its first frame immediately, a bad device path surfaces its error
  tile from the start, and no path leaks open devices. Priming failure is
  NOT fatal — the setup use case includes plugging cameras in while watch
  runs.
- `close()` latches a terminal `_closed` flag under the lock: subsequent
  `frame()` calls return `None` immediately, never touching or rebuilding
  the reader. Without this, the un-joined daemon stream threads would call
  `frame()` after shutdown, and since `_OpenCVCameraReader.close()` empties
  its cap map, the reader's next `__call__` would physically REOPEN the
  V4L2 devices during interpreter teardown.
- `close()` on all supervisors in `serve()`'s `finally`.

Clock and sleep are injected (`clock_fn`, `sleep_fn`) so recovery pacing and
stream pacing are testable without real time.

Known and accepted stall behavior (document in the module docstring, not a
bug): the supervisor lock is held across the reader call, and on a degraded
camera `_latest`'s internal retry sleeps ~0.5 s (the default factory injects
no fake clock into the reader), while a rebuild holds the lock through the
old reader's ~2 s drain-thread join plus the new open's negotiation and
warm-up (multiple seconds). During that window all streams of THAT camera
stall together and placeholder tiles arrive at the serialized `_latest`
cost, not at `FPS`. Healthy cameras are unaffected (per-camera locks); for a
setup tool this is acceptable.

### Server shutdown

Verified against CPython source (3.10-3.14): stock `ThreadingHTTPServer`
already sets `daemon_threads = True`, and `socketserver._Threads.append()`
discards daemon threads, so `server_close()` joins nothing and a plain
`finally: server.server_close()` neither hangs on open browser tabs nor
leaks the listen socket. No `block_on_close`/`daemon_threads` overrides —
they would be dead configuration. The module still defines
`_WatchServer(ThreadingHTTPServer)`, purely as the typed home for shared
state (supervisors, cv2 module, pacing functions): handlers reach it via
`self.server` typed as `_WatchServer`, no casts on an untyped stub. The
handler also overrides `log_message` to a no-op: the default writes one
stderr line per request (noise for operators, surprise lines for capsys
tests). `serve()` runs `try: server.serve_forever() / except
KeyboardInterrupt: pass / finally: server.server_close(); close all
supervisors` and returns 0 on the SIGINT path. Daemon stream threads then
die with the process; their post-close `frame()` calls hit the supervisors'
terminal closed state.

### HTTP surface

- `GET /` — minimal static HTML: three `<img>` tags (one per camera, labeled,
  `DEFAULT_CAMERAS` order) pointing at the stream endpoints, rig hostname in
  `<title>`. Inline CSS only, dark background; no JS.
- `GET /stream/<name>` — `multipart/x-mixed-replace; boundary=frame` MJPEG:
  loop { `supervisor.frame()`; on `None` synthesize the placeholder tile
  (black, error text via the injected cv2's `putText`); flip RGB to BGR with
  `frame[..., ::-1]` (readers return RGB; `imencode` expects BGR — an
  unflipped stream is red/blue-swapped video, the one failure a camera-aiming
  tool half-hides); `imencode('.jpg')`; write part; `sleep_fn(1 / FPS)` }
  until the client disconnects (`BrokenPipeError`/`ConnectionResetError`
  swallowed per connection). `FPS = 10`, module constant.
- Unknown paths: 404.

### Injection and coverage (100% line+branch gate)

```python
def serve(
    cfg: YamConfig,
    *,
    port: int,
    bind: str,
    bind_was_explicit: bool,
    reader_factory: ReaderFactory = _default_reader_factory,
    cv2_module: Any | None = None,           # default via _import_cv2 at first use
    server_factory: ... = _WatchServer,
    hostname_fn: Callable[[], str] = socket.gethostname,
    clock_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int: ...
```

Handler tests do not go through a real socket, and the plan pins ONE shim
(no alternatives): `BaseHTTPRequestHandler(request, client_address, server)`
where `request` is a fake socket exposing `makefile("rb", ...)` (BytesIO
primed with the raw request bytes) and `sendall` — with `wbufsize = 0`, the
`StreamRequestHandler` class default, `setup()` never calls
`makefile("wb")`; writes go through `socketserver._SocketWriter`, which
needs only `sendall` — plus a `("127.0.0.1", 0)` client address and a real
`_WatchServer`-shaped test double for `server` (same attributes, no socket
bind). Running the full constructor flow means `send_response`'s
`command`/`requestline`/`log_request` needs are met for free. The full
handler logic (index HTML, stream loop, 404, disconnect swallowing) runs
against fakes.

Tests cover: index page lists the three names; stream endpoint emits
multipart boundaries and the canned JPEG bytes; BGR flip applied (assert the
fake `imencode` saw reversed channels); placeholder tile on supervisor
`None`; client disconnect closes only that connection; 404; supervisor
recovery (raising reader replaced after `RECOVER_INTERVAL_S`, not before;
rebuilt reader's frames flow; old reader's `close()` called); first-call
serialization asserted as NON-CONCURRENT ENTRY into a blocking fake reader
from two threads (factory call counts cannot show this: the factory runs at
supervisor construction, not first `frame()`); closed supervisor returns
`None` without touching the reader; priming failure non-fatal; `serve()`
closes supervisors and calls `server_close()` in `finally`; `OSError` from
the server factory returns 2 with a stderr message (and no supervisor is
ever constructed on that path); SIGINT from the injected server's
`serve_forever` raising `KeyboardInterrupt` returns 0; stderr URL line uses
`hostname_fn`, or the bind address when explicit; the `cv2_module=None`
default path via a monkeypatched `_import_cv2` (precedent:
`test_default_montage_uses_the_existing_import_seam`). CLI-level tests
monkeypatching `watch.serve` cover the dispatch point (no torque warning
printed on watch), the new exit-2 causes (watch+skip/json, watch-without-cameras,
`--port`/`--bind` sans watch, port out of range — plus serve's own
bind-failure 2), explicit `--watch --port N --bind ADDR` forwarding
(port, bind, `bind_was_explicit=True`), resolution substitution in the
watch branch (defaults replaced per key; `-E cam_width=1280` alone yields
1280x480), and the one-shot path still passing 224x224 to `run`. The
conftest `FakeCv2` docstring claim ("used by the camera reader and health
montage") widens to cover `imencode`.

Nothing new is pragma'd: the only real-world defaults are
`ThreadingHTTPServer`, `socket.gethostname`, `time.*` (all injectable and
exercised via fakes) and cv2 via the already-pragma'd `_import_cv2`.

## Docs

- README: extend the health-check section with a "Live view: aim the
  cameras" subsection (URL example, Ctrl-C to stop, watch never touches the
  motors, unauthenticated stream + `--bind` note). Style rules apply (no em
  dashes, minimal bold).
- `src/inspect_robots_yam/CLAUDE.md`: add the `watch.py` row.
- `health.py` module docstring (including its "One-shot ..." summary line,
  which stops being the whole truth) + argparse description as above. The
  exit-2 enumeration gains all five new causes, and the bind-failure 2
  (returned by `serve`, not routed through `parser.error`) is called out
  explicitly.
- No `uv lock` (stdlib only), no new CI jobs, nothing added to `__all__`.

## Out of scope

Motor state in the live view, recording/snapshots from the stream, TLS/auth
beyond the `--bind` narrowing, camera controls (exposure/focus), fixing
`_OpenCVCameraReader._drain`'s no-sleep spin upstream (worth its own issue),
and the one-shot report semantics (unchanged).
