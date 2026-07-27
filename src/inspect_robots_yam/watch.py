"""Serve configured camera views as a browser-readable MJPEG page.

Each camera has its own supervisor and lock. The lock is deliberately held
across a reader call, whose degraded-camera retries sleep for about 0.5
seconds, and across recovery, whose old-reader drain join takes about 2
seconds before the new reader reopens and warms up. All streams for that
camera stall together during those windows and receive placeholders at the
serialized retry rate. Other cameras have separate locks and remain
unaffected.
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from inspect_robots_yam import embodiment, health
from inspect_robots_yam.config import DEFAULT_CAMERAS, YamConfig
from inspect_robots_yam.health import (
    HealthCameraReader,
    Image,
    ReaderFactory,
    _default_reader_factory,
)

FPS = 10
RECOVER_INTERVAL_S = 2.0


class _CameraSupervisor:
    """Serialize one camera reader and replace it after paced read failures."""

    def __init__(
        self,
        cfg: YamConfig,
        name: str,
        device: str,
        *,
        reader_factory: ReaderFactory,
        clock_fn: Callable[[], float],
    ) -> None:
        self._cfg = cfg
        self._name = name
        self._device = device
        self._reader_factory = reader_factory
        self._clock = clock_fn
        self._lock = threading.Lock()
        self._reader: HealthCameraReader = reader_factory(name, device)
        self._last_rebuild_s = clock_fn()
        self._last_error = ""
        self._closed = False

    def frame(self) -> Image | None:
        """Return one RGB uint8 frame, containing and pacing reader recovery."""
        with self._lock:
            if self._closed:
                return None
            try:
                return np.asarray(self._reader(self._cfg)[self._name], dtype=np.uint8)
            except Exception as exc:
                self._last_error = str(exc)
                now = self._clock()
                if now - self._last_rebuild_s >= RECOVER_INTERVAL_S:
                    self._last_rebuild_s = now
                    try:
                        self._reader.close()
                        self._reader = self._reader_factory(self._name, self._device)
                    except Exception as rebuild_exc:
                        self._last_error = str(rebuild_exc)
                return None

    def last_error(self) -> str:
        """Return the most recent read or rebuild error for placeholder text."""
        with self._lock:
            return self._last_error

    def close(self) -> None:
        """Prevent future reads or rebuilds and release the current reader."""
        with self._lock:
            self._closed = True
            self._reader.close()


class _WatchServer(ThreadingHTTPServer):
    """Carry the typed shared state consumed by request handlers."""

    supervisors: dict[str, _CameraSupervisor]
    cv2: Any
    sleep_fn: Callable[[float], None]
    hostname: str


class _WatchRequestHandler(BaseHTTPRequestHandler):
    """Serve the camera index and long-lived MJPEG stream responses."""

    server: _WatchServer

    def log_message(self, format: str, *args: object) -> None:
        """Suppress the standard per-request stderr access log."""

    def do_GET(self) -> None:
        """Serve a route, swallowing client disconnects on every response path.

        The guard lives here rather than only around the stream loop so that a
        client dropping mid-write on the index or 404 paths cannot escape into
        ``BaseServer.handle_error``'s stderr traceback banner.
        """
        try:
            self._route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True

    def _route(self) -> None:
        """Dispatch the index and known camera streams, rejecting every other path."""
        if self.path == "/":
            self._serve_index()
            return
        supervisor = None
        if self.path.startswith("/stream/"):
            supervisor = self.server.supervisors.get(self.path.removeprefix("/stream/"))
        if supervisor is None:
            self.send_error(404)
            return
        # Last statement on purpose: the stream loop exits only via a client
        # disconnect exception, so a trailing return would be dead code.
        self._serve_stream(supervisor)

    def _serve_index(self) -> None:
        """Write a static, script-free page containing cameras in canonical order."""
        cameras = "".join(
            f"<figure><figcaption>{name}</figcaption>"
            f'<img src="/stream/{name}" alt="{name}"></figure>'
            for name in DEFAULT_CAMERAS
        )
        body = (
            "<!doctype html><html><head>"
            f"<title>{self.server.hostname} camera watch</title>"
            "<style>body{background:#111;color:#eee;font-family:sans-serif}"
            "img{max-width:100%;height:auto}figure{margin:1rem}</style>"
            f"</head><body>{cameras}</body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self, supervisor: _CameraSupervisor) -> None:
        """Write MJPEG parts until this client disconnects (guarded in ``do_GET``)."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        self._write_stream(supervisor)

    def _write_stream(self, supervisor: _CameraSupervisor) -> None:
        """Encode RGB frames or error placeholders at the configured stream rate."""
        last_shape: tuple[int, ...] | None = None
        placeholder_next = False
        while True:
            frame = supervisor.frame()
            if frame is not None:
                last_shape = frame.shape
            if frame is None or placeholder_next:
                shape = last_shape if last_shape is not None else (480, 640, 3)
                frame = np.zeros(shape, dtype=np.uint8)
                self.server.cv2.putText(
                    frame,
                    supervisor.last_error(),
                    (8, 20),
                    self.server.cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            ok, encoded = self.server.cv2.imencode(".jpg", frame[..., ::-1])
            placeholder_next = not ok
            if ok:
                payload = np.asarray(encoded, dtype=np.uint8).tobytes()
                part = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                    + payload
                    + b"\r\n"
                )
                self.wfile.write(part)
            self.server.sleep_fn(1 / FPS)


ServerFactory = Callable[
    [tuple[str, int], type[BaseHTTPRequestHandler]],
    _WatchServer,
]


def serve(
    cfg: YamConfig,
    *,
    port: int,
    bind: str,
    bind_was_explicit: bool,
    reader_factory: ReaderFactory = _default_reader_factory,
    cv2_module: Any | None = None,
    server_factory: ServerFactory = _WatchServer,
    hostname_fn: Callable[[], str] = socket.gethostname,
    clock_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Bind first, then stream supervised cameras until interruption and close all handles."""
    try:
        server = server_factory((bind, port), _WatchRequestHandler)
    except OSError as exc:
        print(f"failed to bind {bind}:{port}: {exc}", file=sys.stderr)
        return 2

    supervisors: dict[str, _CameraSupervisor] = {}
    try:
        for name, device in health._camera_devices(cfg):
            supervisors[name] = _CameraSupervisor(
                cfg,
                name,
                device,
                reader_factory=reader_factory,
                clock_fn=clock_fn,
            )
        for supervisor in supervisors.values():
            supervisor.frame()

        server.supervisors = supervisors
        server.cv2 = cv2_module if cv2_module is not None else embodiment._import_cv2()
        server.sleep_fn = sleep_fn
        server.hostname = hostname_fn()
        host = bind if bind_was_explicit else server.hostname
        print(f"serving http://{host}:{port}/", file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 0
    finally:
        server.server_close()
        for supervisor in supervisors.values():
            supervisor.close()
    return 0
