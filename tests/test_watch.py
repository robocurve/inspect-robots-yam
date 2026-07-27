"""Live camera supervision, HTTP handling, and server lifecycle behavior."""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from conftest import FakeCv2
from inspect_robots_yam.config import DEFAULT_CAMERAS, YamConfig
from inspect_robots_yam.watch import (
    RECOVER_INTERVAL_S,
    _CameraSupervisor,
    _WatchRequestHandler,
    serve,
)

CAMERA_DEVICES = {
    "top_cam_device": "/dev/top",
    "left_cam_device": "/dev/left",
    "right_cam_device": "/dev/right",
}
JPEG = b"\xff\xd8fake-jpeg\xff\xd9"


def image(
    pixel: tuple[int, int, int] = (1, 2, 3),
    shape: tuple[int, int, int] = (2, 3, 3),
) -> np.ndarray:
    """Return a recognizable RGB uint8 frame."""
    return np.full(shape, pixel, dtype=np.uint8)


def configured() -> YamConfig:
    """Return a config with every canonical camera device."""
    return YamConfig(**CAMERA_DEVICES)


class FakeSocket:
    """Provide the pinned handler socket shim and capture response bytes."""

    def __init__(
        self,
        request: bytes,
        *,
        fail_on_send: int | None = None,
        disconnect: type[ConnectionError] = BrokenPipeError,
    ) -> None:
        self._request = io.BytesIO(request)
        self.fail_on_send = fail_on_send
        self.disconnect = disconnect
        self.send_calls = 0
        self.sent = bytearray()

    def makefile(self, mode: str, buffering: int = -1) -> io.BytesIO:
        """Return raw request bytes for the handler's read side."""
        assert mode == "rb"
        assert buffering == -1
        return self._request

    def sendall(self, data: bytes) -> None:
        """Accumulate writes until the configured disconnect."""
        self.send_calls += 1
        if self.fail_on_send == self.send_calls:
            raise self.disconnect("client left")
        self.sent.extend(data)


class StubSupervisor:
    """Return scripted frames and expose fixed placeholder error text."""

    def __init__(self, frames: list[np.ndarray | None], error: str = "camera lost") -> None:
        self.frames = frames
        self.error = error
        self.calls = 0

    def frame(self) -> np.ndarray | None:
        """Return the next frame, repeating the final scripted value."""
        value = self.frames[min(self.calls, len(self.frames) - 1)]
        self.calls += 1
        return value

    def last_error(self) -> str:
        """Return the placeholder label."""
        return self.error


class HandlerServer:
    """Mirror the request handler's typed server attributes without binding."""

    def __init__(
        self,
        supervisors: dict[str, Any],
        cv2: FakeCv2,
        *,
        sleep_fn: Callable[[float], None] = lambda _seconds: None,
    ) -> None:
        self.supervisors = supervisors
        self.cv2 = cv2
        self.sleep_fn = sleep_fn
        self.hostname = "rig-host"


def request(
    path: str,
    server: HandlerServer,
    *,
    fail_on_send: int | None = None,
    disconnect: type[ConnectionError] = BrokenPipeError,
) -> tuple[_WatchRequestHandler, FakeSocket]:
    """Run the real handler constructor against the single pinned socket shim."""
    raw = f"GET {path} HTTP/1.1\r\nHost: rig\r\nConnection: close\r\n\r\n".encode()
    sock = FakeSocket(raw, fail_on_send=fail_on_send, disconnect=disconnect)
    handler = _WatchRequestHandler(sock, ("127.0.0.1", 0), server)
    return handler, sock


def test_index_lists_canonical_labeled_streams() -> None:
    """The index contains the rig title and all three ordered stream URLs."""
    handler, sock = request("/", HandlerServer({}, FakeCv2()))

    response = bytes(sock.sent)
    assert b"200 OK" in response
    assert b"<title>rig-host camera watch</title>" in response
    offsets = [response.index(f"/stream/{name}".encode()) for name in DEFAULT_CAMERAS]
    assert offsets == sorted(offsets)
    for name in DEFAULT_CAMERAS:
        assert f"<figcaption>{name}</figcaption>".encode() in response
    assert handler.close_connection


def test_index_client_disconnect_is_swallowed_and_closes_connection() -> None:
    """A client dropping mid-index-write neither raises nor reaches handle_error."""
    handler, sock = request("/", HandlerServer({}, FakeCv2()), fail_on_send=1)

    assert handler.close_connection
    assert sock.sent == bytearray()


def test_stream_emits_multipart_jpeg_and_flips_rgb_to_bgr() -> None:
    """A known stream writes JPEG parts after reversing the frame channels."""
    rgb = image((1, 2, 3))
    cv2 = FakeCv2()
    supervisor = StubSupervisor([rgb])

    handler, sock = request(
        "/stream/top_cam",
        HandlerServer({"top_cam": supervisor}, cv2),
        fail_on_send=3,
    )

    response = bytes(sock.sent)
    assert b"multipart/x-mixed-replace; boundary=frame" in response
    assert b"--frame\r\n" in response
    assert JPEG in response
    assert cv2.encodes[0][0] == ".jpg"
    assert np.array_equal(cv2.encodes[0][1], rgb[..., ::-1])
    assert handler.close_connection


def test_placeholder_uses_default_shape_before_any_good_frame() -> None:
    """A first read failure renders its error on a 480 by 640 black tile."""
    cv2 = FakeCv2()
    supervisor = StubSupervisor([None], "unplugged")

    request(
        "/stream/top_cam",
        HandlerServer({"top_cam": supervisor}, cv2),
        fail_on_send=3,
    )

    assert cv2.put_text_calls[0] == ("unplugged", True)
    placeholder = cv2.encodes[0][1]
    assert placeholder.shape == (480, 640, 3)
    assert np.count_nonzero(placeholder[1:]) == 0


def test_placeholder_reuses_the_last_good_frame_shape() -> None:
    """A later read failure keeps the established stream dimensions."""
    cv2 = FakeCv2()
    supervisor = StubSupervisor([image(shape=(4, 5, 3)), None], "stale")

    request(
        "/stream/top_cam",
        HandlerServer({"top_cam": supervisor}, cv2),
        fail_on_send=4,
    )

    assert cv2.put_text_calls[0] == ("stale", True)
    assert cv2.encodes[1][1].shape == (4, 5, 3)
    assert np.count_nonzero(cv2.encodes[1][1][1:]) == 0


def test_encode_failure_sends_a_placeholder_on_the_next_iteration() -> None:
    """An encoder false result suppresses that part and schedules a placeholder."""
    cv2 = FakeCv2()
    cv2.encode_ok = False

    def recover_encoder(_seconds: float) -> None:
        """Let later iterations encode after the first deliberate failure."""
        cv2.encode_ok = True

    supervisor = StubSupervisor([image()])
    _, sock = request(
        "/stream/top_cam",
        HandlerServer(
            {"top_cam": supervisor},
            cv2,
            sleep_fn=recover_encoder,
        ),
        fail_on_send=3,
    )

    assert cv2.put_text_calls == [("camera lost", True)]
    assert bytes(sock.sent).count(b"--frame\r\n") == 1
    assert np.count_nonzero(cv2.encodes[1][1][1:]) == 0


@pytest.mark.parametrize("disconnect", [BrokenPipeError, ConnectionResetError])
def test_client_disconnect_is_swallowed_and_closes_connection(
    disconnect: type[ConnectionError],
) -> None:
    """Both common browser disconnects end only the affected stream handler."""
    handler, _ = request(
        "/stream/top_cam",
        HandlerServer({"top_cam": StubSupervisor([image()])}, FakeCv2()),
        fail_on_send=2,
        disconnect=disconnect,
    )

    assert handler.close_connection


@pytest.mark.parametrize("path", ["/missing", "/stream/not_a_camera"])
def test_unknown_paths_return_404_without_logging(
    path: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown general and stream paths both return 404 without access logs."""
    _, sock = request(path, HandlerServer({}, FakeCv2()))

    captured = capsys.readouterr()
    assert b"404 Not Found" in bytes(sock.sent)
    assert captured.err == ""


class ScriptedReader:
    """Return or raise scripted camera values and count lifecycle calls."""

    def __init__(self, name: str, script: list[np.ndarray | Exception]) -> None:
        self.name = name
        self.script = script
        self.calls = 0
        self.closed = 0

    def __call__(self, _cfg: YamConfig) -> dict[str, np.ndarray]:
        """Return the next script item under this camera name."""
        value = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return {self.name: value}

    def close(self) -> None:
        """Record release."""
        self.closed += 1


class FakeClock:
    """Expose explicitly advanced monotonic time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        """Return current fake time."""
        return self.now


def test_supervisor_paces_recovery_and_uses_rebuilt_reader_next_call() -> None:
    """Recovery waits two seconds, closes the old reader, and defers its retry."""
    clock = FakeClock()
    old = ScriptedReader("top_cam", [RuntimeError("dead")])
    new = ScriptedReader("top_cam", [image((9, 8, 7))])
    readers = [old, new]
    supervisor = _CameraSupervisor(
        configured(),
        "top_cam",
        "/dev/top",
        reader_factory=lambda _name, _device: readers.pop(0),
        clock_fn=clock,
    )

    assert supervisor.frame() is None
    clock.now = RECOVER_INTERVAL_S - 0.01
    assert supervisor.frame() is None
    assert old.closed == 0
    clock.now = RECOVER_INTERVAL_S
    assert supervisor.frame() is None
    assert old.closed == 1
    assert supervisor.last_error() == "dead"
    assert np.array_equal(supervisor.frame(), image((9, 8, 7)))


def test_supervisor_contains_rebuild_factory_errors() -> None:
    """A factory failure during recovery is recorded instead of escaping."""
    clock = FakeClock()
    old = ScriptedReader("top_cam", [RuntimeError("read failed")])
    calls = 0

    def factory(_name: str, _device: str) -> ScriptedReader:
        """Build the initial reader, then fail the recovery attempt."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("factory failed")
        return old

    supervisor = _CameraSupervisor(
        configured(),
        "top_cam",
        "/dev/top",
        reader_factory=factory,
        clock_fn=clock,
    )
    clock.now = RECOVER_INTERVAL_S

    assert supervisor.frame() is None
    assert supervisor.last_error() == "factory failed"


class BlockingReader:
    """Block the first entry while measuring concurrent reader calls."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.closed = 0

    def __call__(self, _cfg: YamConfig) -> dict[str, np.ndarray]:
        """Record entry, block, then return one frame."""
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls += 1
        self.entered.set()
        self.release.wait(timeout=2)
        with self._state_lock:
            self.active -= 1
        return {"top_cam": image()}

    def close(self) -> None:
        """Record release."""
        self.closed += 1


def test_supervisor_serializes_first_reader_entry() -> None:
    """Two simultaneous first frames never enter the lazy reader concurrently."""
    reader = BlockingReader()
    supervisor = _CameraSupervisor(
        configured(),
        "top_cam",
        "/dev/top",
        reader_factory=lambda _name, _device: reader,
        clock_fn=lambda: 0.0,
    )
    second_started = threading.Event()

    first = threading.Thread(target=supervisor.frame)

    def second_frame() -> None:
        """Signal the second attempt before waiting on the supervisor lock."""
        second_started.set()
        supervisor.frame()

    second = threading.Thread(target=second_frame)
    first.start()
    assert reader.entered.wait(timeout=1)
    second.start()
    assert second_started.wait(timeout=1)
    time.sleep(0.02)
    assert reader.calls == 1
    reader.release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive() and not second.is_alive()
    assert reader.max_active == 1
    assert reader.calls == 2


def test_closed_supervisor_never_touches_or_rebuilds_reader() -> None:
    """The terminal close latch makes every later frame an inert None."""
    reader = ScriptedReader("top_cam", [image()])
    factory_calls = 0

    def factory(_name: str, _device: str) -> ScriptedReader:
        """Count construction so a post-close rebuild would be visible."""
        nonlocal factory_calls
        factory_calls += 1
        return reader

    supervisor = _CameraSupervisor(
        configured(),
        "top_cam",
        "/dev/top",
        reader_factory=factory,
        clock_fn=lambda: 0.0,
    )
    supervisor.close()

    assert supervisor.frame() is None
    assert reader.calls == 0
    assert reader.closed == 1
    assert factory_calls == 1


class ServeReader:
    """Return a fixed frame or raise while recording close."""

    def __init__(self, name: str, *, error: Exception | None = None) -> None:
        self.name = name
        self.error = error
        self.calls = 0
        self.closed = 0

    def __call__(self, _cfg: YamConfig) -> dict[str, np.ndarray]:
        """Return the fixed frame unless configured to fail."""
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {self.name: image()}

    def close(self) -> None:
        """Record release."""
        self.closed += 1


class FakeServer:
    """Record serve and close calls while accepting assigned watch state."""

    def __init__(self, *, serve_error: BaseException | None = None) -> None:
        self.serve_error = serve_error
        self.serve_calls = 0
        self.close_calls = 0
        self.supervisors: dict[str, Any] = {}
        self.cv2: Any = None
        self.sleep_fn: Callable[[float], None] = lambda _seconds: None
        self.hostname = ""

    def serve_forever(self) -> None:
        """Return or raise the configured server outcome."""
        self.serve_calls += 1
        if self.serve_error is not None:
            raise self.serve_error

    def server_close(self) -> None:
        """Record listening-socket release."""
        self.close_calls += 1


def serve_fakes(
    *,
    serve_error: BaseException | None = None,
    reader_error: Exception | None = None,
    cv2_module: Any | None = None,
    bind_was_explicit: bool = False,
    hostname_fn: Callable[[], str] = lambda: "rig-host",
) -> tuple[int, FakeServer, list[ServeReader], list[tuple[tuple[str, int], type[Any]]]]:
    """Run serve with a captured server and one fake reader per camera."""
    server = FakeServer(serve_error=serve_error)
    readers: list[ServeReader] = []
    constructed: list[tuple[tuple[str, int], type[Any]]] = []

    def reader_factory(name: str, _device: str) -> ServeReader:
        """Build and retain one reader."""
        reader = ServeReader(name, error=reader_error)
        readers.append(reader)
        return reader

    def server_factory(address: tuple[str, int], handler: type[Any]) -> Any:
        """Capture bind arguments and return the fake server."""
        constructed.append((address, handler))
        return server

    code = serve(
        configured(),
        port=8807,
        bind="127.0.0.1",
        bind_was_explicit=bind_was_explicit,
        reader_factory=reader_factory,
        cv2_module=cv2_module if cv2_module is not None else FakeCv2(),
        server_factory=server_factory,
        hostname_fn=hostname_fn,
        clock_fn=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )
    return code, server, readers, constructed


def test_serve_closes_server_and_supervisors_when_serving_raises() -> None:
    """Unexpected server failures still release the socket and every camera."""
    server = FakeServer(serve_error=RuntimeError("server failed"))
    readers: list[ServeReader] = []

    def reader_factory(name: str, _device: str) -> ServeReader:
        """Build and retain one reader for cleanup assertions."""
        reader = ServeReader(name)
        readers.append(reader)
        return reader

    def server_factory(_address: tuple[str, int], _handler: type[Any]) -> Any:
        """Return the server that raises from its serving loop."""
        return server

    with pytest.raises(RuntimeError, match="server failed"):
        serve(
            configured(),
            port=8807,
            bind="0.0.0.0",
            bind_was_explicit=False,
            reader_factory=reader_factory,
            cv2_module=FakeCv2(),
            server_factory=server_factory,
            hostname_fn=lambda: "rig",
            clock_fn=lambda: 0.0,
            sleep_fn=lambda _seconds: None,
        )

    assert server.close_calls == 1
    assert all(reader.closed == 1 for reader in readers)


def test_bind_oserror_returns_two_before_constructing_readers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A bind failure reports its reason and never constructs a supervisor."""
    reader_calls = 0

    def fail_server(_address: tuple[str, int], _handler: type[Any]) -> Any:
        """Raise the bind error from server construction."""
        raise OSError("address already in use")

    def reader_factory(_name: str, _device: str) -> ServeReader:
        """Make any camera construction on this path fail the assertion."""
        nonlocal reader_calls
        reader_calls += 1
        return ServeReader("unused")

    code = serve(
        configured(),
        port=8807,
        bind="0.0.0.0",
        bind_was_explicit=False,
        reader_factory=reader_factory,
        server_factory=fail_server,
    )

    assert code == 2
    assert reader_calls == 0
    assert "address already in use" in capsys.readouterr().err


def test_keyboard_interrupt_returns_zero_and_closes_everything() -> None:
    """Ctrl-C is a clean shutdown that closes the server and all readers."""
    code, server, readers, constructed = serve_fakes(
        serve_error=KeyboardInterrupt(),
        cv2_module=FakeCv2(),
    )

    assert code == 0
    assert constructed[0][0] == ("127.0.0.1", 8807)
    assert constructed[0][1] is _WatchRequestHandler
    assert server.serve_calls == 1
    assert server.close_calls == 1
    assert all(reader.calls == 1 and reader.closed == 1 for reader in readers)


def test_priming_failure_is_nonfatal() -> None:
    """Camera prime errors become supervisor state while serving still starts."""
    code, server, readers, _ = serve_fakes(reader_error=RuntimeError("camera absent"))

    assert code == 0
    assert server.serve_calls == 1
    assert all(reader.calls == 1 and reader.closed == 1 for reader in readers)


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [(False, "serving http://rig-host:8807/"), (True, "serving http://127.0.0.1:8807/")],
)
def test_serving_url_uses_hostname_unless_bind_was_explicit(
    explicit: bool,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The operator URL distinguishes the default all-interface bind from an explicit one."""
    code, server, readers, _ = serve_fakes(bind_was_explicit=explicit)

    assert code == 0
    assert expected in capsys.readouterr().err
    assert server.close_calls == 1
    assert all(reader.closed == 1 for reader in readers)


def test_default_cv2_path_uses_embodiment_import_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting cv2 resolves the existing lazy embodiment seam after binding."""
    cv2 = FakeCv2()
    monkeypatch.setattr("inspect_robots_yam.watch.embodiment._import_cv2", lambda: cv2)
    server = FakeServer()

    def server_factory(_address: tuple[str, int], _handler: type[Any]) -> Any:
        """Return the captured fake server."""
        return server

    code = serve(
        configured(),
        port=8807,
        bind="0.0.0.0",
        bind_was_explicit=False,
        reader_factory=lambda name, _device: ServeReader(name),
        cv2_module=None,
        server_factory=server_factory,
        hostname_fn=lambda: "rig",
        clock_fn=lambda: 0.0,
        sleep_fn=lambda _seconds: None,
    )

    assert code == 0
    assert server.cv2 is cv2
