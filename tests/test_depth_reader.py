"""Tests for _RealsenseDepthReader and YAMEmbodiment depth integration.

Uses a fake pyrealsense2 namespace so everything runs without hardware or the
real SDK.
"""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.embodiment import (
    YAMEmbodiment,
    _PublishedDepth,
    _RealsenseDepthReader,
)
from inspect_robots_yam.operator import OperatorIO

_DEPTH_SCALE = 0.001  # 1 count = 1 mm


# ---------------------------------------------------------------------------
# Fake pyrealsense2 helpers
# ---------------------------------------------------------------------------


def _make_intr(width: int = 640, height: int = 480) -> Any:
    return types.SimpleNamespace(
        fx=600.0,
        fy=600.0,
        ppx=320.0,
        ppy=240.0,
        width=width,
        height=height,
        model="Brown Conrady",
        coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )


def _make_frameset(
    depth_data: np.ndarray | None = None,
    width: int = 640,
    height: int = 480,
    *,
    falsy_depth: bool = False,
    falsy_color: bool = False,
) -> Any:
    if depth_data is None:
        depth_data = np.ones((height, width), dtype=np.uint16) * 1000
    intr = _make_intr(width, height)
    vsp = types.SimpleNamespace(intrinsics=intr)
    cp = types.SimpleNamespace(as_video_stream_profile=lambda: vsp)
    df: Any = 0 if falsy_depth else types.SimpleNamespace(get_data=lambda: depth_data.copy())
    cf: Any = 0 if falsy_color else types.SimpleNamespace(profile=cp)
    return types.SimpleNamespace(get_depth_frame=lambda: df, get_color_frame=lambda: cf)


class _FakePipeline:
    def __init__(
        self,
        framesets: list[Any] | None = None,
        *,
        depth_scale: float = _DEPTH_SCALE,
        raise_on_wait: Exception | None = None,
    ) -> None:
        self._fs = list(framesets or [_make_frameset()])
        self._ds = depth_scale
        self._row = raise_on_wait
        self._i = 0
        self.stopped = False
        self.started = False

    def start(self, cfg: Any) -> Any:
        self.started = True
        ds = types.SimpleNamespace(get_depth_scale=lambda: self._ds)
        dev = types.SimpleNamespace(first_depth_sensor=lambda: ds)
        return types.SimpleNamespace(get_device=lambda: dev)

    def wait_for_frames(self, timeout_ms: int = 1000) -> Any:
        if self._row is not None:
            raise self._row
        fs = self._fs[self._i % len(self._fs)]
        self._i += 1
        return fs

    def stop(self) -> None:
        self.stopped = True


class _FakeAlign:
    def process(self, frames: Any) -> Any:
        return frames


def _make_rs(pipelines: dict[str, _FakePipeline] | None = None) -> Any:
    pl: list[_FakePipeline] = list(pipelines.values()) if pipelines else []
    cc = [0]

    def _pf() -> Any:
        i = cc[0]
        cc[0] += 1
        return pl[i] if i < len(pl) else _FakePipeline()

    rs_cfg = types.SimpleNamespace(enable_device=lambda s: None, enable_stream=lambda *a: None)
    return types.SimpleNamespace(
        pipeline=_pf,
        config=lambda: rs_cfg,
        align=lambda _: _FakeAlign(),
        stream=types.SimpleNamespace(color="c", depth="d"),
        format=types.SimpleNamespace(bgr8="bgr8", z16="z16"),
    )


_SER: dict[str, str] = {"top_cam": "S1", "left_cam": "S2", "right_cam": "S3"}


def _rdr(
    serials: dict[str, str] | None = None,
    rs: Any = None,
    clock: Any = None,
    sleep_fn: Any = None,
) -> _RealsenseDepthReader:
    return _RealsenseDepthReader(
        serials or _SER,
        rs_module=rs or _make_rs(),
        clock=clock or (lambda: 0.0),
        sleep_fn=sleep_fn or (lambda _: None),
    )


def _cfg() -> YamConfig:
    return YamConfig(cam_height=4, cam_width=4)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_depth_and_intrinsics_returned() -> None:
    dd = np.full((480, 640), 500, dtype=np.uint16)
    fs = _make_frameset(dd)
    r = _rdr(rs=_make_rs({k: _FakePipeline(framesets=[fs]) for k in _SER}))
    ex = r(_cfg())
    r.close()
    for n in _SER:
        assert f"{n}_depth" in ex and f"{n}_intrinsics" in ex
        assert ex[f"{n}_depth"].dtype == np.float32
        assert np.allclose(ex[f"{n}_depth"], 0.5, atol=1e-5)


def test_intrinsics_matrix() -> None:
    r = _rdr()
    ex = r(_cfg())
    r.close()
    k = ex["top_cam_intrinsics"]
    assert isinstance(k, np.ndarray)
    assert k.shape == (3, 3)
    assert k.dtype == np.float32
    assert k[0, 0] == 600.0  # fx
    assert k[1, 1] == 600.0  # fy
    assert k[0, 2] == 320.0  # cx
    assert k[1, 2] == 240.0  # cy
    assert k[2, 2] == 1.0


def test_transient_timeout_recovers() -> None:
    class _TimeoutThenOkPipeline(_FakePipeline):
        _called = False

        def wait_for_frames(self, timeout_ms: int = 1000) -> Any:
            if not self._called:
                self._called = True
                raise RuntimeError("Frame didn't arrive within 1000ms")
            return _make_frameset()

    pipelines = {k: _TimeoutThenOkPipeline() for k in _SER}
    r = _rdr(rs=_make_rs(pipelines), sleep_fn=lambda _: None)
    ex = r(_cfg())
    r.close()
    assert "top_cam_depth" in ex


def test_depth_copy() -> None:
    r = _rdr()
    e1 = r(_cfg())
    e2 = r(_cfg())
    r.close()
    assert e1["top_cam_depth"] is not e2["top_cam_depth"]


def test_second_call_cached() -> None:
    ps = {k: _FakePipeline() for k in _SER}
    r = _rdr(rs=_make_rs(ps))
    r(_cfg())
    r(_cfg())
    r.close()
    for p in ps.values():
        assert p.started


# ---------------------------------------------------------------------------
# Close behaviour
# ---------------------------------------------------------------------------


def test_close_stops_pipelines() -> None:
    ps = {k: _FakePipeline() for k in _SER}
    r = _rdr(rs=_make_rs(ps))
    r(_cfg())
    r.close()
    for p in ps.values():
        assert p.stopped


def test_close_before_open() -> None:
    _rdr().close()


def test_close_idempotent() -> None:
    ps = {k: _FakePipeline() for k in _SER}
    r = _rdr(rs=_make_rs(ps))
    r(_cfg())
    r.close()
    r.close()


def test_stop_error_swallowed() -> None:
    class _FS(_FakePipeline):
        def stop(self) -> None:
            raise RuntimeError("x")

    r = _rdr(rs=_make_rs({k: _FS() for k in _SER}))
    r(_cfg())
    r.close()


# ---------------------------------------------------------------------------
# Stale frame
# ---------------------------------------------------------------------------


def test_stale_raises() -> None:
    now = [0.0]
    r = _rdr(clock=lambda: now[0], sleep_fn=lambda _: None)
    r(_cfg())
    now[0] = _RealsenseDepthReader.MAX_FRAME_AGE_S + 1.0
    r._stop.set()  # stop drain threads so they cannot refresh
    with pytest.raises(RuntimeError, match="depth frame read failed"):
        r(_cfg())
    r.close()


def test_drain_fault_latched_and_reraised() -> None:
    class _FaultingPipeline(_FakePipeline):
        def wait_for_frames(self, timeout_ms: int = 1000) -> Any:
            raise RuntimeError("sensor hardware failure")

    pipelines = {k: _FaultingPipeline() for k in _SER}
    r = _rdr(rs=_make_rs(pipelines), sleep_fn=lambda _: None)
    # The drain thread starts immediately on open and hits wait_for_frames
    with pytest.raises(RuntimeError, match="stopped reading"):
        r(_cfg())
    r.close()


# ---------------------------------------------------------------------------
# Falsy frames
# ---------------------------------------------------------------------------


def test_falsy_depth_raises() -> None:
    fs = _make_frameset(falsy_depth=True)
    sl: list[float] = []
    r = _rdr(rs=_make_rs({k: _FakePipeline(framesets=[fs]) for k in _SER}), sleep_fn=sl.append)
    with pytest.raises(RuntimeError, match="depth frame read failed"):
        r(_cfg())
    r.close()
    assert len(sl) >= 10


def test_falsy_color_raises() -> None:
    fs = _make_frameset(falsy_color=True)
    sl: list[float] = []
    r = _rdr(rs=_make_rs({k: _FakePipeline(framesets=[fs]) for k in _SER}), sleep_fn=sl.append)
    with pytest.raises(RuntimeError, match="depth frame read failed"):
        r(_cfg())
    r.close()


# ---------------------------------------------------------------------------
# Generation counter: zombie cannot write to new cycle
# ---------------------------------------------------------------------------


def test_zombie_rejected() -> None:
    r = _rdr()
    r(_cfg())
    r.close()
    fake = np.zeros((480, 640), dtype=np.float32)
    pub = _PublishedDepth(depth=fake, intrinsics={}, published_s=9999.0)
    with r._lock:  # type: ignore[attr-defined]
        r._published["top_cam"] = pub  # type: ignore[attr-defined]
    r(_cfg())
    with r._lock:  # type: ignore[attr-defined]
        pa = r._published.get("top_cam")  # type: ignore[attr-defined]
    assert pa is None or pa.published_s != 9999.0
    r.close()


# ---------------------------------------------------------------------------
# All-or-nothing open rollback
# ---------------------------------------------------------------------------


def test_open_failure_rollback() -> None:
    opened: list[_FakePipeline] = []

    class _T(_FakePipeline):
        def start(self, cfg: Any) -> Any:
            opened.append(self)
            return super().start(cfg)

    class _F(_FakePipeline):
        def start(self, cfg: Any) -> Any:
            raise RuntimeError("device busy")

    order: list[Any] = [_T(), _T(), _F()]
    cc = [0]

    def _pf() -> Any:
        i = cc[0]
        cc[0] += 1
        return order[i]

    rc = types.SimpleNamespace(enable_device=lambda s: None, enable_stream=lambda *a: None)
    rs = types.SimpleNamespace(
        pipeline=_pf,
        config=lambda: rc,
        align=lambda _: _FakeAlign(),
        stream=types.SimpleNamespace(color="c", depth="d"),
        format=types.SimpleNamespace(bgr8="b", z16="z"),
    )
    with pytest.raises(RuntimeError, match="device busy"):
        _rdr(rs=rs)(_cfg())
    for p in opened:
        assert p.stopped


# ---------------------------------------------------------------------------
# YamConfig depth-serial validation
# ---------------------------------------------------------------------------


def test_yamconfig_all_serials() -> None:
    cfg = YamConfig(top_depth_serial="S1", left_depth_serial="S2", right_depth_serial="S3")
    assert cfg.top_depth_serial == "S1"


def test_yamconfig_partial_rejected() -> None:
    with pytest.raises(ValueError, match="depth serial numbers must be set all three or none"):
        YamConfig(top_depth_serial="S1")


def test_yamconfig_none_ok() -> None:
    assert YamConfig().top_depth_serial is None


# ---------------------------------------------------------------------------
# YAMEmbodiment integration
# ---------------------------------------------------------------------------


def _cameras(_c: YamConfig) -> dict[str, Any]:
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    return {"top_cam": img, "left_cam": img, "right_cam": img}


def _driver_factory(_c: YamConfig) -> Any:
    class _D:
        def get_joint_pos(self) -> np.ndarray:
            return np.zeros(14)

        def command_joint_pos(self, t: np.ndarray) -> None:
            pass

        def close(self) -> None:
            pass

    return _D()


def _silent() -> OperatorIO:
    return OperatorIO(input_fn=lambda _: "", output_fn=lambda _: None)


def _emb(dr: Any = None) -> YAMEmbodiment:
    return YAMEmbodiment(
        YamConfig(cam_height=4, cam_width=4),
        driver_factory=_driver_factory,
        camera_reader=_cameras,
        depth_reader=dr,
        operator=_silent(),
        poll_end=lambda: False,
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
        status_fn=lambda _: None,
    )


class _Action:
    data = np.zeros(14)


def test_observe_no_depth_reader() -> None:
    from inspect_robots.scene import Scene

    emb = _emb()
    emb.reset(Scene(id="s", instruction="t"))
    result = emb.step(_Action())
    emb.close()
    obs = result.observation
    assert not hasattr(obs, "extra") or obs.extra is None or obs.extra == {}


def test_observe_with_depth_reader() -> None:
    from inspect_robots.scene import Scene

    de: dict[str, Any] = {
        "top_cam_depth": np.zeros((4, 4), dtype=np.float32),
        "top_cam_intrinsics": {"fx": 1.0},
        "left_cam_depth": np.zeros((4, 4), dtype=np.float32),
        "left_cam_intrinsics": {"fx": 1.0},
        "right_cam_depth": np.zeros((4, 4), dtype=np.float32),
        "right_cam_intrinsics": {"fx": 1.0},
    }
    emb = _emb(dr=lambda _: de)
    emb.reset(Scene(id="s", instruction="t"))
    result = emb.step(_Action())
    emb.close()
    obs = result.observation
    assert obs.extra is not None and "top_cam_depth" in obs.extra


def test_close_releases_depth_reader() -> None:
    closed: list[bool] = []

    class _DR:
        def __call__(self, _c: YamConfig) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            closed.append(True)

    _emb(dr=_DR()).close()
    assert closed == [True]


def test_emb_auto_constructs_depth_reader_from_config() -> None:
    cfg = YamConfig(
        cam_height=4,
        cam_width=4,
        top_depth_serial="S1",
        left_depth_serial="S2",
        right_depth_serial="S3",
    )
    emb = YAMEmbodiment(
        cfg,
        driver_factory=_driver_factory,
        camera_reader=_cameras,
        operator=_silent(),
        poll_end=lambda: False,
        sleep_fn=lambda _: None,
        clock=lambda: 0.0,
        status_fn=lambda _: None,
    )
    assert emb._depth_reader is not None  # type: ignore[attr-defined]
    assert "Depth: when depth serial numbers are configured" in emb.info.docs
    emb.close()


def test_close_depth_reader_release_error_swallowed() -> None:
    class _FailingDepthReader:
        def __call__(self, _c: YamConfig) -> dict[str, Any]:
            return {}

        def close(self) -> None:
            raise RuntimeError("release failed")

    emb = _emb(dr=_FailingDepthReader())
    emb.close()  # must not raise
