"""One-shot rig health checks and CLI behavior."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import FakeCv2
from inspect_robots_yam import packing
from inspect_robots_yam.config import DEFAULT_CAMERAS, YamConfig
from inspect_robots_yam.health import (
    CheckResult,
    HealthReport,
    UncheckedCamera,
    _camera_devices,
    _default_reader_factory,
    _default_write_montage,
    _format_human,
    _parse_scalar,
    main,
    run_health,
)

CAMERA_KWARGS = {
    "top_cam_device": "/dev/top",
    "left_cam_device": "/dev/left",
    "right_cam_device": "/dev/right",
}
CAMERA_ARGS = [
    "--top-cam",
    "/dev/top",
    "--left-cam",
    "/dev/left",
    "--right-cam",
    "/dev/right",
]


def write_config_file(
    path: Path,
    values: Mapping[str, str],
    *,
    owner: str = "yam_arms",
) -> Path:
    """Write one wizard-style embodiment config at an explicit path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = "\n".join(f"{key} = {value}" for key, value in values.items())
    path.write_text(
        f"[defaults]\nembodiment = {owner}\n[embodiment.args]\n{args}\n",
        encoding="utf-8",
    )
    return path


def write_config(
    tmp_path: Path,
    values: Mapping[str, str],
    *,
    owner: str = "yam_arms",
) -> tuple[dict[str, str], Path]:
    """Write one XDG-discoverable config and return its isolated env and path."""
    path = write_config_file(
        tmp_path / "inspect-robots" / "config.ini",
        values,
        owner=owner,
    )
    return {"XDG_CONFIG_HOME": str(tmp_path)}, path


def image(seed: int = 0, shape: tuple[int, int, int] = (3, 4, 3)) -> np.ndarray:
    """Build a non-uniform RGB frame whose bytes differ by seed."""
    values = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    return values + seed


class FakeReader:
    """A per-camera reader returning or raising its scripted values in order."""

    def __init__(self, name: str, script: list[np.ndarray | Exception]) -> None:
        self.name = name
        self.script = script
        self.calls = 0
        self.closed = 0

    def __call__(self, cfg: YamConfig) -> dict[str, np.ndarray]:
        value = self.script[self.calls]
        self.calls += 1
        if isinstance(value, Exception):
            raise value
        return {self.name: value}

    def close(self) -> None:
        self.closed += 1


class ReaderFactory:
    """Build scripted readers and retain each one for lifecycle assertions."""

    def __init__(
        self,
        scripts: Mapping[str, list[np.ndarray | Exception]] | None = None,
        *,
        factory_error: Mapping[str, Exception] | None = None,
    ) -> None:
        self.scripts = {} if scripts is None else dict(scripts)
        self.factory_error = {} if factory_error is None else dict(factory_error)
        self.readers: list[FakeReader] = []
        self.devices: list[tuple[str, str]] = []

    def __call__(self, name: str, device: str) -> FakeReader:
        self.devices.append((name, device))
        if name in self.factory_error:
            raise self.factory_error[name]
        script = self.scripts.get(name, [image(1), image(2)])
        reader = FakeReader(name, script)
        self.readers.append(reader)
        return reader


class FakeDriver:
    """A bimanual driver exposing one canned joint vector and a close marker."""

    def __init__(
        self,
        positions: np.ndarray,
        *,
        read_error: Exception | None = None,
    ) -> None:
        self.positions = positions
        self.read_error = read_error
        self.closed = 0

    def get_joint_pos(self) -> np.ndarray:
        if self.read_error is not None:
            raise self.read_error
        return self.positions.copy()

    def get_joint_eff(self) -> np.ndarray:
        return np.zeros(14)

    def close(self) -> None:
        self.closed += 1


def configured() -> YamConfig:
    """Return a config with all three required camera devices."""
    return YamConfig(**CAMERA_KWARGS)


def good_positions() -> np.ndarray:
    """Return finite in-range values for all packed slots."""
    return np.zeros(packing.TOTAL_DIM)


def run(
    *,
    cfg: YamConfig | None = None,
    readers: ReaderFactory | None = None,
    driver: FakeDriver | None = None,
    out_path: str | None = "health.jpg",
    skip_cameras: bool = False,
    skip_motors: bool = False,
    joint_epsilon: float = 0.02,
    montage: Callable[[str, Mapping[str, np.ndarray], frozenset[str]], None] | None = None,
    sleeps: list[float] | None = None,
) -> HealthReport:
    """Run health against deterministic fakes."""
    readers = ReaderFactory() if readers is None else readers
    driver = FakeDriver(good_positions()) if driver is None else driver
    montage = (lambda _path, _frames, _faulted: None) if montage is None else montage
    sleeps = [] if sleeps is None else sleeps
    return run_health(
        configured() if cfg is None else cfg,
        out_path=out_path,
        settle_s=0.4,
        joint_epsilon=joint_epsilon,
        skip_cameras=skip_cameras,
        skip_motors=skip_motors,
        reader_factory=readers,
        driver_factory=lambda _cfg: driver,
        write_montage=montage,
        sleep_fn=sleeps.append,
    )


def cli_code(report: HealthReport, capsys: pytest.CaptureFixture[str]) -> int:
    """Return the main exit code for a precomputed report without hardware."""
    code = main(["--skip-cameras"], run=lambda _cfg, **_kwargs: report)
    capsys.readouterr()
    return code


def test_all_pass_closes_hardware_sleeps_and_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    readers = ReaderFactory()
    driver = FakeDriver(good_positions())
    sleeps: list[float] = []
    written: list[tuple[str, set[str]]] = []

    report = run(
        readers=readers,
        driver=driver,
        sleeps=sleeps,
        montage=lambda path, _frames, faulted: written.append((path, set(faulted))),
    )

    assert report.ok
    assert [result.name for result in report.cameras] == list(DEFAULT_CAMERAS)
    assert [result.name for result in report.joints] == list(packing.DIM_LABELS)
    assert all(result.detail == "" for result in (*report.cameras, *report.joints))
    assert readers.devices == list(zip(DEFAULT_CAMERAS, CAMERA_KWARGS.values(), strict=True))
    assert all(reader.closed == 1 for reader in readers.readers)
    assert driver.closed == 1
    assert sleeps == [0.4, 0.4, 0.4]
    assert written == [("health.jpg", set())]
    assert report.montage_path == "health.jpg"
    assert cli_code(report, capsys) == 0


@pytest.mark.parametrize(
    ("scripts", "factory_error", "detail"),
    [
        ({}, {"left_cam": RuntimeError("dead device")}, "dead device"),
        (
            {"left_cam": [image(1), np.zeros((3, 4, 3), dtype=np.uint8)]},
            {},
            "uniform frame",
        ),
        ({"left_cam": [image(1), image(1).copy()]}, {}, "frozen stream"),
        (
            {"left_cam": [image(1), RuntimeError("frame read failed: stale")]},
            {},
            "frame read failed: stale",
        ),
    ],
    ids=["dead-device", "uniform", "frozen", "stale"],
)
def test_camera_fault_classes_exit_one(
    scripts: Mapping[str, list[np.ndarray | Exception]],
    factory_error: Mapping[str, Exception],
    detail: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readers = ReaderFactory(scripts, factory_error=factory_error)

    report = run(readers=readers)

    left = next(result for result in report.cameras if result.name == "left_cam")
    assert not left.ok
    assert left.detail == detail
    others = [result for result in report.cameras if result.name != "left_cam"]
    assert len(others) == 2 and all(result.ok for result in others)
    assert cli_code(report, capsys) == 1


def test_every_constructed_reader_closes_when_a_read_raises() -> None:
    readers = ReaderFactory({"left_cam": [RuntimeError("USB disconnected")]})

    report = run(readers=readers)

    assert not report.ok
    assert all(reader.closed == 1 for reader in readers.readers)


@pytest.mark.parametrize(
    ("mutate", "detail"),
    [
        (lambda values, cfg: values.__setitem__(2, np.nan), "non-finite"),
        (lambda values, cfg: values.__setitem__(3, cfg.joint_high[3] + 0.5), "outside"),
        (lambda values, cfg: values.__setitem__(packing.ARM_DOF, np.nan), "non-finite"),
    ],
    ids=["nan-joint", "out-of-range-joint", "nan-gripper"],
)
def test_motor_fault_classes_exit_one(
    mutate: Callable[[np.ndarray, YamConfig], None],
    detail: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = configured()
    positions = good_positions()
    mutate(positions, cfg)

    report = run(cfg=cfg, driver=FakeDriver(positions))

    fault = next(result for result in report.joints if not result.ok)
    assert detail in fault.detail
    assert cli_code(report, capsys) == 1


def test_joint_epsilon_accepts_inside_and_faults_outside(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = configured()
    inside = good_positions()
    inside[0] = cfg.joint_high[0] + 0.019
    inside_report = run(cfg=cfg, driver=FakeDriver(inside), joint_epsilon=0.02)
    assert inside_report.joints[0].ok

    outside = good_positions()
    outside[0] = cfg.joint_high[0] + 0.021
    outside_report = run(cfg=cfg, driver=FakeDriver(outside), joint_epsilon=0.02)
    assert not outside_report.joints[0].ok
    assert cli_code(outside_report, capsys) == 1


def test_driver_factory_exception_is_one_synthetic_fault_and_constructs_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def fail(_cfg: YamConfig) -> FakeDriver:
        nonlocal calls
        calls += 1
        raise RuntimeError("CAN bus unavailable")

    report = run_health(
        configured(),
        out_path=None,
        settle_s=0.4,
        joint_epsilon=0.02,
        skip_cameras=True,
        skip_motors=False,
        driver_factory=fail,
    )

    assert calls == 1
    assert report.joints == (CheckResult("driver", False, "CAN bus unavailable"),)
    assert cli_code(report, capsys) == 1


def test_driver_read_exception_closes_and_becomes_driver_fault() -> None:
    driver = FakeDriver(good_positions(), read_error=RuntimeError("CAN read timed out"))

    report = run(driver=driver, skip_cameras=True)

    assert report.joints == (CheckResult("driver", False, "CAN read timed out"),)
    assert driver.closed == 1


def test_motor_temperatures_are_a_separate_always_ok_row() -> None:
    class TempsDriver(FakeDriver):
        """Expose a canned thermal snapshot in addition to joint positions."""

        def __init__(self, positions: np.ndarray, temperatures: np.ndarray) -> None:
            super().__init__(positions)
            self.temperatures = temperatures

        def get_motor_temps(self) -> np.ndarray:
            return self.temperatures.copy()

    temperatures = np.full(14, 35.0)
    temperatures[13] = 48.5

    report = run(
        driver=TempsDriver(good_positions(), temperatures),
        skip_cameras=True,
    )

    assert report.joints[-1] == CheckResult("temps", True, "max 48.5 C @ right_gripper")
    assert all(result.ok for result in report.joints[:-1])


def test_motor_temperature_failure_is_an_always_ok_unavailable_row() -> None:
    class FailingTempsDriver(FakeDriver):
        """Fail only the optional thermal snapshot."""

        def get_motor_temps(self) -> np.ndarray:
            raise RuntimeError("temperature read timed out")

    report = run(driver=FailingTempsDriver(good_positions()), skip_cameras=True)

    assert [result.name for result in report.joints[:-1]] == list(packing.DIM_LABELS)
    assert all(result.ok for result in report.joints[:-1])
    assert report.joints[-1] == CheckResult(
        "temps", True, "unavailable: temperature read timed out"
    )
    assert report.ok


def test_motor_temperature_sentinels_report_no_data() -> None:
    class SentinelTempsDriver(FakeDriver):
        """Expose only the driver's no-temperature-data sentinel."""

        def get_motor_temps(self) -> np.ndarray:
            return np.full(14, -1.0)

    report = run(driver=SentinelTempsDriver(good_positions()), skip_cameras=True)

    assert [result.name for result in report.joints[:-1]] == list(packing.DIM_LABELS)
    assert report.joints[-1] == CheckResult("temps", True, "no data")
    assert report.ok


def test_motor_temperature_row_is_absent_for_legacy_driver() -> None:
    class LegacyDriver:
        def __init__(self) -> None:
            self.closed = False

        def get_joint_pos(self) -> np.ndarray:
            return good_positions()

        def close(self) -> None:
            self.closed = True

    driver = LegacyDriver()
    report = run_health(
        configured(),
        out_path=None,
        settle_s=0.4,
        joint_epsilon=0.02,
        skip_cameras=True,
        skip_motors=False,
        driver_factory=lambda _cfg: driver,  # type: ignore[arg-type, return-value]
    )

    assert [result.name for result in report.joints] == list(packing.DIM_LABELS)
    assert driver.closed


def test_gripper_native_units_are_not_range_checked() -> None:
    positions = good_positions()
    positions[packing.ARM_DOF] = 100.0
    positions[packing.ARM_WIDTH + packing.ARM_DOF] = -100.0

    report = run(driver=FakeDriver(positions))

    assert report.joints[packing.ARM_DOF].ok
    assert report.joints[packing.ARM_WIDTH + packing.ARM_DOF].ok


def test_skipped_sections_are_reported_and_broken_factories_are_not_called() -> None:
    calls: list[str] = []

    def bad_reader(_name: str, _device: str) -> FakeReader:
        calls.append("camera")
        raise AssertionError("skipped camera factory ran")

    def bad_driver(_cfg: YamConfig) -> FakeDriver:
        calls.append("motor")
        raise AssertionError("skipped driver factory ran")

    cameras_skipped = run_health(
        configured(),
        out_path="unused.jpg",
        settle_s=0.4,
        joint_epsilon=0.02,
        skip_cameras=True,
        skip_motors=False,
        reader_factory=bad_reader,
        driver_factory=lambda _cfg: FakeDriver(good_positions()),
    )
    motors_skipped = run_health(
        configured(),
        out_path="unused.jpg",
        settle_s=0.4,
        joint_epsilon=0.02,
        skip_cameras=False,
        skip_motors=True,
        reader_factory=ReaderFactory(),
        driver_factory=bad_driver,
        write_montage=lambda _path, _frames, _faulted: None,
        sleep_fn=lambda _seconds: None,
    )

    assert cameras_skipped.cameras_skipped and cameras_skipped.cameras == ()
    assert cameras_skipped.ok
    assert motors_skipped.joints_skipped and motors_skipped.joints == ()
    assert motors_skipped.ok
    assert calls == []


def test_no_configured_cameras_are_skipped_by_run_health() -> None:
    report = run(cfg=YamConfig(), skip_motors=False)
    assert report.cameras_skipped
    assert report.cameras == ()
    assert report.montage_path is None


def test_montage_runs_only_after_a_camera_capture() -> None:
    writes: list[str] = []

    def writer(path: str, _frames: Mapping[str, np.ndarray], _faulted: frozenset[str]) -> None:
        writes.append(path)

    run(skip_cameras=True, montage=writer)
    run(
        readers=ReaderFactory(
            factory_error={name: RuntimeError("dead") for name in DEFAULT_CAMERAS}
        ),
        montage=writer,
    )
    run(readers=ReaderFactory({"top_cam": [image(1), RuntimeError("stale")]}), montage=writer)
    run(out_path=None, montage=writer)

    assert writes == ["health.jpg"]


def test_faulted_cameras_are_identified_for_placeholder_tiles() -> None:
    recorded: list[tuple[set[str], set[str]]] = []
    readers = ReaderFactory({"left_cam": [image(1), np.zeros((3, 4, 3), dtype=np.uint8)]})

    run(
        readers=readers,
        montage=lambda _path, frames, faulted: recorded.append((set(frames), set(faulted))),
    )

    assert recorded == [(set(DEFAULT_CAMERAS), {"left_cam"})]


def test_default_montage_labels_normalizes_swaps_and_uses_placeholders() -> None:
    cv2 = FakeCv2()
    top = np.full((2, 3, 3), (1, 2, 3), dtype=np.uint8)
    left_fault = np.full((4, 5, 3), 99, dtype=np.uint8)
    right = np.full((1, 2, 3), (4, 5, 6), dtype=np.uint8)

    _default_write_montage(
        "rig.jpg",
        {"top_cam": top, "left_cam": left_fault, "right_cam": right},
        frozenset({"left_cam"}),
        cv2_module=cv2,
    )

    assert cv2.put_text_calls == [(name, True) for name in DEFAULT_CAMERAS]
    path, written = cv2.writes[0]
    assert path == "rig.jpg"
    assert written.shape == (2, 9, 3)
    assert list(written[0, 0]) == [30, 20, 10]
    assert np.all(written[:, 3:6][1:, :] == 0)
    assert list(written[1, 6]) == [6, 5, 4]


def test_default_montage_uses_the_existing_import_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv2 = FakeCv2()
    monkeypatch.setattr("inspect_robots_yam.health.embodiment._import_cv2", lambda: cv2)
    frames = {name: image(index) for index, name in enumerate(DEFAULT_CAMERAS)}

    _default_write_montage("seam.jpg", frames, frozenset())

    assert cv2.writes[0][0] == "seam.jpg"


def test_default_montage_reports_an_encoder_write_failure() -> None:
    cv2 = FakeCv2()
    cv2.write_ok = False
    frames = {name: image(index) for index, name in enumerate(DEFAULT_CAMERAS)}

    with pytest.raises(RuntimeError, match="failed to write montage"):
        _default_write_montage("bad.jpg", frames, frozenset(), cv2_module=cv2)


def test_default_reader_factory_is_inert() -> None:
    reader = _default_reader_factory("top_cam", "/dev/top")
    reader.close()


def test_montage_write_failure_faults_cameras_but_not_motors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = FakeDriver(good_positions())

    def raising_montage(
        path: str, frames: Mapping[str, np.ndarray], faulted: frozenset[str]
    ) -> None:
        raise RuntimeError("failed to write montage to /no/such/dir.jpg")

    report = run(driver=driver, montage=raising_montage)

    montage = next(result for result in report.cameras if result.name == "montage")
    assert not montage.ok
    assert "failed to write montage" in montage.detail
    assert report.montage_path is None
    assert [result.name for result in report.joints] == list(packing.DIM_LABELS)
    assert driver.closed == 1
    assert cli_code(report, capsys) == 1


def test_extra_camera_device_values_stay_strings() -> None:
    seen: list[YamConfig] = []

    def capture(cfg: YamConfig, **_kwargs: object) -> HealthReport:
        seen.append(cfg)
        return HealthReport(
            cameras=(), cameras_skipped=False, joints=(), joints_skipped=True, montage_path=None
        )

    code = main(
        [
            "--skip-motors",
            "-E",
            "top_cam_device=0",
            "-E",
            "left_cam_device=1",
            "-E",
            "right_cam_device=2",
        ],
        run=capture,
    )

    assert code == 0
    assert (seen[0].top_cam_device, seen[0].left_cam_device, seen[0].right_cam_device) == (
        "0",
        "1",
        "2",
    )


def test_cwd_dotenv_pins_config_for_real_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pinned_path = write_config_file(tmp_path / "pinned" / "config.ini", CAMERA_KWARGS)
    (tmp_path / ".env").write_text(
        f"INSPECT_ROBOTS_CONFIG={pinned_path}\n",
        encoding="utf-8",
    )
    copied_environ = dict(os.environ)
    monkeypatch.setattr(os, "environ", copied_environ)
    seen: list[YamConfig] = []

    def capture(cfg: YamConfig, **_kwargs: object) -> HealthReport:
        seen.append(cfg)
        return HealthReport((), False, (), True, None)

    assert main(["--skip-motors"], run=capture) == 0

    assert _camera_devices(seen[0]) == (
        ("top_cam", "/dev/top"),
        ("left_cam", "/dev/left"),
        ("right_cam", "/dev/right"),
    )
    assert capsys.readouterr().err.startswith(
        f"devices: from {pinned_path} (embodiment yam_arms)\n"
    )


def test_exported_config_pin_precedes_cwd_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dotenv_path = write_config_file(
        tmp_path / "pinned-dotenv" / "config.ini",
        {
            "top_cam_device": "/dev/dotenv-top",
            "left_cam_device": "/dev/dotenv-left",
            "right_cam_device": "/dev/dotenv-right",
        },
    )
    exported_path = write_config_file(
        tmp_path / "pinned-exported" / "config.ini",
        {
            "top_cam_device": "/dev/exported-top",
            "left_cam_device": "/dev/exported-left",
            "right_cam_device": "/dev/exported-right",
        },
    )
    (tmp_path / ".env").write_text(
        f"INSPECT_ROBOTS_CONFIG={dotenv_path}\n",
        encoding="utf-8",
    )
    copied_environ = dict(os.environ)
    copied_environ["INSPECT_ROBOTS_CONFIG"] = str(exported_path)
    monkeypatch.setattr(os, "environ", copied_environ)
    seen: list[YamConfig] = []

    def capture(cfg: YamConfig, **_kwargs: object) -> HealthReport:
        seen.append(cfg)
        return HealthReport((), False, (), True, None)

    assert main(["--skip-motors"], run=capture) == 0

    assert seen[0].top_cam_device == "/dev/exported-top"
    assert capsys.readouterr().err.startswith(
        f"devices: from {exported_path} (embodiment yam_arms)\n"
    )


def test_explicit_env_bypasses_cwd_dotenv_without_mutating_process_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dotenv_path = write_config_file(
        tmp_path / "pinned-dotenv" / "config.ini",
        {
            "top_cam_device": "/dev/dotenv-top",
            "left_cam_device": "/dev/dotenv-left",
            "right_cam_device": "/dev/dotenv-right",
        },
    )
    explicit_path = write_config_file(
        tmp_path / "pinned-explicit" / "config.ini",
        {
            "top_cam_device": "/dev/explicit-top",
            "left_cam_device": "/dev/explicit-left",
            "right_cam_device": "/dev/explicit-right",
        },
    )
    (tmp_path / ".env").write_text(
        f"INSPECT_ROBOTS_CONFIG={dotenv_path}\n",
        encoding="utf-8",
    )
    copied_environ = dict(os.environ)
    monkeypatch.setattr(os, "environ", copied_environ)
    environ_before = copied_environ.copy()
    seen: list[YamConfig] = []

    def capture(cfg: YamConfig, **_kwargs: object) -> HealthReport:
        seen.append(cfg)
        return HealthReport((), False, (), True, None)

    assert (
        main(
            ["--skip-motors"],
            env={"INSPECT_ROBOTS_CONFIG": str(explicit_path)},
            run=capture,
        )
        == 0
    )

    assert seen[0].top_cam_device == "/dev/explicit-top"
    assert copied_environ == environ_before


def test_bare_invocation_uses_rgb_wizard_devices_and_attributes_them(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env, path = write_config(
        tmp_path,
        {
            **CAMERA_KWARGS,
            "left_channel": "can_left",
            "right_channel": "can_right",
        },
    )
    seen: list[tuple[YamConfig, dict[str, object]]] = []

    def capture(cfg: YamConfig, **kwargs: object) -> HealthReport:
        seen.append((cfg, kwargs))
        return HealthReport((), False, (), False, None)

    assert main([], env=env, run=capture) == 0

    cfg, kwargs = seen[0]
    assert _camera_devices(cfg) == (
        ("top_cam", "/dev/top"),
        ("left_cam", "/dev/left"),
        ("right_cam", "/dev/right"),
    )
    assert (cfg.left_channel, cfg.right_channel) == ("can_left", "can_right")
    assert kwargs["skip_cameras"] is False
    assert capsys.readouterr().err.startswith(f"devices: from {path} (embodiment yam_arms)\n")


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (["--top-cam", "/dev/flag-top"], "/dev/flag-top"),
        (["-E", "top_cam_device=/dev/extra-top"], "/dev/extra-top"),
    ],
    ids=["flag", "extra"],
)
def test_explicit_camera_values_override_wizard_config(
    override: list[str],
    expected: str,
    tmp_path: Path,
) -> None:
    env, _ = write_config(tmp_path, CAMERA_KWARGS)
    seen: list[YamConfig] = []

    def capture(cfg: YamConfig, **_kwargs: object) -> HealthReport:
        seen.append(cfg)
        return HealthReport((), False, (), True, None)

    assert main([*override, "--skip-motors"], env=env, run=capture) == 0
    assert seen[0].top_cam_device == expected


def test_explicit_depth_serial_supersedes_both_config_keys_for_its_slot(
    tmp_path: Path,
) -> None:
    env, _ = write_config(tmp_path, CAMERA_KWARGS)
    seen: list[YamConfig] = []

    def capture(cfg: YamConfig, **_kwargs: object) -> HealthReport:
        seen.append(cfg)
        return HealthReport((), False, (), True, None)

    assert (
        main(
            [
                "-E",
                "top_depth_serial=838212071234",
                "--skip-motors",
            ],
            env=env,
            run=capture,
        )
        == 0
    )
    cfg = seen[0]
    assert cfg.top_cam_device is None
    assert cfg.top_depth_serial == "838212071234"
    assert isinstance(cfg.top_depth_serial, str)


def test_mixed_camera_config_checks_only_rgb_slots_and_writes_their_montage() -> None:
    cfg = YamConfig(
        top_cam_device="/dev/top",
        left_depth_serial="left-depth",
        right_cam_device="/dev/right",
    )
    readers = ReaderFactory()
    written: list[tuple[str, set[str], set[str]]] = []

    report = run(
        cfg=cfg,
        readers=readers,
        montage=lambda path, frames, faulted: written.append((path, set(frames), set(faulted))),
    )

    assert report.ok
    assert [result.name for result in report.cameras] == ["top_cam", "right_cam"]
    assert report.unchecked_cameras == (
        UncheckedCamera("left_cam", "depth-configured; not checked by this tool"),
    )
    assert readers.devices == [("top_cam", "/dev/top"), ("right_cam", "/dev/right")]
    assert written == [("health.jpg", {"top_cam", "right_cam"}, set())]


def test_default_montage_accepts_a_mixed_rig_camera_subset() -> None:
    cv2 = FakeCv2()

    _default_write_montage(
        "mixed.jpg",
        {"top_cam": image(1), "right_cam": image(2)},
        frozenset(),
        cv2_module=cv2,
    )

    assert cv2.put_text_calls == [("top_cam", True), ("right_cam", True)]
    assert cv2.writes[0][1].shape == (3, 8, 3)


def test_json_lists_unchecked_depth_cameras(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = HealthReport(
        cameras=(CheckResult("top_cam", True, ""),),
        cameras_skipped=False,
        joints=(),
        joints_skipped=True,
        montage_path=None,
        unchecked_cameras=(
            UncheckedCamera("left_cam", "depth-configured; not checked by this tool"),
        ),
    )

    assert (
        main(
            [*CAMERA_ARGS, "--skip-motors", "--json"],
            run=lambda _cfg, **_kwargs: report,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["unchecked_cameras"] == [
        {
            "name": "left_cam",
            "reason": "depth-configured; not checked by this tool",
        }
    ]


def test_all_depth_config_skips_cameras_but_checks_motors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env, _ = write_config(
        tmp_path,
        {
            "top_depth_serial": "top-depth",
            "left_depth_serial": "left-depth",
            "right_depth_serial": "right-depth",
        },
    )
    reports: list[HealthReport] = []

    def capture(cfg: YamConfig, **kwargs: object) -> HealthReport:
        report = run_health(
            cfg,
            out_path=None,
            settle_s=0.2,
            joint_epsilon=0.02,
            skip_cameras=bool(kwargs["skip_cameras"]),
            skip_motors=bool(kwargs["skip_motors"]),
            reader_factory=lambda _name, _device: pytest.fail(
                "depth slot constructed a V4L2 reader"
            ),
            driver_factory=lambda _cfg: FakeDriver(good_positions()),
        )
        reports.append(report)
        return report

    assert main([], env=env, run=capture) == 0

    report = reports[0]
    assert report.cameras_skipped
    assert report.cameras == ()
    assert len(report.unchecked_cameras) == 3
    assert len(report.joints) == packing.TOTAL_DIM
    captured = capsys.readouterr()
    assert "top_cam: skipped (depth-configured; not checked by this tool)" in captured.err
    assert "camera checks are skipped because configured slots are depth-configured" in captured.err
    assert "cameras               SKIPPED" in captured.out
    assert "configured slots are depth-configured" in captured.out


def test_cross_layer_duplicate_device_attributes_config_before_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env, path = write_config(tmp_path, CAMERA_KWARGS)

    with pytest.raises(SystemExit) as exc_info:
        main(["--top-cam", "/dev/left"], env=env)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    attribution = f"devices: from {path} (embodiment yam_arms)"
    assert attribution in captured.err
    assert captured.err.index(attribution) < captured.err.index("duplicate camera device")


def test_skip_cameras_strips_config_only_camera_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env, _ = write_config(tmp_path, CAMERA_KWARGS)
    report = HealthReport((), True, (), False, None)
    seen: list[YamConfig] = []

    def capture(cfg: YamConfig, **_kwargs: object) -> HealthReport:
        seen.append(cfg)
        return report

    assert main(["--skip-cameras"], env=env, run=capture) == 0
    # The strip is observable: no config camera keys reach the config, and a
    # camera-only contribution earns no attribution line.
    assert seen[0].top_cam_device is None
    assert seen[0].left_cam_device is None
    assert seen[0].right_cam_device is None
    assert "devices: from" not in capsys.readouterr().err


def test_skip_cameras_survives_stale_duplicate_device_config(tmp_path: Path) -> None:
    env, _ = write_config(
        tmp_path,
        {
            "top_cam_device": "/dev/same",
            "left_cam_device": "/dev/same",
            "right_cam_device": "/dev/same",
        },
    )
    report = HealthReport((), True, (), False, None)

    assert main(["--skip-cameras"], env=env, run=lambda _cfg, **_kwargs: report) == 0


def test_skip_cameras_keeps_config_depth_slots_out_of_the_report(tmp_path: Path) -> None:
    env, _ = write_config(
        tmp_path,
        {
            "top_depth_serial": "s-top",
            "left_depth_serial": "s-left",
            "right_depth_serial": "s-right",
        },
    )
    seen: list[YamConfig] = []

    def capture(cfg: YamConfig, **_kwargs: object) -> HealthReport:
        seen.append(cfg)
        return HealthReport((), True, (), False, None)

    assert main(["--skip-cameras"], env=env, run=capture) == 0
    assert seen[0].top_depth_serial is None


def test_skip_motors_strips_config_channel_keys(tmp_path: Path) -> None:
    env, _ = write_config(
        tmp_path,
        {**CAMERA_KWARGS, "left_channel": "can_left", "right_channel": "can_right"},
    )
    seen: list[YamConfig] = []

    def capture(cfg: YamConfig, **_kwargs: object) -> HealthReport:
        seen.append(cfg)
        return HealthReport((), False, (), True, None)

    # Cameras keep the run alive; the channel strip must leave the builtin
    # channel names, not the wizard's.
    assert main(["--skip-motors"], env=env, run=capture) == 0
    assert seen[0].left_channel == "can0"
    assert seen[0].right_channel == "can1"
    assert seen[0].top_cam_device == CAMERA_KWARGS["top_cam_device"]


def test_skip_cameras_rejects_an_explicit_depth_serial() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--skip-cameras", "-E", "top_depth_serial=123"])

    assert exc_info.value.code == 2


def test_watch_uses_mixed_config_and_dispatches_only_rgb_devices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _ = write_config(
        tmp_path,
        {
            "top_cam_device": "/dev/top",
            "left_depth_serial": "left-depth",
            "right_cam_device": "/dev/right",
        },
    )
    seen: list[YamConfig] = []

    def fake_serve(cfg: YamConfig, **_kwargs: object) -> int:
        seen.append(cfg)
        return 0

    monkeypatch.setattr("inspect_robots_yam.watch.serve", fake_serve)

    assert main(["--watch"], env=env) == 0
    assert _camera_devices(seen[0]) == (
        ("top_cam", "/dev/top"),
        ("right_cam", "/dev/right"),
    )


def test_all_depth_watch_error_is_depth_aware(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env, _ = write_config(
        tmp_path,
        {
            "top_depth_serial": "top-depth",
            "left_depth_serial": "left-depth",
            "right_depth_serial": "right-depth",
        },
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--watch"], env=env)

    assert exc_info.value.code == 2
    assert "configured slots are depth-configured" in capsys.readouterr().err


def test_no_config_restores_watch_without_devices_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env, _ = write_config(tmp_path, CAMERA_KWARGS)

    with pytest.raises(SystemExit) as exc_info:
        main(["--watch", "--no-config"], env=env)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--watch requires configured camera devices" in captured.err
    assert "devices: from" not in captured.err


def test_attribution_is_omitted_when_every_config_key_is_superseded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env, _ = write_config(tmp_path, CAMERA_KWARGS)
    report = HealthReport((), False, (), True, None)

    assert main([*CAMERA_ARGS, "--skip-motors"], env=env, run=lambda _c, **_k: report) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["--skip-cameras", "--skip-motors"],
        ["--skip-motors"],
        ["--skip-cameras", "--top-cam", "/dev/top"],
        ["--skip-cameras", "-E", "top_cam_device=/dev/top"],
        ["--settle-s", "0.19"],
        ["--settle-s", "nan"],
        ["--settle-s", "inf"],
        ["--joint-epsilon", "nan"],
        ["--joint-epsilon", "-0.1"],
        ["--top-cam", "/dev/top"],
        ["--top-cam", "/dev/top", "-E", "top_cam_device=/dev/other"],
        ["-E", "unknown_key=value"],
        ["-E", "missing-equals"],
        ["-E", "=empty-key"],
        ["--watch", *CAMERA_ARGS, "--skip-cameras"],
        ["--watch", *CAMERA_ARGS, "--skip-motors"],
        ["--watch", *CAMERA_ARGS, "--json"],
        ["--watch"],
        ["--port", "8808"],
        ["--bind", "127.0.0.1"],
        ["--watch", *CAMERA_ARGS, "--port", "0"],
        ["--port", "65536"],
    ],
    ids=[
        "both-skips",
        "skip-motors-no-cameras",
        "skip-cameras-flag-device",
        "skip-cameras-extra-device",
        "settle-floor",
        "settle-nan",
        "settle-inf",
        "epsilon-nan",
        "epsilon-negative",
        "partial-cameras",
        "flag-extra-conflict",
        "unknown-extra",
        "malformed-extra",
        "empty-extra-key",
        "watch-skip-cameras",
        "watch-skip-motors",
        "watch-json",
        "watch-no-cameras",
        "port-without-watch",
        "bind-without-watch",
        "port-zero",
        "port-too-high",
    ],
)
def test_usage_and_config_errors_exit_two(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv, run=lambda _cfg, **_kwargs: pytest.fail("run should not be called"))
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("true", True),
        ("FALSE", False),
        ("17", 17),
        ("-2.5", -2.5),
        ("camera", "camera"),
    ],
)
def test_parse_scalar(text: str, expected: object) -> None:
    assert _parse_scalar(text) == expected


def test_json_stdout_stays_parseable_when_stderr_warnings_fire(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = HealthReport(
        cameras=(),
        cameras_skipped=True,
        joints=(CheckResult("left_j0", True, ""),),
        joints_skipped=False,
        montage_path=None,
    )

    code = main(["--json"], run=lambda _cfg, **_kwargs: report)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["cameras_skipped"] is True
    assert "no camera devices configured" in captured.err
    assert "drops motor torque" in captured.err


def test_cli_maps_camera_flags_and_extras_to_config_and_run_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[tuple[YamConfig, dict[str, Any]]] = []
    report = HealthReport((), False, (), True, None)

    def capture(cfg: YamConfig, **kwargs: Any) -> HealthReport:
        seen.append((cfg, kwargs))
        return report

    code = main(
        [
            *CAMERA_ARGS,
            "--skip-motors",
            "--out",
            "custom.jpg",
            "--settle-s",
            "0.3",
            "--joint-epsilon",
            "0.04",
            "-E",
            "zero_gravity_mode=false",
            "-E",
            "control_hz=20",
        ],
        run=capture,
    )

    captured = capsys.readouterr()
    cfg, kwargs = seen[0]
    assert code == 0
    assert cfg.top_cam_device == "/dev/top"
    assert cfg.zero_gravity_mode is False
    assert cfg.control_hz == 20
    assert kwargs == {
        "out_path": "custom.jpg",
        "settle_s": 0.3,
        "joint_epsilon": 0.04,
        "skip_cameras": False,
        "skip_motors": True,
    }
    assert captured.err == ""
    assert "SKIPPED" in captured.out


def test_cli_accepts_camera_devices_from_extras(capsys: pytest.CaptureFixture[str]) -> None:
    report = HealthReport((), False, (), True, None)
    argv = [
        "-E",
        "top_cam_device=/dev/top",
        "-E",
        "left_cam_device=/dev/left",
        "-E",
        "right_cam_device=/dev/right",
        "--skip-motors",
        "--json",
    ]

    assert main(argv, run=lambda _cfg, **_kwargs: report) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_watch_dispatch_forwards_explicit_network_options_before_warnings(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Watch returns serve's code and bypasses all one-shot notices and warnings."""
    seen: list[tuple[YamConfig, dict[str, Any]]] = []

    def fake_serve(cfg: YamConfig, **kwargs: Any) -> int:
        """Capture the resolved watch call."""
        seen.append((cfg, kwargs))
        return 7

    monkeypatch.setattr("inspect_robots_yam.watch.serve", fake_serve)

    code = main(["--watch", *CAMERA_ARGS, "--port", "8811", "--bind", "127.0.0.2"])

    captured = capsys.readouterr()
    assert code == 7
    assert seen[0][1] == {
        "port": 8811,
        "bind": "127.0.0.2",
        "bind_was_explicit": True,
    }
    assert captured.err == ""


@pytest.mark.parametrize(
    ("extra_args", "expected_size"),
    [([], (640, 480)), (["-E", "cam_width=1280"], (1280, 480))],
    ids=["native-defaults", "explicit-width"],
)
def test_watch_substitutes_only_unset_resolution_keys(
    extra_args: list[str],
    expected_size: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watch uses native defaults per key while preserving explicit size extras."""
    seen: list[tuple[YamConfig, dict[str, Any]]] = []

    def fake_serve(cfg: YamConfig, **kwargs: Any) -> int:
        """Capture the watch config and resolved network defaults."""
        seen.append((cfg, kwargs))
        return 0

    monkeypatch.setattr("inspect_robots_yam.watch.serve", fake_serve)

    assert main(["--watch", *CAMERA_ARGS, *extra_args]) == 0

    cfg, kwargs = seen[0]
    assert (cfg.cam_width, cfg.cam_height) == expected_size
    assert kwargs == {
        "port": 8807,
        "bind": "0.0.0.0",
        "bind_was_explicit": False,
    }


def test_one_shot_keeps_thumbnail_resolution() -> None:
    """The watch-only native-size substitution never changes one-shot configs."""
    seen: list[YamConfig] = []
    report = HealthReport((), False, (), True, None)

    def capture(cfg: YamConfig, **_kwargs: Any) -> HealthReport:
        """Capture the config received by the unchanged one-shot path."""
        seen.append(cfg)
        return report

    assert main([*CAMERA_ARGS, "--skip-motors"], run=capture) == 0
    assert (seen[0].cam_width, seen[0].cam_height) == (224, 224)


def test_explicit_camera_skip_does_not_print_the_auto_skip_note(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = HealthReport((), True, (CheckResult("left_j0", True, ""),), False, None)

    main(["--skip-cameras"], run=lambda _cfg, **_kwargs: report)

    captured = capsys.readouterr()
    assert "no camera devices configured" not in captured.err
    assert "drops motor torque" in captured.err


def test_human_format_shows_faults_skips_details_and_montage() -> None:
    report = HealthReport(
        cameras=(CheckResult("top_cam", False, "dead"),),
        cameras_skipped=False,
        joints=(),
        joints_skipped=True,
        montage_path="health.jpg",
    )

    text = _format_human(report)

    assert "top_cam" in text
    assert "FAULT" in text
    assert "dead" in text
    assert "motors" in text and "SKIPPED" in text
    assert "montage: health.jpg" in text


def test_human_format_shows_camera_skip_and_healthy_joint() -> None:
    report = HealthReport(
        cameras=(),
        cameras_skipped=True,
        joints=(CheckResult("left_j0", True, ""),),
        joints_skipped=False,
        montage_path=None,
    )

    text = _format_human(report)

    assert "cameras" in text and "SKIPPED" in text
    assert "left_j0" in text and "OK" in text
    assert "montage: (not written)" in text
