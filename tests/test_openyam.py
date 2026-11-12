"""OpenYAM support: config, the i2rt gripper CAN ID remap, and the read-only motor survey."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inspect_robots_yam import motors
from inspect_robots_yam._i2rt import gripper_can_id
from inspect_robots_yam.config import YamConfig
from inspect_robots_yam.motors import (
    QUERY_ARBITRATION_ID,
    STATUS_DISABLED,
    STATUS_ENABLED,
    Finding,
    assess,
    format_report,
    main,
    parse_reply,
    query_frame,
    survey_channel,
)

# ---------------------------------------------------------------------- config


def test_arm_model_defaults_and_gripper_ids() -> None:
    assert YamConfig().arm_model == "yam"
    assert YamConfig().effective_gripper_can_id == 0x07
    assert YamConfig(arm_model="openyam").effective_gripper_can_id == 0x08
    assert YamConfig(arm_model="openyam", gripper_can_id=0x07).effective_gripper_can_id == 0x07


def test_config_parses_hex_ids_and_limits_from_flags() -> None:
    cfg = YamConfig.from_kwargs(
        arm_model="openyam", gripper_can_id="0x08", gripper_limits="0.1,-2.5"
    )
    assert cfg.gripper_can_id == 8
    assert cfg.gripper_limits == (0.1, -2.5)
    assert YamConfig.from_kwargs(gripper_can_id=9).gripper_can_id == 9


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"arm_model": "so101"}, "arm_model 'so101'"),
        ({"gripper_can_id": "eight"}, "integer such as 0x08"),
        ({"gripper_can_id": 3}, "no arm joint uses"),
        ({"gripper_can_id": 0x20}, "0x07 to 0x0F"),
        ({"gripper_can_id": True}, "gripper_can_id"),
        ({"gripper_limits": "1.0"}, "two finite"),
        ({"gripper_limits": "1.0,1.0"}, "distinct"),
        ({"gripper_limits": "nan,1.0"}, "finite"),
    ],
)
def test_config_rejects_bad_openyam_settings(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        YamConfig.from_kwargs(**kwargs)


# ---------------------------------------------------------------------- remap


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[list[list[Any]]] = []

    def __call__(self, motor_list: Any, *args: Any, **kwargs: Any) -> str:
        self.calls.append(motor_list)
        return "chain"


ARM_AND_GRIPPER = [[i, "DM4340" if i <= 3 else "DM4310"] for i in range(1, 7)] + [[0x07, "DM4310"]]


def test_gripper_can_id_remaps_only_the_arm_gripper_chain_and_restores() -> None:
    recorder = _Recorder()
    module = SimpleNamespace(DMChainCanInterface=recorder)
    with gripper_can_id(module, 0x08):
        assert module.DMChainCanInterface is not recorder
        assert module.DMChainCanInterface(ARM_AND_GRIPPER, [0.0] * 7, channel="can0") == "chain"
        module.DMChainCanInterface([[0x07, "DM4310"]], [0.0])  # gripper-only chain: untouched
        module.DMChainCanInterface(ARM_AND_GRIPPER[:6], [0.0] * 6)  # no gripper: untouched
    assert module.DMChainCanInterface is recorder
    assert recorder.calls[0][-1] == [0x08, "DM4310"]
    assert ARM_AND_GRIPPER[-1] == [0x07, "DM4310"]  # the caller's list is not mutated
    assert recorder.calls[1] == [[0x07, "DM4310"]]
    assert len(recorder.calls[2]) == 6


def test_gripper_can_id_restores_after_an_error_and_is_a_no_op_for_0x07() -> None:
    recorder = _Recorder()
    module = SimpleNamespace(DMChainCanInterface=recorder)
    with pytest.raises(RuntimeError), gripper_can_id(module, 0x08):
        raise RuntimeError("bring-up failed")
    assert module.DMChainCanInterface is recorder
    with gripper_can_id(module, 0x07):
        assert module.DMChainCanInterface is recorder


# ---------------------------------------------------------------------- survey


def frame(status: int, position_raw: int = 0x8000, temps: tuple[int, int] = (30, 32)) -> bytes:
    return bytes(
        [(status << 4) | 0x1, position_raw >> 8, position_raw & 0xFF, 0x80, 0x08, 0x00, *temps]
    )


class FakeLink:
    """Answers the read-only query for the motors it holds, like a Damiao bus."""

    def __init__(self, motors_: dict[int, int], *, stray: bool = False) -> None:
        self.motors = motors_
        self.sent: list[tuple[int, bytes]] = []
        self.pending: list[tuple[int, bytes]] = []
        self.closed = False
        self.stray = stray
        self.close_error: Exception | None = None

    def send(self, arbitration_id: int, data: bytes) -> None:
        self.sent.append((arbitration_id, data))
        can_id = data[0]
        if self.stray:
            self.pending.append((0x123, b"\x00" * 8))
        if can_id in self.motors:
            self.pending.append((can_id + 0x10, frame(self.motors[can_id])))

    def recv(self, timeout: float) -> tuple[int, bytes] | None:
        return self.pending.pop(0) if self.pending else None

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


OPENYAM_BUS = {**dict.fromkeys(range(1, 7), STATUS_DISABLED), 0x08: STATUS_DISABLED}


def test_query_frame_and_parse_reply() -> None:
    assert query_frame(0x08) == bytes([0x08, 0x00, 0xCC, 0, 0, 0, 0, 0])
    reply = parse_reply(0x02, frame(STATUS_ENABLED, 0xFFFF, (41, 45)))
    assert reply.status == STATUS_ENABLED and reply.status_name == "enabled"
    assert reply.position == pytest.approx(12.5)
    assert (reply.temp_mos, reply.temp_rotor) == (41.0, 45.0)
    assert parse_reply(0x05, frame(0x0, 0)).position == pytest.approx(-12.5)
    assert parse_reply(0x05, frame(0x7)).status_name == "unknown status 0x7"
    with pytest.raises(ValueError, match="short feedback"):
        parse_reply(0x01, b"\x01\x02")


def test_survey_sends_only_the_read_only_query() -> None:
    link = FakeLink(OPENYAM_BUS, stray=True)
    replies = survey_channel(link)
    assert sorted(replies) == [1, 2, 3, 4, 5, 6, 8]
    assert {arb for arb, _ in link.sent} == {QUERY_ARBITRATION_ID}
    assert {data[2] for _, data in link.sent} == {0xCC}
    assert len(link.sent) == 15


def test_survey_stops_waiting_at_the_deadline() -> None:
    ticks = iter([0.0, 0.0, 1.0])
    link = FakeLink({}, stray=True)
    assert survey_channel(link, ids=(1,), reply_timeout_s=0.5, clock=lambda: next(ticks)) == {}


def test_assess_ready_openyam() -> None:
    cfg = YamConfig(arm_model="openyam")
    findings = assess(cfg, "can0", survey_channel(FakeLink(OPENYAM_BUS)))
    assert all(f.level == "ok" for f in findings)
    assert "0x08 gripper: disabled" in findings[-1].detail
    assert format_report(findings).endswith("nothing was enabled or moved)")


def test_assess_flags_enabled_faulted_missing_and_suggests_the_gripper_id() -> None:
    bus = {1: STATUS_ENABLED, 2: 0xC, 3: STATUS_DISABLED, 4: STATUS_DISABLED, 5: STATUS_DISABLED}
    bus[0x07] = STATUS_DISABLED
    findings = assess(YamConfig(arm_model="openyam"), "can1", survey_channel(FakeLink(bus)))
    text = format_report(findings)
    assert "already enabled" in text
    assert "motor over-temperature" in text
    assert "0x07 unexpected" in text
    assert "joint motor 0x06 did not answer" in text
    assert "try -E gripper_can_id=0x07" in text
    assert text.endswith("NOT READY on can1")


def test_assess_without_a_unique_gripper_hint_and_with_no_replies() -> None:
    bus = dict.fromkeys(range(1, 7), STATUS_DISABLED) | {0x09: 0, 0x0A: 0}
    findings = assess(YamConfig(), "can0", survey_channel(FakeLink(bus)))
    assert findings[-1].detail == "no gripper motor on 0x07 (arm_model=yam)"
    assert assess(YamConfig(), "can0", {})[0].level == "fail"


# ---------------------------------------------------------------------- CLI


def test_main_reports_each_channel(capsys: pytest.CaptureFixture[str]) -> None:
    links = {"can0": FakeLink(OPENYAM_BUS), "can1": FakeLink(OPENYAM_BUS)}
    assert main(["-E", "arm_model=openyam"], env={}, opener=links.__getitem__) == 0
    out = capsys.readouterr().out
    assert "can0" in out and "can1" in out and "READY" in out
    assert all(link.closed for link in links.values())


def test_main_json_channel_override_and_failures(capsys: pytest.CaptureFixture[str]) -> None:
    def opener(channel: str) -> FakeLink:
        if channel == "bad":
            raise OSError("No such device")
        if channel == "broken":
            link = FakeLink({})
            link.recv = None  # type: ignore[assignment,method-assign]
            link.close_error = OSError("close failed")
            return link
        return FakeLink(OPENYAM_BUS)

    args = ["--json", "--channel", "bad", "--channel", "broken", "--channel", "ok"]
    args += ["-E", "arm_model=openyam", "-E", "gripper_can_id=none", "--no-config"]
    assert main(args, opener=opener) == 1
    captured = capsys.readouterr()
    rows = json.loads(captured.out)  # the whole of stdout is valid JSON
    assert rows[0] == {
        "channel": "bad",
        "level": "fail",
        "detail": "could not open the channel: OSError('No such device')",
    }
    assert rows[1]["channel"] == "broken" and "survey failed" in rows[1]["detail"]
    assert "warning: closing broken failed" in captured.err
    assert all(row["level"] == "ok" for row in rows[2:])


def test_main_uses_saved_channels_and_arm_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "inspect-robots" / "config.ini"
    path.parent.mkdir(parents=True)
    path.write_text(
        "[defaults]\nembodiment = yam_arms\n[embodiment.args]\n"
        "left_channel = can_l\nright_channel = can_r\narm_model = openyam\n",
        encoding="utf-8",
    )
    env = {"XDG_CONFIG_HOME": str(tmp_path)}
    opened: list[str] = []

    def opener(channel: str) -> FakeLink:
        opened.append(channel)
        return FakeLink(OPENYAM_BUS)

    assert main([], env=env, opener=opener) == 0
    assert opened == ["can_l", "can_r"]
    assert f"settings: from {path}" in capsys.readouterr().err
    opened.clear()
    assert main(["-E", "right_channel=can9"], env=env, opener=opener) == 0
    assert opened == ["can_l", "can9"]
    opened.clear()
    assert main(["--no-config"], env=env, opener=opener) == 1  # YAM gripper 0x07 is absent
    assert opened == ["can0", "can1"]


def test_main_rejects_bad_options() -> None:
    with pytest.raises(SystemExit, match="key=value"):
        main(["-E", "arm_model"], env={}, opener=lambda _: FakeLink({}))
    with pytest.raises(SystemExit):
        main(["-E", "arm_model=nope"], env={}, opener=lambda _: FakeLink({}))


def test_finding_is_plain_data() -> None:
    assert Finding("can0", "ok", "x").level == "ok"
    assert motors.SURVEY_IDS[0] == 1 and motors.SURVEY_IDS[-1] == 0x0F
