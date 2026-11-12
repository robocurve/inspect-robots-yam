"""Read-only CAN motor survey for YAM and OpenYAM arms.

Run it before the first bring-up of an unfamiliar arm, with the arms resting
or supported and nothing else driving them::

    inspect-robots-yam-motors -E arm_model=openyam

For every channel it asks each Damiao motor ID from 0x01 to 0x0F for one
status frame with the read-only query (``0xCC`` on ``0x7FF``). It never sends
enable, disable, zero, clear-error or control frames, so no motor changes
state and nothing moves. It reports which IDs answer, whether each motor is
already enabled (another controller is driving it) or reports a fault, the
positions and temperatures, and where the gripper motor actually is. The exit
code is 1 if the adapter's bring-up would fail or be unsafe.

Damiao firmware that predates the read-only query does not answer it; then
no motor replies and the report says so rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from inspect_robots_yam.config import ARM_MOTOR_CAN_IDS, YamConfig

QUERY_ARBITRATION_ID = 0x7FF
QUERY_COMMAND = 0xCC
REPLY_OFFSET = 0x10
SURVEY_IDS: tuple[int, ...] = tuple(range(0x01, 0x10))

# Damiao status nibble (high 4 bits of byte 0), as i2rt's MotorErrorCode names them.
STATUS_DISABLED = 0x0
STATUS_ENABLED = 0x1
STATUS_NAMES = {
    0x0: "disabled",
    0x1: "enabled",
    0x8: "over-voltage",
    0x9: "under-voltage",
    0xA: "over-current",
    0xB: "MOSFET over-temperature",
    0xC: "motor over-temperature",
    0xD: "lost communication",
    0xE: "overload",
}
# Position, velocity and torque scaling (i2rt MotorType constants).
_DM4340 = (12.5, 10.0, 28.0)
_DM4310 = (12.5, 30.0, 10.0)
_BIG_JOINT_IDS = frozenset({0x01, 0x02, 0x03})


class CanLink(Protocol):
    """The three calls the survey needs from a CAN channel."""

    def send(self, arbitration_id: int, data: bytes) -> None:
        """Send one standard-ID frame."""
        ...

    def recv(self, timeout: float) -> tuple[int, bytes] | None:
        """Return the next frame, or None after ``timeout`` seconds."""
        ...

    def close(self) -> None:
        """Release the channel."""
        ...


CanOpener = Callable[[str], CanLink]


@dataclass(frozen=True)
class MotorReply:
    """One motor's answer to the read-only query."""

    can_id: int
    status: int
    position: float
    temp_mos: float
    temp_rotor: float

    @property
    def status_name(self) -> str:
        """Human name of the status nibble."""
        return STATUS_NAMES.get(self.status, f"unknown status 0x{self.status:X}")


@dataclass(frozen=True)
class Finding:
    """One report line. ``level`` is ``ok``, ``warn`` or ``fail``."""

    channel: str
    level: str
    detail: str


def query_frame(can_id: int) -> bytes:
    """The Damiao read-only status query for ``can_id``."""
    return bytes([can_id & 0xFF, (can_id >> 8) & 0xFF, QUERY_COMMAND, 0, 0, 0, 0, 0])


def _uint_to_float(value: int, limit: float, bits: int) -> float:
    return value * (2 * limit) / ((1 << bits) - 1) - limit


def parse_reply(can_id: int, data: bytes) -> MotorReply:
    """Decode a Damiao feedback frame (the layout i2rt's parser reads)."""
    if len(data) < 8:
        raise ValueError(f"short feedback frame from 0x{can_id:02X}: {data.hex()}")
    p_max = (_DM4340 if can_id in _BIG_JOINT_IDS else _DM4310)[0]
    position = _uint_to_float((data[1] << 8) | data[2], p_max, 16)
    return MotorReply(
        can_id=can_id,
        status=(data[0] & 0xF0) >> 4,
        position=round(position, 4),
        temp_mos=float(data[6]),
        temp_rotor=float(data[7]),
    )


def survey_channel(
    link: CanLink,
    *,
    ids: Sequence[int] = SURVEY_IDS,
    reply_timeout_s: float = 0.02,
    clock: Callable[[], float] = time.monotonic,
) -> dict[int, MotorReply]:
    """Query every ID once and collect the replies that arrive in time.

    A reply is matched on its arbitration ID (``can_id + 0x10``, i2rt's p16
    receive mode); stray traffic is ignored.
    """
    replies: dict[int, MotorReply] = {}
    for can_id in ids:
        link.send(QUERY_ARBITRATION_ID, query_frame(can_id))
        deadline = clock() + reply_timeout_s
        while (remaining := deadline - clock()) > 0:
            frame = link.recv(remaining)
            if frame is None:
                break
            arbitration_id, data = frame
            replier = arbitration_id - REPLY_OFFSET
            if replier in ids and replier not in replies:
                replies[replier] = parse_reply(replier, data)
            if replier == can_id:
                break
    return replies


def assess(cfg: YamConfig, channel: str, replies: dict[int, MotorReply]) -> list[Finding]:
    """Turn one channel's replies into findings, in report order."""
    if not replies:
        return [
            Finding(
                channel,
                "fail",
                "no motor answered the read-only query: check power, the channel name and "
                "1 Mbit/s bitrate, or the firmware predates the query",
            )
        ]
    findings: list[Finding] = []
    gripper = cfg.effective_gripper_can_id
    expected = (*ARM_MOTOR_CAN_IDS, gripper)
    for reply in sorted(replies.values(), key=lambda r: r.can_id):
        role = (
            f"joint {reply.can_id}"
            if reply.can_id in ARM_MOTOR_CAN_IDS
            else "gripper"
            if reply.can_id == gripper
            else "unexpected"
        )
        detail = (
            f"0x{reply.can_id:02X} {role}: {reply.status_name}, "
            f"position {reply.position:+.3f} rad, "
            f"{max(reply.temp_mos, reply.temp_rotor):.0f} C"
        )
        if reply.status == STATUS_ENABLED:
            findings.append(
                Finding(
                    channel,
                    "fail",
                    detail + ": already enabled, so another controller is driving it; stop "
                    "that software (for example Anvil's stack) before running ours",
                )
            )
        elif reply.status != STATUS_DISABLED:
            findings.append(Finding(channel, "fail", detail))
        else:
            findings.append(Finding(channel, "ok" if role != "unexpected" else "warn", detail))
    missing = [can_id for can_id in expected if can_id not in replies]
    for can_id in missing:
        if can_id == gripper:
            others = [c for c in replies if c not in ARM_MOTOR_CAN_IDS]
            hint = (
                f"; a motor answers on 0x{others[0]:02X}, "
                f"so try -E gripper_can_id=0x{others[0]:02X}"
                if len(others) == 1
                else ""
            )
            findings.append(
                Finding(
                    channel,
                    "fail",
                    f"no gripper motor on 0x{gripper:02X} (arm_model={cfg.arm_model}){hint}",
                )
            )
        else:
            findings.append(Finding(channel, "fail", f"joint motor 0x{can_id:02X} did not answer"))
    return findings


def format_report(findings: Sequence[Finding]) -> str:
    """One line per finding, with a verdict last."""
    lines = [f"{f.level.upper():<5} {f.channel:<6} {f.detail}" for f in findings]
    failed = sorted({f.channel for f in findings if f.level == "fail"})
    verdict = (
        "NOT READY on " + ", ".join(failed)
        if failed
        else "READY: every motor answered, disabled and fault-free (nothing was enabled or moved)"
    )
    return "\n".join([*lines, verdict])


def _parse_options(items: Sequence[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"-E expects key=value; got {item!r}")
        options[key] = value
    return options


def main(argv: Sequence[str] | None = None, *, opener: CanOpener | None = None) -> int:
    """Survey each configured channel and print the report; 1 if not ready."""
    parser = argparse.ArgumentParser(
        prog="inspect-robots-yam-motors",
        description=(
            "Read-only survey of the Damiao motors on each YAM/OpenYAM CAN channel. "
            "Sends only the status query: nothing is enabled or moved."
        ),
    )
    parser.add_argument(
        "-E",
        dest="options",
        action="append",
        default=[],
        help="yam_arms option key=value (arm_model, gripper_can_id, left_channel, right_channel)",
    )
    parser.add_argument(
        "--channel",
        action="append",
        default=None,
        help="survey only this CAN channel (repeatable); default left_channel and right_channel",
    )
    parser.add_argument("--json", action="store_true", help="print the findings as JSON")
    args = parser.parse_args(argv)
    options = _parse_options(args.options)
    if "gripper_can_id" in options and options["gripper_can_id"].lower() in ("", "none"):
        del options["gripper_can_id"]
    try:
        cfg = YamConfig.from_kwargs(**options)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    channels = args.channel or [cfg.left_channel, cfg.right_channel]
    open_link = opener or open_socketcan
    findings: list[Finding] = []
    for channel in channels:
        try:
            link = open_link(channel)
        except Exception as exc:
            findings.append(Finding(channel, "fail", f"could not open the channel: {exc!r}"))
            continue
        try:
            replies = survey_channel(link)
        except Exception as exc:
            findings.append(Finding(channel, "fail", f"survey failed: {exc!r}"))
        else:
            findings.extend(assess(cfg, channel, replies))
        finally:
            try:
                link.close()
            except Exception as exc:
                print(f"warning: closing {channel} failed: {exc!r}")
    if args.json:
        payload = [{"channel": f.channel, "level": f.level, "detail": f.detail} for f in findings]
        print(json.dumps(payload, indent=2))
    else:
        print(format_report(findings))
    return 1 if any(f.level == "fail" for f in findings) else 0


def open_socketcan(channel: str) -> CanLink:  # pragma: no cover - needs a CAN interface
    """Open a SocketCAN channel at 1 Mbit/s with python-can (an i2rt dependency)."""
    import can

    bus = can.interface.Bus(channel=channel, interface="socketcan", bitrate=1_000_000)

    class _Link:
        def send(self, arbitration_id: int, data: bytes) -> None:
            bus.send(can.Message(arbitration_id=arbitration_id, data=data, is_extended_id=False))

        def recv(self, timeout: float) -> tuple[int, bytes] | None:
            message = bus.recv(timeout=timeout)
            if message is None:
                return None
            return int(message.arbitration_id), bytes(message.data)

        def close(self) -> None:
            bus.shutdown()

    return _Link()
