# 0036: Anvil OpenYAM arms

Status: implemented on `achu/rc-355-openyam`, not yet run on an OpenYAM.

## Why

Anvil's OpenYAM arms are expected at the Pantograph robot safety hackathon
(14 Nov 2026, RC-355). Neither Anvil nor Dimensional has said how the lent
arms are controlled, so this follows public sources only.

## What the public sources say

- The arm motors match the I2RT YAM exactly: Damiao DM4340 at CAN IDs
  0x01 to 0x03, DM4310 at 0x04 to 0x06, replies at ID + 0x10, the same kp/kd.
  Source: DimOS `dimos/robot/manipulators/openyam/config.py` (Apache-2.0)
  against i2rt `robots/config/yam.yml` at our pinned commit.
- The gripper is a DM4310 at CAN ID 0x08 (reply 0x18). i2rt hardcodes the
  YAM gripper at 0x07 in `get_yam_robot`.
- Anvil's own stack runs ROS 2 on its Devbox and warns that third-party
  control apps may recalibrate motors. DimOS bypasses it and drives the motors
  directly over CAN.
- DimOS notes that disabling the motors removes torque, so the arms drop.

## Decision

Drive OpenYAM through the existing i2rt path instead of a new driver:

- `arm_model=openyam` selects gripper CAN ID 0x08 (`gripper_can_id` overrides
  either model). `_i2rt.gripper_can_id` wraps i2rt's `DMChainCanInterface` for
  the duration of bring-up and moves the arm chain's 0x07 gripper entry to the
  configured ID. Everything else (gravity compensation, limits, gains, the
  clamp, parking, cameras, agent and EEF modes) is shared with YAM.
- i2rt's gripper calibration stays on. It only applies a 0.2 N·m test torque
  and reads where the jaw stops; it writes nothing to motor memory, which is
  what Anvil's warning is about. `gripper_limits=closed,open` skips it when a
  range is known, but the DM4310 turn count resets at power-off, so a fixed
  range only holds until the next power cycle.
- A new read-only CLI, `inspect-robots-yam-motors`, sends only the Damiao
  status query (0xCC to 0x7FF) for IDs 0x01 to 0x0F on each channel and
  reports which motors answer, whether any is already enabled (another
  controller is driving it), faults, positions and temperatures, and where the
  gripper actually is. It suggests `gripper_can_id` when the gripper answers
  somewhere else.

## Unverified until hardware

- Whether the lent arms are reachable over USB-CAN from our laptop or only
  through Anvil's Devbox.
- The gripper ID, direction and travel; whether i2rt's LINEAR_4310 gains and
  calibration suit Anvil's gripper.
- Zero offsets and joint limits (YAM's are assumed).
- Whether the Damiao firmware answers the read-only status query. Older
  firmware does not; then the survey reports no replies.
- EEF mode uses the I2RT YAM model and LINEAR_4310 gripper geometry, so the
  tool point may be off by the difference between the grippers.
