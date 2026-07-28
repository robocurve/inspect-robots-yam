# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased

### Changed

- Bare `inspect-robots-yam-health` now checks the wizard-configured rig and
  announces contributed configuration with a stderr attribution line;
  `--no-config` restores flag-only behavior.
- `inspect-robots-yam-holdcheck` accepts `left` and `right`, resolving their CAN
  channels through the wizard config.
- Depth-configured camera slots are outside health-check and `--watch` scope and
  are reported as unchecked.
- Arm drivers now initialize concurrently, reducing driver initialization
  wall-clock to approximately the slower arm's time instead of their sum.

### Fixed

- An initialization failure on one arm no longer leaks the other arm's control
  thread, CAN socket, and torque-enabled motors.
