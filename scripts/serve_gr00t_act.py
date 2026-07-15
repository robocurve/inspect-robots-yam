"""Serve an Isaac-GR00T YAM fine-tune through the package's ``/act`` protocol."""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

EMBODIMENT_TAG = "new_embodiment"
TOTAL_DIM = 14
CLIENT_CAMERA_NAMES = frozenset({"top_cam", "left_cam", "right_cam"})
DEFAULT_CAMERA_MAP = "top_cam:base_view,left_cam:left_wrist_view,right_cam:right_wrist_view"
CANONICAL_SLICES: Mapping[str, slice] = {
    "left_arm": slice(0, 6),
    "left_gripper": slice(6, 7),
    "right_arm": slice(7, 13),
    "right_gripper": slice(13, 14),
}

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse the model, wire mapping, advertised timing, and bind address."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local model path")
    parser.add_argument(
        "--camera-map",
        default=DEFAULT_CAMERA_MAP,
        help="Comma-separated client:checkpoint camera names",
    )
    parser.add_argument(
        "--dt-ms",
        type=float,
        default=0.0,
        help="Advertised action interval in milliseconds; 0 leaves pacing to the embodiment",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8203, help="Server bind port")
    return parser.parse_args()


def _resolve_model(model: str) -> Path:
    """Return a local model directory, downloading Hugging Face repo ids."""
    local = Path(model).expanduser()
    if local.exists():
        return local.resolve()
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    return Path(snapshot_download(repo_id=model))


def _parse_camera_map(raw: str) -> dict[str, str]:
    """Parse and validate the client side of the camera mapping."""
    try:
        pairs = [entry.split(":", maxsplit=1) for entry in raw.split(",")]
        if any(len(pair) != 2 or not pair[0] or not pair[1] for pair in pairs):
            raise ValueError
    except ValueError:
        raise ValueError(
            "--camera-map must contain comma-separated source:target pairs; "
            f"valid source names are {sorted(CLIENT_CAMERA_NAMES)}"
        ) from None

    sources = {pair[0] for pair in pairs}
    if len(pairs) != len(CLIENT_CAMERA_NAMES) or sources != CLIENT_CAMERA_NAMES:
        raise ValueError(
            "--camera-map source names must equal "
            f"{sorted(CLIENT_CAMERA_NAMES)}, got {sorted(sources)}"
        )
    targets = [pair[1] for pair in pairs]
    if len(set(targets)) != len(targets):
        raise ValueError("--camera-map target names must be distinct")
    return dict(pairs)


def _validate_modality_configs(
    modality_configs: Mapping[str, Any], camera_map: Mapping[str, str]
) -> None:
    """Reject checkpoint modalities incompatible with the stateless 14-D shim."""
    video_config = modality_configs["video"]
    state_config = modality_configs["state"]
    action_config = modality_configs["action"]

    for modality_name, config in (("video", video_config), ("state", state_config)):
        if len(config.delta_indices) != 1:
            raise ValueError(
                f"{modality_name} delta_indices must contain exactly one frame for this "
                "stateless shim (frame-history checkpoints are unservable)"
            )

    video_keys = set(video_config.modality_keys)
    camera_targets = set(camera_map.values())
    if camera_targets != video_keys:
        raise ValueError(
            "--camera-map target names must equal checkpoint video keys "
            f"{sorted(video_keys)}, got {sorted(camera_targets)}"
        )

    canonical_keys = set(CANONICAL_SLICES)
    for modality_name, config in (("state", state_config), ("action", action_config)):
        actual_keys = set(config.modality_keys)
        if actual_keys != canonical_keys:
            raise ValueError(
                f"checkpoint {modality_name} keys must equal {sorted(canonical_keys)}, "
                f"got {sorted(actual_keys)}"
            )

    covered = [
        index
        for part_slice in CANONICAL_SLICES.values()
        for index in range(part_slice.start, part_slice.stop)
    ]
    if sorted(covered) != list(range(TOTAL_DIM)):
        raise ValueError("canonical state/action slices must exactly partition indices 0..13")


def _load_statistics(model_dir: Path) -> Mapping[str, Any]:
    """Load the checkpoint dataset statistics used for width and units checks."""
    stats_path = model_dir / "experiment_cfg" / "dataset_statistics.json"
    with stats_path.open(encoding="utf-8") as stats_file:
        loaded: Mapping[str, Any] = json.load(stats_file)
    return loaded


def _validate_statistics(stats: Mapping[str, Any], modality_configs: Mapping[str, Any]) -> None:
    """Validate canonical widths and safe radians/normalized-units ranges."""
    tagged_stats = stats[EMBODIMENT_TAG]
    arm_limit = math.pi + 0.05
    gripper_low = -0.05
    gripper_high = 1.05

    for modality_name in ("state", "action"):
        modality_stats = tagged_stats[modality_name]
        for key in modality_configs[modality_name].modality_keys:
            key_stats = modality_stats[key]
            expected_width = CANONICAL_SLICES[key].stop - CANONICAL_SLICES[key].start
            width = len(key_stats["mean"])
            if width != expected_width:
                raise ValueError(
                    f"{modality_name} key {key!r} has width {width}; expected {expected_width}"
                )

            minimum = np.asarray(key_stats["min"], dtype=float)
            maximum = np.asarray(key_stats["max"], dtype=float)
            bounds = np.concatenate((minimum.reshape(-1), maximum.reshape(-1)))
            if not np.isfinite(bounds).all():
                raise ValueError(f"{modality_name} key {key!r} has non-finite min/max stats")
            if key.endswith("_arm"):
                if np.any(np.abs(bounds) > arm_limit):
                    raise ValueError(
                        f"{modality_name} key {key!r} min/max exceed radians limit {arm_limit}"
                    )
            elif np.any(bounds < gripper_low) or np.any(bounds > gripper_high):
                raise ValueError(
                    f"{modality_name} key {key!r} min/max fall outside "
                    f"[{gripper_low}, {gripper_high}]"
                )


def _build_observation(
    payload: Mapping[str, Any],
    modality_configs: Mapping[str, Any],
    camera_map: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Build GR00T's nested, batched observation from one ``/act`` request."""
    video_config = modality_configs["video"]
    state_config = modality_configs["state"]
    language_config = modality_configs["language"]
    video_t = len(video_config.delta_indices)
    state_t = len(state_config.delta_indices)

    if "state" not in payload:
        raise ValueError("/act request is missing field 'state'")
    state = np.asarray(payload["state"], dtype=np.float32)
    if state.shape != (TOTAL_DIM,):
        raise ValueError(f"state must have shape ({TOTAL_DIM},), got {state.shape}")

    video_observation: dict[str, Any] = {}
    for source, target in camera_map.items():
        if source not in payload:
            raise ValueError(f"/act request is missing field {source!r}")
        frame = np.asarray(payload[source], dtype=np.uint8)
        video_observation[target] = np.repeat(frame[None, None, ...], video_t, axis=1)

    state_observation: dict[str, Any] = {}
    for key in state_config.modality_keys:
        part = state[CANONICAL_SLICES[key]]
        state_observation[key] = np.repeat(part[None, None, :], state_t, axis=1)

    if "instruction" not in payload:
        raise ValueError("/act request is missing field 'instruction'")
    instruction = str(payload["instruction"])
    language_observation = {key: [[instruction]] for key in language_config.modality_keys}
    return {
        "video": video_observation,
        "state": state_observation,
        "language": language_observation,
    }


def _pack_actions(actions: Mapping[str, Any], action_keys: list[str]) -> npt.NDArray[np.float32]:
    """Scatter name-keyed GR00T actions into the canonical packed 14-D chunk."""
    if set(actions) != set(action_keys):
        raise ValueError(
            f"returned action keys must equal {sorted(action_keys)}, got {sorted(actions)}"
        )

    chunk_lengths = {actions[key].shape[1] for key in action_keys}
    if len(chunk_lengths) != 1:
        raise ValueError(f"returned action keys have inconsistent chunk lengths: {chunk_lengths}")
    chunk_len = chunk_lengths.pop()
    packed = np.full((chunk_len, TOTAL_DIM), np.nan, dtype=np.float32)
    for key in action_keys:
        packed[:, CANONICAL_SLICES[key]] = np.asarray(actions[key], dtype=np.float32)[0]
    return packed


def _create_app(
    policy: Any,
    modality_configs: Mapping[str, Any],
    camera_map: Mapping[str, str],
    model_label: str,
    dt_ms: float,
) -> Any:
    """Create the health and inference endpoints around one loaded policy."""
    import json_numpy
    from fastapi import FastAPI  # type: ignore[import-not-found]
    from fastapi.concurrency import run_in_threadpool  # type: ignore[import-not-found]
    from fastapi.responses import JSONResponse, Response  # type: ignore[import-not-found]

    app = FastAPI()
    inference_lock = threading.Lock()
    num_steps_logged = False
    action_keys = list(modality_configs["action"].modality_keys)

    @app.get("/act")  # type: ignore[untyped-decorator]
    def health() -> dict[str, str]:
        """Report server health and the requested model id or path."""
        return {"status": "ok", "model": model_label}

    async def act(request: Any) -> Any:
        """Run one locked GR00T inference and return a packed action chunk.

        Registered through Starlette's ``add_route`` (below), NEVER through
        FastAPI's decorator: json_numpy's patched ``json.loads`` makes a
        JSON-shaped body poison FastAPI's parse-and-validate path (ndarrays in
        the parsed dict fail pydantic validation, then the 422 encoder crashes
        on them too, so every request dies as a 500 on fastapi<=0.115), and a
        function-local ``Request`` annotation is unresolvable under
        ``from __future__ import annotations`` anyway. The raw-route handler
        receives the Starlette request directly and decodes the bytes itself.
        """
        nonlocal num_steps_logged
        try:
            payload = json.loads(await request.body())
            if not isinstance(payload, dict):
                raise ValueError("/act request body must be a JSON object")
            observation = _build_observation(payload, modality_configs, camera_map)
            if "num_steps" in payload and not num_steps_logged:
                LOGGER.info(
                    "Ignoring request num_steps=%r; GR00T uses its checkpoint setting",
                    payload["num_steps"],
                )
                num_steps_logged = True

            def _locked_inference() -> Any:
                # Threadpool keeps GPU inference off the event loop, so the
                # lock is genuinely load-bearing and health stays responsive.
                with inference_lock:
                    actions, _info = policy.get_action(observation)
                return actions

            actions = await run_in_threadpool(_locked_inference)
            packed = _pack_actions(actions, action_keys)
            response = {"actions": packed, "dt_ms": dt_ms}
            return Response(content=json_numpy.dumps(response), media_type="application/json")
        except Exception as exc:
            LOGGER.exception("GR00T /act inference failed")
            return JSONResponse(status_code=500, content={"message": str(exc)})

    app.add_route("/act", act, methods=["POST"])
    return app


def main() -> None:
    """Load and validate the checkpoint, then serve its ``/act`` endpoints."""
    import json_numpy
    import uvicorn  # type: ignore[import-not-found]
    from gr00t.policy.gr00t_policy import Gr00tPolicy  # type: ignore[import-not-found]

    logging.basicConfig(level=logging.INFO)
    json_numpy.patch()
    args = _parse_args()
    if args.dt_ms < 0:
        raise ValueError("--dt-ms must be non-negative")

    camera_map = _parse_camera_map(args.camera_map)
    model_dir = _resolve_model(args.model)
    policy = Gr00tPolicy(
        embodiment_tag=EMBODIMENT_TAG,
        model_path=str(model_dir),
        device="cuda",
    )
    modality_configs = policy.get_modality_config()
    _validate_modality_configs(modality_configs, camera_map)
    _validate_statistics(_load_statistics(model_dir), modality_configs)
    app = _create_app(policy, modality_configs, camera_map, args.model, args.dt_ms)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
