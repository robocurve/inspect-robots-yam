"""Keep the GitHub-only hardware installation reproducible."""

from __future__ import annotations

import re
from pathlib import Path

from inspect_robots_yam._i2rt import I2RT_INSTALL_COMMAND

ROOT = Path(__file__).resolve().parents[1]

I2RT_SOURCE_PATTERN = (
    r"i2rt @ git\+https://github\.com/i2rt-robotics/i2rt@(?P<revision>[0-9a-f]{40})"
)


def test_i2rt_revision_is_pinned_consistently() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    runtime_match = re.fullmatch(
        r"echo 'scikit-build-core<0\.10' > build-constraints\.txt && "
        r"uv pip install --build-constraints build-constraints\.txt \""
        + I2RT_SOURCE_PATTERN
        + r'"',
        I2RT_INSTALL_COMMAND,
    )
    readme_match = re.search(
        r"echo 'scikit-build-core<0\.10' > build-constraints\.txt\n"
        r"uv pip install --build-constraints build-constraints\.txt \""
        + I2RT_SOURCE_PATTERN
        + r'"',
        readme,
    )

    assert runtime_match is not None
    assert readme_match is not None
    assert runtime_match.group("revision") == readme_match.group("revision")
