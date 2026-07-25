"""Build locally or verify the exact prebuilt SDK wheel in credentialed CI."""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def expected_wheel() -> Path:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    name = project["name"].replace("-", "_")
    version = project["version"]
    return ROOT / "dist" / f"{name}-{version}-py3-none-any.whl"


def main() -> None:
    wheel = expected_wheel()
    if os.getenv("AAI_USE_PREBUILT_WHEEL") == "1":
        if not wheel.is_file():
            raise FileNotFoundError(f"Expected prebuilt wheel: {wheel}")
        return
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=ROOT,
        check=True,
    )
    if not wheel.is_file():
        raise FileNotFoundError(f"Build did not create expected wheel: {wheel}")


if __name__ == "__main__":
    main()
