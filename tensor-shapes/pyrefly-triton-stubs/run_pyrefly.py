#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Check the experimental Triton stubs without a Triton installation or GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shape_testing import check_suites, pyrefly_command, Suite  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parent


def main() -> int:
    """Check the stubs, regression tests, and examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyrefly", type=Path)
    parser.add_argument("--buck", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--nocapture", action="store_true")
    args = parser.parse_args()
    return check_suites(
        pyrefly=pyrefly_command(
            explicit=args.pyrefly, buck=args.buck, release=args.release
        ),
        package_root=PACKAGE_ROOT,
        suites=[
            Suite(name="regressions", patterns=("test/test_*.py",), expectations=True),
            Suite(name="examples", patterns=("examples/*.py",), expectations=True),
        ],
        nocapture=args.nocapture,
    )


if __name__ == "__main__":
    sys.exit(main())
