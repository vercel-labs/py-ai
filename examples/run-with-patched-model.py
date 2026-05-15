#!/usr/bin/env python3
"""Run a Python file with ``ai.get_model()`` patched to always return a fixed model.

Useful for re-running an example against a different model without
editing it.

Usage (from repo root):

    uv run examples/run-with-patched-model.py <model> <file.py>

Example:

    uv run examples/run-with-patched-model.py \\
        gateway:openai/gpt-5.4-mini \\
        examples/samples/stream.py
"""

import argparse
import runpy
import sys
from typing import Any

import ai
from ai import models
from ai.models import core
from ai.models.core import model as _model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model", help="model id, e.g. 'gateway:anthropic/claude-sonnet-4.6'"
    )
    parser.add_argument("file", help="path to a python file to execute")
    args = parser.parse_args()

    original = _model.get_model

    def patched(*_args: Any, **_kwargs: Any) -> ai.Model:
        return original(args.model)

    ai.get_model = patched
    models.get_model = patched
    core.get_model = patched
    _model.get_model = patched

    sys.argv = [args.file]
    runpy.run_path(args.file, run_name="__main__")


if __name__ == "__main__":
    main()
