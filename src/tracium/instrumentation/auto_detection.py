"""
Automatic detection of agent names and model IDs from context.
"""

from __future__ import annotations

import inspect
from pathlib import Path


def detect_agent_name(default: str = "app") -> str:
    """
    Automatically detect agent name from:
    1. Entry point function name
    2. Script/module name
    3. Project directory name
    4. Default fallback
    """
    frame = inspect.currentframe()
    if frame is None:
        return default
    frame = frame.f_back

    try:
        entry_function = None
        entry_file = None
        entry_patterns = ["main", "run_", "pipeline", "workflow", "process", "execute", "__main__"]

        while frame is not None:
            filename = frame.f_code.co_filename
            function_name = frame.f_code.co_name

            if (
                "src/tracium/" in filename
                or "site-packages" in filename
                or (
                    any(
                        skip in filename
                        for skip in ["openai", "anthropic", "google", "langchain", "langgraph"]
                    )
                    and "site-packages" in filename
                )
            ):
                frame = frame.f_back
                continue

            if any(pattern in function_name.lower() for pattern in entry_patterns):
                entry_function = function_name
                entry_file = filename
                break

            if function_name == "<module>":
                entry_file = filename
                entry_function = Path(filename).stem
                if entry_function == "__main__":
                    entry_function = Path(filename).name
                    if entry_function.endswith(".py"):
                        entry_function = entry_function[:-3]
                break

            if entry_file is None:
                entry_file = filename
                entry_function = (
                    function_name if function_name != "<module>" else Path(filename).stem
                )

            frame = frame.f_back

        if entry_function and entry_function != "<module>":
            name = entry_function.replace("_", "-")
            if name.startswith("test-"):
                name = name[5:]
            if name.endswith("-main"):
                name = name[:-5]
            if name and name != "main":
                return name

        if entry_file:
            name = Path(entry_file).stem.replace("_", "-")
            if name and name != "__main__":
                return name

        try:
            cwd = Path.cwd()
            if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
                name = cwd.name.replace("_", "-")
                if name and name not in ["src", "lib", "app"]:
                    return name
        except Exception:
            pass

    finally:
        del frame

    return default


def detect_model_id_from_call(kwargs: dict) -> str | None:
    """
    Extract model ID from LLM call kwargs.
    This is already partially done in integrations, but we can make it more robust.
    """
    model_keys = ["model", "model_id", "model_name", "model_name_for_completion"]

    for key in model_keys:
        if key in kwargs:
            model = kwargs[key]
            if isinstance(model, str) and model:
                return model

    return None
