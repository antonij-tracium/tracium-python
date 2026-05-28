"""
Automatic, human-readable agent name detection from runtime context.

The agent name shows up in dashboards as "what is this trace?", so it has to be
*recognizable* — ``customer-support-agent`` rather than ``run`` or ``app``. We
walk the call stack from the user's frame outward, applying a small set of
rules in priority order:

1. **Web request route** — if a known framework (Flask, FastAPI, Django, …)
   is dispatching, use the route as the name.
2. **Class context** — if the outer user frame is bound to a class instance
   (``self``) or a classmethod (``cls``), prefer ``ClassName`` as the agent
   name. Class names are already human-meaningful in agent codebases
   (``CustomerSupportAgent``, ``ResearchAgent``).
3. **Outermost user function** — the entry-point function name.
4. **Script / module name** — for CLI scripts and ad-hoc runs.
5. **Project directory** — last-resort fallback for ``__main__`` entries.
6. **Default** — never raise; always return a string.

Names are normalized to lowercase ``kebab-case`` and stripped of obvious noise
(``test_``, ``_main``).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

# Path to the installed tracium package — frames originating from here are
# skipped (the user's code is what we want to identify, not our own wrappers).
_TRACIUM_PKG_ROOT = str(Path(__file__).resolve().parent.parent) + "/"

# Frames matching any of these substrings are skipped during the walk.
_SKIP_FILE_SUBSTRINGS = (
    "/site-packages/",
    "/asyncio/",
    "/concurrent/",
    "/threading.py",
    "/werkzeug/",
    "/starlette/",
    "/uvicorn/",
    "/gunicorn/",
)

# Function names that indicate framework internals, not user agents.
_SKIP_FUNCTION_NAMES = frozenset(
    {
        "__init__",
        "__call__",
        "__enter__",
        "__exit__",
        "__aenter__",
        "__aexit__",
        "wrapper",
        "_wrapper",
        "decorator",
        "dispatch_request",
        "view_function",
        "run_endpoint_function",
        "get_response",
        "_handle",
    }
)

# Names that mean "unknown caller" — keep walking past them.
_TOO_GENERIC = frozenset({"main", "run", "execute", "handle", "process", "app"})


def detect_agent_name(default: str = "app") -> str:
    """Walk the call stack and return a human-readable agent name.

    Never raises; returns ``default`` if nothing usable is found.
    """
    try:
        web = _detect_web_route()
        if web:
            return _normalize(web) or default

        candidate = _walk_user_frames()
        if candidate:
            return _normalize(candidate) or default

        project = _detect_project_dir()
        if project:
            return _normalize(project) or default
    except Exception:
        pass
    return default


def detect_model_id_from_call(kwargs: dict) -> str | None:
    """Pull the model ID out of common LLM-call kwargs."""
    for key in ("model", "model_id", "model_name", "deployment", "engine"):
        value = kwargs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# --------------------------------------------------------------------------- #
# Stack walking                                                                #
# --------------------------------------------------------------------------- #


def _walk_user_frames() -> str | None:
    """Return the best agent-name candidate found while walking outward.

    The first user frame we encounter is privileged: if it's a class method
    (``self``/``cls`` in locals), we use the class name. This is the strongest
    signal in agent codebases. Test classes (``Test*``) are deliberately
    excluded so the detection doesn't return ``test-something`` from pytest
    scaffolding.

    For non-method first frames, we use the function name and keep walking
    only to find a more specific outer function if the inner one is generic
    (``main``, ``run``, etc.).
    """
    frame = inspect.currentframe()
    if frame is None:
        return None
    frame = frame.f_back  # skip ourselves

    first_user_frame = True
    best_function: str | None = None
    best_module: str | None = None
    try:
        while frame is not None:
            filename = frame.f_code.co_filename
            function_name = frame.f_code.co_name

            if _is_skip_file(filename) or function_name in _SKIP_FUNCTION_NAMES:
                frame = frame.f_back
                continue

            if first_user_frame:
                first_user_frame = False
                class_name = _class_name_for_frame(frame)
                if class_name and not _looks_like_test_class(class_name):
                    return class_name

            if function_name == "<module>":
                best_module = best_module or _module_name_from_file(filename)
                frame = frame.f_back
                continue

            if function_name and function_name not in _TOO_GENERIC:
                if best_function is None:
                    best_function = function_name

            frame = frame.f_back
    finally:
        del frame

    return best_function or best_module


def _looks_like_test_class(name: str) -> bool:
    return name.startswith("Test") or name.endswith("TestCase") or name.endswith("Tests")


def _class_name_for_frame(frame) -> str | None:
    """Detect a class context (instance method or classmethod) on ``frame``.

    We look at the frame's local variables: ``self`` for instance methods,
    ``cls`` for classmethods. Both yield the bound class. Static methods don't
    expose this, but those are rare in agent code.
    """
    try:
        local_self = frame.f_locals.get("self")
        if local_self is not None:
            cls = type(local_self)
            if cls.__module__ != "builtins":
                return cls.__name__

        local_cls = frame.f_locals.get("cls")
        if isinstance(local_cls, type) and local_cls.__module__ != "builtins":
            return local_cls.__name__
    except Exception:
        pass
    return None


def _is_skip_file(filename: str) -> bool:
    if filename.startswith(_TRACIUM_PKG_ROOT):
        return True
    return any(s in filename for s in _SKIP_FILE_SUBSTRINGS)


def _module_name_from_file(filename: str) -> str | None:
    try:
        stem = Path(filename).stem
        if not stem or stem in {"__main__", "__init__"}:
            # Try the parent dir name in this case.
            return Path(filename).parent.name or None
        return stem
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Web route detection                                                          #
# --------------------------------------------------------------------------- #


def _detect_web_route() -> str | None:
    """Try framework integrations to extract a route name."""
    try:
        from .web_frameworks import get_web_route_info

        info = get_web_route_info()
        if info is not None:
            _route_path, display_name = info
            return display_name
    except Exception:
        pass
    return None


def _detect_project_dir() -> str | None:
    try:
        cwd = Path.cwd()
        if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
            name = cwd.name
            if name and name not in {"src", "lib", "app"}:
                return name
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Normalization                                                                #
# --------------------------------------------------------------------------- #


_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _normalize(name: str) -> str | None:
    """Normalize a detected name to lowercase kebab-case.

    ``CustomerSupportAgent`` → ``customer-support-agent``
    ``my_chat_handler``     → ``my-chat-handler``
    ``test_search-fn``      → ``search-fn``
    """
    if not name:
        return None
    # Split CamelCase, then collapse separators.
    cleaned = _CAMEL_RE.sub("-", name).replace("_", "-").lower()
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        return None
    if cleaned.startswith("test-"):
        cleaned = cleaned[len("test-") :]
    if cleaned.endswith("-main"):
        cleaned = cleaned[: -len("-main")]
    if cleaned in _TOO_GENERIC or not cleaned:
        return None
    return cleaned
