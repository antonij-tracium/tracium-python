# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.5.8] - 2026-05-07

### Added

- **Tool call tracking**: The SDK now links tool-use spans back to the LLM span that originated them. A new `tool_call_registry` tracks tool call IDs across Anthropic, OpenAI, and Responses API formats; spans that invoke tools register their IDs, and subsequent tool-result spans resolve the correct parent automatically.
- **`extract_tool_calls` for Anthropic**: New utility that extracts `tool_use` content blocks from Anthropic responses, returning structured dicts with `id`, `name`, and `input` fields for use in span linking and tracing.
- **Google Gemini tool tracking**: Tool call IDs from Gemini responses are now registered in the registry, enabling parent-span resolution for Gemini tool flows.

### Fixed

- **OpenAI Responses API patching**: Submodules not re-exported from `openai.resources.__init__` (e.g. `responses`) are now resolved via `importlib.import_module` as a fallback, so Responses API endpoints are always patched correctly.
- **Responses API cached token extraction**: `input_tokens_details.cached_tokens` (Responses API) is now checked alongside `prompt_tokens_details.cached_tokens` (Chat Completions API), ensuring accurate cached token accounting for both API surfaces.

## [1.5.3] - 2026-03-25

### Fixed

- **OpenAI cached token accounting**: OpenAI includes cached tokens within `prompt_tokens`. The SDK now subtracts cached tokens from the prompt total so that `input_tokens` represents non-cached input only, matching Anthropic semantics and giving accurate cost breakdowns.

### Added

- **Automatic `stream_options` for OpenAI streaming**: Streaming OpenAI calls now automatically include `{"include_usage": true}` in `stream_options` (when not already set) so token usage data is always captured, even if the caller didn't opt in.
- **LLM summary on failed traces**: `fail_agent_trace` now sends the `llm_summary` (model, system prompt, and LLM steps) to the backend, just like successful trace completions. This ensures auto-versioning data is preserved even when a trace fails.

## [1.5.2] - 2026-03-16

### Added

- **LLM trace summary and auto-versioning**: Completed agent traces now send aggregated LLM metadata to the backend so that versioning can reflect model and prompt changes. The SDK captures model ID and system prompt from each LLM span (across OpenAI, Anthropic, Google Gemini, and LangChain/LangGraph formats), combines them per trace, and sends a summary on trace completion (`model`, `system_prompt`, and per-step `llm_steps`). Any change in model or system prompt across spans produces a different fingerprint for accurate auto-versioning.
- **System prompt extraction**: New internal helper `_extract_system_prompt()` in span handling that supports Anthropic (top-level `system`), OpenAI (messages with `role: system` and multipart content), Google Gemini (`system_instruction`), and LangChain-style messages (`type`/`role` and `content`/`text`).
- **Anthropic integration**: `normalize_messages` now preserves the `system` argument in the traced payload; span input records the full messages payload (including system) so the system prompt is available for extraction and auto-versioning.
- **API**: `complete_agent_trace` now accepts an optional `llm_summary` parameter (`LLMTraceSummary`) and sends `model`, `system_prompt`, and `llm_steps` to the backend (with length limits: model 255 chars, system_prompt 10000 chars).
- **Tests**: New test modules under `tests/test_models/` for `_extract_system_prompt` and for `_combine_llm_info` / `LLMTraceSummary` (single-span, multi-span, ordering, and fingerprinting behavior).

### Changed

- **Trace completion**: When an agent trace is completed (via `AgentTraceHandle` or `AgentTraceManager`), the combined LLM info from all LLM spans in that trace is sent as `llm_summary` to the backend.
- **Trace state**: `TraceState` now maintains a shared `_llm_info` list of `(span_name, model, system_prompt)` tuples; thread copies share the same list so all LLM spans contribute to the same summary.

## [1.0.5] - 2026-03-02

### Added

- **OpenAI integration (expanded)**: Instrumentation now covers all major OpenAI API surfaces. Every call is captured as a span:
  - Chat completions and legacy completions (sync and async)
  - Embeddings
  - Images (generate, edit, create_variation)
  - Audio: transcriptions, translations, and speech
  - Moderations
  - Responses API
  - Beta Threads: `create_and_run`, `create_and_run_poll`, `create_and_run_stream`
  - Beta Threads Runs: `create`, `create_and_poll`, `stream`
  - Streaming responses with full output and token usage capture (including cached tokens)

## [1.0.4] - 2026-02-21

### Added

- **Version check on init**: The SDK now checks PyPI for newer versions when the client is initialized and logs a one-time warning if an upgrade is available. Skips the check when using test/mock/localhost base URLs. Silent on network failure to avoid breaking initialization.

## [1.0.3] - 2026-02-09

### Fixed

- **Payload size**: Span input and output are now always sanitized and capped at 100KB to avoid "payload too large" errors when sending to the backend. Base64 and data-URL content (e.g. images) are replaced with a short placeholder; long strings are truncated. If the sanitized payload still exceeds 100KB, it is replaced with a single placeholder. Users still see that a document or image was used, prompt text, and the LLM response, without hitting backend size limits.

## [1.0.2] - 2026-02-08

### Added

- **Workspace scoping**: `workspace_id` on `tracium.trace()` and `tracium.init()` to set the workspace for all traces and spans created by that instance. Optional `default_workspace_id` in config (or `TRACIUM_WORKSPACE_ID` env var) applies when not passed per call.
## [1.0.1] - 2026-01-27

### Changed

- **Optional Dependencies**: All LLM providers and web frameworks are now optional dependencies
  - Base installation (`pip install tracium`) now only includes core dependencies (httpx, python-dotenv, typing-extensions)
  - Users can install specific integrations as needed:
    - LLM providers: `pip install tracium[openai]`, `tracium[anthropic]`, `tracium[google]`, `tracium[langchain]`, `tracium[langgraph]`
    - Web frameworks: `pip install tracium[flask]`, `tracium[django]`, `tracium[fastapi]`, `tracium[celery]`
    - Mix and match: `pip install tracium[openai,flask]`
    - All integrations: `pip install tracium[all]`
  - Tracium automatically detects which packages are installed and instruments them accordingly
  - No changes required to existing code - if you already have the packages installed, everything works the same
  - Significantly reduces installation size and dependency conflicts for users who only need specific integrations

## [1.0.0] - 2026-01-27

### Added

- **Event Loss Prevention**: Comprehensive improvements to prevent telemetry data loss
  - Added `block_on_full_queue` configuration option to wait for queue space instead of dropping events
  - Added `max_queue_size` configuration (default: 10,000) to customize queue capacity
  - Added `queue_warning_threshold` configuration (default: 0.8) to warn when queue approaches capacity
  - Added `queue_timeout` configuration (default: 5.0s) for maximum blocking time
  - Added `get_queue_stats()` API to monitor queue health and event counts
  - Added comprehensive statistics tracking (enqueued, sent, failed, dropped events)
  - Improved warning messages with actionable guidance when events are at risk
  - Added health indicators and success/drop rate metrics

### Changed

- **Background Sender**: Enhanced error handling and statistics tracking
  - Now tracks total enqueued, sent, failed, and dropped events
  - Warns when queue reaches configurable capacity threshold (default: 80%)
  - Rate-limits warnings to once per minute to avoid log spam
  - Changed dropped event log level from DEBUG to ERROR for better visibility
  - Failed request log level changed from DEBUG to WARNING with failure count

### Example Configuration

```python
import tracium

# Prevent event loss with blocking mode
client = tracium.init(
    api_key="...",
    config=tracium.TraciumClientConfig(
        max_queue_size=20000,          # Increase capacity
        block_on_full_queue=True,      # Wait instead of dropping
        queue_warning_threshold=0.9,   # Warn at 90% capacity
        queue_timeout=10.0             # Wait up to 10s
    )
)

# Monitor queue health
stats = tracium.get_queue_stats()
print(f"Queue: {stats['capacity_percent']:.1f}% full")
print(f"Dropped: {stats['total_dropped']} events")
print(f"Success rate: {stats['success_rate']:.1%}")
```

## [0.2.0] - 2025-12-25

### Added

- **Web Framework Support**: Added comprehensive web framework instrumentation for automatic tracing
  - Flask integration with automatic route detection and response tracking
  - Django integration with request/response lifecycle tracking
  - FastAPI/Starlette integration with ASGI support (works with uvicorn)
  - Celery integration for background task tracking
  - Generic WSGI middleware support for compatibility with various WSGI servers
  - Automatic trace closure on request completion for all supported frameworks

### Fixed

- Fixed various existing bugs to improve stability and reliability

## [0.1.2] - 2025-12-22

### Fixed

- **LangChain Integration**: Made `TraciumLangChainHandler` optional when LangChain is not installed, preventing import errors when the LangChain package is unavailable
- Fixed LangChain integration to gracefully handle missing `BaseCallbackHandler` class

### Changed

- **CI/CD**: Updated GitHub Actions workflow to deploy to TestPyPI instead of production PyPI for safer release testing
- Updated release workflow to use TestPyPI repository URL and environment

## [0.1.1] - 2025-12-22

### Added

- Support for the new `google-genai` Python SDK (v0.1.0+) alongside the deprecated `google-generativeai`.
- Automatic detection and instrumentation for `google-genai`.

### Fixed

- Suppressed `FutureWarning` from `google-generativeai` during initialization.

## [0.1.0] - 2025-12-17

### Added

- Initial public release
- Comprehensive test suite with unit tests for core components
- Test fixtures and configuration in `tests/conftest.py`
- Tests for client initialization, HTTP client, validation, retry logic, and context management
- Unit tests for tenant context and trace context management
- Test coverage for retry mechanisms and validation utilities
- **Automatic Instrumentation**
  - Automatic instrumentation for OpenAI (GPT-4, GPT-3.5, and all OpenAI models)
  - Automatic instrumentation for Anthropic (Claude models)
  - Automatic instrumentation for Google Gemini (all Gemini models)
  - Automatic instrumentation for LangChain chains and agents
  - Automatic instrumentation for LangGraph workflows
  - Library auto-detection to enable only installed integrations
- **Trace and Span Management**
  - `tracium.trace()` - One-line setup for automatic tracing
  - `tracium.init()` - Advanced initialization with full configuration
  - `tracium.start_trace()` - Manual trace creation
  - `AgentTraceHandle` and `AgentTraceManager` for trace lifecycle management
  - Span creation and management APIs
  - Support for custom trace IDs
- **Context Management**
  - Thread-local trace context management
  - Automatic context propagation across threads via patched `ThreadPoolExecutor` and `Thread` classes
  - Context propagation across async boundaries
  - Multi-tenant support with `set_tenant()` and `get_current_tenant()`
- **Configuration and Customization**
  - `TraciumClientConfig` for advanced client configuration
  - `RetryConfig` for customizable retry policies
  - Default agent names, versions, tags, and metadata
  - Configurable auto-instrumentation per library type
  - Custom HTTP transport support
- **Retry and Resilience**
  - Exponential backoff retry logic for API calls
  - Separate sync and async retry implementations
  - Configurable retry attempts and backoff parameters
  - Fail-open behavior to prevent SDK errors from breaking user code
- **Security Features**
  - Rate limiting with token bucket algorithm
  - Sensitive data redaction in logs
  - `SecurityConfig` for security settings
  - API key validation
- **Validation**
  - Comprehensive input validation for all API parameters
  - Validation utilities for agent names, trace IDs, span IDs, tags, metadata, and error messages
  - Type checking and format validation
- **Logging and Observability**
  - Configurable logging with `configure_logging()`
  - Sensitive data redaction in log output
  - Debug logging for instrumentation and API calls
  - Custom logger support
- **Advanced Features**
  - Parallel execution tracking with `parallel_tracker`
  - Call hierarchy tracking
  - Decorator-based instrumentation (`@agent_trace`, `@agent_span`, `@span`)
  - Span registry for span lifecycle management
  - DateTime utilities for timestamp handling
  - Tag utilities for tag management
- **Developer Experience**
  - PEP 561 compliant type stubs (`py.typed` marker)
  - Comprehensive type hints throughout the codebase
  - Google-style docstrings for all public APIs
  - Clear error messages and validation feedback

### Features

- **One-line setup**: `tracium.trace(api_key="...")` enables automatic tracing for all supported libraries
- **Zero-configuration**: Works out of the box with sensible defaults
- **Automatic library detection**: Only instruments libraries that are installed
- **Thread-safe**: Automatic context propagation across threads and async boundaries
- **Fail-open**: SDK errors never break user code
- **Configurable**: Extensive configuration options for advanced use cases
- **Type-safe**: Full type hints and mypy support

### Dependencies

- Python 3.9+ support
- Core dependencies: `httpx`, `python-dotenv`
- Integration dependencies: `openai`, `anthropic`, `google-generativeai`, `langchain`, `langchain-core`, `langgraph`

---

## Version History

- **1.5.8**: Tool call tracking across Anthropic/OpenAI/Gemini, fix Responses API patching and cached token extraction
- **1.5.3**: Fix OpenAI cached token accounting, auto-include stream_options for token usage, send LLM summary on failed traces
- **1.5.2**: LLM trace summary and auto-versioning - send model/system_prompt/llm_steps on trace completion; system prompt extraction across OpenAI, Anthropic, Gemini, LangChain; Anthropic preserves system in payload
- **1.0.5**: Expanded OpenAI integration - all major API surfaces (chat, completions, embeddings, images, audio, moderations, responses, beta threads) with streaming and token usage
- **1.0.4**: Version check on init - warns when a newer SDK is available on PyPI
- **1.0.3**: Payload size sanitization - span input/output capped at 100KB to avoid backend errors
- **1.0.2**: Added workspace_id to trace() and init() for workspace-scoped tracing
- **0.2.0**: Added web framework support (Flask, Django, FastAPI, Celery) with uvicorn and WSGI compatibility, fixed existing bugs
- **0.1.2**: Fixed LangChain optional dependency handling, updated CI/CD to use TestPyPI
- **0.1.0**: Initial public release with full feature set including automatic instrumentation, context propagation, and comprehensive tooling

## Upgrade Guide

This is the first public release. No upgrade path from previous versions.

## Notes

- All dates are in YYYY-MM-DD format
- Breaking changes are marked with ⚠️
- Deprecations are marked with ⚠️ (deprecated)
- This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
- For detailed API documentation, see [https://docs.tracium.ai](https://docs.tracium.ai)
