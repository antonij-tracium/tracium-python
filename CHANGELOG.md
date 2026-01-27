# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
