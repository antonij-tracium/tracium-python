# Tracium Python SDK - Developer Guide

[![License](https://img.shields.io/pypi/l/tracium-sdk)](https://pypi.org/project/tracium-sdk/)
[![Test](https://github.com/AntonijSimonovski/tracium-python/actions/workflows/test.yml/badge.svg)](https://github.com/AntonijSimonovski/tracium-python/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/tracium-sdk)](https://pypi.org/project/tracium-sdk/)

This is the development guide for the Tracium Python SDK. For user documentation, see [https://docs.tracium.ai](https://docs.tracium.ai).

**Version:** 0.1.0

## Project Overview

The Tracium Python SDK provides automatic instrumentation and tracing for LLM applications. It supports OpenAI, Anthropic, Google Gemini, LangChain, and LangGraph with minimal configuration.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- pip
- git

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/AntonijSimonovski/tracium-python.git
cd tracium-python/tracium

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in editable mode with dev dependencies
# This automatically handles PYTHONPATH for src/ directory
pip install -e ".[dev]"
```

### Environment Variables

For local development and testing, you may need to set:

```bash
export TRACIUM_API_KEY="sk_test_..."  # Required for integration tests only
export TRACIUM_BASE_URL="https://api.tracium.ai"  # Optional: override API endpoint
```

## Project Structure

```
tracium/
├── src/
│   └── tracium/
│       ├── __init__.py              # Public API entrypoint
│       ├── api/                     # HTTP client and API endpoints
│       │   ├── endpoints.py         # API endpoint definitions
│       │   └── http_client.py       # HTTP client implementation
│       ├── context/                 # Context management
│       │   ├── context_propagation.py  # Thread context propagation
│       │   ├── tenant_context.py    # Multi-tenant context
│       │   └── trace_context.py     # Trace context management
│       ├── core/                    # Core SDK components
│       │   ├── client.py            # Main TraciumClient class
│       │   ├── config.py            # Configuration classes
│       │   └── version.py           # Version information
│       ├── helpers/                 # Utility modules
│       │   ├── call_hierarchy.py    # Call hierarchy tracking
│       │   ├── embeddings.py        # Embedding computation
│       │   ├── global_state.py      # Global SDK state
│       │   ├── logging_config.py    # Logging configuration
│       │   ├── parallel_tracker.py # Parallel execution tracking
│       │   ├── retry_async.py       # Async retry logic
│       │   ├── retry.py              # Sync retry logic
│       │   ├── security.py          # Security utilities
│       │   ├── thread_helpers.py    # Threading utilities
│       │   └── validation.py        # Input validation
│       ├── instrumentation/         # Auto-instrumentation
│       │   ├── auto_detection.py    # Library detection
│       │   ├── auto_instrumentation.py  # Main instrumentation logic
│       │   ├── auto_trace_tracker.py    # Automatic trace tracking
│       │   └── decorators.py        # Decorator-based instrumentation
│       ├── integrations/            # Library-specific integrations
│       │   ├── anthropic.py         # Anthropic/Claude integration
│       │   ├── google.py            # Google Gemini integration
│       │   ├── langchain.py         # LangChain integration
│       │   ├── langgraph.py        # LangGraph integration
│       │   └── openai.py            # OpenAI integration
│       ├── models/                  # Data models
│       │   ├── span_handle.py       # Span management
│       │   ├── trace_handle.py      # Trace management
│       │   └── trace_state.py       # Trace state management
│       └── utils/                   # Additional utilities
│           ├── datetime_utils.py    # Date/time utilities
│           ├── span_registry.py     # Span registry
│           ├── tags.py              # Tag utilities
│           └── validation.py        # Additional validation
├── tests/                           # Test suite (if exists)
├── pyproject.toml                   # Project configuration
├── LICENSE                          # MIT License
└── README.md                        # This file
```

## Key Modules

### Core Components

- **`core/client.py`**: Main `TraciumClient` class that handles all API communication
- **`core/config.py`**: Configuration classes (`TraciumClientConfig`, `RetryConfig`)
- **`__init__.py`**: Public API surface with convenience functions (`init()`, `trace()`, `start_trace()`)

### Instrumentation

- **`instrumentation/auto_instrumentation.py`**: Main orchestration for auto-instrumentation
- **`instrumentation/auto_detection.py`**: Detects which libraries are installed
- **`integrations/*.py`**: Library-specific instrumentation hooks

### Context Management

- **`context/trace_context.py`**: Manages active trace context (thread-local)
- **`context/context_propagation.py`**: Propagates context across threads/async boundaries
- **`context/tenant_context.py`**: Multi-tenant context management

### Helpers

- **`helpers/global_state.py`**: Global SDK state (client instance, default options)
- **`helpers/retry.py`** & **`helpers/retry_async.py`**: Retry logic with exponential backoff
- **`helpers/validation.py`**: Input validation utilities
- **`helpers/parallel_tracker.py`**: Tracks parallel execution contexts

## Development Workflow

### Running Tests

If you installed with `pip install -e ".[dev]"`, `pytest` will automatically find the `src` package.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tracium --cov-report=html

# Run specific test file
pytest tests/test_specific.py

# Run with verbose output
pytest -v
```

If you encounter `ModuleNotFoundError: No module named 'tracium'`, ensure you are in the virtual environment and have installed the package in editable mode, or run:
```bash
PYTHONPATH=src pytest
```

### Code Quality

```bash
# Run linter
ruff check tracium/

# Auto-fix linting issues
ruff check --fix tracium/

# Run type checker
mypy tracium/

# Format code
ruff format tracium/
```

### Pre-commit Checks

Before committing, ensure:

1. All tests pass: `pytest`
2. Linting passes: `ruff check tracium/`
3. Type checking passes: `mypy tracium/`
4. Code is formatted: `ruff format tracium/`

## Architecture Notes

### Thread Context Propagation

The SDK patches `ThreadPoolExecutor` and `Thread` classes at import time to automatically propagate trace context across threads. This happens in `__init__.py` before any user code runs.

### Auto-Instrumentation

Auto-instrumentation works by:

1. Detecting installed libraries (`auto_detection.py`)
2. Monkey-patching library methods to intercept calls
3. Creating traces/spans automatically
4. Sending data to Tracium API

### Global State

The SDK uses a global state pattern (`helpers/global_state.py`) to store:

- The initialized client instance
- Default options (agent name, version, tags, metadata)
- Instrumentation configuration

### Retry Logic

Both sync and async retry logic are implemented:

- Exponential backoff with configurable parameters
- Max retry attempts
- Fail-open behavior (errors don't break user code)

## Building and Releasing

### Building the Package

```bash
# Build source distribution
python -m build

# Build wheel
python -m build --wheel
```

### Version Management

Version is defined in `src/tracium/core/version.py`. Update this file when releasing a new version.

### Release Checklist

1. Update version in `src/tracium/core/version.py`
2. Update version in `pyproject.toml`
3. Update `CHANGELOG.md` (if exists)
4. Run all tests: `pytest`
5. Run linting: `ruff check tracium/`
6. Run type checking: `mypy tracium/`
7. Build package: `python -m build`
8. Test installation: `pip install dist/tracium_sdk-*.whl`
9. Tag release: `git tag v0.1.0`
10. Push tags: `git push --tags`
11. Publish to PyPI: `twine upload dist/*`

## Testing Strategy

### Unit Tests

Test individual functions and classes in isolation.

### Integration Tests

Test interactions between components (e.g., client → API, instrumentation → client).

### Library Integration Tests

Test auto-instrumentation with actual library calls (OpenAI, Anthropic, etc.). These may require API keys.

### Mocking

Use mocks for:

- HTTP requests to Tracium API
- External library calls when testing instrumentation
- Time-dependent operations

## Code Style

- **Line length**: 100 characters (configured in `pyproject.toml`)
- **Formatter**: Ruff (black-compatible)
- **Linter**: Ruff
- **Type checker**: mypy
- **Python version**: 3.9+

### Type Hints

Use type hints throughout the codebase. The package includes `py.typed` marker for PEP 561 compliance.

### Docstrings

Use Google-style docstrings for public APIs:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
```

## Common Development Tasks

### Adding a New Integration

1. Create a new file in `integrations/` (e.g., `integrations/new_library.py`)
2. Implement instrumentation hooks
3. Register in `instrumentation/auto_instrumentation.py`
4. Add detection logic in `instrumentation/auto_detection.py`
5. Add tests
6. Update documentation

### Adding a New API Endpoint

1. Define endpoint in `api/endpoints.py`
2. Add method to `TraciumClient` in `core/client.py`
3. Update HTTP client if needed (`api/http_client.py`)
4. Add tests
5. Update public API in `__init__.py` if needed

### Debugging

Enable debug logging:

```python
from tracium import configure_logging
import logging

configure_logging(level=logging.DEBUG)
```

This will show:

- API request/response details (with sensitive data redacted)
- Retry attempts
- Instrumentation activity
- Context propagation

## Dependencies

### Core Dependencies

- `httpx>=0.27,<0.29`: HTTP client
- `python-dotenv>=1.0.0`: Environment variable management

### Integration Dependencies

These are optional and only needed if you want to test specific integrations:

- `openai>=1.0.0`: OpenAI integration
- `anthropic>=0.18.0`: Anthropic integration
- `google-generativeai>=0.3.0`: Google Gemini integration
- `langchain>=0.1.0`: LangChain integration
- `langgraph>=0.0.1`: LangGraph integration

### Dev Dependencies

- `pytest>=7.0`: Testing framework
- `pytest-cov>=4.0`: Coverage reporting
- `pytest-asyncio>=0.21.0`: Async test support
- `ruff>=0.1.0`: Linting and formatting
- `mypy>=1.0`: Type checking
- `types-httpx`: Type stubs for httpx

## Troubleshooting

### Import Errors

If you see import errors, ensure:

1. Virtual environment is activated
2. Package is installed in editable mode: `pip install -e ".[dev]"`
3. Python path includes `src/` directory

### Type Checking Errors

Run `mypy tracium/` to see specific type errors. Some may require type annotations or `# type: ignore` comments.

### Test Failures

- Check that required environment variables are set
- Ensure all dependencies are installed
- Check for API key issues (use test keys for integration tests)

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run tests: `pytest`
4. Run linting: `ruff check tracium/`
5. Run type checking: `mypy tracium/`
6. Format code: `ruff format tracium/`
7. Commit with descriptive messages
8. Push and create a pull request

## Resources

- **Repository**: https://github.com/AntonijSimonovski/tracium-python
- **Issue Tracker**: https://github.com/AntonijSimonovski/tracium-python/issues
- **User Documentation**: https://docs.tracium.ai
- **PyPI Package**: https://pypi.org/project/tracium-sdk/

## License

MIT © Tracium
