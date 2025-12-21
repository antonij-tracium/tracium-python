# Contributing to Tracium Python SDK

Thank you for your interest in contributing to the Tracium Python SDK! We welcome contributions from the community.

## Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AntonijSimonovski/tracium-python.git
    cd tracium-python/tracium
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies in editable mode:**
    ```bash
    pip install -e ".[dev]"
    ```
    This installs the package in editable mode along with all development dependencies (pytest, ruff, mypy, etc.).

## Running Tests

To run the test suite, use `pytest`.

**Important:** Ensure your `PYTHONPATH` includes the `src` directory if you encounter import errors, although installing in editable mode (`pip install -e .`) should handle this automatically.

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=tracium --cov-report=term-missing
```

## Code Quality

We use `ruff` for linting and formatting, and `mypy` for static type checking.

```bash
# Format code
ruff format src/ tests/

# Check specific linting rules
ruff check src/ tests/

# Run static type checking
mypy src/
```

Please ensure all check pass before submitting a Pull Request.

## Pull Request Process

1.  Fork the repository and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  Ensure the test suite passes.
4.  Make sure your code lints.
5.  Issue that pull request!

## license

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
