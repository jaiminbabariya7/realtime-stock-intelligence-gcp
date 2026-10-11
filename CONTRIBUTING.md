# Contributing to Stock Intelligence Platform

Thank you for your interest in contributing! This document outlines the process for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/realtime-stock-intelligence-gcp`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Install dev dependencies: `make install-dev`

## Development Workflow

```bash
# Install dependencies
make install-dev

# Run linting
make lint

# Auto-format code
make format

# Run tests
make test
```

## Code Standards

- **Python style**: [Black](https://black.readthedocs.io/) formatter, line length 100
- **Import order**: [isort](https://pycqa.github.io/isort/) with Black profile
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style docstrings for all modules, classes, and functions
- **Tests**: All new features must include unit tests (target >80% coverage)

## Submitting a Pull Request

1. Ensure all tests pass: `make test`
2. Ensure no lint errors: `make lint`
3. Write a clear PR description explaining the change and motivation
4. Reference any related issues

## Reporting Issues

Use GitHub Issues with the appropriate label: `bug`, `enhancement`, or `question`.
