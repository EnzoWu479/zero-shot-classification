# Pytest Configuration and Commands for Deep RL Neural Network Project

## Installation
pytest>=8.0.0
pytest-cov>=6.0.0

## Basic Commands

### Run all tests
uv run pytest tests/ --ignore=tests/unit/test_persistence_pytest.py

### Run with coverage
uv run pytest tests/ --ignore=tests/unit/test_persistence_pytest.py --cov=src --cov-report=term-missing

### Run by test type
uv run pytest tests/ -m "unit" --ignore=tests/unit/test_persistence_pytest.py
uv run pytest tests/ -m "integration"
uv run pytest tests/ -m "acceptance"

### Run by component
uv run pytest tests/ -m "matrix"
uv run pytest tests/ -m "neural"
uv run pytest tests/ -m "rl"

### Quick test (no slow tests)
uv run pytest tests/ -m "not slow" --ignore=tests/unit/test_persistence_pytest.py

### Verbose output
uv run pytest tests/ -v --ignore=tests/unit/test_persistence_pytest.py

### Run specific test file
uv run pytest tests/test_basic_implementation.py -v

### Run specific test
uv run pytest tests/test_basic_implementation.py::test_matrix_creation -v

## Test Markers Available
- unit: Unit tests for individual components
- integration: Integration tests for component interaction  
- acceptance: End-to-end acceptance tests
- slow: Tests that take longer to run
- neural: Tests related to neural network components
- rl: Tests related to reinforcement learning
- matrix: Tests related to matrix operations

## Coverage Reports
- Terminal: --cov-report=term-missing
- HTML: --cov-report=html (creates htmlcov/ directory)
- XML: --cov-report=xml (creates coverage.xml)

## Current Test Status
- ✅ 108 tests passing
- ❌ 2 tests failing (early stopping logic)
- ⚠️ 1 warning (return vs assert in acceptance test)
- 📊 67% code coverage

## Test Structure
```
tests/
├── acceptance/          # End-to-end tests
├── integration/         # Integration tests
├── unit/               # Unit tests
├── test_*.py           # Legacy test files (being converted)
└── conftest.py         # Pytest configuration
```

## Performance
- Unit tests: ~4 seconds
- Integration tests: ~30 seconds  
- Acceptance tests: ~2-3 minutes
- Full suite: ~10-15 minutes
