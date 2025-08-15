# Contributing to DeepRL Neural

Thank you for your interest in contributing to DeepRL Neural! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Git
- Basic understanding of neural networks and reinforcement learning

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/deeprl-neural.git
   cd deeprl-neural
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## 🛠️ Development Workflow

### Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our coding standards

3. Write tests for your changes

4. Run the test suite:
   ```bash
   pytest tests/
   ```

5. Run code quality checks:
   ```bash
   black deeprl_neural/
   flake8 deeprl_neural/
   mypy deeprl_neural/
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

7. Push to your fork and create a pull request

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` adding or updating tests
- `refactor:` code refactoring
- `style:` formatting changes
- `chore:` maintenance tasks

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m "unit"
pytest tests/ -m "integration"
pytest tests/ -m "acceptance"

# Run with coverage
pytest tests/ --cov=deeprl_neural --cov-report=html
```

### Writing Tests

- Place unit tests in `tests/unit/`
- Place integration tests in `tests/integration/`
- Place acceptance tests in `tests/acceptance/`
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)

Example test:
```python
def test_neural_network_forward_pass():
    # Arrange
    network = NeuralNetwork([2, 3, 1])
    input_data = Matrix([[1.0, 0.5]])
    
    # Act
    output = network.forward(input_data)
    
    # Assert
    assert output.rows == 1
    assert output.cols == 1
```

## 📝 Documentation

### Code Documentation

- Use clear, descriptive docstrings
- Follow Google docstring format
- Include examples in docstrings when helpful
- Document all public methods and classes

Example:
```python
def forward(self, input_data: Matrix) -> Matrix:
    """
    Perform forward pass through the neural network.
    
    Args:
        input_data: Input matrix with shape (batch_size, input_size)
        
    Returns:
        Output matrix with shape (batch_size, output_size)
        
    Example:
        >>> network = NeuralNetwork([2, 3, 1])
        >>> input_data = Matrix([[1.0, 0.5]])
        >>> output = network.forward(input_data)
    """
```

### README Updates

When adding new features, update the relevant documentation:
- Main README.md
- API documentation
- Example notebooks

## 🎯 Code Style

### Python Style Guide

We follow PEP 8 with these additional guidelines:

- Maximum line length: 88 characters (Black default)
- Use type hints for all function parameters and return values
- Prefer descriptive variable names
- Use f-strings for string formatting

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import List, Dict, Optional

# Third-party imports (when available)
import numpy as np

# Local imports
from .matrix import Matrix
from .activation import sigmoid
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Minimal code example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, etc.

## 💡 Feature Requests

For new features:

1. **Description**: Clear description of the feature
2. **Motivation**: Why is this feature needed?
3. **API Design**: How should the API look?
4. **Implementation**: Any implementation details
5. **Tests**: How will this be tested?

## 📦 Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Publish to PyPI

## 🏗️ Architecture Guidelines

### Adding New Components

1. **Neural Network Components**: Add to `deeprl_neural/neural_network/`
2. **RL Algorithms**: Add to `deeprl_neural/reinforcement_learning/`
3. **Environments**: Add to `deeprl_neural/environment/`
4. **Utilities**: Add to `deeprl_neural/utils/`

### Design Principles

- **Pure Python**: No external dependencies
- **Educational**: Code should be readable and well-documented
- **Modular**: Components should be loosely coupled
- **Testable**: All code should be easily testable
- **Extensible**: Easy to add new algorithms/components

## 🤝 Code Review Process

1. All changes require review from a maintainer
2. Ensure tests pass and coverage is maintained
3. Code style checks must pass
4. Documentation must be updated if needed
5. Performance implications should be considered

## 📧 Getting Help

- Create an issue for bugs or feature requests
- Use GitHub Discussions for questions
- Email maintainers for security issues

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

Thank you for contributing to DeepRL Neural! 🎉
