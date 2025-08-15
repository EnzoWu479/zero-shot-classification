# Gitflow Documentation - Deep RL Neural Network Project

## 📋 Overview

Este projeto segue uma estratégia de gitflow adaptada para desenvolvimento de machine learning, com foco em:
- **TDD (Test-Driven Development)**: Todos os componentes são testados antes da implementação
- **Commits frequentes**: Cada funcionalidade ou correção é commitada separadamente
- **Branches organizadas**: Estrutura clara para desenvolvimento e releases

## 🌿 Branch Strategy

### Main Branches

#### `main`
- **Propósito**: Código de produção estável
- **Proteção**: Apenas aceita merges via Pull Request
- **Deploy**: Automaticamente deployável
- **Commits**: Apenas merges de `develop` ou hotfixes

#### `develop`
- **Propósito**: Branch de integração para desenvolvimento
- **Base**: Todas as features são desenvolvidas a partir desta branch
- **Estado**: Sempre deve estar em estado compilável e testável
- **Merge**: Features são merged aqui após aprovação

### Supporting Branches

#### Feature Branches (`feature/feature-name`)
- **Naming**: `feature/matrix-operations`, `feature/neural-network`, `feature/q-learning`
- **Base**: Criadas a partir de `develop`
- **Merge**: Volta para `develop` via Pull Request
- **Lifetime**: Temporárias, deletadas após merge
- **Examples**:
  - `feature/matrix-class`
  - `feature/activation-functions`
  - `feature/experience-replay`
  - `feature/dqn-agent`

#### Test Branches (`test/test-name`)
- **Naming**: `test/matrix-tests`, `test/integration-tests`
- **Propósito**: Desenvolvimento de testes específicos
- **Base**: Criadas a partir de `develop`
- **Merge**: Volta para `develop` após validação

#### Release Branches (`release/version`)
- **Naming**: `release/v1.0.0`, `release/v1.1.0`
- **Propósito**: Preparação para release
- **Base**: Criadas a partir de `develop`
- **Merge**: Para `main` e `develop` simultaneamente
- **Conteúdo**: Bug fixes, documentação, version bumping

#### Hotfix Branches (`hotfix/issue-name`)
- **Naming**: `hotfix/critical-bug`, `hotfix/memory-leak`
- **Propósito**: Correções urgentes em produção
- **Base**: Criadas a partir de `main`
- **Merge**: Para `main` e `develop` simultaneamente

## 🔄 TDD Workflow

### 1. Red Phase (Write Failing Test)
```bash
# Criar branch para nova feature
git checkout develop
git checkout -b feature/new-component

# Escrever teste que falha
touch tests/test_new_component.py
# Implementar teste
git add tests/test_new_component.py
git commit -m "test: add failing test for new component"
```

### 2. Green Phase (Make Test Pass)
```bash
# Implementar código mínimo para passar no teste
touch src/path/new_component.py
# Implementar funcionalidade
git add src/path/new_component.py
git commit -m "feat: implement new component to pass tests"
```

### 3. Refactor Phase (Improve Code)
```bash
# Refatorar código mantendo testes passando
git add .
git commit -m "refactor: improve new component implementation"
```

## 📝 Commit Message Convention

### Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- **feat**: Nova funcionalidade
- **fix**: Correção de bug
- **test**: Adição ou modificação de testes
- **refactor**: Refatoração sem mudança de funcionalidade
- **docs**: Mudanças na documentação
- **style**: Mudanças de formatação/estilo
- **perf**: Melhorias de performance
- **chore**: Tarefas de manutenção

### Scopes (Examples)
- **matrix**: Operações de matriz
- **neural**: Rede neural
- **rl**: Reinforcement learning
- **agent**: Agente de RL
- **env**: Ambiente
- **buffer**: Experience replay buffer
- **policy**: Políticas de exploração

### Examples
```bash
feat(matrix): implement matrix multiplication
test(neural): add tests for forward propagation
fix(agent): correct Q-value calculation bug
refactor(buffer): optimize memory usage in replay buffer
docs(readme): add installation instructions
perf(neural): vectorize activation function calculations
```

## 🚀 Development Workflow

### Starting New Feature
```bash
# 1. Update develop branch
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feature/feature-name

# 3. Write tests first (TDD)
# 4. Implement feature
# 5. Commit frequently

# 6. Push and create PR
git push origin feature/feature-name
```

### Daily Development
```bash
# Morning: sync with develop
git checkout develop
git pull origin develop
git checkout feature/current-feature
git merge develop

# Work: commit frequently
git add .
git commit -m "type(scope): brief description"

# Evening: push work
git push origin feature/current-feature
```

### Finishing Feature
```bash
# 1. Ensure all tests pass
python -m pytest tests/

# 2. Update documentation
git add .
git commit -m "docs(feature): update documentation"

# 3. Create Pull Request to develop
# 4. After review and approval, merge
# 5. Delete feature branch
git branch -d feature/feature-name
```

## 🧪 Testing Strategy

### Test Structure
```
tests/
├── unit/                   # Unit tests for individual components
│   ├── test_matrix.py
│   ├── test_activation.py
│   ├── test_neuron.py
│   └── test_layer.py
├── integration/            # Integration tests
│   ├── test_neural_network.py
│   ├── test_rl_agent.py
│   └── test_full_pipeline.py
├── acceptance/             # End-to-end tests
│   ├── test_training_pipeline.py
│   └── test_model_persistence.py
└── fixtures/               # Test data and mocks
    ├── sample_data.py
    └── mock_environments.py
```

### Test Commands
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/unit/test_matrix.py

# Run with coverage
python -m pytest tests/ --cov=src

# Run tests continuously during development
python -m pytest tests/ --watch
```

## 📊 Release Process

### 1. Prepare Release
```bash
# Create release branch
git checkout develop
git checkout -b release/v1.0.0

# Update version numbers
# Update CHANGELOG.md
# Final testing
git add .
git commit -m "chore(release): prepare v1.0.0"
```

### 2. Merge to Main
```bash
# Merge to main
git checkout main
git merge release/v1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0"

# Merge back to develop
git checkout develop
git merge release/v1.0.0

# Delete release branch
git branch -d release/v1.0.0
```

## 🔥 Hotfix Process

### Emergency Fix
```bash
# Create hotfix from main
git checkout main
git checkout -b hotfix/critical-bug

# Fix the issue
git add .
git commit -m "fix(critical): resolve memory leak in neural network"

# Merge to main
git checkout main
git merge hotfix/critical-bug
git tag -a v1.0.1 -m "Hotfix version 1.0.1"

# Merge to develop
git checkout develop
git merge hotfix/critical-bug

# Delete hotfix branch
git branch -d hotfix/critical-bug
```

## 📁 File Organization

### Project Structure with Git Considerations
```
zero-shot-classification/
├── .git/                   # Git repository
├── .gitignore             # Git ignore rules
├── .github/               # GitHub specific files
│   ├── workflows/         # CI/CD workflows
│   └── PULL_REQUEST_TEMPLATE.md
├── docs/                  # Documentation
│   ├── GITFLOW.md         # This file
│   ├── API.md
│   └── CONTRIBUTING.md
├── src/                   # Source code
├── tests/                 # Test files
├── examples/              # Example scripts
├── models/                # Saved models (gitignored)
├── logs/                  # Log files (gitignored)
├── CHANGELOG.md           # Version history
├── README.md              # Main documentation
└── requirements.txt       # Dependencies
```

## 🚫 Git Ignore Strategy

### Files to Ignore
- **Models**: `models/*.json`, `models/*.pkl`
- **Logs**: `logs/*.log`, `logs/*.txt`
- **Temp**: `*.tmp`, `*.temp`
- **IDE**: `.vscode/`, `.idea/`
- **Python**: `__pycache__/`, `*.pyc`, `.pytest_cache/`
- **OS**: `.DS_Store`, `Thumbs.db`

## 🏷️ Tagging Strategy

### Version Format: `vMAJOR.MINOR.PATCH`

#### MAJOR
- Breaking changes in API
- Major architecture changes
- New main features

#### MINOR
- New features (backward compatible)
- Performance improvements
- New algorithms

#### PATCH
- Bug fixes
- Documentation updates
- Small improvements

### Examples
- `v1.0.0`: Initial release with basic neural network
- `v1.1.0`: Add Q-learning agent
- `v1.1.1`: Fix bug in experience replay
- `v2.0.0`: Major API refactor

## 🤝 Collaboration Guidelines

### Code Review Checklist
- [ ] All tests pass
- [ ] Code follows project style
- [ ] Documentation is updated
- [ ] Performance is acceptable
- [ ] No security issues
- [ ] Git history is clean

### Branch Protection Rules
- **main**: Require PR, require status checks, no direct pushes
- **develop**: Require PR for external contributors
- **feature/***: No restrictions, encourage frequent pushes

## 📈 Continuous Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.11
      - name: Run tests
        run: python -m pytest tests/
```

## 📝 Documentation Standards

### Code Documentation
- **Docstrings**: All public functions and classes
- **Type hints**: Use typing module
- **Comments**: Complex logic and algorithms
- **README**: Keep updated with current features

### Git Documentation
- **Commit messages**: Follow convention
- **PR descriptions**: Include context and testing
- **Issue templates**: Standardize bug reports and feature requests

---

**Last Updated**: August 15, 2025  
**Version**: 1.0  
**Maintainer**: Deep RL Team
