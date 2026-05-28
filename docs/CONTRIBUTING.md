# Contributing to Customer Churn Prediction

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome people of all backgrounds and skill levels
- Focus on constructive feedback
- Report issues respectfully and professionally

## How to Contribute

### 1. Report Issues

Found a bug or have a suggestion? Please create an issue with:
- Clear, descriptive title
- Detailed description of the problem
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### 2. Fork and Clone

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
git remote add upstream https://github.com/original-owner/customer-churn-prediction.git
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or for bug fixes:
git checkout -b fix/bug-description
```

### 4. Make Your Changes

#### Code Style

- Follow PEP 8 guidelines
- Use 4 spaces for indentation
- Keep line length ≤ 88 characters
- Use descriptive variable names

#### Documentation

- Add docstrings to all functions and classes
- Include type hints
- Add comments for complex logic
- Update README if adding new features

#### Example Function

```python
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Computes accuracy, precision, recall, and F1-score for binary classification.
    
    Args:
        y_true (np.ndarray): True labels (0 or 1)
        y_pred (np.ndarray): Predicted labels (0 or 1)
        
    Returns:
        Dict[str, float]: Dictionary with metrics
        
    Raises:
        ValueError: If inputs have different lengths
        
    Examples:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> metrics['accuracy']
        0.75
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    # Implementation here
    metrics = {}
    return metrics
```

### 5. Test Your Changes

```bash
# Run any existing tests
pytest

# Test your specific changes
python -m pytest tests/test_your_module.py -v

# Check code style
flake8 src/
pylint src/
```

### 6. Commit and Push

```bash
# Stage your changes
git add .

# Write a clear commit message
git commit -m "Add feature: description of changes"

# Push to your fork
git push origin feature/your-feature-name
```

### 7. Create a Pull Request

1. Go to GitHub and create a pull request
2. Write a clear title and description
3. Reference any related issues (#123)
4. Ensure CI/CD checks pass

## Pull Request Guidelines

### PR Title
- Be descriptive and concise
- Use conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, etc.

Examples:
- `feat: add SHAP values for model explainability`
- `fix: handle missing values in TotalCharges column`
- `docs: update README with new features`

### PR Description

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Related Issues
Fixes #123

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
How was this tested?
- [ ] Unit tests added
- [ ] Manual testing

## Checklist
- [ ] Code follows PEP 8 style
- [ ] Docstrings added
- [ ] Comments added for complex logic
- [ ] README updated if needed
- [ ] Tests pass locally
```

## Feature Request Template

```markdown
## Description
What feature would you like to add?

## Use Case
Why do you need this feature?

## Suggested Solution
How should it be implemented?

## Alternative Solutions
Any other ways to solve this?

## Additional Context
Any other relevant information?
```

## Areas for Contribution

### Code Improvements
- [ ] Add more models (XGBoost, LightGBM, SVM)
- [ ] Implement SHAP for model explainability
- [ ] Add hyperparameter optimization
- [ ] Improve code efficiency
- [ ] Add unit tests

### Data & Features
- [ ] Add feature selection techniques
- [ ] Handle class imbalance (SMOTE, class weights)
- [ ] Create new engineered features
- [ ] Document feature creation rationale

### Documentation
- [ ] Improve README sections
- [ ] Add more code examples
- [ ] Create tutorials
- [ ] Fix typos and grammar

### Infrastructure
- [ ] Deploy to cloud platforms
- [ ] Create Docker configuration
- [ ] Set up CI/CD pipeline
- [ ] Add monitoring and logging

## Development Environment

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest flake8 pylint black
```

### Useful Commands

```bash
# Format code
black src/ notebooks/

# Check style
flake8 src/
pylint src/

# Run tests
pytest -v
pytest src/ --cov

# Build documentation
sphinx-build -b html docs/ docs/_build/
```

## Documentation Standards

### Module Docstring
```python
"""
Module description: what this module does.

This module handles:
- Task 1
- Task 2
- Task 3
"""
```

### Function Docstring
```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description (one line).
    
    Longer description if needed. Explain what the function does,
    why it's useful, and any important details.
    
    Args:
        param1 (str): Description of param1
        param2 (int): Description of param2
        
    Returns:
        bool: Description of return value
        
    Raises:
        ValueError: When this condition occurs
        TypeError: When that condition occurs
        
    Examples:
        >>> result = function_name("test", 42)
        >>> result
        True
    """
```

### Class Docstring
```python
class MyClass:
    """
    Brief description of the class.
    
    Longer description explaining the class purpose, usage, and behavior.
    
    Attributes:
        attr1 (str): Description of attribute 1
        attr2 (int): Description of attribute 2
        
    Examples:
        >>> obj = MyClass("value")
        >>> obj.method()
    """
```

## Communication

- **Questions?** Open an issue with tag `question`
- **Found a bug?** Create an issue with tag `bug`
- **Have an idea?** Create an issue with tag `enhancement`
- **Need help?** Ask in discussions or contact maintainers

## Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Mentioned in release notes
- Thanked in project README

## Questions?

Feel free to reach out:
- GitHub Issues
- Email: contact@example.com
- Discord: [Join our server]

---

Thank you for contributing! 🎉
