# Tests

This directory contains all unit and integration tests for the EV Supply Chain GAT project.

## Running Tests

### Install Test Dependencies
```bash
pip install pytest pytest-cov
```

### Run All Tests (Fast)
```bash
# From project root
pytest tests/ -v

# Or specific test file
pytest tests/test_market_data.py -v
```

### Run with Coverage Report
```bash
pytest tests/ --cov=utils --cov=models --cov-report=html
```

### Run Integration Tests (Slow - hits real APIs)
```bash
pytest tests/ -v -m integration
```

### Skip Integration Tests (Default)
```bash
pytest tests/ -v -m "not slow"
```

## Test Structure

- **Unit Tests**: Mock external APIs, test logic in isolation
- **Integration Tests**: Hit real APIs (marked with `@pytest.mark.integration`)

## Test Markers

- `@pytest.mark.slow` - Tests that take >5 seconds
- `@pytest.mark.integration` - Tests that require external APIs

## Coverage Goals

- **Minimum**: 85% code coverage
- **Target**: 90%+ code coverage

## Writing New Tests

Follow this pattern:

```python
import pytest
from your_module import YourClass

class TestYourClass:
    @pytest.fixture
    def your_fixture(self):
        """Setup test data"""
        return YourClass()
    
    def test_something(self, your_fixture):
        """Test description"""
        result = your_fixture.method()
        assert result == expected_value
```

## Current Test Files

- `test_market_data.py` - Tests for Market Data Collector
- (More will be added as we build other components)
