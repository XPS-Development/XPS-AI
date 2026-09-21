# Tests

## Run all

```bash
uv run pytest
```

## With coverage

```bash
uv run pytest --cov --cov-report=term-missing
```

## UI tests

Require `QApplication`; CI sets `QT_QPA_PLATFORM=offscreen`.

```bash
uv run pytest tests/ui/ -v
```

## Optimization benchmarks

```bash
# All tests (quality + latency + stress)
uv run pytest tests/core/fitting/benchmark_optimization.py -v

# Benchmarks only (skips quality tests)
uv run pytest tests/core/fitting/benchmark_optimization.py --benchmark-only

# Filter by category
uv run pytest tests/core/fitting/benchmark_optimization.py -k "quality" -v
uv run pytest tests/core/fitting/benchmark_optimization.py -k "latency" --benchmark-only
uv run pytest tests/core/fitting/benchmark_optimization.py -k "stress" --benchmark-only
```
