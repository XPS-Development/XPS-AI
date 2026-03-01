# All tests (quality + latency + stress)
pytest tests/tools/benchmark_optimization.py -v

# Benchmarks only (skips quality tests)
pytest tests/tools/benchmark_optimization.py --benchmark-only

# Filter by category
pytest tests/tools/benchmark_optimization.py -k "quality" -v
pytest tests/tools/benchmark_optimization.py -k "latency" --benchmark-only
pytest tests/tools/benchmark_optimization.py -k "stress" --benchmark-only