# Compiler Explorer Compilation Worker Stress Test Framework

## Project Overview
This is a comprehensive stress testing framework designed to test the performance and scalability of Compiler Explorer's SQS-based compilation worker infrastructure. The framework provides API-based testing with multiple workload scenarios, load patterns, and comprehensive metrics collection.

## Key Components

### Dependencies and Environment
- **Python**: 3.10+ required
- **Package Manager**: Poetry (required - do not use pip)
- **Key Libraries**: aiohttp, asyncio-throttle, pandas, matplotlib, seaborn, rich, click

### Main Modules
- `src/api_client.py`: Compiler Explorer API client with rate limiting and retry logic
- `src/scenarios.py`: Workload scenario definitions and library comment parsing
- `src/load_patterns.py`: Load generation patterns (steady, burst, ramp, wave, sustained)
- `src/metrics.py`: Metrics collection and analysis
- `src/reporting.py`: HTML reporting with visualizations
- `src/framework.py`: Main stress test framework orchestration

### Workload Examples
Located in `examples/workloads/`, all scenarios are stored as physical `.cpp` files with metadata comments:
- `baseline_test.cpp`: Simple baseline test
- `cpu_intensive.cpp`: CTRE-based regex processing
- `boost_heavy.cpp`: Heavy Boost library compilation
- `memory_heavy.cpp`, `io_intensive.cpp`: Various load types
- Error scenarios for testing failure cases

## Important Configuration Notes

### Library Support
Libraries are specified in workload files using comment format:
```cpp
// lib: library_name/version_id
```
Examples:
- `// lib: ctre/trunk`
- `// lib: boost/187` (for Boost 1.87.0)

### API Endpoints
- **Production**: `https://compiler-explorer.com` (default)
- **Beta Testing**: `https://compiler-explorer.com/beta` (when testing beta features)
- **Avoid**: `https://beta.compiler-explorer.com` (misleading name, not actually beta)

### Compiler Configuration
- Default compiler: `g151` (GCC 15.1)
- All requests include `bypassCache: 1` to avoid caching speedups
- User-Agent: `CE-StressTest/1.0` for request identification

## Usage Commands

### Running Tests
```bash
# Use Poetry for all operations
poetry install
poetry run python run_stress_test.py steady --rps 2.0 --duration 30 --scenarios boost_heavy,cpu_intensive

# Common test patterns
poetry run python run_stress_test.py burst --peak-rps 10.0 --duration 60
poetry run python run_stress_test.py ramp --start-rps 1.0 --end-rps 5.0 --duration 120
```

### Code Quality
```bash
poetry run black src/ tests/
poetry run ruff check src/ tests/
poetry run mypy src/
```

## Type Annotations
All code uses Python 3.10+ type annotations. Type stubs are included for pandas and seaborn to resolve mypy issues.

## Performance Characteristics

### Expected Baseline Times
- Simple baseline: 100-500ms
- CPU intensive (CTRE): 1000-3000ms  
- Boost heavy: 4000-8000ms (can be much longer due to scheduling)

### Production Scheduling Issues
Current production scheduling can cause significant delays when requests queue behind long-running compilations (e.g., a 4s request can take 90s+ if queued behind two 30s requests). The new scheduling system should address these bottlenecks.

## Code Storage Policy
- **Always use physical files** for C++ code examples - never inline strings in Python
- Store all workload scenarios as separate `.cpp` files in `examples/workloads/`
- Parse metadata from comment headers in the scenario files

## Architecture Notes
- Async Python with aiohttp for concurrent HTTP requests
- Rate limiting with asyncio-throttle
- Comprehensive error handling and retry logic
- Real-time metrics collection with percentile calculations
- HTML report generation with matplotlib/seaborn visualizations
- Rich terminal UI for progress tracking