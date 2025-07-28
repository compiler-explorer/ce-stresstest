# Compiler Explorer Stress Testing Framework

A comprehensive stress testing framework for Compiler Explorer's compilation workers. This framework tests the performance and scalability of SQS-based compilation worker infrastructure using realistic workloads and various load patterns.

## Features

- **Multiple Load Patterns**: Steady, burst, ramp, wave, and sustained high load patterns
- **Realistic Workloads**: CPU-intensive, memory-heavy, IO-intensive, quick, and error scenarios
- **Scaling Analysis**: Test performance across different worker instance counts
- **Real-time Monitoring**: Live dashboard with metrics and alerts
- **Comprehensive Reporting**: HTML reports with visualizations and analysis
- **Baseline Validation**: Automatic detection of performance regressions
- **CLI and Interactive Modes**: Flexible usage options

## Quick Start

### Installation

1. Install dependencies using Poetry:
```bash
cd stress-test
poetry install
```

2. Or using pip:
```bash
pip install -r requirements.txt  # If you had created this instead of pyproject.toml
```

### Basic Usage

#### Command Line Interface

Run a simple steady load test:
```bash
poetry run python run_stress_test.py steady --rps 10 --duration 300
```

Run a scaling test:
```bash
poetry run python run_stress_test.py scaling --instances 2,4,6,8 --rps-per-instance 5 --duration 180
```

Run a burst load test:
```bash
poetry run python run_stress_test.py burst --baseline-rps 5 --burst-rps 20 --duration 600
```

#### Interactive Mode

For guided configuration:
```bash
poetry run python run_stress_test.py interactive
```

#### Using the Framework Directly

```python
import asyncio
from src.framework import CompilerStressTest, TestConfiguration

async def main():
    config = TestConfiguration(
        endpoint="https://beta.compiler-explorer.com",
        compiler="g122",
        scenarios=["cpu_intensive", "simple", "memory_heavy"]
    )
    
    async with CompilerStressTest(config) as tester:
        # Run steady load test
        summary = await tester.steady_load_test(rps=10, duration_seconds=300)
        
        # Run scaling test
        results = await tester.scaling_test(
            instance_counts=[2, 4, 6, 8],
            rps_per_instance=5,
            duration_per_test=180
        )

asyncio.run(main())
```

## Available Test Patterns

### Steady Load
Maintains constant requests per second throughout the test duration.
```bash
poetry run python run_stress_test.py steady --rps 15 --duration 300
```

### Burst Load
Baseline traffic with periodic bursts of higher load.
```bash
poetry run python run_stress_test.py burst \
    --baseline-rps 5 \
    --burst-rps 25 \
    --duration 600 \
    --burst-duration 30 \
    --burst-interval 120
```

### Ramp Tests
Gradually increase (ramp up) or decrease (ramp down) load over time.
```bash
# Ramp up
poetry run python run_stress_test.py ramp --min-rps 1 --max-rps 20 --duration 300

# Ramp down
poetry run python run_stress_test.py ramp --min-rps 1 --max-rps 20 --duration 300 --ramp-down
```

### Wave Pattern
Sinusoidal load variation for testing response to cyclical traffic.
```bash
poetry run python run_stress_test.py wave \
    --base-rps 10 \
    --amplitude-rps 5 \
    --duration 600 \
    --period 300
```

### Scaling Tests
Test performance across different worker instance counts.
```bash
poetry run python run_stress_test.py scaling \
    --instances 2,4,6,8,10 \
    --rps-per-instance 5 \
    --duration 180
```

## Workload Scenarios

The framework includes several built-in workload scenarios:

### CPU-intensive Scenarios
- **cpu_intensive**: Heavy template metaprogramming with recursive instantiation
- Complex optimization with loop unrolling and LTO
- Recursive constexpr computation

### Memory-intensive Scenarios  
- **memory_heavy**: Large static array allocations and deep template instantiation
- Heavy macro expansion
- Deep template instantiation trees

### IO-intensive Scenarios
- **io_intensive**: Many standard library includes and large generated output
- Large debug symbol generation

### Quick Scenarios
- **simple**: Basic "Hello World" program
- Simple mathematical operations
- Basic class definitions

### Error Scenarios
- **error_syntax**: Syntax error compilation failures
- **error_template**: Template instantiation errors
- Missing header file errors

## Configuration

### Environment Variables
- `DEBUG_API=1`: Enable detailed API response logging
- `VERBOSE=1`: Enable verbose output

### Custom Scenarios
Add your own C++ files to `examples/workloads/` with metadata comments:

```cpp
// compile: -O3 -std=c++20
// baseline_min_ms: 500
// baseline_max_ms: 2000
// weight: 0.8
// description: Custom CPU-intensive workload

#include <iostream>
// Your code here...
```

### Configuration Files
Use YAML configuration files for complex test setups:

```yaml
# examples/configs/my_test.yaml
test_name: "custom_test"
endpoint: "https://beta.compiler-explorer.com"
compiler: "g122"
scenarios: ["cpu_intensive", "simple"]
test_patterns:
  - type: "steady"
    rps: 10.0
    duration_seconds: 300
```

## CLI Commands

### Available Commands

```bash
# List all available scenarios
poetry run python run_stress_test.py list-scenarios

# Generate report from results
poetry run python run_stress_test.py report results/test_results.json

# Interactive configuration mode
poetry run python run_stress_test.py interactive
```

### Global Options

Most commands support these options:
- `--endpoint`: API endpoint (default: https://beta.compiler-explorer.com)
- `--compiler`: Compiler ID (default: g122)
- `--concurrent`: Max concurrent requests (default: 50)
- `--scenarios`: Comma-separated scenario names
- `--workload-dir`: Custom workload directory
- `--results-dir`: Results output directory (default: results)
- `--no-dashboard`: Disable live dashboard
- `--test-name`: Custom test name

## Output and Reporting

### Results Structure
```
results/
├── test_name_timestamp_summary.json    # Test summary
├── test_name_timestamp_raw.json        # Raw request/response data
└── reports/
    ├── test_name_report.html           # HTML report
    └── test_name_charts/               # Generated charts
        ├── latency_distribution.png
        ├── throughput_timeline.png
        ├── success_rate_comparison.png
        ├── error_breakdown.png
        ├── scaling_analysis.png
        └── performance_heatmap.png
```

### Metrics Collected
- **Latency**: Mean, median, P95, P99, min, max response times
- **Throughput**: Requests per second, successful RPS
- **Success Rate**: Percentage of successful compilations
- **Error Breakdown**: Categorized error types and frequencies
- **Baseline Violations**: Responses outside expected time ranges
- **Scaling Efficiency**: Performance across instance counts

### HTML Reports
Comprehensive reports include:
- Executive summary with key metrics
- Interactive charts and visualizations
- Detailed test results breakdown
- Alerts and performance issues
- Scaling analysis (when applicable)

## Advanced Usage

### Custom Load Patterns
```python
from src.load_patterns import LoadPatternFactory

# Create custom pattern
pattern = LoadPatternFactory.create_pattern_from_config({
    "type": "burst",
    "baseline_rps": 5,
    "burst_rps": 30,
    "duration_seconds": 600,
    "burst_duration_seconds": 45,
    "burst_interval_seconds": 150
}, scenarios)
```

### Performance Analysis
```python
from src.metrics import PerformanceAnalyzer

# Detect regressions
regressions = PerformanceAnalyzer.detect_performance_regression(
    current_metrics, baseline_metrics, regression_threshold=0.15
)

# Analyze scaling efficiency
scaling_analysis = PerformanceAnalyzer.analyze_scaling_efficiency(
    scaling_results
)
```

### Custom Scenarios
```python
from src.scenarios import create_custom_scenario
from pathlib import Path

scenario = create_custom_scenario(
    name="my_custom_test",
    source_file=Path("my_code.cpp"),
    compiler_options="-O3 -std=c++20",
    baseline_min_ms=200,
    baseline_max_ms=1000,
    description="Custom optimization test"
)
```

## Best Practices

### Test Design
1. **Start Small**: Begin with low RPS and short durations
2. **Baseline First**: Establish performance baselines before stress testing
3. **Gradual Scaling**: Increase load gradually to find breaking points
4. **Monitor Resources**: Watch for API rate limits and server constraints
5. **Realistic Workloads**: Use scenarios that match your actual usage patterns

### Performance Monitoring
1. **Live Dashboard**: Use the live dashboard for real-time monitoring
2. **Baseline Validation**: Set appropriate baseline thresholds for your scenarios
3. **Alert Thresholds**: Configure alerts for critical performance metrics
4. **Regular Testing**: Run tests regularly to detect performance regressions

### Troubleshooting
1. **API Limits**: Reduce `--rate-limit-rps` if hitting API limits
2. **Timeouts**: Increase `--timeout` for slow compilations
3. **Memory Issues**: Reduce `--concurrent` for large workloads
4. **Network Issues**: Use `--debug-api` to diagnose API problems

## Examples

### Comprehensive Baseline Test
```bash
poetry run python run_stress_test.py steady \
    --rps 8 \
    --duration 600 \
    --scenarios "simple,cpu_intensive,memory_heavy,io_intensive" \
    --test-name "nightly_baseline" \
    --results-dir "results/baseline"
```

### Performance Regression Testing
```bash
# Run current build
poetry run python run_stress_test.py steady --rps 10 --duration 300 --test-name "current_build"

# Compare with baseline (manual process using reports)
poetry run python run_stress_test.py report results/current_build_*_summary.json
```

### Scaling Analysis
```bash
poetry run python run_stress_test.py scaling \
    --instances 2,4,6,8,10,12 \
    --rps-per-instance 4 \
    --duration 240 \
    --scenarios "simple,cpu_intensive" \
    --test-name "scaling_analysis_v2"
```

### High Load Stress Test
```bash
poetry run python run_stress_test.py burst \
    --baseline-rps 15 \
    --burst-rps 75 \
    --duration 900 \
    --burst-duration 60 \
    --burst-interval 180 \
    --concurrent 100 \
    --test-name "high_load_stress"
```

## Contributing

1. Add new workload scenarios to `examples/workloads/`
2. Extend load patterns in `src/load_patterns.py`
3. Improve metrics collection in `src/metrics.py`
4. Enhance reporting in `src/reporting.py`

## License

This framework is provided as-is for testing Compiler Explorer infrastructure. Use responsibly and respect API rate limits.