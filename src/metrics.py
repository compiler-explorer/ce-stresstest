"""
Metrics collection and analysis system for stress testing
"""

import time
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

from .api_client import CompilationResult, CompilationStatus
from .scenarios import ScenarioConfig


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    current_value: float
    threshold_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """Summary of metrics for a test run"""

    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float

    # Latency metrics (in milliseconds)
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    min_successful_latency_ms: Optional[float]
    min_failed_latency_ms: Optional[float]

    # Throughput metrics
    requests_per_second: float
    successful_rps: float

    # Error breakdown
    error_breakdown: Dict[str, int]

    # Duration
    test_duration_seconds: float

    # Baseline violations
    baseline_violations: int
    baseline_violations_too_fast: int
    baseline_violations_too_slow: int
    total_alerts: int

    # Scenario breakdown
    scenario_breakdown: Dict[str, Dict[str, Any]]


class MetricsCollector:
    """Collects and analyzes compilation metrics in real-time"""

    def __init__(self, scenarios: List[ScenarioConfig]):
        self.scenarios = {s.name: s for s in scenarios}
        self.results: List[CompilationResult] = []
        self.alerts: List[Alert] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Real-time metrics tracking
        self.latency_window: deque[float] = deque(maxlen=1000)  # Last 1000 requests
        self.success_window: deque[int] = deque(maxlen=1000)  # Last 1000 requests
        self.throughput_window: deque[float] = deque(maxlen=60)  # Last 60 seconds

        # Counters by scenario
        self.scenario_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "latencies": [],
                "baseline_violations": 0,
                "baseline_violations_too_fast": 0,
                "baseline_violations_too_slow": 0,
            }
        )

        # Time series data for visualization
        self.time_series_data: list[dict[str, Any]] = []
        self._last_metrics_update = 0.0

    def start_collection(self) -> None:
        """Start metrics collection"""
        self.start_time = time.time()
        self._last_metrics_update = self.start_time

    def stop_collection(self) -> None:
        """Stop metrics collection"""
        self.end_time = time.time()

    def record_result(self, result: CompilationResult, scenario_name: str) -> None:
        """Record a compilation result and update metrics"""
        self.results.append(result)

        # Update scenario stats
        stats = self.scenario_stats[scenario_name]
        stats["total"] += 1
        stats["latencies"].append(result.total_time_ms)

        if result.status == CompilationStatus.SUCCESS:
            stats["successful"] += 1
            self.success_window.append(1)
        else:
            stats["failed"] += 1
            self.success_window.append(0)

        # Update real-time windows
        self.latency_window.append(result.total_time_ms)

        # Check baseline violations
        scenario_config = self.scenarios.get(scenario_name)
        if scenario_config:
            if result.total_time_ms < scenario_config.baseline_min_ms:
                stats["baseline_violations"] += 1
                stats["baseline_violations_too_fast"] += 1
                self._create_baseline_alert(result, scenario_config)
            elif result.total_time_ms > scenario_config.baseline_max_ms:
                stats["baseline_violations"] += 1
                stats["baseline_violations_too_slow"] += 1
                self._create_baseline_alert(result, scenario_config)

        # Update time series data periodically
        current_time = time.time()
        if current_time - self._last_metrics_update >= 1.0:  # Every second
            self._update_time_series()
            self._last_metrics_update = current_time

    def _create_baseline_alert(
        self, result: CompilationResult, scenario: ScenarioConfig
    ) -> None:
        """Create an alert for baseline violation"""
        if result.total_time_ms < scenario.baseline_min_ms:
            level = AlertLevel.WARNING
            message = f"Response time too fast for {scenario.name}: {result.total_time_ms:.1f}ms < {scenario.baseline_min_ms}ms"
            threshold = scenario.baseline_min_ms
        else:
            level = AlertLevel.CRITICAL
            message = f"Response time too slow for {scenario.name}: {result.total_time_ms:.1f}ms > {scenario.baseline_max_ms}ms"
            threshold = scenario.baseline_max_ms

        alert = Alert(
            level=level,
            message=message,
            timestamp=result.timestamp,
            metric_name="response_time_baseline",
            current_value=result.total_time_ms,
            threshold_value=threshold,
            metadata={
                "scenario": scenario.name,
                "request_id": result.request_id,
                "status": result.status.value,
            },
        )
        self.alerts.append(alert)

    def _update_time_series(self) -> None:
        """Update time series data for visualization"""
        current_time = time.time()

        # Calculate current metrics
        current_rps = len(
            [r for r in self.results if current_time - r.timestamp <= 1.0]
        )

        current_success_rate = (
            statistics.mean(self.success_window) if self.success_window else 0
        )

        current_p95 = (
            statistics.quantiles(list(self.latency_window), n=20)[18]
            if len(self.latency_window) >= 20
            else 0
        )

        self.time_series_data.append(
            {
                "timestamp": current_time,
                "rps": current_rps,
                "success_rate": current_success_rate,
                "p95_latency_ms": current_p95,
                "active_requests": len(self.latency_window),
            }
        )

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        if not self.results:
            return {
                "rps": 0,
                "success_rate": 0,
                "p95_latency_ms": 0,
                "total_requests": 0,
                "alerts": 0,
            }

        current_time = time.time()
        recent_results = [
            r for r in self.results if current_time - r.timestamp <= 60.0
        ]  # Last minute

        if not recent_results:
            return {
                "rps": 0,
                "success_rate": 0,
                "p95_latency_ms": 0,
                "total_requests": len(self.results),
                "alerts": len(self.alerts),
            }

        success_count = sum(
            1 for r in recent_results if r.status == CompilationStatus.SUCCESS
        )
        success_rate = success_count / len(recent_results)

        # Calculate RPS over last 60 seconds
        rps = len(recent_results) / min(
            60.0, current_time - self.start_time if self.start_time else 60.0
        )

        # Calculate P95 latency
        latencies = [r.total_time_ms for r in recent_results]
        p95_latency = (
            statistics.quantiles(latencies, n=20)[18]
            if len(latencies) >= 20
            else statistics.mean(latencies)
            if latencies
            else 0
        )

        return {
            "rps": rps,
            "success_rate": success_rate,
            "p95_latency_ms": p95_latency,
            "total_requests": len(self.results),
            "alerts": len(
                [a for a in self.alerts if current_time - a.timestamp <= 300]
            ),  # Last 5 min
        }

    def generate_summary(self) -> MetricsSummary:
        """Generate comprehensive metrics summary"""
        if not self.results:
            return MetricsSummary(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate=0.0,
                mean_latency_ms=0.0,
                median_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                min_latency_ms=0.0,
                max_latency_ms=0.0,
                min_successful_latency_ms=None,
                min_failed_latency_ms=None,
                requests_per_second=0.0,
                successful_rps=0.0,
                error_breakdown={},
                test_duration_seconds=0.0,
                baseline_violations=0,
                baseline_violations_too_fast=0,
                baseline_violations_too_slow=0,
                total_alerts=0,
                scenario_breakdown={},
            )

        # Basic counts
        total_requests = len(self.results)
        successful_requests = sum(
            1 for r in self.results if r.status == CompilationStatus.SUCCESS
        )
        failed_requests = total_requests - successful_requests
        success_rate = successful_requests / total_requests if total_requests > 0 else 0

        # Latency analysis
        latencies = [r.total_time_ms for r in self.results]
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        # Separate latencies by success/failure
        successful_latencies = [r.total_time_ms for r in self.results if r.status == CompilationStatus.SUCCESS]
        failed_latencies = [r.total_time_ms for r in self.results if r.status != CompilationStatus.SUCCESS]
        
        min_successful_latency = min(successful_latencies) if successful_latencies else None
        min_failed_latency = min(failed_latencies) if failed_latencies else None

        # Percentiles
        if len(latencies) >= 100:
            quantiles = statistics.quantiles(latencies, n=100)
            p95_latency = quantiles[94]  # 95th percentile
            p99_latency = quantiles[98]  # 99th percentile
        elif len(latencies) >= 20:
            quantiles = statistics.quantiles(latencies, n=20)
            p95_latency = quantiles[18]  # 95th percentile (19th quantile out of 19)
            p99_latency = quantiles[-1]  # Use the highest quantile as approximation
        else:
            p95_latency = max_latency
            p99_latency = max_latency

        # Throughput
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        requests_per_second = total_requests / duration if duration > 0 else 0
        successful_rps = successful_requests / duration if duration > 0 else 0

        # Error breakdown
        error_breakdown: dict[str, int] = defaultdict(int)
        for result in self.results:
            if result.status != CompilationStatus.SUCCESS:
                error_breakdown[result.status.value] += 1

        # Baseline violations
        baseline_violations = sum(
            stats["baseline_violations"] for stats in self.scenario_stats.values()
        )
        baseline_violations_too_fast = sum(
            stats["baseline_violations_too_fast"] for stats in self.scenario_stats.values()
        )
        baseline_violations_too_slow = sum(
            stats["baseline_violations_too_slow"] for stats in self.scenario_stats.values()
        )

        # Scenario breakdown
        scenario_breakdown = {}
        for scenario_name, stats in self.scenario_stats.items():
            if stats["latencies"]:
                scenario_breakdown[scenario_name] = {
                    "total_requests": stats["total"],
                    "successful_requests": stats["successful"],
                    "failed_requests": stats["failed"],
                    "success_rate": stats["successful"] / stats["total"],
                    "mean_latency_ms": statistics.mean(stats["latencies"]),
                    "median_latency_ms": statistics.median(stats["latencies"]),
                    "baseline_violations": stats["baseline_violations"],
                    "baseline_violations_too_fast": stats["baseline_violations_too_fast"],
                    "baseline_violations_too_slow": stats["baseline_violations_too_slow"],
                }

        return MetricsSummary(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            success_rate=success_rate,
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            min_successful_latency_ms=min_successful_latency,
            min_failed_latency_ms=min_failed_latency,
            requests_per_second=requests_per_second,
            successful_rps=successful_rps,
            error_breakdown=dict(error_breakdown),
            test_duration_seconds=duration,
            baseline_violations=baseline_violations,
            baseline_violations_too_fast=baseline_violations_too_fast,
            baseline_violations_too_slow=baseline_violations_too_slow,
            total_alerts=len(self.alerts),
            scenario_breakdown=scenario_breakdown,
        )

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        since_timestamp: Optional[float] = None,
    ) -> List[Alert]:
        """Get alerts, optionally filtered by level and time"""
        alerts = self.alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        if since_timestamp:
            alerts = [a for a in alerts if a.timestamp >= since_timestamp]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def export_raw_data(self) -> Dict[str, Any]:
        """Export raw data for external analysis"""
        return {
            "results": [
                {
                    "timestamp": r.timestamp,
                    "request_id": r.request_id,
                    "status": r.status.value,
                    "total_time_ms": r.total_time_ms,
                    "compile_time_ms": r.compile_time_ms,
                    "execution_time_ms": r.execution_time_ms,
                    "exit_code": r.exit_code,
                    "error_message": r.error_message,
                }
                for r in self.results
            ],
            "alerts": [
                {
                    "level": a.level.value,
                    "message": a.message,
                    "timestamp": a.timestamp,
                    "metric_name": a.metric_name,
                    "current_value": a.current_value,
                    "threshold_value": a.threshold_value,
                    "metadata": a.metadata,
                }
                for a in self.alerts
            ],
            "time_series": self.time_series_data,
            "scenario_stats": dict(self.scenario_stats),
            "test_metadata": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "total_scenarios": len(self.scenarios),
            },
        }

    def save_to_file(self, filepath: str) -> None:
        """Save metrics data to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.export_raw_data(), f, indent=2)


class PerformanceAnalyzer:
    """Analyzes performance trends and detects anomalies"""

    @staticmethod
    def detect_performance_regression(
        current_metrics: MetricsSummary,
        baseline_metrics: MetricsSummary,
        regression_threshold: float = 0.2,  # 20% degradation
    ) -> List[str]:
        """Detect performance regressions compared to baseline"""
        regressions = []

        # Check latency regression
        if current_metrics.mean_latency_ms > baseline_metrics.mean_latency_ms * (
            1 + regression_threshold
        ):
            regressions.append(
                f"Mean latency regression: {current_metrics.mean_latency_ms:.1f}ms "
                f"vs baseline {baseline_metrics.mean_latency_ms:.1f}ms "
                f"({(current_metrics.mean_latency_ms / baseline_metrics.mean_latency_ms - 1) * 100:.1f}% increase)"
            )

        # Check throughput regression
        if (
            current_metrics.requests_per_second
            < baseline_metrics.requests_per_second * (1 - regression_threshold)
        ):
            regressions.append(
                f"Throughput regression: {current_metrics.requests_per_second:.1f} RPS "
                f"vs baseline {baseline_metrics.requests_per_second:.1f} RPS "
                f"({(1 - current_metrics.requests_per_second / baseline_metrics.requests_per_second) * 100:.1f}% decrease)"
            )

        # Check success rate regression
        if (
            current_metrics.success_rate
            < baseline_metrics.success_rate - regression_threshold
        ):
            regressions.append(
                f"Success rate regression: {current_metrics.success_rate:.1%} "
                f"vs baseline {baseline_metrics.success_rate:.1%}"
            )

        return regressions

    @staticmethod
    def analyze_scaling_efficiency(
        scaling_results: List[Tuple[int, MetricsSummary]]
    ) -> Dict[str, Any]:
        """Analyze scaling efficiency across different instance counts"""
        if len(scaling_results) < 2:
            return {"error": "Need at least 2 data points for scaling analysis"}

        # Sort by instance count
        scaling_results.sort(key=lambda x: x[0])

        # Calculate scaling efficiency
        base_instances, base_metrics = scaling_results[0]
        base_throughput = base_metrics.requests_per_second

        scaling_analysis: Dict[str, Any] = {
            "base_instances": base_instances,
            "base_throughput_rps": base_throughput,
            "scaling_points": [],
            "ideal_scaling_slope": base_throughput,  # Perfect linear scaling
            "actual_scaling_efficiency": [],
        }

        for instances, metrics in scaling_results:
            expected_throughput = base_throughput * (instances / base_instances)
            actual_throughput = metrics.requests_per_second
            efficiency = (
                actual_throughput / expected_throughput
                if expected_throughput > 0
                else 0
            )

            scaling_analysis["scaling_points"].append(
                {
                    "instances": instances,
                    "actual_throughput_rps": actual_throughput,
                    "expected_throughput_rps": expected_throughput,
                    "scaling_efficiency": efficiency,
                    "mean_latency_ms": metrics.mean_latency_ms,
                    "success_rate": metrics.success_rate,
                }
            )

            scaling_analysis["actual_scaling_efficiency"].append(efficiency)

        # Calculate overall scaling efficiency
        scaling_analysis["average_efficiency"] = statistics.mean(
            scaling_analysis["actual_scaling_efficiency"]
        )

        return scaling_analysis
