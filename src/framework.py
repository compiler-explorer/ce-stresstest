"""
Main stress test framework for Compiler Explorer compilation workers
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .api_client import CompilerExplorerClient, CompilationResult
from .scenarios import WorkloadScenarios, ScenarioConfig
from .load_patterns import LoadPattern, LoadPatternFactory, LoadEvent
from .metrics import MetricsCollector, MetricsSummary, PerformanceAnalyzer


@dataclass
class TestConfiguration:
    """Configuration for a stress test"""

    endpoint: str = "https://beta.compiler-explorer.com"
    compiler: str = "g122"
    max_concurrent_requests: int = 50
    request_timeout_seconds: int = 30
    rate_limit_rps: float = 10.0
    scenarios: Optional[List[str]] = None  # Scenario names to use
    workload_dir: Optional[str] = None
    results_dir: str = "results"
    enable_live_dashboard: bool = True
    save_raw_data: bool = True


class CompilerStressTest:
    """Main stress testing framework"""

    def __init__(self, config: TestConfiguration):
        self.config = config
        self.console = Console()
        self.workload_scenarios = WorkloadScenarios(
            Path(config.workload_dir) if config.workload_dir else None
        )
        self.metrics_collector: Optional[MetricsCollector] = None
        self.client: Optional[CompilerExplorerClient] = None

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Ensure results directory exists
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    async def __aenter__(self) -> StressTestFramework:  # type: ignore
        self.client = CompilerExplorerClient(
            base_url=self.config.endpoint,
            max_requests_per_second=self.config.rate_limit_rps,
            timeout_seconds=self.config.request_timeout_seconds,
        )
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    def _load_scenarios(self) -> List[ScenarioConfig]:
        """Load scenarios based on configuration"""
        if self.config.scenarios:
            # Load specific scenarios by name
            scenarios = []
            for scenario_name in self.config.scenarios:
                scenario = self.workload_scenarios.get_scenario_by_name(scenario_name)
                if scenario:
                    scenarios.append(scenario)
                else:
                    self.logger.warning(f"Scenario '{scenario_name}' not found")
            return scenarios
        else:
            # Load all available scenarios
            return self.workload_scenarios.load_all_scenarios()

    async def steady_load_test(
        self, rps: float, duration_seconds: int, test_name: Optional[str] = None
    ) -> MetricsSummary:
        """Run a steady load test"""
        scenarios = self._load_scenarios()
        if not scenarios:
            raise ValueError("No scenarios available for testing")

        pattern = LoadPatternFactory.create_steady_load(
            scenarios, duration_seconds, rps
        )
        return await self._execute_load_test(
            pattern, test_name or f"steady_{rps}rps_{duration_seconds}s"
        )

    async def burst_load_test(
        self,
        baseline_rps: float,
        burst_rps: float,
        duration_seconds: int,
        burst_duration_seconds: int = 30,
        burst_interval_seconds: int = 120,
        test_name: Optional[str] = None,
    ) -> MetricsSummary:
        """Run a burst load test"""
        scenarios = self._load_scenarios()
        if not scenarios:
            raise ValueError("No scenarios available for testing")

        pattern = LoadPatternFactory.create_burst_load(
            scenarios,
            duration_seconds,
            baseline_rps,
            burst_rps,
            burst_duration_seconds,
            burst_interval_seconds,
        )
        return await self._execute_load_test(
            pattern,
            test_name or f"burst_{baseline_rps}to{burst_rps}rps_{duration_seconds}s",
        )

    async def ramp_test(
        self,
        min_rps: float,
        max_rps: float,
        duration_seconds: int,
        ramp_up: bool = True,
        test_name: Optional[str] = None,
    ) -> MetricsSummary:
        """Run a ramp up or ramp down test"""
        scenarios = self._load_scenarios()
        if not scenarios:
            raise ValueError("No scenarios available for testing")

        if ramp_up:
            pattern: LoadPattern = LoadPatternFactory.create_ramp_up(
                scenarios, duration_seconds, min_rps, max_rps
            )
            default_name = f"ramp_up_{min_rps}to{max_rps}rps_{duration_seconds}s"
        else:
            pattern = LoadPatternFactory.create_ramp_down(
                scenarios, duration_seconds, max_rps, min_rps
            )
            default_name = f"ramp_down_{max_rps}to{min_rps}rps_{duration_seconds}s"

        return await self._execute_load_test(pattern, test_name or default_name)

    async def wave_test(
        self,
        base_rps: float,
        amplitude_rps: float,
        duration_seconds: int,
        period_seconds: int = 300,
        test_name: Optional[str] = None,
    ) -> MetricsSummary:
        """Run a wave pattern test"""
        scenarios = self._load_scenarios()
        if not scenarios:
            raise ValueError("No scenarios available for testing")

        pattern = LoadPatternFactory.create_wave_pattern(
            scenarios, duration_seconds, base_rps, amplitude_rps, period_seconds
        )
        return await self._execute_load_test(
            pattern,
            test_name or f"wave_{base_rps}±{amplitude_rps}rps_{duration_seconds}s",
        )

    async def sustained_high_load_test(
        self, rps: float, duration_seconds: int, test_name: Optional[str] = None
    ) -> MetricsSummary:
        """Run a sustained high load test"""
        scenarios = self._load_scenarios()
        if not scenarios:
            raise ValueError("No scenarios available for testing")

        pattern = LoadPatternFactory.create_sustained_high_load(
            scenarios, duration_seconds, rps
        )
        return await self._execute_load_test(
            pattern, test_name or f"sustained_{rps}rps_{duration_seconds}s"
        )

    async def scaling_test(
        self,
        instance_counts: List[int],
        rps_per_instance: float,
        duration_per_test: int,
        test_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run scaling tests across different instance counts"""
        scenarios = self._load_scenarios()
        if not scenarios:
            raise ValueError("No scenarios available for testing")

        scaling_results = []
        base_test_name = test_name or f"scaling_{rps_per_instance}rps_per_instance"

        self.console.print(
            f"\n[bold blue]Starting scaling test: {base_test_name}[/bold blue]"
        )
        self.console.print(f"Testing with instance counts: {instance_counts}")
        self.console.print(f"RPS per instance: {rps_per_instance}")
        self.console.print(f"Duration per test: {duration_per_test}s\n")

        for i, instance_count in enumerate(instance_counts, 1):
            total_rps = instance_count * rps_per_instance
            current_test_name = f"{base_test_name}_{instance_count}instances"

            self.console.print(
                f"[yellow]Test {i}/{len(instance_counts)}: {instance_count} instances ({total_rps} RPS)[/yellow]"
            )

            pattern = LoadPatternFactory.create_steady_load(
                scenarios, duration_per_test, total_rps
            )
            summary = await self._execute_load_test(pattern, current_test_name)

            scaling_results.append((instance_count, summary))

            # Brief pause between tests
            if i < len(instance_counts):
                self.console.print("[dim]Pausing 10s between tests...[/dim]")
                await asyncio.sleep(10)

        # Analyze scaling efficiency
        scaling_analysis = PerformanceAnalyzer.analyze_scaling_efficiency(
            scaling_results
        )

        # Save scaling results
        results_file = Path(self.config.results_dir) / f"{base_test_name}_analysis.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "test_name": base_test_name,
                    "scaling_analysis": scaling_analysis,
                    "individual_results": [
                        {
                            "instance_count": count,
                            "summary": {
                                "total_requests": summary.total_requests,
                                "success_rate": summary.success_rate,
                                "mean_latency_ms": summary.mean_latency_ms,
                                "requests_per_second": summary.requests_per_second,
                                "baseline_violations": summary.baseline_violations,
                            },
                        }
                        for count, summary in scaling_results
                    ],
                },
                f,
                indent=2,
            )

        self.console.print(
            f"[green]Scaling test completed. Results saved to {results_file}[/green]"
        )

        return {
            "test_name": base_test_name,
            "scaling_analysis": scaling_analysis,
            "results": scaling_results,
        }

    async def _execute_load_test(
        self, pattern: LoadPattern, test_name: str
    ) -> MetricsSummary:
        """Execute a load test with the given pattern"""
        scenarios = self._load_scenarios()
        self.metrics_collector = MetricsCollector(scenarios)

        self.console.print(f"\n[bold green]Starting test: {test_name}[/bold green]")
        self.console.print(f"Duration: {pattern.duration_seconds}s")
        self.console.print(f"Scenarios: {len(scenarios)}")
        self.console.print(
            f"Max concurrent requests: {self.config.max_concurrent_requests}"
        )

        self.metrics_collector.start_collection()

        # Create progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            main_task = progress.add_task(
                f"Running {test_name}", total=pattern.duration_seconds
            )

            # Semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

            # Task queue for managing concurrent requests
            tasks = []
            completed_requests = 0

            # Live dashboard setup
            if self.config.enable_live_dashboard:
                dashboard_task = asyncio.create_task(self._run_live_dashboard())

            try:
                # Generate and execute load
                load_generator = pattern.generate_load_events()
                async for load_event in load_generator:  # type: ignore
                    # Update progress
                    elapsed = time.time() - (pattern.start_time or time.time())
                    progress.update(
                        main_task, completed=min(elapsed, pattern.duration_seconds)
                    )

                    # Create compilation task
                    task = asyncio.create_task(
                        self._execute_compilation_request(semaphore, load_event)
                    )
                    tasks.append(task)

                    # Clean up completed tasks periodically
                    if len(tasks) >= self.config.max_concurrent_requests * 2:
                        done_tasks = [t for t in tasks if t.done()]
                        for task in done_tasks:
                            tasks.remove(task)
                            completed_requests += 1

                    # Small delay to prevent overwhelming the event loop
                    await asyncio.sleep(0.001)

                # Wait for all remaining tasks to complete
                if tasks:
                    progress.update(
                        main_task, description="Waiting for remaining requests..."
                    )
                    await asyncio.gather(*tasks, return_exceptions=True)

                progress.update(main_task, completed=pattern.duration_seconds)

            finally:
                if self.config.enable_live_dashboard:
                    dashboard_task.cancel()
                    try:
                        await dashboard_task
                    except asyncio.CancelledError:
                        pass

        self.metrics_collector.stop_collection()
        summary = self.metrics_collector.generate_summary()

        # Save results
        await self._save_test_results(test_name, summary)

        # Display summary
        self._display_test_summary(test_name, summary)

        return summary

    async def _execute_compilation_request(
        self, semaphore: asyncio.Semaphore, load_event: LoadEvent
    ) -> CompilationResult:
        """Execute a single compilation request"""
        async with semaphore:
            try:
                if not self.client:
                    raise RuntimeError("Client not initialized")
                result = await self.client.compile_and_execute(
                    source_code=load_event.scenario.source_code,
                    compiler_id=self.config.compiler,
                    options=load_event.scenario.compiler_options,
                    libraries=load_event.scenario.libraries,
                    request_id=load_event.request_id,
                )

                # Record result in metrics collector
                if self.metrics_collector:
                    self.metrics_collector.record_result(result, load_event.scenario.name)

                return result

            except Exception as e:
                self.logger.error(f"Request {load_event.request_id} failed: {e}")
                # Create error result
                from .api_client import CompilationStatus

                error_result = CompilationResult(
                    status=CompilationStatus.API_ERROR,
                    compile_time_ms=None,
                    execution_time_ms=None,
                    total_time_ms=0,
                    compiler_stdout="",
                    compiler_stderr="",
                    program_stdout="",
                    program_stderr="",
                    exit_code=None,
                    error_message=str(e),
                    timestamp=time.time(),
                    request_id=load_event.request_id,
                )
                if self.metrics_collector:
                    self.metrics_collector.record_result(
                        error_result, load_event.scenario.name
                    )
                return error_result

    async def _run_live_dashboard(self) -> None:
        """Run live dashboard display"""
        try:
            while True:
                await asyncio.sleep(1)
                # Dashboard updates are handled by the display methods
        except asyncio.CancelledError:
            pass

    def _create_dashboard_table(self) -> Table:
        """Create live dashboard table"""
        if not self.metrics_collector:
            return Table()

        current_metrics = self.metrics_collector.get_current_metrics()

        table = Table(title="Live Metrics Dashboard", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Requests/sec", f"{current_metrics['rps']:.1f}")
        table.add_row("Success Rate", f"{current_metrics['success_rate']:.1%}")
        table.add_row("P95 Latency", f"{current_metrics['p95_latency_ms']:.1f}ms")
        table.add_row("Total Requests", f"{current_metrics['total_requests']:,}")
        table.add_row("Active Alerts", f"{current_metrics['alerts']}")

        return table

    async def _save_test_results(self, test_name: str, summary: MetricsSummary) -> None:
        """Save test results to files"""
        timestamp = int(time.time())
        base_filename = f"{test_name}_{timestamp}"

        # Save summary as JSON
        summary_file = Path(self.config.results_dir) / f"{base_filename}_summary.json"
        summary_data = {
            "test_name": test_name,
            "timestamp": timestamp,
            "config": {
                "endpoint": self.config.endpoint,
                "compiler": self.config.compiler,
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "rate_limit_rps": self.config.rate_limit_rps,
            },
            "summary": {
                "total_requests": summary.total_requests,
                "successful_requests": summary.successful_requests,
                "failed_requests": summary.failed_requests,
                "success_rate": summary.success_rate,
                "mean_latency_ms": summary.mean_latency_ms,
                "median_latency_ms": summary.median_latency_ms,
                "p95_latency_ms": summary.p95_latency_ms,
                "p99_latency_ms": summary.p99_latency_ms,
                "requests_per_second": summary.requests_per_second,
                "test_duration_seconds": summary.test_duration_seconds,
                "baseline_violations": summary.baseline_violations,
                "total_alerts": summary.total_alerts,
                "error_breakdown": summary.error_breakdown,
                "scenario_breakdown": summary.scenario_breakdown,
            },
        }

        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Save raw data if enabled
        if self.config.save_raw_data and self.metrics_collector:
            raw_data_file = Path(self.config.results_dir) / f"{base_filename}_raw.json"
            self.metrics_collector.save_to_file(str(raw_data_file))

        self.console.print(f"[dim]Results saved to {summary_file}[/dim]")

    def _display_test_summary(self, test_name: str, summary: MetricsSummary) -> None:
        """Display test summary"""
        self.console.print(f"\n[bold green]Test Completed: {test_name}[/bold green]")

        # Create summary table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Requests", f"{summary.total_requests:,}")
        table.add_row("Successful Requests", f"{summary.successful_requests:,}")
        table.add_row("Failed Requests", f"{summary.failed_requests:,}")
        table.add_row("Success Rate", f"{summary.success_rate:.1%}")
        table.add_row("", "")  # Separator
        table.add_row("Mean Latency", f"{summary.mean_latency_ms:.1f}ms")
        table.add_row("Median Latency", f"{summary.median_latency_ms:.1f}ms")
        table.add_row("P95 Latency", f"{summary.p95_latency_ms:.1f}ms")
        table.add_row("P99 Latency", f"{summary.p99_latency_ms:.1f}ms")
        table.add_row("", "")  # Separator
        table.add_row("Throughput", f"{summary.requests_per_second:.1f} RPS")
        table.add_row("Successful RPS", f"{summary.successful_rps:.1f} RPS")
        table.add_row("", "")  # Separator
        table.add_row("Test Duration", f"{summary.test_duration_seconds:.1f}s")
        table.add_row("Baseline Violations", f"{summary.baseline_violations}")
        table.add_row("Total Alerts", f"{summary.total_alerts}")

        self.console.print(table)

        # Show error breakdown if there are errors
        if summary.error_breakdown:
            self.console.print("\n[bold red]Error Breakdown:[/bold red]")
            error_table = Table(show_header=True)
            error_table.add_column("Error Type", style="red")
            error_table.add_column("Count", style="yellow")
            error_table.add_column("Percentage", style="yellow")

            for error_type, count in summary.error_breakdown.items():
                percentage = (
                    (count / summary.total_requests) * 100
                    if summary.total_requests > 0
                    else 0
                )
                error_table.add_row(error_type, str(count), f"{percentage:.1f}%")

            self.console.print(error_table)

    def generate_report(self, results_dir: Optional[str] = None) -> str:
        """Generate a comprehensive test report"""
        # This will be implemented in the reporting module
        report_path = (
            Path(results_dir or self.config.results_dir) / "comprehensive_report.html"
        )
        self.console.print(
            f"[yellow]Report generation not yet implemented. Would save to: {report_path}[/yellow]"
        )
        return str(report_path)
