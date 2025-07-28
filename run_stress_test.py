#!/usr/bin/env python3
"""
Main entry point for Compiler Explorer stress testing framework
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.table import Table
from rich.panel import Panel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.framework import CompilerStressTest, TestConfiguration
from src.scenarios import WorkloadScenarios
from src.reporting import create_quick_report


console = Console()


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Compiler Explorer Stress Testing Framework

    A comprehensive tool for stress testing Compiler Explorer's compilation workers.
    Supports various load patterns, scaling tests, and detailed performance analysis.
    """
    pass


@cli.command()
@click.option(
    "--endpoint",
    default="https://compiler-explorer.com",
    help="Compiler Explorer API endpoint",
)
@click.option("--compiler", default="g151", help="Compiler ID to use")
@click.option("--rps", type=float, default=10.0, help="Requests per second")
@click.option("--duration", type=int, default=300, help="Test duration in seconds")
@click.option("--concurrent", type=int, default=50, help="Max concurrent requests")
@click.option("--scenarios", help="Comma-separated list of scenario names to use")
@click.option(
    "--workload-dir", type=click.Path(exists=True), help="Custom workload directory"
)
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--no-dashboard", is_flag=True, help="Disable live dashboard")
@click.option("--test-name", help="Custom test name")
def steady(
    endpoint,
    compiler,
    rps,
    duration,
    concurrent,
    scenarios,
    workload_dir,
    results_dir,
    no_dashboard,
    test_name,
):
    """Run a steady load test with constant RPS"""

    config = TestConfiguration(
        endpoint=endpoint,
        compiler=compiler,
        max_concurrent_requests=concurrent,
        scenarios=scenarios.split(",") if scenarios else None,
        workload_dir=workload_dir,
        results_dir=results_dir,
        enable_live_dashboard=not no_dashboard,
    )

    asyncio.run(_run_steady_test(config, rps, duration, test_name))


@cli.command()
@click.option(
    "--endpoint",
    default="https://compiler-explorer.com",
    help="Compiler Explorer API endpoint",
)
@click.option("--compiler", default="g151", help="Compiler ID to use")
@click.option("--baseline-rps", type=float, default=5.0, help="Baseline RPS")
@click.option("--burst-rps", type=float, default=20.0, help="Burst RPS")
@click.option(
    "--duration", type=int, default=600, help="Total test duration in seconds"
)
@click.option(
    "--burst-duration", type=int, default=30, help="Burst duration in seconds"
)
@click.option(
    "--burst-interval", type=int, default=120, help="Burst interval in seconds"
)
@click.option("--concurrent", type=int, default=50, help="Max concurrent requests")
@click.option("--scenarios", help="Comma-separated list of scenario names to use")
@click.option(
    "--workload-dir", type=click.Path(exists=True), help="Custom workload directory"
)
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--no-dashboard", is_flag=True, help="Disable live dashboard")
@click.option("--test-name", help="Custom test name")
def burst(
    endpoint,
    compiler,
    baseline_rps,
    burst_rps,
    duration,
    burst_duration,
    burst_interval,
    concurrent,
    scenarios,
    workload_dir,
    results_dir,
    no_dashboard,
    test_name,
):
    """Run a burst load test with periodic traffic spikes"""

    config = TestConfiguration(
        endpoint=endpoint,
        compiler=compiler,
        max_concurrent_requests=concurrent,
        scenarios=scenarios.split(",") if scenarios else None,
        workload_dir=workload_dir,
        results_dir=results_dir,
        enable_live_dashboard=not no_dashboard,
    )

    asyncio.run(
        _run_burst_test(
            config,
            baseline_rps,
            burst_rps,
            duration,
            burst_duration,
            burst_interval,
            test_name,
        )
    )


@cli.command()
@click.option(
    "--endpoint",
    default="https://compiler-explorer.com",
    help="Compiler Explorer API endpoint",
)
@click.option("--compiler", default="g151", help="Compiler ID to use")
@click.option("--min-rps", type=float, default=1.0, help="Minimum RPS")
@click.option("--max-rps", type=float, default=20.0, help="Maximum RPS")
@click.option("--duration", type=int, default=300, help="Test duration in seconds")
@click.option("--ramp-down", is_flag=True, help="Ramp down instead of up")
@click.option("--concurrent", type=int, default=50, help="Max concurrent requests")
@click.option("--scenarios", help="Comma-separated list of scenario names to use")
@click.option(
    "--workload-dir", type=click.Path(exists=True), help="Custom workload directory"
)
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--no-dashboard", is_flag=True, help="Disable live dashboard")
@click.option("--test-name", help="Custom test name")
def ramp(
    endpoint,
    compiler,
    min_rps,
    max_rps,
    duration,
    ramp_down,
    concurrent,
    scenarios,
    workload_dir,
    results_dir,
    no_dashboard,
    test_name,
):
    """Run a ramp test with gradually increasing/decreasing load"""

    config = TestConfiguration(
        endpoint=endpoint,
        compiler=compiler,
        max_concurrent_requests=concurrent,
        scenarios=scenarios.split(",") if scenarios else None,
        workload_dir=workload_dir,
        results_dir=results_dir,
        enable_live_dashboard=not no_dashboard,
    )

    asyncio.run(
        _run_ramp_test(config, min_rps, max_rps, duration, not ramp_down, test_name)
    )


@cli.command()
@click.option(
    "--endpoint",
    default="https://compiler-explorer.com",
    help="Compiler Explorer API endpoint",
)
@click.option("--compiler", default="g151", help="Compiler ID to use")
@click.option("--base-rps", type=float, default=10.0, help="Base RPS")
@click.option("--amplitude-rps", type=float, default=5.0, help="Wave amplitude RPS")
@click.option("--duration", type=int, default=600, help="Test duration in seconds")
@click.option("--period", type=int, default=300, help="Wave period in seconds")
@click.option("--concurrent", type=int, default=50, help="Max concurrent requests")
@click.option("--scenarios", help="Comma-separated list of scenario names to use")
@click.option(
    "--workload-dir", type=click.Path(exists=True), help="Custom workload directory"
)
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--no-dashboard", is_flag=True, help="Disable live dashboard")
@click.option("--test-name", help="Custom test name")
def wave(
    endpoint,
    compiler,
    base_rps,
    amplitude_rps,
    duration,
    period,
    concurrent,
    scenarios,
    workload_dir,
    results_dir,
    no_dashboard,
    test_name,
):
    """Run a wave pattern test with sinusoidal load variation"""

    config = TestConfiguration(
        endpoint=endpoint,
        compiler=compiler,
        max_concurrent_requests=concurrent,
        scenarios=scenarios.split(",") if scenarios else None,
        workload_dir=workload_dir,
        results_dir=results_dir,
        enable_live_dashboard=not no_dashboard,
    )

    asyncio.run(
        _run_wave_test(config, base_rps, amplitude_rps, duration, period, test_name)
    )


@cli.command()
@click.option(
    "--endpoint",
    default="https://compiler-explorer.com",
    help="Compiler Explorer API endpoint",
)
@click.option("--compiler", default="g151", help="Compiler ID to use")
@click.option(
    "--instances", default="2,4,6,8,10", help="Comma-separated instance counts"
)
@click.option("--rps-per-instance", type=float, default=5.0, help="RPS per instance")
@click.option("--duration", type=int, default=180, help="Duration per test in seconds")
@click.option("--concurrent", type=int, default=100, help="Max concurrent requests")
@click.option("--scenarios", help="Comma-separated list of scenario names to use")
@click.option(
    "--workload-dir", type=click.Path(exists=True), help="Custom workload directory"
)
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--no-dashboard", is_flag=True, help="Disable live dashboard")
@click.option("--test-name", help="Custom test name")
def scaling(
    endpoint,
    compiler,
    instances,
    rps_per_instance,
    duration,
    concurrent,
    scenarios,
    workload_dir,
    results_dir,
    no_dashboard,
    test_name,
):
    """Run scaling tests across different instance counts"""

    instance_counts = [int(x.strip()) for x in instances.split(",")]

    config = TestConfiguration(
        endpoint=endpoint,
        compiler=compiler,
        max_concurrent_requests=concurrent,
        scenarios=scenarios.split(",") if scenarios else None,
        workload_dir=workload_dir,
        results_dir=results_dir,
        enable_live_dashboard=not no_dashboard,
    )

    asyncio.run(
        _run_scaling_test(
            config, instance_counts, rps_per_instance, duration, test_name
        )
    )


@cli.command()
@click.option(
    "--workload-dir", type=click.Path(exists=True), help="Custom workload directory"
)
def list_scenarios(workload_dir):
    """List available test scenarios"""

    workload_scenarios = WorkloadScenarios(Path(workload_dir) if workload_dir else None)

    try:
        scenarios = workload_scenarios.load_all_scenarios()

        if not scenarios:
            console.print("[red]No scenarios found[/red]")
            return

        table = Table(title="Available Test Scenarios")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="green")
        table.add_column("Baseline", style="yellow")

        for scenario in scenarios:
            baseline = f"{scenario.baseline_min_ms}-{scenario.baseline_max_ms}ms"
            table.add_row(
                scenario.name,
                scenario.workload_type.value.replace("_", " ").title(),
                scenario.description[:50] + "..."
                if len(scenario.description) > 50
                else scenario.description,
                baseline,
            )

        console.print(table)
        console.print(f"\n[dim]Total scenarios: {len(scenarios)}[/dim]")

    except Exception as e:
        console.print(f"[red]Error loading scenarios: {e}[/red]")


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--output-dir", default="results/reports", help="Report output directory")
def report(results_file, output_dir):
    """Generate a report from test results"""

    results_path = Path(results_file)
    output_path = Path(output_dir)

    console.print(f"[blue]Generating report from {results_path}[/blue]")

    try:
        report_file = create_quick_report(results_path, output_path)
        console.print(f"[green]Report generated: {report_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")


@cli.command()
def interactive():
    """Interactive mode for configuring and running tests"""

    console.print(
        Panel.fit(
            "[bold blue]Compiler Explorer Stress Testing Framework[/bold blue]\n"
            "[green]Interactive Configuration Mode[/green]",
            border_style="blue",
        )
    )

    # Basic configuration
    endpoint = Prompt.ask("API Endpoint", default="https://beta.compiler-explorer.com")
    compiler = Prompt.ask("Compiler ID", default="g122")

    # Test type selection
    test_types = {
        "1": "Steady Load",
        "2": "Burst Load",
        "3": "Ramp Test",
        "4": "Wave Pattern",
        "5": "Scaling Test",
    }

    console.print("\n[bold]Available Test Types:[/bold]")
    for key, value in test_types.items():
        console.print(f"  {key}. {value}")

    test_choice = Prompt.ask(
        "Select test type", choices=list(test_types.keys()), default="1"
    )

    # Common parameters
    duration = IntPrompt.ask("Test duration (seconds)", default=300)
    concurrent = IntPrompt.ask("Max concurrent requests", default=50)

    # Test-specific parameters
    config = TestConfiguration(
        endpoint=endpoint,
        compiler=compiler,
        max_concurrent_requests=concurrent,
        enable_live_dashboard=True,
    )

    if test_choice == "1":  # Steady Load
        rps = FloatPrompt.ask("Requests per second", default=10.0)
        asyncio.run(_run_steady_test(config, rps, duration))

    elif test_choice == "2":  # Burst Load
        baseline_rps = FloatPrompt.ask("Baseline RPS", default=5.0)
        burst_rps = FloatPrompt.ask("Burst RPS", default=20.0)
        burst_duration = IntPrompt.ask("Burst duration (seconds)", default=30)
        burst_interval = IntPrompt.ask("Burst interval (seconds)", default=120)
        asyncio.run(
            _run_burst_test(
                config,
                baseline_rps,
                burst_rps,
                duration,
                burst_duration,
                burst_interval,
            )
        )

    elif test_choice == "3":  # Ramp Test
        min_rps = FloatPrompt.ask("Minimum RPS", default=1.0)
        max_rps = FloatPrompt.ask("Maximum RPS", default=20.0)
        ramp_up = (
            Prompt.ask("Ramp direction", choices=["up", "down"], default="up") == "up"
        )
        asyncio.run(_run_ramp_test(config, min_rps, max_rps, duration, ramp_up))

    elif test_choice == "4":  # Wave Pattern
        base_rps = FloatPrompt.ask("Base RPS", default=10.0)
        amplitude = FloatPrompt.ask("Wave amplitude RPS", default=5.0)
        period = IntPrompt.ask("Wave period (seconds)", default=300)
        asyncio.run(_run_wave_test(config, base_rps, amplitude, duration, period))

    elif test_choice == "5":  # Scaling Test
        instances_str = Prompt.ask(
            "Instance counts (comma-separated)", default="2,4,6,8"
        )
        instance_counts = [int(x.strip()) for x in instances_str.split(",")]
        rps_per_instance = FloatPrompt.ask("RPS per instance", default=5.0)
        test_duration = IntPrompt.ask(
            "Duration per scaling test (seconds)", default=180
        )
        asyncio.run(
            _run_scaling_test(config, instance_counts, rps_per_instance, test_duration)
        )


# Helper functions for running tests


async def _run_steady_test(
    config: TestConfiguration,
    rps: float,
    duration: int,
    test_name: Optional[str] = None,
):
    """Run steady load test"""
    async with CompilerStressTest(config) as tester:
        await tester.steady_load_test(rps, duration, test_name)


async def _run_burst_test(
    config: TestConfiguration,
    baseline_rps: float,
    burst_rps: float,
    duration: int,
    burst_duration: int,
    burst_interval: int,
    test_name: Optional[str] = None,
):
    """Run burst load test"""
    async with CompilerStressTest(config) as tester:
        await tester.burst_load_test(
            baseline_rps, burst_rps, duration, burst_duration, burst_interval, test_name
        )


async def _run_ramp_test(
    config: TestConfiguration,
    min_rps: float,
    max_rps: float,
    duration: int,
    ramp_up: bool,
    test_name: Optional[str] = None,
):
    """Run ramp test"""
    async with CompilerStressTest(config) as tester:
        await tester.ramp_test(min_rps, max_rps, duration, ramp_up, test_name)


async def _run_wave_test(
    config: TestConfiguration,
    base_rps: float,
    amplitude_rps: float,
    duration: int,
    period: int,
    test_name: Optional[str] = None,
):
    """Run wave pattern test"""
    async with CompilerStressTest(config) as tester:
        await tester.wave_test(base_rps, amplitude_rps, duration, period, test_name)


async def _run_scaling_test(
    config: TestConfiguration,
    instance_counts: List[int],
    rps_per_instance: float,
    duration: int,
    test_name: Optional[str] = None,
):
    """Run scaling test"""
    async with CompilerStressTest(config) as tester:
        await tester.scaling_test(
            instance_counts, rps_per_instance, duration, test_name
        )


@cli.command()
@click.option(
    "--endpoint",
    default="https://compiler-explorer.com",
    help="Compiler Explorer API endpoint",
)
@click.option("--compiler", default="g151", help="Compiler ID to use")
@click.option("--scenario", default="minimal", help="Scenario to test")
@click.option("--concurrent", type=int, default=50, help="Max concurrent requests")
@click.option(
    "--workload-dir", type=click.Path(exists=True), help="Custom workload directory"
)
@click.option("--results-dir", default="results", help="Results output directory")
@click.option("--no-dashboard", is_flag=True, help="Disable live dashboard")
@click.option("--test-name", help="Custom test name")
def instance_match(
    endpoint,
    compiler,
    scenario,
    concurrent,
    workload_dir,
    results_dir,
    no_dashboard,
    test_name,
):
    """Run requests matching the number of active production instances"""
    
    config = TestConfiguration(
        endpoint=endpoint,
        compiler=compiler,
        max_concurrent_requests=concurrent,
        scenarios=[scenario],
        workload_dir=workload_dir,
        results_dir=results_dir,
        enable_live_dashboard=not no_dashboard,
    )

    asyncio.run(_run_instance_match_test(config, scenario, test_name))


async def _run_instance_match_test(
    config: TestConfiguration,
    scenario: str,
    test_name: Optional[str] = None,
):
    """Run instance match test"""
    async with CompilerStressTest(config) as tester:
        # Get current instance count - only main production instances
        if tester.client:
            instance_status = await tester.client.get_instance_status()
            
            # Only count main production instances (exclude GPU, ARM64, Windows)
            prod_instances = 0
            for inst in instance_status:
                if (inst.status == "Online" and 
                    inst.environment_name in ["prod-blue", "prod-green"] and
                    not any(x in inst.environment_name for x in ["gpu", "aarch64", "win"])):
                    prod_instances += inst.healthy_targets
            
            print(f"Found {prod_instances} main production instances, running {prod_instances} requests...")
            
            # Run steady test with same number of requests as instances
            # Use high RPS to send requests quickly, let concurrency limit manage the flow  
            rps = min(prod_instances * 2.0, 50.0)  # Up to 50 RPS max
            duration = max(int(prod_instances / rps) + 2, 3)  # Ensure enough time + buffer
            
            final_test_name = test_name or f"instance_match_{prod_instances}_{scenario}"
            
            await tester.steady_load_test(rps, duration, final_test_name)
        else:
            raise RuntimeError("Client not initialized")


if __name__ == "__main__":
    cli()
