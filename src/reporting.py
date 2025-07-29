"""
Report generation and visualization for stress test results
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import statistics

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ReportGenerator:
    """Generates comprehensive HTML reports with visualizations"""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "reports"
        self.output_dir.mkdir(exist_ok=True)

        # Configure matplotlib for better looking plots
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def generate_comprehensive_report(
        self, test_results: List[Dict[str, Any]], report_name: Optional[str] = None
    ) -> str:
        """Generate a comprehensive HTML report"""

        if not report_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"stress_test_report_{timestamp}"

        report_file = self.output_dir / f"{report_name}.html"

        # Generate visualizations
        charts = self._generate_charts(test_results, report_name)

        # Generate HTML report
        html_content = self._generate_html_report(test_results, charts, report_name)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(report_file)

    def _generate_charts(
        self, test_results: List[Dict[str, Any]], report_name: str
    ) -> Dict[str, str]:
        """Generate all visualization charts"""
        charts: Dict[str, str] = {}

        if not test_results:
            return charts

        # Create charts directory
        charts_dir = self.output_dir / f"{report_name}_charts"
        charts_dir.mkdir(exist_ok=True)

        try:
            # 1. Response time distribution
            charts["latency_distribution"] = self._create_latency_distribution_chart(
                test_results, charts_dir / "latency_distribution.png"
            )

            # 2. Throughput over time
            charts["throughput_timeline"] = self._create_throughput_timeline_chart(
                test_results, charts_dir / "throughput_timeline.png"
            )

            # 3. Success rate comparison
            charts["success_rate_comparison"] = self._create_success_rate_chart(
                test_results, charts_dir / "success_rate_comparison.png"
            )

            # 4. Error breakdown
            charts["error_breakdown"] = self._create_error_breakdown_chart(
                test_results, charts_dir / "error_breakdown.png"
            )

            # 5. Scaling analysis (if available)
            scaling_data = self._extract_scaling_data(test_results)
            if scaling_data:
                charts["scaling_analysis"] = self._create_scaling_analysis_chart(
                    scaling_data, charts_dir / "scaling_analysis.png"
                )

            # 6. Performance heatmap
            charts["performance_heatmap"] = self._create_performance_heatmap(
                test_results, charts_dir / "performance_heatmap.png"
            )

        except Exception as e:
            print(f"Warning: Chart generation failed: {e}")

        return charts

    def _create_latency_distribution_chart(
        self, test_results: List[Dict[str, Any]], output_file: Path
    ) -> str:
        """Create latency distribution histogram"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Collect latency data from all tests
        all_latencies = []
        test_names = []

        for result in test_results:
            if "summary" in result:
                summary = result["summary"]
                # For this chart, we'll use summary statistics
                latencies = [
                    summary.get("mean_latency_ms", 0),
                    summary.get("median_latency_ms", 0),
                    summary.get("p95_latency_ms", 0),
                    summary.get("p99_latency_ms", 0),
                ]
                all_latencies.extend(latencies)
                test_names.append(result.get("test_name", "Unknown"))

        if all_latencies:
            # Histogram
            ax1.hist(all_latencies, bins=30, alpha=0.7, edgecolor="black")
            ax1.set_xlabel("Latency (ms)")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Latency Distribution")
            ax1.grid(True, alpha=0.3)

            # Box plot by test
            if len(test_results) > 1:
                latency_by_test = []
                labels = []
                for result in test_results:
                    if "summary" in result:
                        summary = result["summary"]
                        latency_by_test.append(
                            [
                                summary.get("mean_latency_ms", 0),
                                summary.get("median_latency_ms", 0),
                                summary.get("p95_latency_ms", 0),
                                summary.get("p99_latency_ms", 0),
                            ]
                        )
                        labels.append(
                            result.get("test_name", "Unknown")[:15]
                        )  # Truncate long names

                ax2.boxplot(latency_by_test, labels=labels)
                ax2.set_ylabel("Latency (ms)")
                ax2.set_title("Latency by Test")
                ax2.tick_params(axis="x", rotation=45)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Single test - no comparison available",
                    ha="center",
                    va="center",
                    transform=ax2.transAxes,
                )
                ax2.set_title("Test Comparison")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_file.relative_to(self.output_dir))

    def _create_throughput_timeline_chart(
        self, test_results: List[Dict[str, Any]], output_file: Path
    ) -> str:
        """Create throughput timeline chart"""
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, result in enumerate(test_results):
            test_name = result.get("test_name", f"Test {i+1}")
            summary = result.get("summary", {})

            # Create a simple timeline representation
            duration = summary.get("test_duration_seconds", 0)
            rps = summary.get("requests_per_second", 0)

            if duration > 0:
                # Simple representation - could be enhanced with actual time series data
                times = [0, duration]
                throughputs = [rps, rps]
                ax.plot(times, throughputs, marker="o", linewidth=2, label=test_name)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Requests per Second")
        ax.set_title("Throughput Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_file.relative_to(self.output_dir))

    def _create_success_rate_chart(
        self, test_results: List[Dict[str, Any]], output_file: Path
    ) -> str:
        """Create success rate comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 6))

        test_names = []
        success_rates = []

        for result in test_results:
            test_names.append(result.get("test_name", "Unknown"))
            summary = result.get("summary", {})
            success_rates.append(
                summary.get("success_rate", 0) * 100
            )  # Convert to percentage

        if success_rates:
            cmap = plt.cm.get_cmap("RdYlGn")
            colors = cmap(
                [rate / 100 for rate in success_rates]
            )  # Color based on success rate
            bars = ax.bar(range(len(test_names)), success_rates, color=colors)

            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                )

            ax.set_xlabel("Test")
            ax.set_ylabel("Success Rate (%)")
            ax.set_title("Success Rate by Test")
            ax.set_ylim(0, 105)  # Give some headroom
            ax.set_xticks(range(len(test_names)))
            ax.set_xticklabels(
                [name[:15] for name in test_names], rotation=45, ha="right"
            )
            ax.grid(True, alpha=0.3, axis="y")

            # Add horizontal line at 95% (common SLA threshold)
            ax.axhline(y=95, color="red", linestyle="--", alpha=0.7, label="95% SLA")
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_file.relative_to(self.output_dir))

    def _create_error_breakdown_chart(
        self, test_results: List[Dict[str, Any]], output_file: Path
    ) -> str:
        """Create error breakdown pie chart"""
        fig, axes = plt.subplots(1, min(len(test_results), 3), figsize=(15, 5))
        if len(test_results) == 1:
            axes = [axes]

        for i, result in enumerate(test_results[:3]):  # Limit to first 3 tests
            ax = axes[i] if len(test_results) > 1 else axes[0]

            test_name = result.get("test_name", f"Test {i+1}")
            summary = result.get("summary", {})
            error_breakdown = summary.get("error_breakdown", {})

            if error_breakdown:
                labels = list(error_breakdown.keys())
                sizes = list(error_breakdown.values())

                # Create pie chart
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, autopct="%1.1f%%", startangle=90
                )
                ax.set_title(f"Errors - {test_name[:20]}")

                # Improve text readability
                for autotext in autotexts:
                    autotext.set_color("white")
                    autotext.set_fontweight("bold")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No errors recorded",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Errors - {test_name[:20]}")

        # Hide unused subplots
        for j in range(len(test_results), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_file.relative_to(self.output_dir))

    def _extract_scaling_data(
        self, test_results: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract scaling analysis data if available"""
        for result in test_results:
            if "scaling_analysis" in result:
                return result["scaling_analysis"]  # type: ignore
        return None

    def _create_scaling_analysis_chart(
        self, scaling_data: Dict[str, Any], output_file: Path
    ) -> str:
        """Create scaling analysis charts"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        scaling_points = scaling_data.get("scaling_points", [])

        if scaling_points:
            instances = [point["instances"] for point in scaling_points]
            actual_throughput = [
                point["actual_throughput_rps"] for point in scaling_points
            ]
            expected_throughput = [
                point["expected_throughput_rps"] for point in scaling_points
            ]
            efficiency = [point["scaling_efficiency"] for point in scaling_points]

            # Throughput scaling chart
            ax1.plot(
                instances,
                actual_throughput,
                "o-",
                label="Actual Throughput",
                linewidth=2,
                markersize=8,
            )
            ax1.plot(
                instances,
                expected_throughput,
                "--",
                label="Ideal Linear Scaling",
                linewidth=2,
            )
            ax1.set_xlabel("Instance Count")
            ax1.set_ylabel("Throughput (RPS)")
            ax1.set_title("Throughput Scaling")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Scaling efficiency chart
            ax2.plot(
                instances,
                [e * 100 for e in efficiency],
                "o-",
                color="red",
                linewidth=2,
                markersize=8,
            )
            ax2.set_xlabel("Instance Count")
            ax2.set_ylabel("Scaling Efficiency (%)")
            ax2.set_title("Scaling Efficiency")
            ax2.axhline(
                y=100, color="green", linestyle="--", alpha=0.7, label="Perfect Scaling"
            )
            ax2.axhline(
                y=80, color="orange", linestyle="--", alpha=0.7, label="80% Efficiency"
            )
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 110)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_file.relative_to(self.output_dir))

    def _create_performance_heatmap(
        self, test_results: List[Dict[str, Any]], output_file: Path
    ) -> str:
        """Create performance heatmap across scenarios"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Collect scenario performance data
        scenario_data: Dict[str, List[float]] = {}
        test_names = []

        for result in test_results:
            test_name = result.get("test_name", "Unknown")[:15]  # Truncate long names
            test_names.append(test_name)
            summary = result.get("summary", {})
            scenario_breakdown = summary.get("scenario_breakdown", {})

            for scenario_name, scenario_stats in scenario_breakdown.items():
                if scenario_name not in scenario_data:
                    scenario_data[scenario_name] = []
                scenario_data[scenario_name].append(
                    scenario_stats.get("mean_latency_ms", 0)
                )

        if scenario_data:
            # Create DataFrame for heatmap
            df_data = []
            scenario_names = list(scenario_data.keys())

            for test_name in test_names:
                row = []
                for scenario_name in scenario_names:
                    # Find corresponding latency for this test
                    test_idx = test_names.index(test_name)
                    if test_idx < len(scenario_data[scenario_name]):
                        row.append(scenario_data[scenario_name][test_idx])
                    else:
                        row.append(0)
                df_data.append(row)

            df = pd.DataFrame(df_data, index=test_names, columns=scenario_names)

            # Create heatmap
            sns.heatmap(
                df,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                ax=ax,
                cbar_kws={"label": "Latency (ms)"},
            )
            ax.set_title("Performance Heatmap: Mean Latency by Test and Scenario")
            ax.set_xlabel("Scenario")
            ax.set_ylabel("Test")
        else:
            ax.text(
                0.5,
                0.5,
                "No scenario data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Heatmap")

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        return str(output_file.relative_to(self.output_dir))

    def _generate_html_report(
        self,
        test_results: List[Dict[str, Any]],
        charts: Dict[str, str],
        report_name: str,
    ) -> str:
        """Generate the HTML report content"""

        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compiler Explorer Stress Test Report - {report_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section {{
            background: white;
            margin-bottom: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .section-header {{
            background: #f8f9fa;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid #e9ecef;
            font-weight: bold;
            font-size: 1.2rem;
        }}
        .section-content {{
            padding: 1.5rem;
        }}
        .chart {{
            text-align: center;
            margin: 1rem 0;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .test-details {{
            display: grid;
            gap: 1rem;
        }}
        .test-item {{
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 1rem;
        }}
        .test-title {{
            font-weight: bold;
            color: #495057;
            margin-bottom: 0.5rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .metric-item {{
            background: #f8f9fa;
            padding: 0.75rem;
            border-radius: 4px;
            text-align: center;
        }}
        .status-success {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-error {{ color: #dc3545; }}
        .timestamp {{
            color: #6c757d;
            font-size: 0.9rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
            margin-top: 3rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Compiler Explorer Stress Test Report</h1>
        <h2>{report_name}</h2>
        <p class="timestamp">Generated on {timestamp}</p>
    </div>

    <div class="summary-cards">
        {summary_cards}
    </div>

    <div class="section">
        <div class="section-header">📊 Performance Visualizations</div>
        <div class="section-content">
            {charts_html}
        </div>
    </div>

    <div class="section">
        <div class="section-header">📋 Test Results Details</div>
        <div class="section-content">
            <div class="test-details">
                {test_details}
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-header">⚠️ Alerts and Issues</div>
        <div class="section-content">
            {alerts_html}
        </div>
    </div>

    <div class="footer">
        <p>Report generated by Compiler Explorer Stress Testing Framework</p>
        <p>Framework version 1.0 | {timestamp}</p>
    </div>
</body>
</html>
        """

        # Generate content sections
        summary_cards = self._generate_summary_cards(test_results)
        charts_html = self._generate_charts_html(charts)
        test_details = self._generate_test_details_html(test_results)
        alerts_html = self._generate_alerts_html(test_results)

        return html_template.format(
            report_name=report_name,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary_cards=summary_cards,
            charts_html=charts_html,
            test_details=test_details,
            alerts_html=alerts_html,
        )

    def _generate_summary_cards(self, test_results: List[Dict[str, Any]]) -> str:
        """Generate summary cards HTML"""
        if not test_results:
            return '<div class="card"><div class="metric-label">No Data</div><div class="metric-value">0</div></div>'

        # Aggregate metrics across all tests
        total_requests = sum(
            result.get("summary", {}).get("total_requests", 0)
            for result in test_results
        )
        avg_success_rate = (
            statistics.mean(
                [
                    result.get("summary", {}).get("success_rate", 0)
                    for result in test_results
                ]
            )
            * 100
        )
        avg_latency = statistics.mean(
            [
                result.get("summary", {}).get("mean_latency_ms", 0)
                for result in test_results
            ]
        )
        total_violations = sum(
            result.get("summary", {}).get("baseline_violations", 0)
            for result in test_results
        )

        cards = [
            f'<div class="card"><div class="metric-label">Total Requests</div><div class="metric-value">{total_requests:,}</div></div>',
            f'<div class="card"><div class="metric-label">Average Success Rate</div><div class="metric-value">{avg_success_rate:.1f}%</div></div>',
            f'<div class="card"><div class="metric-label">Average Latency</div><div class="metric-value">{avg_latency:.0f}ms</div></div>',
            f'<div class="card"><div class="metric-label">Baseline Violations</div><div class="metric-value">{total_violations}</div></div>',
            f'<div class="card"><div class="metric-label">Tests Executed</div><div class="metric-value">{len(test_results)}</div></div>',
        ]

        return "\n".join(cards)

    def _generate_charts_html(self, charts: Dict[str, str]) -> str:
        """Generate charts HTML"""
        if not charts:
            return "<p>No charts available</p>"

        charts_html = []
        for chart_name, chart_path in charts.items():
            title = chart_name.replace("_", " ").title()
            charts_html.append(
                f"""
            <div class="chart">
                <h3>{title}</h3>
                <img src="{chart_path}" alt="{title}" />
            </div>
            """
            )

        return "\n".join(charts_html)

    def _generate_test_details_html(self, test_results: List[Dict[str, Any]]) -> str:
        """Generate test details HTML"""
        if not test_results:
            return "<p>No test results available</p>"

        details_html = []
        for result in test_results:
            test_name = result.get("test_name", "Unknown Test")
            summary = result.get("summary", {})

            success_rate = summary.get("success_rate", 0) * 100
            status_class = (
                "status-success"
                if success_rate >= 95
                else "status-warning"
                if success_rate >= 90
                else "status-error"
            )

            details_html.append(
                f"""
            <div class="test-item">
                <div class="test-title">{test_name}</div>
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-label">Requests</div>
                        <div class="metric-value">{summary.get('total_requests', 0):,}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Success Rate</div>
                        <div class="metric-value {status_class}">{success_rate:.1f}%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Mean Latency</div>
                        <div class="metric-value">{summary.get('mean_latency_ms', 0):.1f}ms</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">P95 Latency</div>
                        <div class="metric-value">{summary.get('p95_latency_ms', 0):.1f}ms</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Throughput</div>
                        <div class="metric-value">{summary.get('requests_per_second', 0):.1f} RPS</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Duration</div>
                        <div class="metric-value">{summary.get('test_duration_seconds', 0):.1f}s</div>
                    </div>
                </div>
            </div>
            """
            )

        return "\n".join(details_html)

    def _generate_alerts_html(self, test_results: List[Dict[str, Any]]) -> str:
        """Generate alerts HTML"""
        total_alerts = sum(
            result.get("summary", {}).get("total_alerts", 0) for result in test_results
        )
        total_violations = sum(
            result.get("summary", {}).get("baseline_violations", 0)
            for result in test_results
        )

        if total_alerts == 0 and total_violations == 0:
            return '<p class="status-success">✅ No alerts or baseline violations detected during testing.</p>'

        alerts_html = []
        if total_violations > 0:
            alerts_html.append(
                f'<p class="status-warning">⚠️ {total_violations} baseline violations detected across all tests.</p>'
            )

        if total_alerts > 0:
            alerts_html.append(
                f'<p class="status-error">🚨 {total_alerts} alerts generated during testing.</p>'
            )

        alerts_html.append(
            "<p>Check individual test logs for detailed alert information.</p>"
        )

        return "\n".join(alerts_html)


def create_quick_report(results_file: Path, output_dir: Path) -> str:
    """Create a quick summary report from a single results file"""
    with open(results_file, "r") as f:
        data = json.load(f)

    generator = ReportGenerator(output_dir)
    report_file = generator.generate_comprehensive_report(
        [data], results_file.stem + "_report"
    )

    return report_file
