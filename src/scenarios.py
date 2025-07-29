"""
Workload scenario definitions for stress testing
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class WorkloadType(Enum):
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    QUICK = "quick"
    ERROR_CASES = "error_cases"
    MIXED = "mixed"


@dataclass
class ScenarioConfig:
    name: str
    workload_type: WorkloadType
    source_code: str
    compiler_options: str
    baseline_min_ms: int
    baseline_max_ms: int
    weight: float = 1.0
    description: str = ""
    libraries: List[dict] = field(default_factory=list)
    language: str = "c++"


class WorkloadScenarios:
    """Workload scenario manager that loads from physical files"""

    def __init__(self, workload_dir: Optional[Path] = None):
        if workload_dir is None:
            # Default to examples/workloads relative to this file
            self.workload_dir = Path(__file__).parent.parent / "examples" / "workloads"
        else:
            self.workload_dir = Path(workload_dir)

    def _parse_metadata_from_source(
        self, source_code: str
    ) -> Tuple[str, int, int, float, str, List[dict]]:
        """Parse metadata from source code comments"""
        lines = source_code.split("\n")

        compiler_options = "-O2 -std=c++17"  # Default
        baseline_min_ms = 100  # Default
        baseline_max_ms = 1000  # Default
        weight = 1.0  # Default
        description = ""  # Default
        libraries = []  # Default

        for line in lines[:10]:  # Only check first 10 lines
            line = line.strip()
            if line.startswith("//"):
                # Remove // and strip
                content = line[2:].strip()

                if content.startswith("compile:"):
                    compiler_options = content[8:].strip()
                elif content.startswith("lib:"):
                    # Parse library specification: lib: name/version
                    lib_spec = content[4:].strip()
                    if "/" in lib_spec:
                        lib_name, version = lib_spec.split("/", 1)
                        libraries.append(
                            {"id": lib_name.strip(), "version": version.strip()}
                        )
                    else:
                        # Just library name, use default version
                        libraries.append({"id": lib_spec.strip(), "version": "trunk"})
                elif content.startswith("baseline_min_ms:"):
                    try:
                        baseline_min_ms = int(content[16:].strip())
                    except ValueError:
                        pass
                elif content.startswith("baseline_max_ms:"):
                    try:
                        baseline_max_ms = int(content[16:].strip())
                    except ValueError:
                        pass
                elif content.startswith("weight:"):
                    try:
                        weight = float(content[7:].strip())
                    except ValueError:
                        pass
                elif content.startswith("description:"):
                    description = content[12:].strip()

        return (
            compiler_options,
            baseline_min_ms,
            baseline_max_ms,
            weight,
            description,
            libraries,
        )

    def _determine_workload_type(self, filename: str) -> WorkloadType:
        """Determine workload type from filename"""
        filename_lower = filename.lower()

        if "cpu" in filename_lower or "intensive" in filename_lower:
            return WorkloadType.CPU_INTENSIVE
        elif "memory" in filename_lower or "heavy" in filename_lower:
            return WorkloadType.MEMORY_INTENSIVE
        elif "io" in filename_lower:
            return WorkloadType.IO_INTENSIVE
        elif (
            "simple" in filename_lower
            or "hello" in filename_lower
            or "quick" in filename_lower
        ):
            return WorkloadType.QUICK
        elif "error" in filename_lower:
            return WorkloadType.ERROR_CASES
        else:
            return WorkloadType.MIXED

    def load_all_scenarios(self) -> List[ScenarioConfig]:
        """Load all scenario configurations from files"""
        scenarios = []

        if not self.workload_dir.exists():
            raise FileNotFoundError(
                f"Workload directory not found: {self.workload_dir}"
            )

        # Load both .cpp and .cu files
        source_files = list(self.workload_dir.glob("*.cpp")) + list(
            self.workload_dir.glob("*.cu")
        )

        for source_file in source_files:
            try:
                with open(source_file, "r", encoding="utf-8") as f:
                    source_code = f.read()

                # Parse metadata from source
                (
                    compiler_options,
                    baseline_min_ms,
                    baseline_max_ms,
                    weight,
                    description,
                    libraries,
                ) = self._parse_metadata_from_source(source_code)

                # Determine workload type
                workload_type = self._determine_workload_type(source_file.stem)

                # Determine language from file extension
                language = "cuda" if source_file.suffix == ".cu" else "c++"

                scenario = ScenarioConfig(
                    name=source_file.stem,
                    workload_type=workload_type,
                    source_code=source_code,
                    compiler_options=compiler_options,
                    baseline_min_ms=baseline_min_ms,
                    baseline_max_ms=baseline_max_ms,
                    weight=weight,
                    description=description or f"Workload from {source_file.name}",
                    libraries=libraries,
                    language=language,
                )

                scenarios.append(scenario)

            except Exception as e:
                print(f"Warning: Failed to load scenario from {source_file}: {e}")
                continue

        return sorted(scenarios, key=lambda x: x.name)

    def get_scenarios_by_type(
        self, workload_type: WorkloadType
    ) -> List[ScenarioConfig]:
        """Get scenarios filtered by workload type"""
        all_scenarios = self.load_all_scenarios()
        return [s for s in all_scenarios if s.workload_type == workload_type]

    def get_cpu_intensive_scenarios(self) -> List[ScenarioConfig]:
        """Get CPU-intensive scenarios"""
        return self.get_scenarios_by_type(WorkloadType.CPU_INTENSIVE)

    def get_memory_intensive_scenarios(self) -> List[ScenarioConfig]:
        """Get memory-intensive scenarios"""
        return self.get_scenarios_by_type(WorkloadType.MEMORY_INTENSIVE)

    def get_io_intensive_scenarios(self) -> List[ScenarioConfig]:
        """Get IO-intensive scenarios"""
        return self.get_scenarios_by_type(WorkloadType.IO_INTENSIVE)

    def get_quick_scenarios(self) -> List[ScenarioConfig]:
        """Get quick scenarios"""
        return self.get_scenarios_by_type(WorkloadType.QUICK)

    def get_error_scenarios(self) -> List[ScenarioConfig]:
        """Get error case scenarios"""
        return self.get_scenarios_by_type(WorkloadType.ERROR_CASES)

    def get_mixed_scenarios(self) -> List[ScenarioConfig]:
        """Get a mixed set of scenarios representing realistic workload"""
        all_scenarios = self.load_all_scenarios()

        # Group by type
        scenarios_by_type: dict[WorkloadType, list[ScenarioConfig]] = {}
        for scenario in all_scenarios:
            if scenario.workload_type not in scenarios_by_type:
                scenarios_by_type[scenario.workload_type] = []
            scenarios_by_type[scenario.workload_type].append(scenario)

        # Select weighted samples from each category
        mixed = []

        # Add more quick scenarios (common case)
        quick_scenarios = scenarios_by_type.get(WorkloadType.QUICK, [])
        mixed.extend(quick_scenarios[:2])

        # Add some CPU intensive
        cpu_scenarios = scenarios_by_type.get(WorkloadType.CPU_INTENSIVE, [])
        mixed.extend(cpu_scenarios[:1])

        # Add some memory intensive
        memory_scenarios = scenarios_by_type.get(WorkloadType.MEMORY_INTENSIVE, [])
        mixed.extend(memory_scenarios[:1])

        # Add some IO intensive
        io_scenarios = scenarios_by_type.get(WorkloadType.IO_INTENSIVE, [])
        mixed.extend(io_scenarios[:1])

        # Add some error cases (less common)
        error_scenarios = scenarios_by_type.get(WorkloadType.ERROR_CASES, [])
        mixed.extend(error_scenarios[:1])

        return mixed

    def get_scenario_by_name(self, name: str) -> Optional[ScenarioConfig]:
        """Get a specific scenario by name"""
        all_scenarios = self.load_all_scenarios()
        for scenario in all_scenarios:
            if scenario.name == name:
                return scenario
        return None

    def list_available_scenarios(self) -> List[str]:
        """List names of all available scenarios"""
        all_scenarios = self.load_all_scenarios()
        return [s.name for s in all_scenarios]


def create_custom_scenario(
    name: str,
    source_file: Path,
    compiler_options: str = "-O2 -std=c++17",
    baseline_min_ms: int = 100,
    baseline_max_ms: int = 1000,
    weight: float = 1.0,
    description: str = "",
) -> ScenarioConfig:
    """Create a custom scenario from a source file"""

    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")

    with open(source_file, "r", encoding="utf-8") as f:
        source_code = f.read()

    # Try to determine workload type from filename or content
    workload_type = WorkloadType.MIXED
    if "cpu" in name.lower() or "intensive" in name.lower():
        workload_type = WorkloadType.CPU_INTENSIVE
    elif "memory" in name.lower():
        workload_type = WorkloadType.MEMORY_INTENSIVE
    elif "io" in name.lower():
        workload_type = WorkloadType.IO_INTENSIVE
    elif "simple" in name.lower() or "hello" in name.lower():
        workload_type = WorkloadType.QUICK
    elif "error" in name.lower():
        workload_type = WorkloadType.ERROR_CASES

    return ScenarioConfig(
        name=name,
        workload_type=workload_type,
        source_code=source_code,
        compiler_options=compiler_options,
        baseline_min_ms=baseline_min_ms,
        baseline_max_ms=baseline_max_ms,
        weight=weight,
        description=description or f"Custom scenario from {source_file.name}",
    )
