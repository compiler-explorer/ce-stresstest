"""
Load generation patterns for stress testing
"""

import asyncio
import time
import math
from typing import AsyncGenerator, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .scenarios import ScenarioConfig


class LoadPatternType(Enum):
    STEADY = "steady"
    BURST = "burst"
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    WAVE = "wave"
    SUSTAINED_HIGH = "sustained_high"


@dataclass
class LoadEvent:
    """Represents a single load event (compilation request)"""

    timestamp: float
    scenario: ScenarioConfig
    request_id: str
    metadata: Dict[str, Any]


class LoadPattern(ABC):
    """Abstract base class for load patterns"""

    def __init__(self, scenarios: List[ScenarioConfig], duration_seconds: int):
        self.scenarios = scenarios
        self.duration_seconds = duration_seconds
        self.start_time: Optional[float] = None

    @abstractmethod
    async def generate_load_events(self) -> AsyncGenerator[LoadEvent, None]:  # type: ignore
        """Generate load events according to the pattern"""
        pass

    def _select_scenario(self, current_time: float) -> ScenarioConfig:
        """Select a scenario based on weights and current time"""
        if not self.scenarios:
            raise ValueError("No scenarios available")

        # Simple weighted random selection
        total_weight = sum(s.weight for s in self.scenarios)
        if total_weight <= 0:
            return self.scenarios[0]

        # Use current time as seed for some determinism
        import random

        random.seed(int(current_time * 1000) % (2**32))

        rand_val = random.random() * total_weight
        current_weight = 0.0

        for scenario in self.scenarios:
            current_weight += scenario.weight
            if rand_val <= current_weight:
                return scenario

        return self.scenarios[-1]  # Fallback

    def _generate_request_id(self, pattern_type: str, sequence: int) -> str:
        """Generate a unique request ID"""
        timestamp = int(time.time() * 1000)
        return f"{pattern_type}_{timestamp}_{sequence}"


class SteadyLoadPattern(LoadPattern):
    """Generates steady load at constant RPS"""

    def __init__(
        self, scenarios: List[ScenarioConfig], duration_seconds: int, rps: float
    ):
        super().__init__(scenarios, duration_seconds)
        self.rps = rps
        self.interval = 1.0 / rps if rps > 0 else 1.0

    async def generate_load_events(self) -> AsyncGenerator[LoadEvent, None]:  # type: ignore
        self.start_time = time.time()
        sequence = 0

        while (time.time() - self.start_time) < self.duration_seconds:
            current_time = time.time()
            scenario = self._select_scenario(current_time)

            yield LoadEvent(
                timestamp=current_time,
                scenario=scenario,
                request_id=self._generate_request_id("steady", sequence),
                metadata={"pattern": "steady", "rps": self.rps, "sequence": sequence},
            )

            sequence += 1
            await asyncio.sleep(self.interval)


class BurstLoadPattern(LoadPattern):
    """Generates baseline traffic with periodic bursts"""

    def __init__(
        self,
        scenarios: List[ScenarioConfig],
        duration_seconds: int,
        baseline_rps: float,
        burst_rps: float,
        burst_duration_seconds: int,
        burst_interval_seconds: int,
    ):
        super().__init__(scenarios, duration_seconds)
        self.baseline_rps = baseline_rps
        self.burst_rps = burst_rps
        self.burst_duration_seconds = burst_duration_seconds
        self.burst_interval_seconds = burst_interval_seconds

    async def generate_load_events(self) -> AsyncGenerator[LoadEvent, None]:  # type: ignore
        self.start_time = time.time()
        sequence = 0

        while (time.time() - self.start_time) < self.duration_seconds:
            elapsed = time.time() - self.start_time
            current_time = time.time()

            # Determine if we're in a burst period
            cycle_position = elapsed % self.burst_interval_seconds
            in_burst = cycle_position < self.burst_duration_seconds

            current_rps = self.burst_rps if in_burst else self.baseline_rps
            interval = 1.0 / current_rps if current_rps > 0 else 1.0

            scenario = self._select_scenario(current_time)

            yield LoadEvent(
                timestamp=current_time,
                scenario=scenario,
                request_id=self._generate_request_id("burst", sequence),
                metadata={
                    "pattern": "burst",
                    "current_rps": current_rps,
                    "in_burst": in_burst,
                    "sequence": sequence,
                },
            )

            sequence += 1
            await asyncio.sleep(interval)


class RampUpLoadPattern(LoadPattern):
    """Gradually increases load from min to max RPS"""

    def __init__(
        self,
        scenarios: List[ScenarioConfig],
        duration_seconds: int,
        min_rps: float,
        max_rps: float,
    ):
        super().__init__(scenarios, duration_seconds)
        self.min_rps = min_rps
        self.max_rps = max_rps

    async def generate_load_events(self) -> AsyncGenerator[LoadEvent, None]:  # type: ignore
        self.start_time = time.time()
        sequence = 0

        while (time.time() - self.start_time) < self.duration_seconds:
            elapsed = time.time() - self.start_time
            current_time = time.time()

            # Calculate current RPS based on linear ramp
            progress = elapsed / self.duration_seconds
            current_rps = self.min_rps + (self.max_rps - self.min_rps) * progress
            interval = 1.0 / current_rps if current_rps > 0 else 1.0

            scenario = self._select_scenario(current_time)

            yield LoadEvent(
                timestamp=current_time,
                scenario=scenario,
                request_id=self._generate_request_id("ramp_up", sequence),
                metadata={
                    "pattern": "ramp_up",
                    "current_rps": current_rps,
                    "progress": progress,
                    "sequence": sequence,
                },
            )

            sequence += 1
            await asyncio.sleep(interval)


class RampDownLoadPattern(LoadPattern):
    """Gradually decreases load from max to min RPS"""

    def __init__(
        self,
        scenarios: List[ScenarioConfig],
        duration_seconds: int,
        max_rps: float,
        min_rps: float,
    ):
        super().__init__(scenarios, duration_seconds)
        self.max_rps = max_rps
        self.min_rps = min_rps

    async def generate_load_events(self) -> AsyncGenerator[LoadEvent, None]:  # type: ignore
        self.start_time = time.time()
        sequence = 0

        while (time.time() - self.start_time) < self.duration_seconds:
            elapsed = time.time() - self.start_time
            current_time = time.time()

            # Calculate current RPS based on linear ramp down
            progress = elapsed / self.duration_seconds
            current_rps = self.max_rps - (self.max_rps - self.min_rps) * progress
            interval = 1.0 / current_rps if current_rps > 0 else 1.0

            scenario = self._select_scenario(current_time)

            yield LoadEvent(
                timestamp=current_time,
                scenario=scenario,
                request_id=self._generate_request_id("ramp_down", sequence),
                metadata={
                    "pattern": "ramp_down",
                    "current_rps": current_rps,
                    "progress": progress,
                    "sequence": sequence,
                },
            )

            sequence += 1
            await asyncio.sleep(interval)


class WaveLoadPattern(LoadPattern):
    """Generates sinusoidal wave pattern of load"""

    def __init__(
        self,
        scenarios: List[ScenarioConfig],
        duration_seconds: int,
        base_rps: float,
        amplitude_rps: float,
        period_seconds: int,
    ):
        super().__init__(scenarios, duration_seconds)
        self.base_rps = base_rps
        self.amplitude_rps = amplitude_rps
        self.period_seconds = period_seconds

    async def generate_load_events(self) -> AsyncGenerator[LoadEvent, None]:  # type: ignore
        self.start_time = time.time()
        sequence = 0

        while (time.time() - self.start_time) < self.duration_seconds:
            elapsed = time.time() - self.start_time
            current_time = time.time()

            # Calculate current RPS using sine wave
            phase = (elapsed / self.period_seconds) * 2 * math.pi
            wave_value = math.sin(phase)
            current_rps = self.base_rps + self.amplitude_rps * wave_value

            # Ensure RPS doesn't go negative
            current_rps = max(0.1, current_rps)
            interval = 1.0 / current_rps

            scenario = self._select_scenario(current_time)

            yield LoadEvent(
                timestamp=current_time,
                scenario=scenario,
                request_id=self._generate_request_id("wave", sequence),
                metadata={
                    "pattern": "wave",
                    "current_rps": current_rps,
                    "wave_value": wave_value,
                    "phase": phase,
                    "sequence": sequence,
                },
            )

            sequence += 1
            await asyncio.sleep(interval)


class SustainedHighLoadPattern(LoadPattern):
    """Generates sustained high load for maximum stress testing"""

    def __init__(
        self, scenarios: List[ScenarioConfig], duration_seconds: int, rps: float
    ):
        super().__init__(scenarios, duration_seconds)
        self.rps = rps
        self.interval = 1.0 / rps if rps > 0 else 0.1

    async def generate_load_events(self) -> AsyncGenerator[LoadEvent, None]:  # type: ignore
        self.start_time = time.time()
        sequence = 0

        while (time.time() - self.start_time) < self.duration_seconds:
            current_time = time.time()
            scenario = self._select_scenario(current_time)

            yield LoadEvent(
                timestamp=current_time,
                scenario=scenario,
                request_id=self._generate_request_id("sustained", sequence),
                metadata={
                    "pattern": "sustained_high",
                    "rps": self.rps,
                    "sequence": sequence,
                },
            )

            sequence += 1
            await asyncio.sleep(self.interval)


class LoadPatternFactory:
    """Factory for creating load patterns"""

    @staticmethod
    def create_steady_load(
        scenarios: List[ScenarioConfig], duration_seconds: int, rps: float
    ) -> SteadyLoadPattern:
        return SteadyLoadPattern(scenarios, duration_seconds, rps)

    @staticmethod
    def create_burst_load(
        scenarios: List[ScenarioConfig],
        duration_seconds: int,
        baseline_rps: float,
        burst_rps: float,
        burst_duration_seconds: int = 30,
        burst_interval_seconds: int = 120,
    ) -> BurstLoadPattern:
        return BurstLoadPattern(
            scenarios,
            duration_seconds,
            baseline_rps,
            burst_rps,
            burst_duration_seconds,
            burst_interval_seconds,
        )

    @staticmethod
    def create_ramp_up(
        scenarios: List[ScenarioConfig],
        duration_seconds: int,
        min_rps: float,
        max_rps: float,
    ) -> RampUpLoadPattern:
        return RampUpLoadPattern(scenarios, duration_seconds, min_rps, max_rps)

    @staticmethod
    def create_ramp_down(
        scenarios: List[ScenarioConfig],
        duration_seconds: int,
        max_rps: float,
        min_rps: float,
    ) -> RampDownLoadPattern:
        return RampDownLoadPattern(scenarios, duration_seconds, max_rps, min_rps)

    @staticmethod
    def create_wave_pattern(
        scenarios: List[ScenarioConfig],
        duration_seconds: int,
        base_rps: float,
        amplitude_rps: float,
        period_seconds: int = 300,
    ) -> WaveLoadPattern:
        return WaveLoadPattern(
            scenarios, duration_seconds, base_rps, amplitude_rps, period_seconds
        )

    @staticmethod
    def create_sustained_high_load(
        scenarios: List[ScenarioConfig], duration_seconds: int, rps: float
    ) -> SustainedHighLoadPattern:
        return SustainedHighLoadPattern(scenarios, duration_seconds, rps)

    @staticmethod
    def create_pattern_from_config(
        pattern_config: Dict[str, Any], scenarios: List[ScenarioConfig]
    ) -> LoadPattern:
        """Create a load pattern from configuration dictionary"""
        pattern_type = pattern_config.get("type", "steady")
        duration = pattern_config.get("duration_seconds", 300)

        if pattern_type == "steady":
            return LoadPatternFactory.create_steady_load(
                scenarios, duration, pattern_config.get("rps", 10)
            )
        elif pattern_type == "burst":
            return LoadPatternFactory.create_burst_load(
                scenarios,
                duration,
                pattern_config.get("baseline_rps", 5),
                pattern_config.get("burst_rps", 20),
                pattern_config.get("burst_duration_seconds", 30),
                pattern_config.get("burst_interval_seconds", 120),
            )
        elif pattern_type == "ramp_up":
            return LoadPatternFactory.create_ramp_up(
                scenarios,
                duration,
                pattern_config.get("min_rps", 1),
                pattern_config.get("max_rps", 20),
            )
        elif pattern_type == "ramp_down":
            return LoadPatternFactory.create_ramp_down(
                scenarios,
                duration,
                pattern_config.get("max_rps", 20),
                pattern_config.get("min_rps", 1),
            )
        elif pattern_type == "wave":
            return LoadPatternFactory.create_wave_pattern(
                scenarios,
                duration,
                pattern_config.get("base_rps", 10),
                pattern_config.get("amplitude_rps", 5),
                pattern_config.get("period_seconds", 300),
            )
        elif pattern_type == "sustained_high":
            return LoadPatternFactory.create_sustained_high_load(
                scenarios, duration, pattern_config.get("rps", 50)
            )
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")


# Convenience functions for common patterns
async def generate_scaling_test_patterns(
    scenarios: List[ScenarioConfig],
    instance_counts: List[int],
    rps_per_instance: float,
    duration_per_test: int,
) -> AsyncGenerator[tuple[int, LoadPattern], None]:
    """Generate load patterns for scaling tests"""

    for instance_count in instance_counts:
        total_rps = instance_count * rps_per_instance
        pattern = LoadPatternFactory.create_steady_load(
            scenarios, duration_per_test, total_rps
        )
        yield instance_count, pattern
