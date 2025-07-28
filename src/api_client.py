"""
Compiler Explorer API client for stress testing
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

import aiohttp
from asyncio_throttle import Throttler


class CompilationStatus(Enum):
    SUCCESS = "success"
    COMPILE_ERROR = "compile_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"


@dataclass
class CompilationResult:
    status: CompilationStatus
    compile_time_ms: Optional[float]
    execution_time_ms: Optional[float]
    total_time_ms: float
    compiler_stdout: str
    compiler_stderr: str
    program_stdout: str
    program_stderr: str
    exit_code: Optional[int]
    error_message: Optional[str]
    timestamp: float
    request_id: str


class CompilerExplorerClient:
    """Async client for Compiler Explorer API with rate limiting and retry logic"""

    def __init__(
        self,
        base_url: str = "https://compiler-explorer.com",
        max_requests_per_second: float = 10.0,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.throttler = Throttler(rate_limit=int(max_requests_per_second))
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "CompilerExplorerClient":
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "CE-StressTest/1.0"},
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.session:
            await self.session.close()

    async def compile_and_execute(
        self,
        source_code: str,
        compiler_id: str,
        options: str = "",
        libraries: Optional[List[Dict[str, str]]] = None,
        request_id: Optional[str] = None,
    ) -> CompilationResult:
        """
        Compile and execute code using Compiler Explorer API

        Args:
            source_code: The source code to compile
            compiler_id: Compiler ID (e.g., 'g122', 'clang1700')
            options: Compiler options string
            request_id: Optional request ID for tracking

        Returns:
            CompilationResult with timing and output information
        """
        if not request_id:
            request_id = f"req_{int(time.time() * 1000)}"

        start_time = time.time()

        payload = {
            "source": source_code,
            "options": {
                "userArguments": options,
                "executeParameters": {"args": [], "stdin": ""},
                "compilerOptions": {"executorRequest": True},
                "filters": {
                    "execute": True,
                    "binary": False,
                    "binaryObject": False,
                    "labels": True,
                    "libraryCode": False,
                    "directives": True,
                    "commentOnly": True,
                    "trim": False,
                    "demangle": True,
                    "intel": True,
                },
                "tools": [],
                "libraries": libraries or [],
            },
            "bypassCache": 1,
        }

        url = f"{self.base_url}/api/compiler/{compiler_id}/compile"

        # Apply rate limiting
        async with self.throttler:
            for attempt in range(self.max_retries + 1):
                try:
                    if self.session is None:
                        raise RuntimeError("Client session not initialized")
                    async with self.session.post(
                        url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                        },
                    ) as response:
                        total_time_ms = (time.time() - start_time) * 1000

                        if response.status == 200:
                            result_data = await response.json()
                            return self._parse_compilation_result(
                                result_data, total_time_ms, request_id
                            )
                        elif response.status == 429:  # Rate limited
                            if attempt < self.max_retries:
                                await asyncio.sleep(2**attempt)  # Exponential backoff
                                continue
                            else:
                                return CompilationResult(
                                    status=CompilationStatus.API_ERROR,
                                    compile_time_ms=None,
                                    execution_time_ms=None,
                                    total_time_ms=total_time_ms,
                                    compiler_stdout="",
                                    compiler_stderr="",
                                    program_stdout="",
                                    program_stderr="",
                                    exit_code=None,
                                    error_message=f"Rate limited after {self.max_retries} retries",
                                    timestamp=start_time,
                                    request_id=request_id,
                                )
                        else:
                            error_text = await response.text()
                            return CompilationResult(
                                status=CompilationStatus.API_ERROR,
                                compile_time_ms=None,
                                execution_time_ms=None,
                                total_time_ms=total_time_ms,
                                compiler_stdout="",
                                compiler_stderr="",
                                program_stdout="",
                                program_stderr="",
                                exit_code=None,
                                error_message=f"HTTP {response.status}: {error_text}",
                                timestamp=start_time,
                                request_id=request_id,
                            )

                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        total_time_ms = (time.time() - start_time) * 1000
                        return CompilationResult(
                            status=CompilationStatus.TIMEOUT,
                            compile_time_ms=None,
                            execution_time_ms=None,
                            total_time_ms=total_time_ms,
                            compiler_stdout="",
                            compiler_stderr="",
                            program_stdout="",
                            program_stderr="",
                            exit_code=None,
                            error_message=f"Request timeout after {self.timeout_seconds}s",
                            timestamp=start_time,
                            request_id=request_id,
                        )

                except Exception as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(2**attempt)
                        continue
                    else:
                        total_time_ms = (time.time() - start_time) * 1000
                        return CompilationResult(
                            status=CompilationStatus.API_ERROR,
                            compile_time_ms=None,
                            execution_time_ms=None,
                            total_time_ms=total_time_ms,
                            compiler_stdout="",
                            compiler_stderr="",
                            program_stdout="",
                            program_stderr="",
                            exit_code=None,
                            error_message=f"Unexpected error: {str(e)}",
                            timestamp=start_time,
                            request_id=request_id,
                        )

        # This should never be reached due to the loop logic, but needed for type safety
        raise RuntimeError("Exhausted all retry attempts without returning a result")

    def _parse_compilation_result(
        self, result_data: Dict[str, Any], total_time_ms: float, request_id: str
    ) -> CompilationResult:
        """Parse API response into CompilationResult"""

        def extract_text_lines(lines: List) -> str:
            """Helper to extract text from API response lines"""
            if not lines:
                return ""
            text_parts = []
            for line in lines:
                if isinstance(line, dict):
                    text_parts.append(line.get("text", ""))
                else:
                    text_parts.append(str(line))
            return "\n".join(text_parts)

        # Extract build result
        build_result = result_data.get("buildResult", {})
        compiler_stdout = extract_text_lines(build_result.get("stdout", []))
        compiler_stderr = extract_text_lines(build_result.get("stderr", []))

        # Check compilation success
        compilation_code = result_data.get("code", 1)

        if compilation_code != 0:
            # Compilation failed
            return CompilationResult(
                status=CompilationStatus.COMPILE_ERROR,
                compile_time_ms=None,
                execution_time_ms=None,
                total_time_ms=total_time_ms,
                compiler_stdout=compiler_stdout,
                compiler_stderr=compiler_stderr,
                program_stdout="",
                program_stderr="",
                exit_code=compilation_code,
                error_message=f"Compilation failed: {compiler_stderr[:2000]}" if compiler_stderr else "Compilation failed",
                timestamp=time.time(),
                request_id=request_id,
            )

        # Extract execution result
        exec_result = result_data.get("execResult", {})

        if exec_result:
            did_execute = exec_result.get("didExecute", False)
            exec_code = exec_result.get("code", 0)
            exec_time_ms = exec_result.get("execTime", 0)

            program_stdout = extract_text_lines(exec_result.get("stdout", []))
            program_stderr = extract_text_lines(exec_result.get("stderr", []))

            if did_execute:
                if exec_code == 0:
                    status = CompilationStatus.SUCCESS
                else:
                    status = CompilationStatus.RUNTIME_ERROR

                return CompilationResult(
                    status=status,
                    compile_time_ms=None,  # Not provided by API
                    execution_time_ms=exec_time_ms,
                    total_time_ms=total_time_ms,
                    compiler_stdout=compiler_stdout,
                    compiler_stderr=compiler_stderr,
                    program_stdout=program_stdout,
                    program_stderr=program_stderr,
                    exit_code=exec_code,
                    error_message=None
                    if status == CompilationStatus.SUCCESS
                    else "Runtime error",
                    timestamp=time.time(),
                    request_id=request_id,
                )

        # Compilation succeeded but no execution result
        return CompilationResult(
            status=CompilationStatus.SUCCESS,
            compile_time_ms=None,
            execution_time_ms=None,
            total_time_ms=total_time_ms,
            compiler_stdout=compiler_stdout,
            compiler_stderr=compiler_stderr,
            program_stdout="",
            program_stderr="",
            exit_code=0,
            error_message=None,
            timestamp=time.time(),
            request_id=request_id,
        )


class CMakeClient:
    """Client for CMake compilation endpoints (if different from regular compile)"""

    def __init__(self, base_client: CompilerExplorerClient):
        self.base_client = base_client

    async def compile_cmake_project(
        self,
        cmake_files: Dict[str, str],  # filename -> content
        compiler_id: str,
        cmake_args: str = "",
        request_id: Optional[str] = None,
    ) -> CompilationResult:
        """
        Compile a CMake project

        Args:
            cmake_files: Dictionary of filename -> file content
            compiler_id: Compiler ID
            cmake_args: CMake arguments
            request_id: Optional request ID

        Returns:
            CompilationResult
        """
        # This is a placeholder - the actual CMake API might be different
        # For now, we'll use the main CMakeLists.txt as source
        main_cmake = cmake_files.get("CMakeLists.txt", "")
        if not main_cmake:
            raise ValueError("CMakeLists.txt is required")

        # Use regular compile endpoint with CMake content
        # This might need adjustment based on actual CMake API
        return await self.base_client.compile_and_execute(
            main_cmake, compiler_id, cmake_args, request_id
        )
