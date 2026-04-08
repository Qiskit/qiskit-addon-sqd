"""
Lightweight resource profiler for QCSC workloads (SBD, fulqrum, etc.).

Captures CPU/GPU memory, CPU utilization, and timing via before/after
snapshots — no background polling.  MPI-aware: rank 0 collects the max
of each metric across all ranks and prints a single summary.

Usage::

    from qiskit_addon_sqd.profiler import ResourceMonitor

    with ResourceMonitor() as mon:
        results = sbd.tpb_diag_from_files(...)
    mon.report()

    # With checkpoints (e.g. SQD loop)
    mon = ResourceMonitor()
    for i in range(max_iterations):
        mon.checkpoint(f"iter-{i}")
        result = diagonalize_fermionic_hamiltonian(...)
    mon.stop()
    mon.report()
"""

from __future__ import annotations

import os
import platform
import resource
import time
from dataclasses import dataclass, field

import psutil

# Optional GPU support
try:
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
        import pynvml

    pynvml.nvmlInit()
    _PYNVML_AVAILABLE = True
except Exception:
    _PYNVML_AVAILABLE = False

# Optional MPI support
try:
    from mpi4py import MPI as _MPI

    _MPI_AVAILABLE = True
except ImportError:
    _MPI = None
    _MPI_AVAILABLE = False


_GB = 1 << 30  # bytes per GB
_KB = 1 << 10


@dataclass
class _Snapshot:
    """A point-in-time resource snapshot."""

    name: str
    wall_time: float
    rss_gb: float
    gpu_used_gb: float | None
    gpu_total_gb: float | None


@dataclass
class _AggregatedMetric:
    """A metric with value and the rank that produced it."""

    value: float
    rank: int


class ResourceMonitor:
    """Lightweight before/after resource profiler with MPI aggregation.

    Attributes (available after ``stop()``):
        wall_time: Elapsed wall-clock seconds.
        cpu_memory_peak_gb: Peak RSS of this process (kernel-tracked).
        cpu_memory_total_gb: Total system RAM.
        cpu_memory_free_gb: Estimated free = total - peak × ranks_on_node.
        cpu_cores_allocated: ranks_on_node × OMP threads.
        cpu_cores_total: ``os.cpu_count()``.
        cpu_utilization_pct: Actual CPU % over the monitored interval.
        gpu_memory_used_gb: GPU memory used (max across ranks), or None.
        gpu_memory_total_gb: GPU total memory, or None.
        gpu_memory_free_gb: GPU free estimate, or None.

    Note:
        CPU peak RSS is tracked by the kernel (``ru_maxrss``) and reflects
        the true high-water mark.  GPU memory, however, is a point-in-time
        snapshot taken at ``stop()`` — NVML does not expose a peak counter.
        For workloads that keep CUDA allocations alive (cuBLAS, Thrust),
        the snapshot is a reasonable approximation.  A future version may
        add optional background polling for true GPU peak tracking.
        mpi_ranks: Total MPI world size (1 if no MPI).
        omp_threads: ``OMP_NUM_THREADS`` (1 if unset).
        checkpoints: List of snapshots taken via ``checkpoint()``.
    """

    def __init__(self) -> None:
        self._process = psutil.Process()

        # MPI info
        if _MPI_AVAILABLE and _MPI.COMM_WORLD.Get_size() > 1:
            self._comm = _MPI.COMM_WORLD
            self._rank = self._comm.Get_rank()
            self._world_size = self._comm.Get_size()
            # Ranks sharing the same physical node
            node_comm = self._comm.Split_type(_MPI.COMM_TYPE_SHARED)
            self._ranks_on_node = node_comm.Get_size()
            node_comm.Free()
        else:
            self._comm = None
            self._rank = 0
            self._world_size = 1
            self._ranks_on_node = 1

        # Static info
        mem = psutil.virtual_memory()
        self.cpu_memory_total_gb: float = mem.total / _GB
        self.cpu_cores_total: int = os.cpu_count() or 1
        self.omp_threads: int = int(os.environ.get("OMP_NUM_THREADS", "1"))
        self.mpi_ranks: int = self._world_size

        # GPU info (local device for this rank)
        self._gpu_handle = None
        self._has_gpu = False
        if _PYNVML_AVAILABLE:
            try:
                gpu_count = pynvml.nvmlDeviceGetCount()
                if gpu_count > 0:
                    gpu_id = self._rank % gpu_count
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    self._has_gpu = True
            except Exception:
                pass

        # State
        self._started = False
        self._stopped = False
        self._start_wall: float = 0.0
        self._start_cpu_times = None
        self.checkpoints: list[_Snapshot] = []

        # Results (populated by stop())
        self.wall_time: float = 0.0
        self.cpu_memory_peak_gb: float = 0.0
        self.cpu_memory_free_gb: float = 0.0
        self.cpu_cores_allocated: int = self._ranks_on_node * self.omp_threads
        self.cpu_utilization_pct: float = 0.0
        self.gpu_memory_used_gb: float | None = None
        self.gpu_memory_total_gb: float | None = None
        self.gpu_memory_free_gb: float | None = None

        # Aggregated results (rank 0 only, after stop)
        self._agg_peak_rss: _AggregatedMetric | None = None
        self._agg_gpu_used: _AggregatedMetric | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ResourceMonitor:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin monitoring."""
        if self._started:
            return
        self._started = True
        self._start_wall = time.perf_counter()
        self._start_cpu_times = self._process.cpu_times()

    def stop(self) -> None:
        """End monitoring, compute metrics, aggregate across MPI ranks."""
        if self._stopped or not self._started:
            return
        self._stopped = True

        # Wall time
        self.wall_time = time.perf_counter() - self._start_wall

        # Peak RSS from kernel (no polling needed)
        ru = resource.getrusage(resource.RUSAGE_SELF)
        if platform.system() == "Darwin":
            # macOS: ru_maxrss is in bytes
            self.cpu_memory_peak_gb = ru.ru_maxrss / _GB
        else:
            # Linux: ru_maxrss is in KB
            self.cpu_memory_peak_gb = ru.ru_maxrss * _KB / _GB

        # CPU utilization
        end_cpu = self._process.cpu_times()
        start_cpu = self._start_cpu_times
        cpu_seconds = (
            (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
        )
        if self.wall_time > 0:
            self.cpu_utilization_pct = (cpu_seconds / self.wall_time) * 100.0
        else:
            self.cpu_utilization_pct = 0.0

        # GPU memory
        if self._has_gpu and self._gpu_handle is not None:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self.gpu_memory_used_gb = info.used / _GB
                self.gpu_memory_total_gb = info.total / _GB
                self.gpu_memory_free_gb = (info.total - info.used) / _GB
            except Exception:
                pass

        # Estimated free CPU memory
        self.cpu_memory_free_gb = (
            self.cpu_memory_total_gb
            - self.cpu_memory_peak_gb * self._ranks_on_node
        )
        if self.cpu_memory_free_gb < 0:
            self.cpu_memory_free_gb = 0.0

        # MPI aggregation — collect max across ranks
        self._aggregate()

    def checkpoint(self, name: str = "") -> None:
        """Take a named snapshot of current resource usage.

        In MPI mode, includes a barrier to synchronize ranks.
        """
        if not self._started:
            self.start()

        if self._comm is not None:
            self._comm.Barrier()

        rss_gb = self._process.memory_info().rss / _GB
        gpu_used_gb = None
        gpu_total_gb = None
        if self._has_gpu and self._gpu_handle is not None:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_used_gb = info.used / _GB
                gpu_total_gb = info.total / _GB
            except Exception:
                pass

        self.checkpoints.append(
            _Snapshot(
                name=name or f"checkpoint-{len(self.checkpoints)}",
                wall_time=time.perf_counter() - self._start_wall,
                rss_gb=rss_gb,
                gpu_used_gb=gpu_used_gb,
                gpu_total_gb=gpu_total_gb,
            )
        )

    # ------------------------------------------------------------------
    # MPI aggregation
    # ------------------------------------------------------------------

    def _aggregate(self) -> None:
        """Collect max metrics across MPI ranks to rank 0."""
        local_rss = self.cpu_memory_peak_gb
        local_gpu = self.gpu_memory_used_gb if self.gpu_memory_used_gb is not None else 0.0

        if self._comm is None:
            self._agg_peak_rss = _AggregatedMetric(local_rss, 0)
            if self.gpu_memory_used_gb is not None:
                self._agg_gpu_used = _AggregatedMetric(local_gpu, 0)
            return

        # MPI_MAXLOC on (value, rank) pairs
        local_rss_pair = (local_rss, self._rank)
        local_gpu_pair = (local_gpu, self._rank)

        max_rss_pair = self._comm.allreduce(local_rss_pair, op=_MPI.MAXLOC)
        self._agg_peak_rss = _AggregatedMetric(max_rss_pair[0], max_rss_pair[1])

        if self.gpu_memory_used_gb is not None:
            max_gpu_pair = self._comm.allreduce(local_gpu_pair, op=_MPI.MAXLOC)
            self._agg_gpu_used = _AggregatedMetric(max_gpu_pair[0], max_gpu_pair[1])

        # Collect total CPU memory used across all ranks (for the summary line)
        total_cpu_used = self._comm.reduce(local_rss, op=_MPI.SUM, root=0)
        if self._rank == 0:
            self._total_cpu_used_gb = total_cpu_used
        else:
            self._total_cpu_used_gb = 0.0

        # Collect max CPU utilization
        local_util_pair = (self.cpu_utilization_pct, self._rank)
        max_util_pair = self._comm.allreduce(local_util_pair, op=_MPI.MAXLOC)
        self._agg_cpu_util = _AggregatedMetric(max_util_pair[0], max_util_pair[1])

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Print resource summary. Only prints on rank 0."""
        if self._rank != 0:
            return

        if not self._stopped:
            print("ResourceMonitor: not stopped yet — call stop() first")
            return

        lines = []
        width = 54

        if self._world_size > 1:
            header = f" Resource Summary (max across {self._world_size} ranks) "
        else:
            header = " Resource Summary "

        lines.append(f"\u250c{header:\u2500^{width}}\u2510")

        # Wall time
        lines.append(self._row("Wall time:", f"{self.wall_time:.1f}s"))

        # CPU cores
        lines.append(
            self._row(
                "CPU cores:",
                f"{self.cpu_cores_allocated} / {self.cpu_cores_total} allocated",
            )
        )

        # CPU utilization
        allocated_pct = self.cpu_cores_allocated * 100.0
        if self._world_size > 1 and hasattr(self, "_agg_cpu_util"):
            util = self._agg_cpu_util.value
        else:
            util = self.cpu_utilization_pct
        if allocated_pct > 0:
            efficiency = util / allocated_pct * 100.0
            lines.append(
                self._row(
                    "CPU utilization:",
                    f"{util:.0f}% ({efficiency:.0f}% of allocated)",
                )
            )
        else:
            lines.append(self._row("CPU utilization:", f"{util:.0f}%"))

        # CPU peak RSS
        if self._agg_peak_rss is not None:
            rss_val = self._agg_peak_rss.value
            rss_rank = self._agg_peak_rss.rank
            if self._world_size > 1:
                lines.append(
                    self._row(
                        "CPU peak RSS:",
                        f"{rss_val:.1f} GB  (rank {rss_rank})",
                    )
                )
            else:
                lines.append(self._row("CPU peak RSS:", f"{rss_val:.1f} GB"))

        # CPU memory total used
        if self._world_size > 1 and hasattr(self, "_total_cpu_used_gb"):
            lines.append(
                self._row(
                    "CPU memory:",
                    f"{self._total_cpu_used_gb:.1f} / {self.cpu_memory_total_gb:.1f} GB used",
                )
            )
        else:
            lines.append(
                self._row(
                    "CPU memory:",
                    f"{self.cpu_memory_peak_gb:.1f} / {self.cpu_memory_total_gb:.1f} GB used",
                )
            )

        # CPU free estimate
        lines.append(
            self._row("CPU free (est):", f"{self.cpu_memory_free_gb:.1f} GB")
        )

        # GPU memory
        if self._agg_gpu_used is not None:
            gpu_val = self._agg_gpu_used.value
            gpu_rank = self._agg_gpu_used.rank
            gpu_total = self.gpu_memory_total_gb or 0.0
            gpu_free = gpu_total - gpu_val
            if self._world_size > 1:
                lines.append(
                    self._row(
                        "GPU memory:",
                        f"{gpu_val:.1f} / {gpu_total:.1f} GB  (rank {gpu_rank})",
                    )
                )
            else:
                lines.append(
                    self._row(
                        "GPU memory:",
                        f"{gpu_val:.1f} / {gpu_total:.1f} GB",
                    )
                )
            lines.append(self._row("GPU free (est):", f"{gpu_free:.1f} GB"))

        # MPI / OMP summary
        lines.append(
            self._row(
                "",
                f"MPI ranks: {self._world_size}   OMP threads/rank: {self.omp_threads}",
            )
        )

        border = "\u2500" * width
        lines.append(f"\u2514{border}\u2518")

        # Checkpoints
        if self.checkpoints:
            lines.append("")
            lines.append("Checkpoints:")
            for cp in self.checkpoints:
                gpu_str = ""
                if cp.gpu_used_gb is not None:
                    gpu_str = f"  GPU: {cp.gpu_used_gb:.2f} GB"
                lines.append(
                    f"  {cp.name:20s}  t={cp.wall_time:7.2f}s  "
                    f"RSS: {cp.rss_gb:.2f} GB{gpu_str}"
                )

        print("\n".join(lines))

    @staticmethod
    def _row(label: str, value: str) -> str:
        """Format a single row of the report table."""
        if label:
            return f"\u2502 {label:20s}{value:32s}\u2502"
        return f"\u2502 {value:52s}\u2502"
