# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Support for collective multi-process execution.

This module exists to support coordinated parallel execution across multiple
isolated processes, with explicit global synchronization and communication
(e.g., MPI-style SPMD collectives).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _ProcessGroup:
    mpi_enabled: bool = False
    mpi_comm: int | None = None
    process_index: int = 0
    process_count: int = 1


def _detect_mpi() -> _ProcessGroup:
    """Automatically detect if running under MPI and return state.

    This function is called at module import time to detect MPI environment.
    It checks for common MPI environment variables and attempts to import mpi4py
    if they are detected.
    """
    # Check for common MPI environment variables
    mpi_env_vars = [
        "OMPI_COMM_WORLD_SIZE",  # OpenMPI
        "PMI_SIZE",  # Intel MPI, MPICH
        "SLURM_NTASKS",  # SLURM
        "MPI_LOCALNRANKS",  # IBM Spectrum MPI
    ]

    mpi_detected = any(var in os.environ for var in mpi_env_vars)

    if not mpi_detected:
        # single-process default
        return _ProcessGroup()

    try:
        from mpi4py import MPI
    except ImportError:
        # mpi4py not available, return single-process default
        return _ProcessGroup()

    comm = MPI.COMM_WORLD

    return _ProcessGroup(
        mpi_enabled=True,
        mpi_comm=comm,
        process_index=comm.Get_rank(),
        process_count=comm.Get_size(),
    )


def is_mpi_execution() -> bool:
    """Check if running in MPI mode.

    Returns:
       ``True`` if MPI is detected.
    """
    return _process_group.mpi_enabled


def _get_comm():
    """Get the MPI communicator.

    Returns:
        MPI communicator if distributed mode is enabled, ``None`` otherwise.
    """
    return _process_group.mpi_comm


def process_index() -> int:
    """Get the index of the current process.

    Returns:
        Process index (0 for the control process).
    """
    return _process_group.process_index


def process_count() -> int:
    """Get the total number of collective processes.

    Returns:
        Number of collective processes (1 if not in distributed mode).
    """
    return _process_group.process_count


def is_control_process() -> bool:
    """Return ``True`` if the current process is the control process.

    Returns:
        ``True`` if this is rank 0 or if not in distributed mode.
    """
    return process_index() == 0


def broadcast(obj: Any, root: int = 0) -> Any:
    """Broadcast a picklable object to all processes.

    Args:
        obj: Object to broadcast (only used on the root process).
        root: Root process index (default: 0 = broadcast from the control process).

    Returns:
        The broadcasted object on all ranks.
    """
    if not is_mpi_execution():
        return obj

    comm = _get_comm()
    return comm.bcast(obj, root=root)


def barrier() -> None:
    """Synchronize all processes.

    This is a no-op if not in distributed mode.
    """
    if is_mpi_execution():
        comm = _get_comm()
        comm.Barrier()


# Automatically detect MPI at module import time
_process_group = _detect_mpi()

# Made with Bob
