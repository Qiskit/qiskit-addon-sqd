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

"""Distributed computing support for MPI-aware solvers."""

from __future__ import annotations

import os
from typing import Any

# Global state for distributed mode
_distributed_state: dict[str, Any] = {
    "enabled": False,
    "comm": None,
    "rank": 0,
    "size": 1,
}


def _detect_mpi() -> None:
    """Automatically detect if running under MPI and initialize state.
    
    This function is called at module import time to detect MPI environment.
    It checks for common MPI environment variables and attempts to import mpi4py.
    """
    # Check for common MPI environment variables
    mpi_env_vars = [
        "OMPI_COMM_WORLD_SIZE",  # OpenMPI
        "PMI_SIZE",               # Intel MPI, MPICH
        "SLURM_NTASKS",          # SLURM
        "MPI_LOCALNRANKS",       # IBM Spectrum MPI
    ]
    
    mpi_detected = any(var in os.environ for var in mpi_env_vars)
    
    if mpi_detected:
        try:
            from mpi4py import MPI
            
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            
            # Only enable distributed mode if we have multiple processes
            if size > 1:
                _distributed_state["enabled"] = True
                _distributed_state["comm"] = comm
                _distributed_state["rank"] = rank
                _distributed_state["size"] = size
        except ImportError:
            # mpi4py not available, stay in single-process mode
            pass


def is_distributed() -> bool:
    """Check if running in distributed mode.
    
    Returns:
        True if MPI is detected and multiple processes are running.
    """
    return _distributed_state["enabled"]


def get_comm():
    """Get the MPI communicator.
    
    Returns:
        MPI communicator if distributed mode is enabled, None otherwise.
    """
    return _distributed_state["comm"]


def get_rank() -> int:
    """Get the MPI rank of the current process.
    
    Returns:
        MPI rank (0 if not in distributed mode).
    """
    return _distributed_state["rank"]


def get_size() -> int:
    """Get the total number of MPI processes.
    
    Returns:
        Number of MPI processes (1 if not in distributed mode).
    """
    return _distributed_state["size"]


def is_main_rank() -> bool:
    """Check if this is the main rank (rank 0).
    
    Returns:
        True if this is rank 0 or not in distributed mode.
    """
    return get_rank() == 0


def broadcast(obj: Any, root: int = 0) -> Any:
    """Broadcast an object from root to all processes.
    
    Args:
        obj: Object to broadcast (only used on root rank).
        root: Root rank for broadcast (default: 0).
        
    Returns:
        The broadcasted object on all ranks.
    """
    if not is_distributed():
        return obj
    
    comm = get_comm()
    return comm.bcast(obj, root=root)


def barrier() -> None:
    """Synchronize all processes.
    
    This is a no-op if not in distributed mode.
    """
    if is_distributed():
        comm = get_comm()
        comm.Barrier()


# Automatically detect MPI at module import time
_detect_mpi()

# Made with Bob
