##############################################################
Support for multi-process and multi-threaded acceleration
##############################################################

This page documents the extent to which ``qiskit-addon-sqd`` supports
multi-threaded and multi-process acceleration, and the assumptions that a
high-performance-computing (HPC) developer can rely on when integrating this
package into an accelerated workload.

The single-threaded assumption
==============================

Unless otherwise specified, the APIs in this package are meant to be called
from a single thread. High-level APIs exposed by this package are not
guaranteed to be re-entrant, and end users should not invoke them from any
thread other than the main thread.

Collective multi-process execution
===================================

This package supports collective multi-process acceleration in the
single-program, multiple-data (SPMD) style, in which the entire program is
launched as multiple isolated processes that run with explicit global
synchronization and communication between them (for example,
``mpirun -n 128 python my_program.py``). While the API of this package is
designed to be general, the current implementation relies on MPI,
the standard message-passing API for HPC systems.  Regardless of implementation,
this package makes the assumption that there is a single thread controlling
each process. In the case of MPI, this corresponds to ``MPI_THREAD_FUNNELED`` or lower.
Because it composes cleanly with job schedulers and other parallel software and
scales to large process counts, this is the recommended model for
high-performance and large-scale workloads, and it is the subject of this page.

Only one function currently supports being called collectively from all
processes: :func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`.
When it is invoked collectively, the eigensolver step is where all processes
participate and contribute work, so an eigensolver implementation can use
every process. The remaining parts of the configuration-recovery loop have no
distributed implementation and run on the control process (rank 0) only.

Some existing eigensolver implementations instead require the calling program
to be outside an MPI/SPMD environment, as they launch and manage their own parallel processes internally, for
example by invoking ``mpirun`` on the user's behalf. That mode is convenient
for interactive and notebook-based work and remains supported; its requirements
on the calling program are described in the API reference of the
:func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` function,
which accepts such a solver.

Every other API in this package must be called from the control
process alone, on its own local data, and carries no collective semantics.

The return value and synchronization semantics of a collective function
belong with that function, so they are documented in its API reference.
In general, the documentation of any collective function
should make clear:

- Whether the function is meant to be called independently by each process on
  its own local data, or collectively by all processes.
- How its arguments must agree across processes.
- How its return value is delivered---whether the result exists only on the
  control process, whether all processes receive the same value, or whether
  each process receives a handle to a local portion of a distributed data
  structure.

Error handling in a multi-process context
=========================================

A function that is called collectively from all processes must not raise an
exception or abort only a single process, because that would leave the
remaining processes deadlocked or in an inconsistent state. Instead, error
handling follows fail-stop semantics for the execution context as a whole: upon
an error, the implementation must be prepared to abort the entire execution
context collectively (for example, by using ``MPI_Abort``).

The API cannot guarantee coordinated error reporting or collective delivery of
exceptions across processes, because MPI implementations do not provide such
guarantees. An implementation might additionally attempt to make an error visible
to all participating processes---for example, by having each process raise an
exception---but this can only be provided on a best-effort basis and must not be
relied upon for correctness or recovery.

These are properties that a collective implementation is permitted to have, not
guarantees about any particular one. In this package, the only collective entry
point is ``sci_solver`` (see the preceding section); its default implementation is not
distributed, so these considerations apply to a custom, collective
``sci_solver`` supplied by the user.
