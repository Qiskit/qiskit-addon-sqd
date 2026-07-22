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
single-program, multiple-data (SPMD) style, in which the same program runs in
multiple isolated processes with explicit global synchronization and
communication between them. The current implementation relies on MPI, with a
single thread controlling each process (``MPI_THREAD_FUNNELED`` or lower).

Only one function currently supports being called collectively from all
processes: :func:`~qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian`.
When it is invoked collectively, the eigensolver step is where all processes
participate and contribute work, so an eigensolver implementation can use
every process. The remaining parts of the configuration-recovery loop have no
distributed implementation and run on the control process (rank 0) only.

Every other API in this package is intended to be called from the control
process alone, on its own local data, and carries no collective semantics.

The precise return-value and synchronization semantics of a collective function
belong with that function, so they are documented in its API reference rather
than repeated here. In general, the documentation of any collective function
should make clear:

- whether the function is meant to be called independently by each process on
  its own local data, or collectively by all processes;
- how its arguments must agree across processes;
- and how its return value is delivered---whether the result exists only on the
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
context collectively (for example, via ``MPI_Abort``).

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
