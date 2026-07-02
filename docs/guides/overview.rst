########
Overview
########

This page summarizes the guides that are available. All guides are written to focus on specific aspects of the package. Examples of end-to-end workflows can be found in the tutorials hosted on the IBM Quantum Platform.

Getting started
---------------

These simple guides help you to get started quickly with the package.

- :doc:`Quickstart <quickstart>`

Beyond the basics
-----------------

These guides provide a deeper explanation of specific concepts and components from this package.

General
"""""""
- :doc:`How to choose the subspace dimension from its impact on eigenvalue estimation accuracy <choose_subspace_dimension>`

Fermionic systems
"""""""""""""""""
- :doc:`Scale SQD workflows for Fermionic systems with the Dice solver <integrate_dice_solver>`
- :doc:`Optimize the Hamiltonian basis with orbital optimization to improve energy estimations <use_oo_to_optimize_hamiltonian_basis>`
- :doc:`Understand open- and closed-shell options and their effect on subspace construction <select_open_closed_shell>`
- :doc:`Augment the pool of electronic configurations using Fermionic transition operators to improve energy estimations and compute energies of excited states <add_fermionic_excitations_to_configuration_pool>`

Spin systems
""""""""""""
- :doc:`Build a projected Hamiltonian matrix for qubit Hamiltonians <project_pauli_operators_onto_hilbert_subspaces>`
- :doc:`Benchmark performance of subspace projection for qubit Hamiltonians <benchmark_pauli_projection>`

.. toctree::
   :hidden:
   :caption: Getting started

   self
   quickstart

.. toctree::
   :hidden:
   :caption: General

   Choose the subspace dimension <choose_subspace_dimension>

.. toctree::
   :hidden:
   :caption: Fermionic systems

   Scale workloads with the Dice solver <integrate_dice_solver>
   Orbital optimization <use_oo_to_optimize_hamiltonian_basis>
   Open- and closed-shell options <select_open_closed_shell>
   Augment pool of electronic configurations <add_fermionic_excitations_to_configuration_pool>

.. toctree::
   :hidden:
   :caption: Spin systems

   Projected qubit Hamiltonians <project_pauli_operators_onto_hilbert_subspaces>
   Benchmark qubit Hamiltonian projection <benchmark_pauli_projection>
