Tests
=====

We currently provide two tests. These are both invocable from the command-line
interface and include the MPI-based parallelism.

These tests also provide examples of calling Cyclops.

Gaussian_test.sh
----------------

This is the first test to be run and should converge very quickly. An initially
stationary Gaussian height field is assumed on a coarse spatial grid.

E_run.sh
--------

This test is much more computationally intensive. It implements the initially
geostrophically balanced initial condition from `The Coherent Structures of
Shallow-Water Turbulence` (Polvani et al, 1994) on a 64 x 64 grid. This provides
an example of the use of Cyclops with externally-generated initial conditions.
