Introduction
============
Cyclops is a program to solve the Rotating Shallow Water Equations (RSWE) using a
Parareal solver, and in particular the novel Asymptotic Parallel-in-Time (APiNT)
method. The APinT method is described by `Haut and Wingate, 2014 <http://arxiv.org/abs/1303.6615>`_.

The program uses a Fourier spectral method and solves the equations on a 2-D
doubly-periodic domain with evenly-spaced gridpoints.

The purpose of this code is to act as a mathematical testbed to investigate the
mathematical basis of the APinT method, particularly the wave averaging for oscillatory-
stiff problems. It is hoped that future collaborators may easily work with and modify
the code to improve the understanding of this class of methods.

Dependencies
------------
- Python 3
- numpy
- mpi4py
- pickle
- pyfftw

Authors
-------
The authors of Cyclops are Adam G. Peddle (ap553@exeter.ac.uk) and Dr. Terry Haut.
