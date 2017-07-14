#!/bin/bash

mpiexec -n 8 python3 ../source/cyclops.py --conv_tol=0.01 --coarse_timestep=0.1 --fine_timestep=0.001 --Nx=16 --working_dir=../test
