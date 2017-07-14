#!/bin/bash

mpiexec -n 64 python3 ../source/cyclops.py --coarse_timestep=0.001 --fine_timestep=0.00001 --filename=E64 --gravity=24.999999254941958 --f_naught=351.85837720205683 --Lx=1.0 --working_dir=../test
