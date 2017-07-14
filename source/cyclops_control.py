#!/usr/bin/env/ python3
"""
This module provides a dictionary through which all computational and physical
parameters are organised and accessed. All have semi-reasonable default values
and may be chosen by the user via the command-line interface.

Functions
---------
- `make_control` : Creates a control object with the default values
- `setup_control` : Updates the default control object with user selections

Parameters
----------
- `filename` : Filename root for input initial conditions <str>
- `Nx` : Number of gridpoints along one direction. Domain is Nx X Nx square <int>
- `Nt` : Number of coarse timesteps. In practice set to number of processes <int>
- `coarse_timestep` : Coarse timestep length <float>
- `fine_timestep` : Fine timestep length <float>
- `Lx` : Side length of the square domain <float>
- `conv_tol` : The convergence criterion for iterative error <float>
- `HMM_T0` : The length of the averaging window (absoltue) <float>
- `mu` : Hyperviscosity coefficient <float>
- `outFileStem` : Optional stem for output filenames <str>
- `f_naught` : Coriolis parameters <float>
- `H_naught` : Mean water depth <float>
- `gravity` : Gravitational acceleration, g <float>

| Author: Adam G. Peddle
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import sys
import getopt
import numpy as np

def make_control():
    """
    Initialises the control object to the default values. New defaults
    should be placed here.

    **Returns**
    - `control` : Default control object
    """

    control = dict()

    control['filename'] = None
    control['Nx'] = 64  # Number of grid points. Domain is square.
    control['Nt'] = 5  # Number of coarse timesteps
    control['coarse_timestep'] = 0.1  # Coarse timestep
    control['fine_timestep'] = 0.0001  # Fine timestep
    control['Lx'] = 2.0*np.pi  # Side length of square domain
    control['conv_tol'] = 01.0e-6  # Tolerance for iterative convergence

    control['HMM_T0'] = 0.5  # Used in wave averaging kernel
    control['mu'] = 1.0e-4  # Hyperviscosity parameter
    control['outFileStem'] = None  # Stem for creation of output files

    control['f_naught'] = 0.001
    control['H_naught'] = 2
    control['gravity'] = 9.8

    return control

def setup_control(invals):
    """
    Creates and updates the default control object with user selections.
    Input should come via stdin and relies on sys.argv for parsing.

    **Parameters**
    - `invals` : command-line input values in Unix-style

    **Returns**
    - `control` : Default control object
    """

    control = make_control()

    opts, args = getopt.gnu_getopt(invals, '', ['filename=', 'working_dir=', 'Nx=', 'Lx=', 'conv_tol=', 'coarse_timestep=', 'fine_timestep=', 'outFileStem=', 'f_naught=', 'H_naught=', 'gravity=', 'Nt=', 'mu=', 'HMM_T0='])
    for o, a in opts:
        if o in ("--filename"):
            control['filename'] = a
        elif o in ("--working_dir"):
            control["working_dir"] = a
        elif o in ("--Nx"):
            control['Nx'] = int(a)
        elif o in ("--Lx"):
            control['Lx'] = float(a)
        elif o in ("--conv_tol"):
            control['conv_tol'] = float(a)
        elif o in ("--Nt"):
            control['Nt'] = int(a)
        elif o in ("--fine_timestep"):
            control['fine_timestep'] = float(a)
        elif o in ("--coarse_timestep"):
            control['coarse_timestep'] = float(a)
        elif o in ("--outFileStem"):
            control['outFileStem'] = a
        elif o in ("--f_naught"):
            control['f_naught'] = float(a)
        elif o in ("--H_naught"):
            control['H_naught'] = float(a)
        elif o in ("--gravity"):
            control['gravity'] = float(a)
        elif o in ("--mu"):
            control['mu'] = float(a)
        elif o in ("--HMM_T0"):
            control['HMM_T0'] = float(a)

    return control


if __name__ == "__main__":
    control = setup_control(sys.argv[1:])
    print(control)
