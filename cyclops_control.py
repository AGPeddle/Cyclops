import sys
import getopt
import numpy as np

def make_control():
    control = dict()

    control['filename'] = None
    control['Nx'] = 64  # Number of grid points. Domain is square.
    control['Nt'] = 5  # Number of grid points. Domain is square.
    control['delta'] = 500  # Number of fine timesteps to take per coarse step
    control['coarse_timestep'] = 0.1  # Coarse timestep
    control['fine_timestep'] = 0.0001  # Fine timestep
    control['final_time'] = 1.0  # Final time for computation
    control['Lx'] = 2.0*np.pi  # Side length of square domain
    control['conv_tol'] = 01.0e-6  # Tolerance for iterative convergence

    control['HMM_T0'] = 0.5  # Used in wave averaging kernel
    control['HMM_T0_L'] = 0.2  # Used in wave averaging kernel
    control['HMM_T0_M'] = 0.2  # Used in wave averaging kernel
    control['mu'] = 1.0e-4  # Hyperviscosity parameter
    control['outFileStem'] = None  # Stem for creation of output files
    control['assim_cycle_length'] = 10  # Time at which EnKF is applied

    control['f_naught'] = 0.001
    control['H_naught'] = 2
    control['gravity'] = 9.8

    return control

def setup_control(invals):

    control = make_control()

    opts, args = getopt.gnu_getopt(invals, '', ['filename=', 'Nx=', 'Lx=', 'coarse_timestep=', 'fine_timestep=', 'final_time=', 'outFileStem=', 'f_naught=', 'H_naught=', 'gravity=', 'Nt=', 'mu=', 'assim_cycle_length=', 'HMM_T0_L=', 'HMM_T0_M='])
    for o, a in opts:
        if o in ("--filename"):
            control['filename'] = a
        elif o in ("--Nx"):
            control['Nx'] = int(a)
        elif o in ("--Lx"):
            control['Lx'] = float(a)
        elif o in ("--Nt"):
            control['Nt'] = int(a)
        elif o in ("--fine_timestep"):
            control['fine_timestep'] = float(a)
        elif o in ("--coarse_timestep"):
            control['coarse_timestep'] = float(a)
        elif o in ("--final_time"):
            control['final_time'] = float(a)
        elif o in ("--outFileStem"):
            control['outFileStem'] = a
        elif o in ("--delta"):
            control['delta'] = int(a)
        elif o in ("--f_naught"):
            control['f_naught'] = float(a)
        elif o in ("--H_naught"):
            control['H_naught'] = float(a)
        elif o in ("--gravity"):
            control['gravity'] = float(a)
        elif o in ("--mu"):
            control['mu'] = float(a)
        elif o in ("--assim_cycle_length"):
            control['assim_cycle_length'] = int(a)
        elif o in ("--HMM_T0_L"):
            control['HMM_T0_L'] = float(a)
        elif o in ("--HMM_T0_M"):
            control['HMM_T0_M'] = float(a)

    return control


if __name__ == "__main__":
    control = setup_control(sys.argv[1:])
    print(control)
