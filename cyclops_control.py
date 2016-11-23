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
    control['mu'] = 1.0e-4  # Hyperviscosity parameter
    control['outFileStem'] = 'test'  # Stem for creation of output files

    control['f_naught'] = 1
    control['H_naught'] = 1
    control['gravity'] = 1

    return control

def setup_control(invals):

    control = make_control()

    opts, args = getopt.gnu_getopt(invals, '', ['filename=', 'Nx=', 'coarse_timestep=', 'fine_timestep=', 'final_time=', 'outFileStem=', 'f_naught=', 'H_naught=', 'gravity=', 'Nt='])
    for o, a in opts:
        if o in ("--filename"):
            control['filename'] = a
        elif o in ("--Nx"):
            control['Nx'] = int(a)
        elif o in ("--Nt"):
            control['Nt'] = int(a)
        elif o in ("--timestep"):
            control['timestep'] = float(a)
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

    return control


if __name__ == "__main__":
    control = setup_control(sys.argv[1:])
    print(control)
