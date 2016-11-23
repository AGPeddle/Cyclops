import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import cyclops_control
import RSWE_direct
from sklearn.gaussian_process import GaussianProcessRegressor
from spectral_toolbox import SpectralToolbox
from rswe_exponential_integrator import *

station_spacing = 14
n_inits = 32

def read_ICs(control):
    filename = control['filename']
    for uvh, suffix in enumerate(('.u', '.v', '.h')):
        with open(filename + suffix, 'rb') as f:
            data=np.fromfile(filename + suffix, dtype=np.dtype('d'), count=-1, sep="")
            f.close()

        TotalBytes = data.nbytes # This is the total size of the data read - in bytes
        BytesPerDouble = data.itemsize # This is the total number of bytes per double
        TotalItems = TotalBytes/BytesPerDouble

        # Nx, Ny, Nz
        nx = int(data[1])
        ny = int(data[2])
        nz = int(data[3])
        assert nx == ny
        assert nz == 1
        control['Nx'] = nx - 1

        # Declare arrays and store x, y and z coordinates
        xcoord = np.zeros(nx)
        ycoord = np.zeros(ny)
        xcoord = data[4:3+nx]
        ycoord = data[4+nx:3+nx+ny]

        # Declare arrays and store the data
        DataD = np.zeros((1, nx-1,ny-1))
                
        # Store the data in a way that makes more sense
        icounter = 3+nx+ny+nz

        for i in np.arange(1, nx):
            for j in np.arange(1, ny):
                icounter += 1
                DataD[0, i-1, j-1]=data[icounter]
            icounter += 1

        try:
            U = np.vstack((U, DataD))
        except UnboundLocalError:
            U = DataD

    return U

def main(control):
    """

    """
    # Hardcoded for now:
    control['final_time'] = 0.1

    # Set up initial (truth) field
    ICs = read_ICs(control)
    truth = ICs

    st = SpectralToolbox(control['Nx'], control['Lx'])
    expInt = ExponentialIntegrator_Dim(control)

    X = np.linspace(0,control['Lx'],control['Nx'])
    x, y = np.meshgrid(X,X)

    # Propagate it through Nts
    plt.figure()
    plt.contourf(x, y,  truth[2, :, :])
    plt.colorbar()
    plt.show()
    fulltime = 0
    for i in range(0, control['Nt']):
        truth[:, :, :] = RSWE_direct.solve('fine_propagator', control, st, expInt, truth[:, :, :])
        fulltime += control['final_time']
        plt.figure()
        plt.contourf(x, y,  truth[2, :, :])
        plt.colorbar()
        plt.show()

        with open("{}_truth_{}.dat".format(control['filename'], i) 'wb') as f:
            data = dict()
            data['time'] = fulltime
            data['uvh'] = truth

            pickle.dump(data, f)

if __name__ == "__main__":
    control_in = cyclops_control.setup_control(sys.argv[1:])
    main(control_in)
