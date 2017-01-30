import numpy as np
import pickle
import sys
import cyclops_control
from rswe_exponential_integrator import *

def read_ICs(control, filename, perturbation = True):
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

        # Find mean water depth and perturbation height
        if uvh == 2 and perturbation:
            mean_wd = np.mean(DataD)
            DataD -= mean_wd

            control['H_naught'] = mean_wd

            DataD = geopotential_transform(control, DataD)

        try:
            U = np.vstack((U, DataD))
        except UnboundLocalError:
            U = DataD

    return U


def geopotential_transform(control, h):
    """

    """

    div_sqrt_phi_naught = 1.0/np.sqrt(control['gravity']*control['H_naught'])
    h_out = h*control['gravity']*div_sqrt_phi_naught

    return h_out

def inv_geopotential_transform(control, h):
    """

    """
    sqrt_phi_naught = np.sqrt(control['gravity']*control['H_naught'])
    h_out = h*sqrt_phi_naught/control['gravity']

    return h_out

def compute_L_2_error(U_hat_ref, U_hat_approx, st):
    error = abs(np.reshape(U_hat_approx[:,:] - U_hat_ref[:,:], np.prod(np.shape(U_hat_ref[:,:]))))
    error = np.sqrt(np.sum(error**2))
    norm_val = abs(np.reshape(U_hat_ref[:,:], np.prod(np.shape(U_hat_ref[:,:]))))
    norm_val = np.sqrt(np.sum(norm_val**2))
    error = error/norm_val
    return error

def compute_L_infty_error(U_hat_ref, U_hat_approx, st):
    v1_hat = U_hat_ref[0,:]
    v2_hat = U_hat_ref[1,:]
    h_hat = U_hat_ref[2,:]
    # compute spatial solution
    v1_space = st.inverse_fft(v1_hat)
    v2_space = st.inverse_fft(v2_hat)
    h_space = st.inverse_fft(h_hat)
    # compute error in Fourier
    v1_hat_err = U_hat_approx[0,:] - U_hat_ref[0,:]
    v2_hat_err = U_hat_approx[1,:] - U_hat_ref[1,:]
    h_hat_err = U_hat_approx[2,:] - U_hat_ref[2,:]
    # compute error in space
    v1_space_err = st.inverse_fft(v1_hat_err)
    v1_space_err = np.max(np.abs(v1_space_err))/np.max(np.abs(v1_space))
    v2_space_err = st.inverse_fft(v2_hat_err)
    v2_space_err = np.max(np.abs(v2_space_err))/np.max(np.abs(v2_space))
    h_space_err = st.inverse_fft(h_hat_err)
    h_space_err = np.max(np.abs(h_space_err))/np.max(np.abs(h_space))
    error = max(v1_space_err,v2_space_err,h_space_err)
    return error


