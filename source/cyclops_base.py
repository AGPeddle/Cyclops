#!/usr/bin/env/ python3
"""
Library of miscellaneous helper functions for Cyclops init/admin tasks.


Functions
---------

- `read_ICs` : Read initial conditions from the Polvani experiments
- `geopotential_transform` : Transform the height field from (u,v,h) to skew-Hermitian (u,v,phi)
- `inv_geopotential_transform` : Transform the height field from skew-Hermitian (u,v,phi) to (u,v,h)
- `compute_L_2_error` : Compute the L_2 error between two vectors
- `compute_L_infty_error` : Compute the L_infty (sup-norm) error between two vectors
- `h_init` : Generate an initially stationary Gaussian height field for testing

| Authors: Adam G. Peddle, Martin Schreiber
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import numpy as np
import pickle
import sys
import cyclops_control

def read_ICs(control, filename, perturbation = True):
    """
    Read and return initial conditions for the Polvani experiments as provided
    by Beth Wingate's code.

    **Parameters**
    - `control` : a control object
    - `filename` : the filename to be read <string>
    - `perturbation` : flag if ICs are not in geopotential perturbation coordinates

    **Returns**
    - `XX` : x-coordinate matrix from the x-coord vector
    - `YY` : y-coordinate matrix from the y-coord vector
    - `U` : initial condition in realspace, sized (3, Nx, Nx) with (u, v, {h || phi}) along the first rank
    """

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

    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])
    XX, YY = np.meshgrid(x_grid, x_grid)

    return XX, YY, U


def geopotential_transform(control, h):
    """
    Transforms the height field from perturbation height to geopotential
    height (phi) defined as

    .. math:: \\phi = \\frac{gh}{\\sqrt{gH_{0}}}.

    This permits a skew-Hermitian formulation for the linear operator.

    **Parameters**
    - `control` : control object
    - `h` : the height field (np.array of size (Nx, Nx))

    **Returns**
    - `h_out` : the geopotential height (np.array of size (Nx, Nx))
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
    """
    Computes and returns the L_2 error.

    The L_2 error is defined as:

    .. math:: L_{2} = \\frac{\\sqrt{\\sum e^{2}}}{\\sqrt{\\sum u_{ref}^{2}}}

    where e is the absolute error.

    The errors are computed in Fourier space but returned in real space. The returned error
    is the L_2 error of all three variables together.

    The reference solution will generally be the solution at the previous iteration, which
    is used for measuring convergence.

    **Parameters**

    - `U_hat_ref` : the solution at the previous timestep (or a reference solution)
    - `U_hat_approx` : the solution at the current timestep
    - `st` : spectral toolbox object

    **Returns**

    - `error` : The computed L_2 error
    """

    error = abs(np.reshape(U_hat_approx[:,:] - U_hat_ref[:,:], np.prod(np.shape(U_hat_ref[:,:]))))
    error = np.sqrt(np.sum(error**2))
    norm_val = abs(np.reshape(U_hat_ref[:,:], np.prod(np.shape(U_hat_ref[:,:]))))
    norm_val = np.sqrt(np.sum(norm_val**2))
    error = error/norm_val
    return error

def compute_L_infty_error(U_hat_ref, U_hat_approx, st):
    """
    Compute the L_infty error at a given timestep.

    The L_infty error is defined as:

    .. math:: L_{\\infty} = \\max\\left|\\frac{U_{new}-U_{old}}{U_{old}}\\right|

    The errors are computed in Fourier space but returned in real space. The returned error
    is the greatest of the errors computed for both velocities and the height.

    The reference solution will generally be the solution at the previous iteration, which
    is used for measuring convergence.

    **Parameters**

    - `U_hat_ref` : the solution at the previous timestep (or a reference solution)
    - `U_hat_approx` : the solution at the current timestep
    - `st` : spectral toolbox object

    **Returns**

    - `error` : The computed L_infty error
    """

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

def h_init(control):
    """
    This function sets up a Gaussian initial condition for the height field.
    Initial flow is stationary.

    **Parameters**
    - `control` : control object

    **Returns**
    - `XX` : x-coordinate matrix from the x-coord vector
    - `YY` : y-coordinate matrix from the y-coord vector
    - `U` : initial condition in realspace, sized (3, Nx, Nx) with (u, v, {h || phi}) along the first rank
    """

    # Scale factor
    # NB must be steep enough to approximate double periodicity (here be Gibb's)
    width = 2.0

    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])
    XX, YY = np.meshgrid(x_grid, x_grid)

    h_space = np.zeros((1,control['Nx'],control['Nx']))
    h_space[0,:,:] = np.exp(-width * ((XX-control['Lx']/2.0)**2 + (YY-control['Lx']/2.0)**2))

    # Initial unknown vector
    U = np.vstack((np.zeros_like(h_space), np.zeros_like(h_space), h_space))

    return XX, YY, U


