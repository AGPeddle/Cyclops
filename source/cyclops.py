#!/usr/bin/env/ python3
"""
This is the main point of access for the time-parallel implementation
of the APinT method. I/O is handled through the cyclops_control module
and pickled dicts, respectively.

Functions
---------
- `main` -- Exposes the high-level implementation

Invocation
----------

See the cyclops_control docs for details of parameters. The test directory
contains example calls to Cyclops.

Dependencies
------------
- `cyclops suite`
- numpy
- pyfftw
- mpi4py

| Author: Adam G. Peddle, Terry Haut
| Contact: ap553@exeter.ac.uk
| Version: 1.0
"""

import numpy as np
import sys
import pickle
import os
import cyclops_control
import cyclops_base
import rswe_direct
from spectral_toolbox import SpectralToolbox
from rswe_exponential_integrator import *
from mpi4py import MPI

def main(control):
    """
    Main program. Exposes the Parareal algorithm. Sub-algos are encapsulated in
    the rswe_direct and rswe_exponential_integrator modules. All control is
    through the control object.
    """

    if 'working_dir' in control: os.chdir(control['working_dir'])

    # Set up MPI communicator
    global comm
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Local parameterisations:
    if control['outFileStem'] is None: control['outFileStem'] = ''
    conv_tol = control['conv_tol']
    control['final_time'] = control['coarse_timestep']  # For generalisabilty of rswe_direct
    control['solver'] = None  # Idiot-proofing
    control['Nt'] = size  # One coarse timestep per process

    # Initialise spectral toolbox object
    st = SpectralToolbox(control['Nx'], control['Lx'])

    control['HMM_M_bar'] = max(25, int(80*control['HMM_T0']))

    # Exponential integrators at different Parareal levels
    # may be different, e.g. in the case of triple scale
    # separation.
    expInt_coarse = ExponentialIntegrator_FullEqs(control)
    expInt_fine = ExponentialIntegrator_FullEqs(control)

    # Set up initial (truth) field
    if control['filename']:
        XX, YY, ICs = cyclops_base.read_ICs(control, control['filename'])
    else:  # Fall back on default initial Gaussian
        XX, YY, ICs = cyclops_base.h_init(control)
        control['filename'] = 'default'

    st = SpectralToolbox(control['Nx'], control['Lx'])

    # Process-local representations of macroscopic, microscopic, and previous iterand
    U_hat_mac_local = np.zeros((3, control['Nx'], control['Nx']), dtype = 'complex')
    U_hat_mic_local = np.zeros((3, control['Nx'], control['Nx']), dtype = 'complex')
    U_hat_old_local = np.zeros((3, control['Nx'], control['Nx']), dtype = 'complex')
    converged = np.zeros((1),dtype=int)

    # U_hat_mac contains the solution at the completion of the
    # coarse parareal timesteps.
    # U_hat_mic contains the solution at the completion of the fine
    # parareal timesteps, but discards the information in between
    # (i.e. matches the timesteps of the coarse solution)
    U_hat_mac = np.zeros((control['Nt'] + 1, 3, control['Nx'], control['Nx']), dtype = 'complex')
    U_hat_mic = np.zeros((control['Nt'] + 1, 3, control['Nx'], control['Nx']), dtype = 'complex')
    U_hat_new = np.zeros((control['Nt'] + 1, 3, control['Nx'], control['Nx']), dtype = 'complex')
    U_hat_old = np.zeros((control['Nt'] + 1, 3, control['Nx'], control['Nx']), dtype = 'complex')

    if rank == 0:  # Root node by convention
        # Create initial condition
        U_hat_new[0,:,:,:] = ICs

        for k in range(3):
            U_hat_new[0,k,:,:] = st.forward_fft(U_hat_new[0,k,:,:])

        U_hat_old[0,:,:,:] = U_hat_new[0,:,:,:]

        # Compute first parareal level here
        # TODO: Parallelise this step. It's embarassingly parallel, but
        # care must be taken not to interfere with the time-parallel
        # coarse solves.
        for j in range(control['Nt']):
            # First parareal level by coarse timestep in serial only
            U_hat_new[j+1, :, :, :] = rswe_direct.solve('coarse_propagator',
                                                        control, st,
                                                        expInt_coarse,
                                                        U_hat_new[j,:,:,:],
                                                        invert_fft = False)
            U_hat_old[j+1, :, :, :] = U_hat_new[j+1, :, :, :]

    # Further parareal levels computed here
    k = 0
    acc_err = 0.
    comm.Barrier()
    while converged[0] == 0:
        #Scatter from 0 to all local olds
        comm.Scatter(np.ascontiguousarray(U_hat_old[:-1, :,:,:]), U_hat_old_local, root = 0)

        # Compute coarse and fine timesteps (parallel)
        # Average computed in serial. It is parallelisable,
        # but ideally this would be on a heterogeneous
        # computing architecture.
        U_hat_mac_local = rswe_direct.solve('coarse_propagator',
                                            control, st,
                                            expInt_coarse, U_hat_old_local,
                                            invert_fft = False)
        U_hat_mic_local = rswe_direct.solve('fine_propagator',
                                            control, st,
                                            expInt_fine, U_hat_old_local,
                                            invert_fft = False)

        #Gather from all local macs and mics to root process
        comm.Gather(np.ascontiguousarray(U_hat_mic_local), U_hat_mic[1:,:,:,:], root = 0)
        comm.Gather(np.ascontiguousarray(U_hat_mac_local), U_hat_mac[1:,:,:,:], root = 0)

        if rank == 0:
            U_hat_new = np.zeros((control['Nt'] + 1, 3, control['Nx'], control['Nx']), dtype = 'complex')
            U_hat_new[0, :, :, :] = U_hat_old[0, :, :, :]

            for j in range(control['Nt']):  # Loop over timesteps
                # Compute and apply Parareal correction (serial)
                U_hat_new[j+1, :, :, :] = rswe_direct.solve('coarse_propagator',
                                                            control, st,
                                                            expInt_coarse,
                                                            U_hat_new[j,:,:,:],
                                                            invert_fft = False)

                U_hat_new[j+1, :, :, :] = U_hat_new[j+1, :, :, :] + (U_hat_mic[j+1, :, :, :] - U_hat_mac[j+1, :, :, :])

                # L_inf
                acc_err = max(acc_err, cyclops_base.compute_L_infty_error(U_hat_old[j+1,:,:,:], U_hat_new[j+1,:,:,:], st))

            # Perform convergence checks (iterative error)
            U_hat_old[:, :, :, :] = U_hat_new[:, :, :, :].copy()  #Overwrite previous solution for convergence tests
            if k > 0 and acc_err < conv_tol:
                print('Converged with acc_err {}'.format(acc_err))
                converged[0] = 1
            else:
                print('Not converged with acc_err {}'.format(acc_err))
                converged[0] = 0
                acc_err = 0.

        comm.Bcast(converged, root = 0)
        comm.Barrier()
        k+=1

    # Post-convergence, handle output
    if rank == 0:
        for i in range(control['Nt'] + 1):
            with open("{}{}_APinT_{}.dat".format(control['filename'], control['outFileStem'], i), 'wb') as f:
                for k in range(3):
                    U_hat_new[i,k,:,:] = st.inverse_fft(U_hat_new[i,k,:,:])

                data = dict()
                data['time'] = i*control['final_time']
                data['u'] = U_hat_new[i, 0, :, :]
                data['v'] = U_hat_new[i, 1, :, :]
                data['h'] = cyclops_base.inv_geopotential_transform(control, U_hat_new[i, 2, :, :])

                data['control'] = control

                pickle.dump(data, f)


if __name__ == "__main__":
    control_in = cyclops_control.setup_control(sys.argv[1:])
    main(control_in)
