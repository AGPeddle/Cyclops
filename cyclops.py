import numpy as np
import pickle
import sys
import cyclops_control
import cyclops_base
import RSWE_direct
import rswe_metrics
from spectral_toolbox import SpectralToolbox
from rswe_exponential_integrator import *
from mpi4py import MPI

def main_serial():
    """
    Implements the Asymptotic Parallel in Time algorithm.

    This methods proceeds by performing a coarse approximation using the coarse propagator
    which is then refined (in theory, in parallel) by the fine propagator. It differs from
    standard parareal methods by the averaging over the fast waves which is performed in
    the coarse timestep. The fine timestep solves the full equations to the desired level
    of accuracy.

    Implementation follows:

    .. math:: U_{n+1}^{k+1} = G(U_{n}^{k+1}) + F(U_{n}^{k}) - G(U_{n}^{k})

    where G refers to the coarse propagator and F to the fine propagator. n is the timestep
    k is the current iteration number. Converges to the accuracy of the fine propgator.

    **Parameters**

    None

    **Returns**

    Dumps errors to file and screen

    **See Also**

    coarse_propagator, fine_propagator

    """
    # U_hat_mac contains the solution at the completion of the
    # coarse parareal timesteps.
    # U_hat_mic contains the solution at the completion of the fine
    # parareal timesteps, but discards the information in between
    # (i.e. matches the timesteps of the coarse solution)
    U_hat_mac = np.zeros((control['Nt'], 3, control['Nx'], control['Nx']), dtype = 'complex')
    U_hat_mic = np.zeros((control['Nt'], 3, control['Nx'], control['Nx']), dtype = 'complex')
    U_hat_new = np.zeros((control['Nt'], 3, control['Nx'], control['Nx']), dtype = 'complex')
    U_hat_old = np.zeros((control['Nt'], 3, control['Nx'], control['Nx']), dtype = 'complex')

    # U_hat_old contains the solution at the previous iteration for
    # use in convergence testing
    errors = [[] for _ in range(control['Nt'])]

    # Create initial condition
    U_hat_new[0,2,:,:] = h_init()

    U_hat_new[0,2,:,:] = st.forward_fft(U_hat_new[0,2,:,:])

    U_hat_old[0,:,:,:] = U_hat_new[0,:,:,:]

    # Compute first parareal level here
    start = time.time()
    for j in range(control['Nt']-1):
        # First parareal level by coarse timestep in serial only
        U_hat_new[j+1, :, :, :] = coarse_propagator(U_hat_new[j, :, :, :])
        U_hat_old[j+1,:,:,:] = U_hat_new[j+1, :, :, :]
    end = time.time()
    logging.info("First APinT level completed in {:.8f} seconds".format(end-start))

    # Further parareal levels computed here
    iterative_error = 100000000000000.
    k = 1
    while iterative_error > control['conv_tol']:
        start = time.time()
        L_inf_buffer = []
        L_2_buffer = []

        U_hat_new = np.zeros((control['Nt'], 3, control['Nx'], control['Nx']), dtype = 'complex')
        U_hat_new[0,:,:,:] = U_hat_old[0,:,:,:]

        for j in range(control['Nt']-1):  # Loop over timesteps
            # Compute coarse and fine timesteps (parallel)
            U_hat_mac[j+1, :, :, :] = coarse_propagator(U_hat_old[j,:,:,:])
            U_hat_mic[j+1, :, :, :] = fine_propagator(U_hat_old[j,:,:,:])

            # Compute and apply correction (serial)
            U_hat_new[j+1, :, :, :] = coarse_propagator(U_hat_new[j, :, :, :])
            U_hat_new[j+1, :, :, :] = U_hat_new[j+1, :,:,:] + (U_hat_mic[j+1, :,:,:] - U_hat_mac[j+1, :,:,:])

            # L_inf, L_2
            error_iteration = compute_errors(U_hat_old[j+1,:,:,:], U_hat_new[j+1,:,:,:])
            L_inf_buffer.append(error_iteration[0])
            L_2_buffer.append(error_iteration[1])

        k += 1
        # Perform convergence checks (iterative error)
        U_hat_old[:, :, :, :] = U_hat_new[:, :, :, :].copy()  #Overwrite previous solution for convergence tests
        iterative_error_old = iterative_error
        iter_err_Linf = max(L_inf_buffer)
        iter_err_L2 = np.sqrt(np.sum([i**2 for i in L_2_buffer]))
        iterative_error = max(iter_err_Linf, iter_err_L2)
        end = time.time()

        logging.info("APinT level {:>2} completed in {:.8f} seconds".format(k,end-start))
        logging.info("L_infty norm = {:.6e},   L_2 norm = {:.6e}".format(iter_err_Linf, iter_err_L2))

        plotfile.write("{:2}\t{:.6e}\t{:.6e}\n".format(k, iter_err_Linf, iter_err_L2))

        if iterative_error > iterative_error_old:
            logging.warning('Possible Numerical Instability Detected.')

    logging.info("APinT Computation Complete in {:>2} iterations.".format(k))


def main_parallel(control):
    """

    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Hardcoded for now:
    if control['outFileStem'] is None: control['outFileStem'] = ''
    conv_tol = control['conv_tol']
    control['final_time'] = control['coarse_timestep']
    control['solver'] = None  # Idiot-proofing
    control['Nt'] = size

    st = SpectralToolbox(control['Nx'], control['Lx'])
    if control['Whitehead']:
        control['HMM_M_bar_M'] = max(25, int(80*control['HMM_T0_M']))
        control['HMM_M_bar_L'] = max(25, int(80*control['HMM_T0_L']))

        expInt_g = ExpInt_Gravitational(control)
        expInt_m = expM(control, None, expInt_g)

        expInt_coarse = (expInt_m, expInt_g)
        expInt_fine = ExponentialIntegrator_FullEqs(control)
    else:
        control['HMM_M_bar'] = max(25, int(80*control['HMM_T0']))
        expInt_coarse = ExponentialIntegrator_FullEqs(control)
        expInt_fine = ExponentialIntegrator_FullEqs(control)

    # Set up initial (truth) field
    #ICs = cyclops_base.read_ICs(control, control['filename'])
    XX, YY, ICs = h_init(control)

    st = SpectralToolbox(control['Nx'], control['Lx'])

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

    if rank == 0:
        # Create initial condition
        U_hat_new[0,:,:,:] = ICs

        for k in range(3):
            U_hat_new[0,k,:,:] = st.forward_fft(U_hat_new[0,k,:,:])

        U_hat_old[0,:,:,:] = U_hat_new[0,:,:,:]

        # Compute first parareal level here
        for j in range(control['Nt']):
            # First parareal level by coarse timestep in serial only
            U_hat_new[j+1, :, :, :] = RSWE_direct.solve('coarse_propagator', control, st, expInt_coarse, U_hat_new[j,:,:,:], invert_fft = False)
            U_hat_old[j+1, :, :, :] = U_hat_new[j+1, :, :, :]
        #print("Done first parareal")

    # Further parareal levels computed here
    k = 0
    acc_err = 0.
    comm.Barrier()
    while converged[0] == 0:
        #print("Running iteration {} on rank {}".format(k, rank))
        #Scatter from 0 to all local olds
        comm.Scatter(np.ascontiguousarray(U_hat_old[:-1, :,:,:]), U_hat_old_local, root = 0)

        # Compute coarse and fine timesteps (parallel)
        U_hat_mac_local = RSWE_direct.solve('coarse_propagator', control, st, expInt_coarse, U_hat_old_local, invert_fft = False)
        U_hat_mic_local = RSWE_direct.solve('fine_propagator', control, st, expInt_fine, U_hat_old_local, invert_fft = False)

        #Gather from all local macs and mics to rank 0
        comm.Gather(np.ascontiguousarray(U_hat_mic_local), U_hat_mic[1:,:,:,:], root = 0)
        comm.Gather(np.ascontiguousarray(U_hat_mac_local), U_hat_mac[1:,:,:,:], root = 0)
        
        if rank == 0:
            U_hat_new = np.zeros((control['Nt'] + 1, 3, control['Nx'], control['Nx']), dtype = 'complex')
            U_hat_new[0, :, :, :] = U_hat_old[0, :, :, :]

            for j in range(control['Nt']):  # Loop over timesteps
                # Compute and apply correction (serial)
                U_hat_new[j+1, :, :, :] = RSWE_direct.solve('coarse_propagator', control, st, expInt_coarse, U_hat_new[j,:,:,:], invert_fft = False)
                U_hat_new[j+1, :, :, :] = U_hat_new[j+1, :, :, :] + (U_hat_mic[j+1, :, :, :] - U_hat_mac[j+1, :, :, :])

                # L_inf
                acc_err = max(acc_err, cyclops_base.compute_L_infty_error(U_hat_old[j+1,:,:,:], U_hat_new[j+1,:,:,:], st))

            #print("Done iteration number {}".format(k))

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
    main_parallel(control_in)
