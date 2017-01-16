import numpy as np
import pickle
import sys
import cyclops_control
import cyclops_base
import RSWE_direct
import rswe_metrics
from spectral_toolbox import SpectralToolbox
from rswe_exponential_integrator import *

def main(control):
    """

    """
    # Hardcoded for now:
    control['final_time'] = 1.0
    control['Nt'] = 4
    solver = 'coarse_propagator'
    #solver = 'fine_propagator'

    # Set up initial (truth) field
    ICs = cyclops_base.read_ICs(control, control['filename'])
    truth = ICs

    st = SpectralToolbox(control['Nx'], control['Lx'])
    if solver == 'coarse_propagator':
        control['HMM_M_bar_M'] = int(50*control['HMM_T0_M'])
        control['HMM_M_bar_L'] = int(50*control['HMM_T0_L'])

        expInt_g = ExpInt_Gravitational(control)
        expInt_m = expM(control, None, expInt_g)

        expInt = (expInt_m, expInt_g)
    else:
        expInt = ExponentialIntegrator_Dim(control)

    metrics = rswe_metrics.Metrics()

    X = np.linspace(0,control['Lx'],control['Nx'])
    x, y = np.meshgrid(X,X)

    # Propagate it through Nts
    plt.figure()
    plt.contourf(x, y,  truth[2, :, :])
    plt.colorbar()
    plt.show()
    fulltime = 0

    for i in range(0, control['Nt']):
        truth[:, :, :] = RSWE_direct.solve(solver,  control, st, expInt, truth[:, :, :])
        fulltime += control['final_time']
        plt.figure()
        plt.contourf(x, y,  truth[2, :, :])
        plt.colorbar()
        plt.show()

        with open("{}_truth_{}.dat".format(control['filename'], i), 'wb') as f:
            data = dict()
            data['time'] = fulltime
            data['u'] = truth[0,:,:]
            data['v'] = truth[1,:,:]
            data['h'] = cyclops_base.inv_geopotential_transform(control, truth[2,:,:])

            metrics.compute_all_metrics(control, data['u'], data['v'], data['h'])
            data['metrics'] = metrics
            data['control'] = control

            print("KE: {} PE: {}".format(metrics.KE, metrics.PE))

            pickle.dump(data, f)

if __name__ == "__main__":
    control_in = cyclops_control.setup_control(sys.argv[1:])
    main(control_in)
