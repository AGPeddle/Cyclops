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
    #control['final_time'] = 1.0
    if control['outFileStem'] is None: control['outFileStem'] = ''
    solver = control['solver']

    # Set up initial (truth) field
    ICs = cyclops_base.read_ICs(control, control['filename'])
    truth = ICs

    print("H_naught = {}".format(control['H_naught']))

    st = SpectralToolbox(control['Nx'], control['Lx'])
    metrics = rswe_metrics.Metrics()

    # Set up initial (truth) field
    ICs = cyclops_base.read_ICs(control, control['filename'])
    truth = ICs

    st = SpectralToolbox(control['Nx'], control['Lx'])
    expInt = ExponentialIntegrator_FullEqs(control)

    metrics = rswe_metrics.Metrics()

    X = np.linspace(0,control['Lx'],control['Nx'])
    x, y = np.meshgrid(X,X)

    # Propagate it through Nts
    fulltime = 0

    for i in range(0, control['Nt']):
        truth[:, :, :] = RSWE_direct.solve('fine_propagator', control, st, expInt, truth[:, :, :])
        fulltime += control['final_time']

        with open("{}{}_truth_{}.dat".format(control['filename'], control['outFileStem'], i), 'wb') as f:
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
