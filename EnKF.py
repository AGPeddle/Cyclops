import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cyclops_control
import RSWE_direct

class Ensemble_Kalman:
    def __init__(self, x_meas, Nx, ensemble_size):
        self.Nx = Nx
        self.N = ensemble_size

        self.H = np.zeros((x_meas.shape[0], self.Nx))
        self.H[np.arange(x_meas.shape[0]), x_meas] = 1

        np.random.seed(0)

    def ensemble_perturbation(self, A):
        X = np.identity(self.N) - (1/self.N)*np.ones((self.N, self.N))
        return np.dot(A, X)

    def ensemble_covariance(self, A):
        A_p = self.ensemble_perturbation(A)

        return (1/(self.N-1))*np.outer(A_p, A_p.T)

    def perturb_observations(self, d):
        
        mean = 0
        var = 0.05

        D = np.tile(d, (1,self.N))
        ens_perts = np.random.normal(mean, var, D.shape)

        return D, ens_perts

    def apply(self, A, d):
        A_p = self.ensemble_perturbation(A)
        P_e = self.ensemble_perturbation(A_p)

        D, ens_perts = self.perturb_observations(d)
        R_e = np.dot(ens_perts, ens_perts.T)/(self.N-1)
        D_p = D - np.dot(self.H, A)

        A2 = np.dot(A_p, A_p.T)
        X = np.dot(self.H, A2)
        X = np.dot(X, self.H.T) + np.dot(ens_perts, ens_perts.T)

        X = np.linalg.inv(X)

        RHS = np.dot(A2, self.H.T)
        RHS = np.dot(RHS, X)
        RHS = np.dot(RHS, D_p)

        return A + RHS


def test(control):

    control['solver'] = "fine_propagator"

    with open('assim_data.dat', 'rb') as f:
        assim_data = pickle.load(f)

        meas_locs = np.hstack((assim_data['x_meas'], control['Nx'] + assim_data['x_meas'], 2*control['Nx'] + assim_data['x_meas']))
        ts = assim_data['t']
        measurements = assim_data['u']
        ICs = assim_data['ICs']
        size = assim_data['size']

    enKF = Ensemble_Kalman(meas_locs, 3*control['Nx'], size)

    x_grid = control['Lx']*np.arange(0,control['Nx'])/float(control['Nx'])
    U = np.zeros((size, 3, control['Nx']))
    for k in range(size):
        U[k, 2, :] = ICs[k, 2, :]

    """
    for i in range(size):
        plt.plot(x_grid, U[i, 2, :])
    plt.show()
    """

    for nt, T in enumerate(ts[1:]):
        print("Running timestep {} ending at {}".format(nt, T))
        control['final_time'] = T - ts[nt]
        #control['Nt'] = 1

        for n in range(size):
            print("Running ensemble {}".format(n))
            U[n, :, :] = RSWE_direct.solve(control, U[n, :, :])

        d = measurements[nt + 1, :, :].reshape(-1,1)
        A = U.reshape(size, -1)

        A = enKF.apply(np.transpose(A), d)
        A = np.transpose(A)
        U = A.reshape(size, 3, control['Nx'])

        plt.figure()
        for i in range(size):
            plt.plot(x_grid, U[i, 2, :])
        plt.xlabel('x-coordinate')
        plt.ylabel('Height')
        plt.title('Ensemble Kalman Filter, {}, at T = {}'.format(control['solver'], T))
        #plt.show()
        plt.savefig("Fine{}".format(nt))


if __name__ == "__main__":
    control_in = cyclops_control.setup_control(sys.argv[1:])
    test(control_in)
