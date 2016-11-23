#!/usr/bin/env/ python3
"""
This module implements the solution to the exponential integrator for the linear operator
of the rotating shallow water equations as used by pypint.

Consider the RSWE, neglecting the nonlinear terms. We may write this as:

.. math :: \\frac{\\partial u}{\\partial t} = Lu

We may solve this via an integrating factor method, yielding:

.. math:: u_{t} = e^{tL}u_{0}

to which we may freely apply various choices of timestepping. The exponential integrator
then requires the efficient computation of the matrix exponential. We write the matrix
exponential in the form:

.. math:: e^{tL} = r_{k}^{\\alpha} e^{t\\omega_{k}^{\\alpha}} (r_{k}^{\\alpha})^{-1}

where :math:`r_{k}^{\\alpha}` is a matrix containing the eigenvectors of the linear
operator (for a corresponding wavenumber, k, as we are working in Fourier space) and
:math:`\\omega_{k}^{\\alpha}` is a vector containing the associated eigenvalues of
the linear operator, such that:

.. math:: \\omega_{k}^{\\alpha} = \\alpha\\sqrt{f_{0}^{2} + 2\\pi g H_{0} |k| / L}

with :math:`\\alpha = -1, 0, 1`. This reduces the problem to an easier problem
of finding the eigenvalues/eigenvectors and applying these. These are pre-computed
for speed as they will not change over the course of the computation for constant
gravity, rotation, and water depth.

Classes
-------
-`ExponentialIntegrator` : Implements the exponential integrator for the RSWE

Notes
-----
We write the linear operator, L, as:

.. math:: L = \\left[\\begin{array}{ccc}
                0 & -f_{0} & g\\partial_{x} \\\\
               f_{0} & 0 & g\\partial_{y} \\\\
               H_{0}\\partial_{x} & H_{0}\\partial_{y} & 0 \\end{array}\\right]

which becomes, in Fourier space:

.. math:: L = \\left[\\begin{array}{ccc}
                0 & -f_{0} & 2\\pi i g k_{1}/L \\\\
               f_{0} & 0 & 2\\pi i g k_{2}/L \\\\
               2\\pi i H_{0} k_{1}/L & 2\\pi i H_{0} k_{2}/L & 0 \\end{array}\\right]

See Also
--------
numpy

| Authors: Terry Haut, Adam G. Peddle
| Contact: ap553@exeter.ac.uk
| Version: 2.0
"""

import numpy as np
import logging

class ExponentialIntegrator:
    """
    Implements the exponential integrator for the Rotating Shallow Water Equations.

    This class implements the exponential integrator objects, which precompute and store
    the eigenvalues and eigenvectors of the linear operator in Fourier space and apply
    the matrix exponential for a given timestep when invoked. See module header for
    explanation of computation.

    **Methods**

    - `call` : Invoke the matrix exponential
    - `project_basis` : Use the computed eigenbasis to project
    """

    def __init__(self):
        raise Exception("This class is not stand-alone")

    def call(self,U_hat,t):
        """
        Call to invoke the exponential integrator on a given initial solution over some time.

        Propagates the solution to the linear problem for some time, t, based on the initial
        condition, U. U is, in general, the value at the beginning of a timestep and t is the
        timestep, although this need not be the case.

        **Parameters**

        - `U_hat` : the Fourier modes of the unknown solution, ordered as below
        - `t` : the time at which the solution is desired (or timestep to be used)

        **Returns**

        - `U_hat_sp` : The computed solution, propagated by time t

        **Notes**

        1) Works in -t direction as well.
        2) Input order: v1[:,:] = U_hat[0,:,:], v2[:,:] = U_hat[1,:,:], h[:,:] = U_hat[2,:,:]

        """

        # Separate unknowns to allow vector multiply for speed
        v1_hat = U_hat[0,:,:]
        v2_hat = U_hat[1,:,:]
        h_hat = U_hat[2,:,:]

        # First eigenvector for L(ik), eigenvalue omega=0
        rk00 = self.eBasis[0,0,:,:]
        rk10 = self.eBasis[1,0,:,:]
        rk20 = self.eBasis[2,0,:,:]

        # Second eigenvector for L(ik),
        # eigenvalue omega = -i*sqrt(F+k^2)
        rk01 = self.eBasis[0,1,:,:]
        rk11 = self.eBasis[1,1,:,:]
        rk21 = self.eBasis[2,1,:,:]

        # Third eigenvector for L(ik),
        # eigenvalue omega = I*sqrt(F+k^2)
        rk02 = self.eBasis[0,2,:,:]
        rk12 = self.eBasis[1,2,:,:]
        rk22 = self.eBasis[2,2,:,:]

        # Convert to eigenvector basis
        U_hat_sp = np.zeros((3,self._Nx,self._Nx), dtype='complex')
        v1_hat_sp = np.conj(rk00) * v1_hat + np.conj(rk10) * v2_hat + np.conj(rk20) * h_hat
        v2_hat_sp = np.conj(rk01) * v1_hat + np.conj(rk11) * v2_hat + np.conj(rk21) * h_hat
        h_hat_sp = np.conj(rk02) * v1_hat + np.conj(rk12) * v2_hat + np.conj(rk22) * h_hat

        # Apply exp(-t*L)
        omega0 = self.eVals[0,:,:]; omega0[:,:] = 1.0 #omega0 = np.exp(-1j*omega0*t/self._eps)
        omega1 = self.eVals[1,:,:]; omega1 = np.exp(-1j*omega1*t*self._diveps)
        omega2 = self.eVals[2,:,:]; omega2 = np.exp(-1j*omega2*t*self._diveps)

        U_hat_sp[0,:,:] = omega0 * rk00 * v1_hat_sp +  omega1 * rk01 * v2_hat_sp + \
                          omega2 * rk02 * h_hat_sp
        U_hat_sp[1,:,:] = omega0 * rk10 * v1_hat_sp +  omega1 * rk11 * v2_hat_sp + \
                          omega2 * rk12 * h_hat_sp
        U_hat_sp[2,:,:] = omega0 * rk20 * v1_hat_sp +  omega1 * rk21 * v2_hat_sp + \
                          omega2 * rk22 * h_hat_sp

        return U_hat_sp

    def project_basis(self,U_hat,t):
        """
        Call to invoke the exponential integrator on a given initial solution over some time.

        Propagates the solution to the linear problem for some time, t, based on the initial
        condition, U. U is, in general, the value at the beginning of a timestep and t is the
        timestep, although this need not be the case.

        **Parameters**

        - `U_hat` : the Fourier modes of the unknown solution, ordered as below
        - `t` : the time at which the solution is desired (or timestep to be used)

        **Returns**

        - `U_hat_sp` : The computed solution, propagated by time t

        **Notes**

        1) Works in -t direction as well.
        2) Input order: v1[:,:] = U_hat[0,:,:], v2[:,:] = U_hat[1,:,:], h[:,:] = U_hat[2,:,:]

        """

        # Separate unknowns to allow vector multiply for speed
        v1_hat = U_hat[0,:,:]
        v2_hat = U_hat[1,:,:]
        h_hat = U_hat[2,:,:]

        # First eigenvector for L(ik), eigenvalue omega=0
        rk00 = self.eBasis[0,0,:,:]
        rk10 = self.eBasis[1,0,:,:]
        rk20 = self.eBasis[2,0,:,:]

        # Second eigenvector for L(ik),
        # eigenvalue omega = -i*sqrt(F+k^2)
        rk01 = self.eBasis[0,1,:,:]
        rk11 = self.eBasis[1,1,:,:]
        rk21 = self.eBasis[2,1,:,:]

        # Third eigenvector for L(ik),
        # eigenvalue omega = I*sqrt(F+k^2)
        rk02 = self.eBasis[0,2,:,:]
        rk12 = self.eBasis[1,2,:,:]
        rk22 = self.eBasis[2,2,:,:]

        # Convert to eigenvector basis
        sigmas = np.zeros((3,self._Nx,self._Nx), dtype = complex)
        sig_0 = np.conj(rk00) * v1_hat + np.conj(rk10) * v2_hat + np.conj(rk20) * h_hat
        sig_m1 = np.conj(rk01) * v1_hat + np.conj(rk11) * v2_hat + np.conj(rk21) * h_hat
        sig_p1 = np.conj(rk02) * v1_hat + np.conj(rk12) * v2_hat + np.conj(rk22) * h_hat

        # Apply exp(-t*L)
        omega0 = self.eVals[0,:,:]; omega0[:,:] = 1.0 #omega0 = np.exp(-1j*omega0*t/self._eps)
        omega1 = self.eVals[1,:,:]; omega1 = np.exp(-1j*omega1*-t*self._diveps)
        omega2 = self.eVals[2,:,:]; omega2 = np.exp(-1j*omega2*-t*self._diveps)

        sigmas[0,:,:] = omega0*sig_0
        sigmas[1,:,:] = omega1*sig_m1
        sigmas[2,:,:] = omega2*sig_p1

        return sigmas


class ExponentialIntegrator_Dim(ExponentialIntegrator):
    """
    Implements the exponential integrator for the dimensional, perturbation-height
    Rotating Shallow Water Equations which have been symmetrised with respect to
    the geopotential height.

    This class implements the exponential integrator objects, which precompute and store
    the eigenvalues and eigenvectors of the linear operator in Fourier space and apply
    the matrix exponential for a given timestep when invoked. See module header for
    explanation of computation.

    **Attributes**

    - `f0` : the Coriolis parameter
    - `g` : the gravitational acceleration
    - `H0` : the mean water depth
    - `deriv_factor` : factor of 2*pi/L due to spectral differentiation
    - `_Nx` : Number of grid points, such that domain is Nx X Nx
    - `eVals` : Vector of eigenvalues, ordered alpha = 0, -1, 1
    - `eBasis` : Corresponding orthonormalised matrix of eigenvectors, arranged in columns

    **Methods**

    - `create_eigenvalues` : Set up the analytically-computed eigenvalues for L
    - `create_eigenbasis` : Set up the analyticall-computed eigenbasis for L
    - `call` : Invoke the matrix exponential

    **Parameters**

    - `control` : Control object containing the relevant parameters/constants for initialisation

    """

    def __init__(self,control):
        # Physical parameters:
        self.H0 = control['H_naught']
        self.f0 = control['f_naught']
        self.g = control['gravity']
        self.phi0 = self.g*self.H0
        self._diveps = 1.0

        # Factor arising from spectral derivative on domain of size L
        self.deriv_factor = 2.0*np.pi/control['Lx']

        self._Nx = control['Nx']
        self.eVals  = np.zeros((3,self._Nx,self._Nx),   dtype = np.float64)
        self.eBasis = np.zeros((3,3,self._Nx,self._Nx), dtype = complex)

        linearOperator = np.zeros(shape = (3,3), dtype = complex)
        ctr_shift = self._Nx//2  # Python arrays are numbered from 0 to N-1, spectrum goes from -N/2 to N/2-1

        for k1 in range(-self._Nx//2, self._Nx//2):
            for k2 in range(-self._Nx//2, self._Nx//2):
                # Create eigenvals/vects for given wavenumber
                # combination. Consider shift between array and
                # spectrum
                self.eVals[:,k1 + ctr_shift, k2 + ctr_shift]    = self.create_eigenvalues(k1, k2)
                self.eBasis[:,:,k1 + ctr_shift, k2 + ctr_shift] = self.create_eigenbasis(k1, k2)

    def create_eigenvalues(self, k1, k2):
        """
        Creates the eigenvalues required for the creation of the eigenbasis (of the linear operator).

        This method returns the eigenvectors of the linear operator of the RSWE as formulated in
        spectral space. These have been found analytically to be:

        .. math:: evals = \\alpha\\sqrt{f_{0}^{2} + \\beta^{2}gH_{0}|k|^{2}}

        where :math:`\\beta = 2\\pi/L` and :math:`\\alpha = 0, -1, +1`, respectively.

        **Parameters**

        - `k1`, `k2` : the wavenumbers in the x- and y- directions

        **Returns**

        - `eVals` : eigenvalues for slow mode and fast waves in - and + directions, in that order

        **Notes**

        Note that these have been found analytically and so an arbitrary L is not currently supported.

        """

        eVals = np.zeros(3, dtype = np.float64)
        eVals[0] = 0.
        # deriv_factor corresponds to beta in the header
        eVals[1] = -np.sqrt(self.f0**2 + (self.deriv_factor**2)*self.phi0*(k1**2 + k2**2))
        eVals[2] =  np.sqrt(self.f0**2 + (self.deriv_factor**2)*self.phi0*(k1**2 + k2**2))

        return eVals

    def create_eigenbasis(self, k1, k2):
        """
        Create the eigenbasis of the linear operator. As with the eigenvalues, it is an implementation of
        an analytical solution.

        This method computes the eigenvector matrix for a given wavenumber pair. It does not solve the
        eigenvalue problem, rather the analytical solution has been found for the representation of the
        linear operator which has been used.

        **Parameters**

        - `k1`, `k2` : the wavenumbers in the x- and y- directions

        **Returns**

        - `A` : eigenvectors of L in columns, with slow mode 1st then fast waves in - and + directions

        **Notes**

        1) Note that these have been found analytically and so an arbitrary L is not currently supported.
        2) The eigenbasis is orthonormalised

        """

        A = np.zeros((3,3), dtype = complex)
        if k1 != 0 or k2 != 0:  # Almost all of the time
            kappa = k1**2 + k2**2
            f = self.f0
            sqp = self.deriv_factor*np.sqrt(self.g*self.H0)  # Factor on spatial derivatives
            omega = np.sqrt(f**2 + sqp*sqp*kappa)
            pm_1_denom = omega*omega/f - f

            # Eigenvectors in columns corresponding to slow mode and
            # fast waves travelling in either direction
            A = np.array([[-1j*sqp*k2/f, sqp*(1j*k2 - omega*k1/f)/pm_1_denom,  sqp*(1j*k2 + omega*k1/f)/pm_1_denom],
                          [1j*sqp*k1/f,  -sqp*(1j*k1 + omega*k2/f)/pm_1_denom, sqp*(-1j*k1 + omega*k2/f)/pm_1_denom],
                          [1.,           1.,                                   1.]], dtype = complex)

            # Orthonormalise slow mode eigenvector
            eig_norm = omega/f
            #eig_norm = np.sqrt(A[0,0]**2 + A[1,0]**2 + A[2,0]**2)
            A[:,0] = A[:,0]/eig_norm

            eig_norm = np.sqrt(2*omega*omega/(sqp*sqp*kappa))  # Same orthonormalisation for both fast waves

            A[:,1] = A[:,1]/eig_norm
            A[:,2] = A[:,2]/eig_norm

            # Check that orthonormalisation has actually orthonormalised
            try:
                assert np.isclose(np.sqrt(A[0,0]*np.conj(A[0,0]) + A[1,0]*np.conj(A[1,0]) + A[2,0]*np.conj(A[2,0])), np.complex(1.0, 0.0), rtol = 1e-4, atol = 1e-6)
                assert np.isclose(np.sqrt(A[0,1]*np.conj(A[0,1]) + A[1,1]*np.conj(A[1,1]) + A[2,1]*np.conj(A[2,1])), np.complex(1.0, 0.0), rtol = 1e-4, atol = 1e-6)
                assert np.isclose(np.sqrt(A[0,2]*np.conj(A[0,2]) + A[1,2]*np.conj(A[1,2]) + A[2,2]*np.conj(A[2,2])), np.complex(1.0, 0.0), rtol = 1e-4, atol = 1e-6)

            except AssertionError:
                logging.error("Orthonormalisation Failure in Exponential Integrator at wavenumber {}, {}". format(k1, k2))
                print(np.sqrt(A[0,0]*np.conj(A[0,0]) + A[1,0]*np.conj(A[1,0]) + A[2,0]*np.conj(A[2,0])))
                print(np.sqrt(A[0,1]*np.conj(A[0,1]) + A[1,1]*np.conj(A[1,1]) + A[2,1]*np.conj(A[2,1])))
                print(np.sqrt(A[0,2]*np.conj(A[0,2]) + A[1,2]*np.conj(A[1,2]) + A[2,2]*np.conj(A[2,2])))

                raise

        else:  # Special case for k1 = k2 = 0
            A = np.array([[0., -1j/np.sqrt(2.), 1j/np.sqrt(2.)],\
                          [0., 1./np.sqrt(2.),  1./np.sqrt(2.)],\
                          [1., 0.,              0.]],dtype = complex)

        return A


class ExponentialIntegrator_nonDim(ExponentialIntegrator):
    """
    Implements the exponential integrator for the nondimensional formulation of
    the Rotating Shallow Water Equations.

    This class implements the exponential integrator objects, which precompute and store
    the eigenvalues and eigenvectors of the linear operator in Fourier space and apply
    the matrix exponential for a given timestep when invoked. See module header for
    explanation of computation.

    **Attributes**

    - `H0` : the mean water depth
    - `deriv_factor` : factor of 2*pi/L due to spectral differentiation
    - `_Nx` : Number of grid points, such that domain is Nx X Nx
    - `eVals` : Vector of eigenvalues, ordered alpha = 0, -1, 1
    - `eBasis` : Corresponding orthonormalised matrix of eigenvectors, arranged in columns

    - `_eps` : Epsilon, used in nondimensional formulation
    - `_F` : Rossby deformation radius (should be O(1))

    **Methods**

    - `create_eigenvalues` : Set up the analytically-computed eigenvalues for L
    - `create_eigenbasis` : Set up the analyticall-computed eigenbasis for L
    - `call` : Invoke the matrix exponential (inherited)

    **Parameters**

    - `control` : Control object containing the relevant parameters/constants for initialisation

    """

    def __init__(self,control):
        # Physical parameters:
        self.H0 = control['H_naught']
        self._eps = control['epsilon']
        self._diveps = 1.0/self._eps
        self._F = control['deformation_radius']

        # Factor arising from spectral derivative on domain of size L
        self.deriv_factor = 2.0*np.pi/control['Lx']

        self._Nx = control['Nx']
        self.eVals  = np.zeros((3,self._Nx,self._Nx),   dtype = np.float64)
        self.eBasis = np.zeros((3,3,self._Nx,self._Nx), dtype = complex)

        linearOperator = np.zeros(shape = (3,3), dtype = complex)
        ctr_shift = self._Nx//2  # Python arrays are numbered from 0 to N-1, spectrum goes from -N/2 to N/2-1

        for k1 in range(-self._Nx//2, self._Nx//2):
            for k2 in range(-self._Nx//2, self._Nx//2):
                self.eVals[:,k1 + ctr_shift, k2 + ctr_shift]    = self.create_eigenvalues(k1, k2)
                self.eBasis[:,:,k1 + ctr_shift, k2 + ctr_shift] = self.create_eigenbasis(k1, k2)

    def create_eigenvalues(self, k1, k2):
        """
        Creates the eigenvalues required for the creation of the eigenbasis (of the linear operator).
        Note that these have been found analytically and so an arbitrary L is not currently supported.

        **Parameters**

        - `k1`, `k2` : the wavenumbers in the x- and y- directions

        **Returns**

        - `eVals` : eigenvalues for slow mode and fast waves in - and + directions, in that order

        **Notes**

        Note that these have been found analytically and so an arbitrary L is not currently supported.

        """

        eVals = np.zeros(3, dtype = np.float64)
        eVals[0] = 0.
        eVals[1] = -np.sqrt(1. + (k1**2 + k2**2)/self._F)
        eVals[2] = np.sqrt(1. + (k1**2 + k2**2)/self._F)

        return eVals

    def create_eigenbasis(self, k1, k2):
        """
        Create the eigenbasis of the linear operator. As with the eigenvalues, it is an implementation of
        an analytical solution.

        **Parameters**

        - `k1`, `k2` : the wavenumbers in the x- and y- directions

        **Returns**

        - `A` : eigenvectors of L in columns, with slow mode 1st then fast waves in - and + directions

        **Notes**

        1) Note that these have been found analytically and so an arbitrary L is not currently supported.
        2) The eigenbasis is orthonormalised
        """

        A = np.zeros((3,3), dtype = complex)
        if k1 != 0 or k2 != 0:
            f = np.sqrt(self._F)
            kappa = k1**2 + k2**2
            omega = 1j*f*np.sqrt(1 + kappa/(f**2))

            # Eigenvectors in columns corresponding to slow mode and
            # fast waves travelling in either direction
            A = np.array([[-1j*k2/f, 1j*(f*k2 + k1*omega)/kappa,            1j*(f*k2 - k1*omega)/kappa],
                          [1j*k1/f,  -1j*(f**2 + k2**2)/(f*k1 + k2*omega),  -1j*(f**2 + k2**2)/(f*k1 - k2*omega)],
                          [1.,       1.,                                    1.]], dtype = complex)

            A[:,0] = A[:,0]/np.sqrt(1 + 1/f**2 * kappa)
            A[:,1] = A[:,1]/(np.sqrt(2) * np.sqrt(1 + f**2 * 1/kappa))
            A[:,2] = A[:,2]/(np.sqrt(2) * np.sqrt(1 + f**2 * 1/kappa))

        else:  # Special case for k1 = k2 = 0
            A = np.array([[0., -1j/np.sqrt(2.), 1j/np.sqrt(2.)],\
                          [0., 1./np.sqrt(2.),  1./np.sqrt(2.)],\
                          [1., 0.,              0.]],dtype = complex)

        return A


