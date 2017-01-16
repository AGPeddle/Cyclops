#!/usr/bin/env/ python3
"""
"""
#from spectral_toolbox import SpectralToolbox
import numpy as np
import time

class Solvers:
    @staticmethod
    def fine_propagator(control, expInt, st, U_hat):
        """
        Implements the fine propagator used in the APinT or Parareal methods. Can handle
        any implemented methods for the linear and nonlinear operator. Calls through to
        appropriate methods.

        The full range of fine timesteps is taken in this module from the previous to the
        next coarse timestep. Only the solution corresponding to the desired coarse timestep
        is returned; all others are discarded.

        **Parameters**

        - `U_hat` : the solution at the current timestep

        **Returns**

        - `u_hat_new` : the solution at the next timestep

        """

        U_hat_new = np.zeros(np.shape(U_hat), dtype = 'complex')
        U_hat_old = U_hat.copy()

        t = 0
        while t < control['final_time']:
            "limit fine timestep size to avoid overshooting the last timestep"
            dt = min(control['fine_timestep'], control['final_time']-t)

            U_hat_new = strang_splitting(U_hat_old, dt, control, expInt, st, exp_L_exp_D, compute_nonlinear)

            U_hat_old = U_hat_new
            t += dt

        return U_hat_new

    @staticmethod
    def coarse_propagator(control, expInt, st, U_hat):
        """
        Implements the coarse propagator used in the APinT method. Can handle any implemented methods
        for the linear and nonlinear operator. Calls through to appropriate methods. Differs from the
        non_asymptotic version in its call to the average force computation.

        **Parameters**

        - `U_hat` : the solution at the current timestep

        **Returns**

        - `u_hat_new` : the solution at the next timestep

        """
        U_hat_old = U_hat.copy()
        t = 0
        while t < control['final_time']:
            "limit fine timestep size to avoid overshooting the last timestep"
            dt = min(control['coarse_timestep'], control['final_time']-t)

            start = time.time()
            U_hat_new = strang_splitting(U_hat_old, dt, control, expInt, st, dissipative_exponential, compute_average_force2)
            #U_hat_new = expInt.call(U_hat_new, dt)
            U_hat_new = expInt[0].call(expInt[1].call(U_hat, dt), dt)
            end = time.time()
            print("Time for one timestep was {} seconds".format(end-start))

            U_hat_old = U_hat_new
            t += dt

        return U_hat_new


########## BEGIN SOLVER (SUB) ROUTINES ##########

def dissipative_exponential(control, expInt, U_hat, t):
    """
    Implements dissipation through a 4th-order hyperviscosity operator
    and matrix exponential.

    **Parameters**

    - `U_hat` : the known solution at the current timestep
    - `t` : the timestep taken

    **Returns**

    - `U_hat_sp` : The solution at the next timestep by dissipation

    """

    U_hat_sp = np.zeros((3, control['Nx'], control['Nx']), dtype = 'complex')
    wavenumbers_x = np.arange(-control['Nx']/2, control['Nx']/2)
    wavenumbers_y = np.arange(-control['Nx']/2, control['Nx']/2)

    wavenums_x, wavenums_y = np.meshgrid(wavenumbers_x, wavenumbers_y)

    exp_D = np.exp(-control['mu']*t*(wavenums_x**4 + wavenums_y**4))
    for k in range(3):
        U_hat_sp[k,:,:] = exp_D*U_hat[k,:,:]
    return U_hat_sp

def exp_L_exp_D(control, expInt, U_hat, t):
    """
    Call-through method for applying the linear solution
    and the dissipative term. (Both via exponential integrator)

    **Parameters**

    - `U_hat` : the known solution at the current timestep
    - `t` : the timestep taken

    **Returns**

    - `U_hat_sp` : The solution propagated by L and D

    """

    U_hat_sp = expInt.call(U_hat, t)  # exp(Lt/epsilon)
    U_hat_sp = dissipative_exponential(control, expInt, U_hat_sp, t)  # Dissipative term
    return U_hat_sp

def strang_splitting(U_hat, delta_t, control, expInt, st, linear, nonlinear):
    """
    Propagates a solution for arbitrary linear and nonlinear operators over a timestep
    delta_t by using Strang Splitting.

    """

    U_hat_new = linear(control, expInt, U_hat, 0.5*delta_t)
    # Computation of midpoint:
    U_NL_hat = nonlinear(U_hat_new, control, st, expInt)
    U_NL_hat1 = -delta_t*U_NL_hat

    U_NL_hat = nonlinear(U_hat_new + 0.5*U_NL_hat1, control, st, expInt)
    U_NL_hat2 = -delta_t*U_NL_hat
    U_hat_new = U_hat_new + U_NL_hat2

    U_hat_new = linear(control, expInt, U_hat_new, 0.5*delta_t)
    return U_hat_new

def compute_nonlinear(U_hat, control, st, expInt = None):
    """
    Function to compute the nonlinear terms of the RSWE. Calls through to some
    spectral toolbox methods to implement derivatives and nonlinear multiplication.

    This function implements the simple solution to the problem, with none of the wave averaging.

    **Parameters**
    -`U_hat` : the components of the unknown vector in Fourier space, ordered u, v, h
    -`linear_operator` : not used here. Required to have identical calling to compute_average_force

    **Returns**
    -`U_NL_hat` : the result of the multiplication in Fourier space

    **See Also**
    compute_average_force

    """
    N1, N2 = np.shape(U_hat)[1::]
    U_NL_hat = np.zeros((3,N1,N2), dtype='complex')

    v1_hat = U_hat[0,:,:]
    v2_hat = U_hat[1,:,:]
    h_hat = U_hat[2,:,:]

    v1_x_hat = st.calc_derivative(v1_hat, 'x')
    v1_y_hat = st.calc_derivative(v1_hat, 'y')

    v2_x_hat = st.calc_derivative(v2_hat, 'x')
    v2_y_hat = st.calc_derivative(v2_hat, 'y')

    # Compute Fourier coefficients of v1 * v1_x1 + v2 * v1_x2
    U_NL_hat1 = st.multiply_nonlinear(v1_x_hat, v1_hat)
    U_NL_hat2 = st.multiply_nonlinear(v1_y_hat, v2_hat)
    U_NL_hat[0,:,:] = U_NL_hat1 + U_NL_hat2

    # Compute Fourier coefficients of  v1 * v2_x1 + v2 * v2_x2
    U_NL_hat1 = st.multiply_nonlinear(v2_x_hat, v1_hat)
    U_NL_hat2 = st.multiply_nonlinear(v2_y_hat, v2_hat)
    U_NL_hat[1,:,:] = U_NL_hat1 + U_NL_hat2

    # Compute Fourier coefficients of (h*v1)_x1 + (h*v2)_x2
    U_NL_hat1 = st.multiply_nonlinear(h_hat, v1_hat)
    U_NL_hat1 = st.calc_derivative(U_NL_hat1, 'x')
    U_NL_hat2 = st.multiply_nonlinear(h_hat, v2_hat)
    U_NL_hat2 = st.calc_derivative(U_NL_hat2, 'y')
    U_NL_hat[2,:,:] = U_NL_hat1 + U_NL_hat2

    return U_NL_hat

########## END SOLVER (SUB) ROUTINES ##########

########## BEGIN WAVE AVERAGING ROUTINES ##########

def filter_kernel_exp(M, s):
    """
    Smooth integration kernel.

    This kernel is used for the integration over the fast waves. It is
    formulated as:

    .. math:: \\rho(s) \\approx \\exp(-50*(s-0.5)^{2})

    and is normalised to have a total integral of unity. This method is
    used for the wave averaging, which is performed by `self.compute_average_force`.

    **Parameters**

    - `M` : The interval over which the average is to be computed
    - `s` : The sample in the interval

    **Returns**

    The computed kernel value at s.

    """

    points = np.arange(1, M)/float(M)
    norm = (1.0/float(M))*sum(np.exp(-50*(points - 0.5)**2))
    return np.exp(-50*(s-0.5)**2)/norm

def compute_average_force2(U_hat, control, st, expInt):
    """
    This method computed the wave-averaged solution for use with the APinT
    coarse timestepping.

    The equation solved by this method is a modified equation for a slowly-varying
    solution (see module header, above). The equation solved is:

    .. math:: \\bar{N}(\\bar{u}) = \\sum \\limits_{m=0}^{M-1} \\rho(s/T_{0})e^{sL}N(e^{-sL}\\bar{u}(t))

    where :math:`\\rho` is the smoothing kernel (`filter_kernel`).

    **Parameters**

    - `U_hat` : the known solution at the current timestep
    - `linear_operator` : the linear operator being used, to be
      passed to the nonlinear computation

    **Returns**

    - `U_hat_averaged` : The predicted averaged solution at the next timestep

    **Notes**

    The smooth kernel is chosen so that the length of the time window over which the averaging is
    performed is as small as possible and the error from the trapezoidal rule is negligible.

    *See Also**
    `filter_kernel`

    """

    T0_L = control['HMM_T0_L']
    T0_M = control['HMM_T0_M']
    M = control['HMM_M_bar_L']
    N = control['HMM_M_bar_M']

    filter_kernel = filter_kernel_exp

    U_hat_NL_averaged = np.zeros(np.shape(U_hat), dtype = 'complex')

    for m in np.arange(1,M):
        for n in np.arange(1,N):
            #print("m,n : {}, {}".format(m, n))
            tm = T0_L*m/float(M)
            tn = T0_M*n/float(N)
            Km = filter_kernel(M, m/float(M))
            Kn = filter_kernel(N, n/float(N))

            U_hat_RHS = expInt[0].call(expInt[1].call(U_hat, tm), tn)
            U_hat_RHS = compute_nonlinear(U_hat_RHS, control, st)
            U_hat_RHS = expInt[0].call(expInt[1].call(U_hat_RHS, -tm), -tn)

            U_hat_NL_averaged += Km*Kn*U_hat_RHS

    return U_hat_NL_averaged/float(M)/float(N)

def compute_average_force(U_hat, control, st, expInt):
    """
    This method computed the wave-averaged solution for use with the APinT
    coarse timestepping.

    The equation solved by this method is a modified equation for a slowly-varying
    solution (see module header, above). The equation solved is:

    .. math:: \\bar{N}(\\bar{u}) = \\sum \\limits_{m=0}^{M-1} \\rho(s/T_{0})e^{sL}N(e^{-sL}\\bar{u}(t))

    where :math:`\\rho` is the smoothing kernel (`filter_kernel`).

    **Parameters**

    - `U_hat` : the known solution at the current timestep
    - `linear_operator` : the linear operator being used, to be
      passed to the nonlinear computation

    **Returns**

    - `U_hat_averaged` : The predicted averaged solution at the next timestep

    **Notes**

    The smooth kernel is chosen so that the length of the time window over which the averaging is
    performed is as small as possible and the error from the trapezoidal rule is negligible.

    *See Also**
    `filter_kernel`

    """

    T0 = control['HMM_T0']
    M = control['HMM_M_bar']
    filter_kernel = filter_kernel_exp

    U_hat_NL_averaged = np.zeros(np.shape(U_hat), dtype = 'complex')

    for m in np.arange(1,M):
        tm = T0*m/float(M)
        Km = filter_kernel(M, m/float(M))
        U_hat_RHS = expInt.call(U_hat, tm)
        U_hat_RHS = compute_nonlinear(U_hat_RHS, control, st)
        U_hat_RHS = expInt.call(U_hat_RHS, -tm)

        U_hat_NL_averaged += Km*U_hat_RHS
    return U_hat_NL_averaged/float(M)

########## END WAVE AVERAGING ROUTINES ##########

def solve(solver_name, control, st, expInt, u_init):

    """
    if solver_name == 'coarse_propagator':
        control['HMM_T0'] = control['HMM_T0_g']
        control['HMM_M_bar'] = 100*control['HMM_T0']
    """

    out_sols = np.zeros((3, control['Nx'], control['Nx']), dtype = 'complex')
    for k in range(3):
        out_sols[k, :, :] = st.forward_fft(u_init[k, :, :])

    out_sols = getattr(Solvers, solver_name)(control, expInt, st, out_sols)

    for k in range(3):
        out_sols[k, :, :] = st.inverse_fft(out_sols[k, :, :])

    return np.real(out_sols)
