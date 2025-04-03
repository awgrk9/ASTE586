import numpy as np
import scipy as sp


def torqueFree_solver(I, omega0, q0, t_span):


    ####################################################################################################
    ### This section will define the necessary functions                                             ###
    ###     1. Definition of the system for the Numerical Solver to chew on                          ###
    ###     4. Function to input a 3-coordinate vector and rotate by a quaternion, returning a vector###
    ####################################################################################################


    # Define diff eq system for Numerical Solver in scipy (combine both eqns into 1 function)
    #   t: time
    #   z: state vector (w1, w2, w3, q1, q2, q3, q4)
    #   I: Inertia matrix
    #   returns dz: (dw1, dw2, dw3, dq1, dq2, dq3, dq4)
    def euler_equations(t, z, I):
        # Computes d(omega)/dt using Euler's equations for a free rigid body.
        z = np.transpose(z)
        dz = [0, 0, 0, 0, 0, 0, 0] # Initialize alpha vector
        # Compute angular acceleration (alpha)
        dz[0] = (I[1] - I[2]) * z[1] * z[2] / I[0]
        dz[1] = (I[2] - I[0]) * z[0] * z[2] / I[1]
        dz[2] = (I[0] - I[1]) * z[0] * z[1] / I[2]

        q1, q2, q3, q4 = z[3:]
         #print('p', q1, q2, q3, q4)
        # M can be found in the notes: "aste586-supplement-8-quaternion-rates-20250226.pdf"
        M = np.array([[ q4, -q3,  q2],
                      [ q3,  q4, -q1],
                      [-q2,  q1,  q4],
                      [-q1, -q2, -q3]])
        #print(M.shape)
        #print(z[:3].shape)
        dq = 0.5 * M @ z[:3]
        dz[3] = dq[0]
        dz[4] = dq[1]
        dz[5] = dq[2]
        dz[6] = dq[3]
        return dz



    state_0 = omega0 + q0 # This is the overall test state initial conditions (w1, w2, w3, q1, q2, q3, q4) at t=0


    # Solve ODE
    solution_numerical = sp.integrate.solve_ivp(euler_equations,
                                                t_span=t_span,
                                                y0=state_0,
                                                args=(I,),
                                                rtol=1E-10,
                                                atol=1E-10)


    # Extract results
    t_vals       = np.transpose(solution_numerical.t)
    omega_vals   = np.transpose(solution_numerical.y[:3, :])
    q_vals       = np.transpose(solution_numerical.y[3:, :])

    return t_vals, omega_vals, q_vals
