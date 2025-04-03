import numpy as np
import scipy as sp



def torqueFree_solver_symm(I, omega0, q0, t_span):



    ####################################################################################################
    ### This section will define the necessary functions                                             ###
    ###     1. Definition of the system for the Numerical Solver to chew on                          ###
    ###     2. Analytical closed-form solution for Angular Velocity components for any time, t       ###
    ###     3. Analytical closed-form solution for Euler-Rodrigues Parameters for any time, t        ###
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


    ## Define function to Compute Analytical Solution for Angular velocity assuming axisymmetry
    #   t: time
    #   omega_0: 1x3 array | Initial Angular Velocity Values
    #   I: inertia matrix
    def euler_analytical(t, omega_0, I):
        omega = np.zeros((3,1)) # initialize omega vector

        # recognize which axes are symmetric
        if I[0] is I[1]:
            symmetry_axis = 2
            first_axis = 0  # First  equal I_x axis
            second_axis = 1 # Second equal I_x axis
            # define lambda based on symmetry axis
            lamd = omega_0[symmetry_axis] * (I[symmetry_axis] - I[0])/I[0]

        elif I[1] is I[2]:
            symmetry_axis = 0
            first_axis = 1   # First  equal I_x axis
            second_axis = 2  # Second equal I_x axis
            # define lambda based on symmetry axis
            lamd = -omega_0[symmetry_axis] * (I[1] - I[symmetry_axis]) / I[1]

        elif I[0] is I[2]:
            symmetry_axis = 1
            first_axis = 0   # First  equal I_x axis
            second_axis = 2  # Second equal I_x axis
            # define lambda based on symmetry axis
            lamd = omega_0[symmetry_axis] * (I[0] - I[symmetry_axis]) / I[0]


        #print('Symmetry axis has been determined to be: {}.'.format(symmetry_axis+1))
        #print('First and second axes are: {}, {}.'.format(first_axis + 1, second_axis + 1))
        ## Analytical solution for Euler's Equations when assuming axisymmetry
        #   Eqns have been coded with variable indecies to handle any of the three possibilities for axis of sym.
        omega[first_axis]  = np.cos(lamd * t) * omega_0[first_axis] - np.sin(lamd * t) * omega_0[second_axis]
        omega[second_axis] = np.sin(lamd * t) * omega_0[first_axis] + np.cos(lamd * t) * omega_0[second_axis]
        omega[symmetry_axis] = omega_0[symmetry_axis]
        #print(omega)
        return np.transpose(omega)


    ## Define function to Compute Analytical Solution for the Quaternion Evolution assuming axisymmetry
    #   t: time
    #   w: 1x3 array | Angular Velocity found at the time step 't' from the analytical solution to Euler's Eqns
    #   q0: 1x4 array | Initial Euler-Rodrigues parameters/quaternion components
    def quat_analytical(t, w, q0):
        # See my report for further explanation of the M_star term and how it was derived
        # For the purposes of code commenting, M_star is just a 4x4 skew-symmetric matrix comprised of previously computed
        #   angular velocity components.
        #print(w)
        M_star = np.array([[ 0   ,  w[2], -w[1],  w[0]],
                           [-w[2],  0   ,  w[0],  w[1]],
                           [ w[1], -w[0],  0.  ,  w[2]],
                           [-w[0], -w[1], -w[2],  0   ]])

        A = 1/2*M_star*t
        eigenvalues, P = np.linalg.eig(A)  # Get eigenvalues and eigenvectors
        D = np.diag(np.exp(eigenvalues))  # Exponentiate eigenvalues
        expm_approx = P @ D @ np.linalg.inv(P)  # Compute expm(A) manually

        #print(expm_approx)  # Should match expm(A)

        #q = sp.linalg.expm(0.5 * M_star * t) @ q0
        q = np.matmul(expm_approx, q0)


        return np.transpose(q)


    ## Define function to rotate a vector by a quaternion (scalar last notation)
    #   v: 3d vector
    #   q: 4d vector (x, y, z, s)
    #   returns a new vector (v rotated by q)
    def rotate_vector_by_quaternion(v, q):
        v = np.array(v)
        qv = np.array(q[:3])  # Vector part of quaternion
        qs = q[3]  # Scalar part of quaternion

        # Compute cross products and dot products
        qv_cross_v = np.cross(qv, v)
        qv_cross_qv_cross_v = np.cross(qv, qv_cross_v)

        # Apply quaternion rotation formula
        v_rot = v + 2 * (qs * qv_cross_v + qv_cross_qv_cross_v)

        return v_rot



    state_0 = omega0 + q0 # This is the overall test state initial conditions (w1, w2, w3, q1, q2, q3, q4) at t=0

    ####################################################################################################
    ### This section will compute and plot the numerical solution for both Differential Eqn Systems  ###
    ####################################################################################################

    # First Solve for angular velocity numerically
    #   Solving numerically first so that the numerical integrator can choose the relevant time steps
    #   I'll then use those same time steps when assessing the analytical solution.

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


    ####################################################################################################
    ### This section will compute and plot the analytical solution for both Differential Eqn Systems ###
    ####################################################################################################


    # Define empty array for analytical omega solution to go into
    omega_analytical = np.zeros((len(t_vals), 3))
    # Compute analytical omega from the 'euler_analytical' function for each time step declared by the numerical integrator
    i = 0
    for time in t_vals:
        #print(omega_analytical[i, :])
        omega_analytical[i, :] = euler_analytical(time, omega0, I)
        #print(omega_analytical[i, :])
        i = i + 1 # indexer to fill the rows of omega_analytical


    # Define empty array for analytical quaternion solution to go into
    q_analytical = np.zeros((len(t_vals), 4))
    # Compute analytical quaternions from the 'quat_analytical' function for each time step declared by the numerical integrator
    i = 0
    for time in t_vals:
        #print(q_analytical[i, :])
        q_analytical[i, :] = quat_analytical(time, omega_analytical[i, :], q0)
        #print(q_analytical[i, :])

        i = i + 1 # indexer to fill the rows of q_analytical
        #print(i)

    ####################################################################################################
    ### Test Case Validation                                                                         ###
    ####################################################################################################

    ## Compute Angular Velocity Difference Numerical-Analytical
    omega_error = omega_vals - omega_analytical
    q_error = q_vals - q_analytical


    ## Compute Quaternion Unity for Numerical and Analytical
    q_unity_numerical  = q_vals[:, 0]**2 + q_vals[:, 1]**2 + q_vals[:, 2]**2 + q_vals[:, 3]**2 - 1
    q_unity_analytical = q_analytical[:, 0]**2 + q_analytical[:, 1]**2 + q_analytical[:, 2]**2 + q_analytical[:, 3]**2 - 1

    ## Compute Angular Momentum Components represented in Body Frame
    h1_c = I[0] * omega_vals[:, 0]
    h2_c = I[1] * omega_vals[:, 1]
    h3_c = I[2] * omega_vals[:, 2]
    H_c = np.zeros((len(t_vals), 3))
    H_c[:, 0] = h1_c
    H_c[:, 1] = h2_c
    H_c[:, 2] = h3_c
    #print(H_c[:10, :10])
    ## Use quaternion at each time step to rotate to Inertial Frame 'F'
    H_f = np.zeros((len(H_c), 3))



    for j in range(0, len(H_f)):
        H_f[j, :] = rotate_vector_by_quaternion(H_c[j, :], q_vals[j, :])

    return t_vals, omega_vals, q_vals, H_c, H_f, omega_error, q_unity_numerical, omega_analytical