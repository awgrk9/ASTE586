## ASTE 586 Computer Project
##      Part 2
## Andrew Gerth
## 20250228

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


####################################################################################################
### This section will define the necessary functions                                             ###
###     1. Restatement of Euler's Equations for the Numerical Solver to chew on                  ###
###     2. Restatement of Euler-Rodrigues Parameter Equations for the Numerical solver           ###
###     3. Analytical closed-form solution for Angular Velocity components for any time, t       ###
###     4. Analytical closed-form solution for Euler-Rodrigues Parameters for any time, t        ###
####################################################################################################

# Define diff eq system for Numerical Solver in scipy
#   t: time
#   omega: angular velocity
#   I: Inertia matrix
def euler_equations(t, omega, I):
    # Computes d(omega)/dt using Euler's equations for a free rigid body.
    omega = np.transpose(omega)
    alpha = np.zeros((3, 1)) # Initialize alpha vector
    # Compute angular acceleration (alpha)
    alpha[0] = (I[1] - I[2]) * omega[1] * omega[2] / I[0]
    alpha[1] = (I[2] - I[0]) * omega[0] * omega[2] / I[1]
    alpha[2] = (I[0] - I[1]) * omega[0] * omega[1] / I[2]

    return np.transpose(alpha)

# Define diff eq system for Numerical Solver in scipy
#   t: time
#   q: 1x4 array | Euler-Rodrigues Parameters 1-4
#   omega_interp: Interpolated angular velocity function to feed the Euler-Rodrigues equation (eqn 1 in handout)
def quaternion_equations(t, q, omega_interp):
    q1, q2, q3, q4 = q
    omega = omega_interp(t)
    # M can be found in the notes: "aste586-supplement-8-quaternion-rates-20250226.pdf"
    M = np.array([[ q4, -q3,  q2],
                  [ q3,  q4, -q1],
                  [-q2,  q1,  q4],
                  [-q1, -q2, -q3]])
    dq_dt = 0.5 * M @ omega

    return dq_dt

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
        lamd = omega_0[symmetry_axis] * (I[1] - I[symmetry_axis]) / I[1]

    elif I[0] is I[2]:
        symmetry_axis = 1
        first_axis = 0   # First  equal I_x axis
        second_axis = 2  # Second equal I_x axis
        # define lambda based on symmetry axis
        lamd = omega_0[symmetry_axis] * (I[0] - I[symmetry_axis]) / I[0]


    print('Symmetry axis has been determined to be: {}.'.format(symmetry_axis+1))
    print('First and second axes are: {}, {}.'.format(first_axis + 1, second_axis + 1))
    ## Analytical solution for Euler's Equations when assuming axisymmetry
    #   Eqns have been coded with variable indecies to handle any of the three possibilities for axis of sym.
    omega[first_axis]  = np.cos(lamd * t) * omega_0[first_axis] - np.sin(lamd * t) * omega_0[second_axis]
    omega[second_axis] = np.sin(lamd * t) * omega_0[first_axis] + np.cos(lamd * t) * omega_0[second_axis]
    omega[symmetry_axis] = omega_0[symmetry_axis]
    print(omega)
    return np.transpose(omega)



## Define function to Compute Analytical Solution for the Quaternion Evolution assuming axisymmetry
#   t: time
#   w: 1x3 array | Angular Velocity found at the time step 't' from the analytical solution to Euler's Eqns
#   q0: 1x4 array | Initial Euler-Rodrigues parameters/quaternion components
def quat_analytical(t, w, q0):
    # See my report for further explanation of the M_star term and how it was derived
    # For the purposes of code commenting, M_star is just a 4x4 skew-symmetric matrix comprised of previously computed
    #   angular velocity components.
    M_star = np.array([[ 0   ,  w[2], -w[1],  w[0]],
                       [-w[2],  0   ,  w[0],  w[1]],
                       [ w[1], -w[0],  0.  ,  w[2]],
                       [-w[0], -w[1], -w[2],  0   ]])
    # General analytical solution. Note I am using sciPy's Matrix Exponential as I didn't have the knowledge to
    #   simplify the matrix exponential to a better form. If this proves to add unacceptable error into the computation,
    #   it will be the first thing on the chopping block...
    #theta = np.linalg.norm(w)
    #M_star_test = np.array([[np.cosh(theta*t/2), -w[2]/theta*np.sinh(theta*t/2), w[1]/theta*np.sinh(theta*t/2), -w[0]/theta*np.sinh(theta*t/2)],
    #                        [w[2]/theta*np.sinh(theta*t/2), np.cosh(theta*t/2), -w[0]/theta*np.sinh(theta*t/2), -w[1]/theta*np.sinh(theta*t/2)],
    #                        [-w[1]/theta*np.sinh(theta*t/2), w[0]/theta*np.sinh(theta*t/2), np.cosh(theta*t/2), -w[2]/theta*np.sinh(theta*t/2)],
    #                        [w[0]/theta*np.sinh(theta*t/2), w[1]/theta*np.sinh(theta*t/2), w[2]/theta*np.sinh(theta*t/2), np.cosh(theta*t/2)]])
    #M_star_test_2 = np.eye(4) + (np.sinh(theta*t/2)/theta) * M_star + (np.cosh(theta*t/2)-1)*np.square(M_star)/theta**2
    #M_star_test_3 = np.cosh(theta*t/2)*np.eye(4) + (np.sinh(theta*t/2)/theta) * M_star
    A = 1/2*M_star*t
    eigenvalues, P = np.linalg.eig(A)  # Get eigenvalues and eigenvectors
    D = np.diag(np.exp(eigenvalues))  # Exponentiate eigenvalues
    expm_approx = P @ D @ np.linalg.inv(P)  # Compute expm(A) manually

    #print(expm_approx)  # Should match expm(A)

    #q = np.linalg.expm(0.5 * M_star * t) @ q0
    q = np.matmul(expm_approx, q0)



    return np.transpose(q)



####################################################################################################
### This section will Define the Parameters for a particular Test Case                           ###
####################################################################################################

I = [5, 4, 5]  # Moments of Inertia
omega0 = [0.0181844, 0.128911, 0]  # Initial angular velocity
q0 = [0, 0, 0.0871557, 0.996195]   # Initial euler-rodrigues parameters
t_span = (0, 1000)  # Integration time range

####################################################################################################
### This section will compute and plot the numerical solution for both Differential Eqn Systems  ###
####################################################################################################

# First Solve for angular velocity numerically
#   Solving numerically first so that the numerical integrator can choose the relevant time steps
#   I'll then use those same time steps when assessing the analytical solution.

# Solve ODE
solution_omega = sp.integrate.solve_ivp(euler_equations,
                                        t_span=t_span,
                                        y0=omega0,
                                        args=(I,),
                                        rtol=1E-8,
                                        atol=1E-8)


## Interpolate omega(t) to get continuous function for use in the quaternion equation.
omega_interp = sp.interpolate.interp1d(solution_omega.t,
                                       solution_omega.y,
                                       axis=1,
                                       kind='cubic',
                                       fill_value="extrapolate")

# Solve ODE
solution_q = sp.integrate.solve_ivp(quaternion_equations,
                                    t_span=t_span,
                                    y0=q0,
                                    args=(omega_interp,),
                                    rtol=1E-8,
                                    atol=1E-8)


# Extract results
t_vals_omega = np.transpose(solution_omega.t)
omega_vals   = np.transpose(solution_omega.y)
t_vals_q     = np.transpose(solution_q.t)
q_vals       = np.transpose(solution_q.y)

## Plotting section
fig1, ax1 = plt.subplots(2, 1, figsize=(12,16))
fig1.suptitle('Plot 1')
fig1.canvas.manager.set_window_title('Plot 1')

ax1[0].plot(t_vals_omega, omega_vals)
ax1[1].plot(t_vals_q, q_vals)
ax1[0].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
ax1[1].legend([r'$\epsilon_1$', r'$\epsilon_2$', r'$\epsilon_3$', r'$\epsilon_4$'])


####################################################################################################
### This section will compute and plot the analytical solution for both Differential Eqn Systems ###
####################################################################################################


# Define empty array for analytical omega solution to go into
omega_analytical = np.zeros((len(t_vals_q), 3))
# Compute analytical omega from the 'euler_analytical' function for each time step declared by the numerical integrator
i = 0
for time in t_vals_q:
    print(omega_analytical[i, :])
    omega_analytical[i, :] = euler_analytical(time, omega0, I)
    print(omega_analytical[i, :])
    i = i + 1 # indexer to fill the rows of omega_analytical





# Define empty array for analytical quaternion solution to go into
q_analytical = np.zeros((len(t_vals_q), 4))
# Compute analytical quaternions from the 'quat_analytical' function for each time step declared by the numerical integrator
i = 0
for time in t_vals_q:
    print(q_analytical[i, :])
    q_analytical[i, :] = quat_analytical(time, omega_analytical[i, :], q0)
    print(q_analytical[i, :])

    i = i + 1 # indexer to fill the rows of q_analytical
    print(i)



## Plotting section
fig2, ax2 = plt.subplots(4, 1, figsize=(12,16))
fig2.suptitle('Plot 2')
fig2.canvas.manager.set_window_title('Euler Solve Analytically')

ax2[0].plot(t_vals_q, omega_analytical)
ax2[1].plot(t_vals_q, q_analytical)
ax2[0].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
ax2[1].legend([r'$\epsilon_1$', r'$\epsilon_2$', r'$\epsilon_3$', r'$\epsilon_4$'])



## Compute Angular Velocity DIfference Numerical-Analytical
omega_error = omega_vals - omega_analytical
fig3, ax3 = plt.subplots(1, 1, figsize=(12,16))
fig3.suptitle('Plot 3')
fig3.canvas.manager.set_window_title('Error Plot')
ax3.plot(t_vals_q, omega_error)

## Compute Quaternion Unity for Numerical and Analytical
q_unity_numerical  = q_vals[:, 0]**2 + q_vals[:, 1]**2 + q_vals[:, 2]**2 + q_vals[:, 3]**2 - 1
q_unity_analytical = q_analytical[:, 0]**2 + q_analytical[:, 1]**2 + q_analytical[:, 2]**2 + q_analytical[:, 3]**2 - 1
ax2[2].plot(t_vals_q, q_unity_numerical)
ax2[3].plot(t_vals_q, q_unity_analytical)

plt.show()



