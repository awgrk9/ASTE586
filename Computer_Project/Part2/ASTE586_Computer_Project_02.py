## ASTE 586 Computer Project
##      Part 2
## Andrew Gerth
## 20250228
import copy

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


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


####################################################################################################
### This section will Define the Parameters for a particular Test Case                           ###
####################################################################################################
test_case = 2

if test_case == 1:
    I = [5, 4, 5]  # Moments of Inertia
    omega0 = [0.0181844, 0.128911, 0]  # Initial angular velocity
    q0 = [0, 0, 0.0871557, 0.996195]   # Initial euler-rodrigues parameters
else:
    I = [5, 4, 4]  # Moments of Inertia
    omega0 = [0.0592384, 0, 0.0740480]  # Initial angular velocity
    q0 = [0, 0.382683, 0, 0.923880]   # Initial euler-rodrigues parameters

t_span = (0, 500)  # Integration time range
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



####################################################################################################
### Compute Euler Angles                                                                         ###
####################################################################################################
# recognize which axes are symmetric
if I[0] is I[1]:
    symmetry_axis = 2
    first_axis = 0  # First  equal I_x axis
    second_axis = 1 # Second equal I_x axis
    e_ref = [0, 0, 1]

elif I[1] is I[2]:
    symmetry_axis = 0
    first_axis = 1   # First  equal I_x axis
    second_axis = 2  # Second equal I_x axis
    e_ref = [1, 0, 0]

elif I[0] is I[2]:
    symmetry_axis = 1
    first_axis = 0   # First  equal I_x axis
    second_axis = 2  # Second equal I_x axis
    e_ref = [0, 1, 0]

theta = np.zeros((len(H_c), 1))
for k in range(0, len(H_c)):
    theta[k] = np.acos(H_c[k, symmetry_axis]/np.linalg.norm(H_c[k, :]))


## Compute Spin Angle
e_c_node = np.cross(H_c[0, :], e_ref)/np.linalg.norm(np.cross(H_c[0, :], e_ref))
print('e_C_node: {}'.format(e_c_node))
e_c_x = np.cross(e_ref, e_c_node)
print('e_C_x:    {}'.format(e_c_x))
u = np.zeros((len(t_vals), 3))
for k in range(0, len(t_vals)):
    u[k, :] = np.cross(H_c[k, :], e_ref)/np.linalg.norm(np.cross(H_c[k, :], e_ref))

phi = -np.atan2(np.dot(u, e_c_x), np.dot(u, e_c_node))


## Compute Precession

## Need to get all of the quantities in the inertial frame, F
u_f     = np.zeros((len(t_vals), 3))
v_f     = np.zeros((len(t_vals), 3))
e_f_ref = np.zeros((len(t_vals), 3))
for k in range(0, len(t_vals)):
    e_f_ref[k, :] = rotate_vector_by_quaternion(e_ref, q_vals[k, :] )
    u_f[k, :] =  np.cross(H_f[k, :], e_f_ref[k, :])/np.linalg.norm(np.cross(H_f[k, :], e_f_ref[k, :]))
    v_f[k, :] =  np.cross(H_f[k, :]/np.linalg.norm(H_f[k, :]), u_f[k, :])


psi = np.atan2(np.dot(u_f, v_f[0,:]), np.dot(u_f,u_f[0,:]))

####################################################################################################
### Plotting Section                                                                             ###
####################################################################################################


fig1, ax1 = plt.subplots(4, 1, figsize=(12,16))
fig1.canvas.manager.set_window_title('State Plots')
plt.subplots_adjust(hspace=0.4)
ax1[0].plot(t_vals, omega_vals)
ax1[1].plot(t_vals, q_vals)
ax1[0].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
ax1[1].legend([r'$\epsilon_1$', r'$\epsilon_2$', r'$\epsilon_3$', r'$\epsilon_4$'])
ax1[0].set_title('Angular Velocity - Numerical Solution')
ax1[1].set_title('Euler.Rodrigues Parameters - Numerical Solution')
ax1[0].set_xlabel('time (s)')
ax1[1].set_ylabel('rad/s')
ax1[0].set_xlabel('time (s)')
ax1[2].plot(t_vals, omega_analytical)
ax1[3].plot(t_vals, q_analytical)
ax1[2].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
ax1[3].legend([r'$\epsilon_1$', r'$\epsilon_2$', r'$\epsilon_3$', r'$\epsilon_4$'])
ax1[2].set_title('Angular Velocity - Analytical Solution')
ax1[3].set_title('Euler.Rodrigues Parameters - Analytical Solution')
ax1[2].set_xlabel('time (s)')
ax1[3].set_ylabel('rad/s')
ax1[2].set_xlabel('time (s)')

fig2, ax2 = plt.subplots(4, 1, figsize=(12,16))
#fig3.suptitle('Plot 3')
fig2.canvas.manager.set_window_title('Error Plot')
plt.subplots_adjust(hspace=0.4)
ax2[0].plot(t_vals, omega_error)
ax2[1].plot(t_vals, q_error)
ax2[0].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
ax2[1].legend([r'$\epsilon_1$', r'$\epsilon_2$', r'$\epsilon_3$', r'$\epsilon_4$'])
ax2[2].plot(t_vals, q_unity_numerical)
ax2[3].plot(t_vals, q_unity_analytical)
ax2[0].set_title('Absolute Error -- Angular Velocity')
ax2[1].set_title('Absolute Error -- Euler.Rodrigues Parameters')
ax2[2].set_title('Quaternion Unity Error -- Numerical Solution')
ax2[3].set_title('Quaternion Unity Error - Analytical Solution')
ax2[0].set_xlabel('time (s)')
ax2[0].set_ylabel('rad/s')
ax2[1].set_xlabel('time (s)')
ax2[2].set_xlabel('time (s)')
ax2[3].set_xlabel('time (s)')

fig3, ax3 = plt.subplots(3, 1, figsize=(12,16))
fig3.canvas.manager.set_window_title('Angular Momentum Plots')
plt.subplots_adjust(hspace=0.4)
ax3[0].plot(t_vals, H_c)
ax3[0].set_title('Angular Momentum represented in the Body Frame, C')
ax3[0].legend([r'$h_1(t)$', r'$h_2(t)$', r'$h_3(t)$'])
ax3[1].plot(t_vals, H_f)
ax3[1].set_title('Angular Momentum represented in the Inertial Frame, F')
ax3[1].legend([r'$h_1(t)$', r'$h_2(t)$', r'$h_3(t)$'])
ax3[2].plot(t_vals, H_f-H_f[0, :])
ax3[2].set_title('Angular Momentum difference from initial Angular Momentum\nRepresented in the Inertial Frame, F')
ax3[2].legend([r'$h_1(t)-h_1(0) $', r'$h_2(t)-h_2(0)$', r'$h_3(t)-h_3(0)$'])
ax3[0].set_xlabel('time (s)')
ax3[0].set_ylabel(r'$kg*m^2/s$')
ax3[1].set_xlabel('time (s)')
ax3[1].set_ylabel(r'$kg*m^2/s$')
ax3[2].set_xlabel('time (s)')
ax3[2].set_ylabel(r'$kg*m^2/s$')

fig4, ax4 = plt.subplots(3, 1, figsize=(12,16))
fig4.canvas.manager.set_window_title('Euler Angle Plots')
plt.subplots_adjust(hspace=0.4)
ax4[0].plot(t_vals, np.degrees(theta))
ax4[0].set_ylim(0, 90)
ax4[0].set_title('Nutation Angle vs. Time')
ax4[0].legend([r'$\theta^\degree$'])
ax4[1].plot(t_vals, np.degrees(psi))
ax4[1].set_title('Precession Angle vs. Time')
ax4[1].legend([r'$\psi^\degree$'])
ax4[2].plot(t_vals, np.degrees(phi))
ax4[2].set_title('Spin Angle vs. Time')
ax4[2].legend([r'$\phi^\degree$'])
ax4[0].set_xlabel('time (s)')
ax4[0].set_ylabel(r'$\degree$')
ax4[1].set_xlabel('time (s)')
ax4[1].set_ylabel(r'$\degree$')
ax4[2].set_xlabel('time (s)')
ax4[2].set_ylabel(r'$\degree$')

fig1.savefig('State_Plots.png')
fig2.savefig('Errors.png')
fig3.savefig('Momentum.png')
fig4.savefig('EulerAngles.png')
plt.show()



