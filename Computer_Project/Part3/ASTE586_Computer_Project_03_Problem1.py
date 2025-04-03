## ASTE 586 Computer Project
##      Part 2
## Andrew Gerth
## 20250228

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D



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
test_case = 1

if test_case == 1:
    I = [1.6975, 1.6975, 2.5272]
    omega0 = [0, 0.41071176, 4.69445692]  # Initial angular velocity
    q0 = [0, 0, 0, 1]   # Initial euler-rodrigues parameters
    t_span = (0, 100)  # Integration time range

else:
    I = [5, 4, 4]  # Moments of Inertia
    omega0 = [0.0592384, 0, 0.0740480]  # Initial angular velocity
    q0 = [0, 0.382683, 0, 0.923880]   # Initial euler-rodrigues parameters
    t_span = (0, 100)  # Integration time range

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
# Report the value of nutation angle in degrees
print('The nutation angle is {:.3f} degrees.'.format(np.degrees(theta[0, 0])))
start_idx   = 0     # Index to start rate calculation
end_idx     = 15    # Index to end rate calculation     NOTE: this method is crude and requires some
                    #       visual inspection to ensure the angle doesn't cross from Quadrant 1-4 or
                    #       4-1 in the set.
precession_amount = psi[end_idx] - psi[start_idx] # Total precession over the set in radians
time_amount = t_vals[end_idx] - t_vals[start_idx] # Time span of the set
psi_dot = precession_amount/time_amount # Precession rate in rad/s
spin_amount = phi[end_idx] - phi[start_idx] # Total spin over the set in radians
phi_dot = spin_amount/time_amount # Spin rate in rad/s
print('The precession rate is {:.3f} degrees/sec.'.format(np.degrees(psi_dot)))
print('The spin rate is {:.3f} degrees/sec.'.format(np.degrees(phi_dot)))

# Report the worst error for each omega axis:
print('The maximum error between numerical & analytical ang. vel for each component:')
print('w_1: {:.3e} rad/s'.format(np.max(np.abs(omega_error[:,0]))))
print('w_2: {:.3e} rad/s'.format(np.max(np.abs(omega_error[:,1]))))
print('w_3: {:.3e} rad/s'.format(np.max(np.abs(omega_error[:,2]))))

# Report the max quaternion unity error
print('\nThe maximum deviation from quaternion unity was: {:.3e}'.format(np.max(q_unity_numerical)))

# Report the worst error for each h_i(t) - h_i(0):
print('\nThe maximum error of h_i(t) - h_i(0):')
print('h_1(t) - h_1(0): {:.3e} kg*m^2/s'.format(np.max(np.abs(H_f[:,0] - H_f[0, 0]))))
print('h_2(t) - h_2(0): {:.3e} kg*m^2/s'.format(np.max(np.abs(H_f[:,1] - H_f[0, 1]))))
print('h_3(t) - h_3(0): {:.3e} kg*m^2/s'.format(np.max(np.abs(H_f[:,2] - H_f[0, 2]))))

fig1, ax1 = plt.subplots(3, 1, figsize=(12,16))
fig1.canvas.manager.set_window_title('State Plots')
plt.subplots_adjust(hspace=0.4)
ax1[0].plot(t_vals, omega_vals)
ax1[2].plot(t_vals, q_vals)
ax1[0].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
ax1[2].legend([r'$\epsilon_1$', r'$\epsilon_2$', r'$\epsilon_3$', r'$\epsilon_4$'])
ax1[0].set_title('Angular Velocity - Numerical Solution')
ax1[2].set_title('Euler.Rodrigues Parameters - Numerical Solution')
ax1[0].set_xlabel('time (s)')
ax1[0].set_ylabel('rad/s')
ax1[1].set_ylabel('rad/s')
ax1[0].set_xlabel('time (s)')
ax1[1].plot(t_vals, omega_analytical)
ax1[1].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
ax1[1].set_title('Angular Velocity - Analytical Solution')
ax1[1].set_xlabel('time (s)')
ax1[1].set_xlabel('time (s)')


fig2, ax2 = plt.subplots(2, 1, figsize=(12,16))
fig2.canvas.manager.set_window_title('Error Plot')
plt.subplots_adjust(hspace=0.4)
ax2[0].plot(t_vals, omega_error)
ax2[0].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
ax2[1].plot(t_vals, q_unity_numerical)
ax2[0].set_title('Absolute Error -- Angular Velocity')
ax2[1].set_title('Quaternion Unity Error -- Numerical Solution')
ax2[0].set_xlabel('time (s)')
ax2[0].set_ylabel('rad/s')
ax2[1].set_xlabel('time (s)')

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

# if test_case == 1:
#     for i, ax in enumerate(ax1):
#         extent = ax.get_tightbbox(fig1.canvas.get_renderer()).transformed(fig1.dpi_scale_trans.inverted())
#         fig1.savefig(f"./Part3_Submission/10subplot_{i+1}.png", bbox_inches=extent, pad_inches=0.1)
#
#     for i, ax in enumerate(ax2):
#         extent = ax.get_tightbbox(fig2.canvas.get_renderer()).transformed(fig2.dpi_scale_trans.inverted())
#         fig2.savefig(f"./Part3_Submission/11subplot_{i+1}.png", bbox_inches=extent, pad_inches=0.1)
#
#     for i, ax in enumerate(ax3):
#         extent = ax.get_tightbbox(fig3.canvas.get_renderer()).transformed(fig3.dpi_scale_trans.inverted())
#         fig3.savefig(f"./Part3_Submission/12subplot_{i+1}.png", bbox_inches=extent, pad_inches=0.1)
#
#     for i, ax in enumerate(ax4):
#         extent = ax.get_tightbbox(fig4.canvas.get_renderer()).transformed(fig4.dpi_scale_trans.inverted())
#         fig4.savefig(f"./Part3_Submission/13subplot_{i+1}.png", bbox_inches=extent, pad_inches=0.1)
# else:
#     for i, ax in enumerate(ax1):
#         extent = ax.get_tightbbox(fig1.canvas.get_renderer()).transformed(fig1.dpi_scale_trans.inverted())
#         fig1.savefig(f"./Part3_Submission/10subplot_{i + 1}.png", bbox_inches=extent, pad_inches=0.1)
#
#     for i, ax in enumerate(ax2):
#         extent = ax.get_tightbbox(fig2.canvas.get_renderer()).transformed(fig2.dpi_scale_trans.inverted())
#         fig2.savefig(f"./Part3_Submission/11subplot_{i + 1}.png", bbox_inches=extent, pad_inches=0.1)
#
#     for i, ax in enumerate(ax3):
#         extent = ax.get_tightbbox(fig3.canvas.get_renderer()).transformed(fig3.dpi_scale_trans.inverted())
#         fig3.savefig(f"./Part3_Submission/12subplot_{i + 1}.png", bbox_inches=extent, pad_inches=0.1)
#
#     for i, ax in enumerate(ax4):
#         extent = ax.get_tightbbox(fig4.canvas.get_renderer()).transformed(fig4.dpi_scale_trans.inverted())
#         fig4.savefig(f"./Part3_Submission/13subplot_{i + 1}.png", bbox_inches=extent, pad_inches=0.1)

#plt.show()


## Use quaternion at each time step to rotate body axes to Inertial Frame 'F'
e_c_1 = np.zeros((len(H_c), 3))
e_c_2 = np.zeros((len(H_c), 3))
e_c_3 = np.zeros((len(H_c), 3))
antenna = np.zeros((len(H_c), 3))



for j in range(0, len(e_c_1)):
    e_c_1[j, :] = rotate_vector_by_quaternion([1,0,0], q_vals[j, :])
    e_c_2[j, :] = rotate_vector_by_quaternion([0,1,0], q_vals[j, :])
    e_c_3[j, :] = rotate_vector_by_quaternion([0,0,1], q_vals[j, :])
    antenna[j,:] = rotate_vector_by_quaternion([0,np.sin(np.radians(5)),np.cos(np.radians(5))], q_vals[j, :])



# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

omega_x = omega_vals[:,0]
omega_y = omega_vals[:,1]
omega_z = omega_vals[:,2]


# Set axis labels
ax.set_xlabel('X Component')
ax.set_ylabel('Y Component')
ax.set_zlabel('Z Component')
ax.set_title('Animated 3D Vector Trajectory')

# Initialize plot elements
line, = ax.plot([], [], [], 'b', label=r'$\omega_0$ trajectory')  # Line for trajectory
point, = ax.plot([], [], [], 'ro')  # Moving point indicator
origin_line, = ax.plot([], [], [], 'r-', linewidth=2)  # Line from origin to point
axis_line, = ax.plot([], [], [], 'r-', linewidth=2)  # Line from origin to point

time_text = ax.text(0, 0, 0, '', fontsize=12, color='black')  # Time text annotation

# Set axis limits
ax.set_xlim([min(omega_x), max(omega_x)])
ax.set_ylim([min(omega_y), max(omega_y)])
ax.set_zlim([0, max(omega_z)])


# Animation function
def update(frame):
    line.set_data(omega_x[:frame], omega_y[:frame])
    line.set_3d_properties(omega_z[:frame])

    # Ensure point takes a sequence (using a list or array with one element)
    point.set_data([omega_x[frame-1]], [omega_y[frame-1]])
    point.set_3d_properties([omega_z[frame-1]])

    # Update line from origin to current point
    origin_line.set_data([0, omega_x[frame-1]], [0, omega_y[frame-1]])
    origin_line.set_3d_properties([0, omega_z[frame-1]])

    # Update time text
    time_text.set_position((1, 0))  # Keep it near origin
    time_text.set_text(f'Time: {t_vals[frame-1]:.2f}s')
    time_text.set_3d_properties(omega_z[frame-1])  # Adjust text height dynamically

    return line, point


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(t_vals), interval=50, blit=False)
axis_line.set_data([0, 0], [0, 0])
axis_line.set_3d_properties([0, max(omega_z)])
plt.legend()
plt.show()




# Create figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

e_c_1_x = e_c_1[:,0]
e_c_1_y = e_c_1[:,1]
e_c_1_z = e_c_1[:,2]

e_c_2_x = e_c_2[:,0]
e_c_2_y = e_c_2[:,1]
e_c_2_z = e_c_2[:,2]

e_c_3_x = e_c_3[:,0]
e_c_3_y = e_c_3[:,1]
e_c_3_z = e_c_3[:,2]

antenna_x = antenna[:,0]
antenna_y = antenna[:,1]
antenna_z = antenna[:,2]  # Set axis labels

ax.set_xlabel('X Component')
ax.set_ylabel('Y Component')
ax.set_zlabel('Z Component')
ax.set_title('Animated 3D Vector Trajectory')

# Initialize plot elements

antenna_traj, = ax.plot([], [], [], 'o', label='antenna trajectory')  # Line for trajectory
antenna_point, = ax.plot([], [], [], 'ro')  # Moving point indicator
e_c_3_traj, = ax.plot([], [], [], 'p', label=r'$e^c_3$ trajectory')  # Line for trajectory
e_c_3_point, = ax.plot([], [], [], 'ro')  # Moving point indicator
axis_line1, = ax.plot([], [], [], 'r-', linewidth=2)  # Line from origin to point
axis_line2, = ax.plot([], [], [], 'r-', linewidth=2)  # Line from origin to point
axis_line3, = ax.plot([], [], [], 'y-', linewidth=2)  # Line from origin to point
nadir_line, = ax.plot([], [], [], 'g-', linewidth=2)  # Line from origin to point
antenna_line, = ax.plot([], [], [], 'b-', linewidth=2)  # Line from origin to point

time_text1 = ax.text(0, 0, 0, '', fontsize=12, color='black')  # Time text annotation

# Set axis limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])


# Animation function
def update(frame):

    antenna_traj.set_data(antenna_x[:frame], antenna_y[:frame])
    antenna_traj.set_3d_properties(antenna_z[:frame])

    e_c_3_traj.set_data(e_c_3_x[:frame], e_c_3_y[:frame])
    e_c_3_traj.set_3d_properties(e_c_3_z[:frame])

    # Ensure point takes a sequence (using a list or array with one element)
    e_c_3_point.set_data([e_c_3_x[frame-1]], [e_c_3_y[frame-1]])
    e_c_3_point.set_3d_properties([e_c_3_z[frame-1]])

    antenna_point.set_data([antenna_x[frame-1]], [antenna_y[frame-1]])
    antenna_point.set_3d_properties([antenna_z[frame-1]])

    # Update line from origin to current point
    axis_line1.set_data([0, e_c_1_x[frame-1]], [0, e_c_1_y[frame-1]])
    axis_line1.set_3d_properties([0, e_c_1_z[frame-1]])

    # Update line from origin to current point
    axis_line2.set_data([0, e_c_2_x[frame-1]], [0, e_c_2_y[frame-1]])
    axis_line2.set_3d_properties([0, e_c_2_z[frame-1]])

    # Update line from origin to current point
    axis_line3.set_data([0, e_c_3_x[frame-1]], [0, e_c_3_y[frame-1]])
    axis_line3.set_3d_properties([0, e_c_3_z[frame-1]])

    antenna_line.set_data([0, antenna_x[frame-1]], [0, antenna_y[frame-1]])
    antenna_line.set_3d_properties([0, antenna_z[frame-1]])

    # Update time text
    time_text.set_position((1, 0))  # Keep it near origin
    time_text.set_text(f'Time: {t_vals[frame-1]:.2f}s')
    time_text.set_3d_properties(omega_z[frame-1])  # Adjust text height dynamically

    return line, point


# Create animation
ani1 = animation.FuncAnimation(fig, update, frames=len(t_vals), interval=50, blit=False)
nadir_line.set_data([0, 0], [0,  np.sin(np.radians(5))])
nadir_line.set_3d_properties([0, np.cos(np.radians(5))])
plt.legend()
plt.show()

# Compute angle between nadir and antenna
antenna_offset = np.zeros(len(t_vals))
for i in range(0, len(t_vals)):
    antenna_offset[i] = np.acos(np.dot(antenna[i,:], [0,np.sin(np.radians(5)),np.cos(np.radians(5))]) /(
                             np.linalg.norm(antenna[i,:]) * np.linalg.norm([0,np.sin(np.radians(5)),np.cos(np.radians(5))])))


fig5, ax5 = plt.subplots(1, 1, figsize=(12,16))
fig5.canvas.manager.set_window_title('Antenna Offset')
ax5.plot(t_vals, np.degrees(antenna_offset))
#ax5.set_ylim(0, 90)
ax5.set_title('Antenna Offset vs. Time')
ax5.legend([r'$\theta^\degree$'])

plt.show()
