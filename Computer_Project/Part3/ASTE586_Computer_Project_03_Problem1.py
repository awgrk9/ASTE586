## ASTE 586 Computer Project
##      Part 3
##      Problem 1
## Andrew Gerth
## 20250401

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from Computer_Project.Part3.torqueFree_solver_symm import torqueFree_solver_symm


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




I = [2.39493, 2.39493, 2.5272]
omega0 = [0, 0.41071176, 4.69445692]  # Initial angular velocity
q0 = [0, 0, 0, 1]   # Initial euler-rodrigues parameters
t_span = (0, 100)  # Integration time range

t_vals, omega_vals, q_vals, H_c, H_f, omega_error, q_unity_numerical, omega_analytical = \
    (torqueFree_solver_symm(I, omega0, q0, t_span))


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
ax.set_xlabel(r'$e^C_1$')
ax.set_ylabel(r'$e^C_2$')
ax.set_zlabel(r'$e^C_3$')
ax.set_title('Body Fixed Reference Frame Motion (Animated)')

# Initialize plot elements
line, = ax.plot([], [], [], 'b', label=r'$\omega$ trajectory')  # Line for trajectory
axis_line, = ax.plot([], [], [], 'g-', linewidth=2)  # Line from origin to point
point, = ax.plot([], [], [], 'bo')  # Moving point indicator
origin_line, = ax.plot([], [], [], 'b-', linewidth=2)  # Line from origin to point
time_text = ax.text(0, 0, 0, '', fontsize=12, color='black')  # Time text annotation

# Set axis limits
ax.set_xlim([min(omega_x), max(omega_x)])
ax.set_ylim([min(omega_y), max(omega_y)])
ax.set_zlim([0, max(omega_z)])


# Animation function
def update(frame):
    index = np.argmin(np.abs(t_vals - frame))  # Find closest index to the time value

    line.set_data(omega_x[:index], omega_y[:index])
    line.set_3d_properties(omega_z[:index])

    # Ensure point takes a sequence (using a list or array with one element)
    point.set_data([omega_x[index-1]], [omega_y[index-1]])
    point.set_3d_properties([omega_z[index-1]])

    # Update line from origin to current point
    origin_line.set_data([0, omega_x[index-1]], [0, omega_y[index-1]])
    origin_line.set_3d_properties([0, omega_z[index-1]])

    # Update time text
    time_text.set_position((1, 0))  # Keep it near origin
    time_text.set_text(f'Time: {t_vals[index-1]:.2f}s')
    time_text.set_3d_properties(omega_z[index-1])  # Adjust text height dynamically

    return line, point

step = 50
indices = np.arange(0, len(t_vals), step)
t_sub = t_vals[indices]

# Create animation
ani = animation.FuncAnimation(fig, update, frames=t_sub, interval=50, blit=False)
axis_line.set_data([0, 0], [0, 0])
axis_line.set_3d_properties([0, max(omega_z)])
plt.legend(['$\omega$', r'$e^c_3$'])
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

ax.set_xlabel(r'$e^F_1$')
ax.set_ylabel(r'$e^F_2$')
ax.set_zlabel(r'$e^F_3$')
ax.set_title('Inertial Ref Frame Motion (Animated)')

# Initialize plot elements

antenna_traj, = ax.plot([], [], [], 'b', label='antenna trajectory')  # Line for trajectory
antenna_point, = ax.plot([], [], [], 'bo')  # Moving point indicator
e_c_3_traj, = ax.plot([], [], [], 'y', label=r'$e^c_3$ trajectory')  # Line for trajectory
e_c_3_point, = ax.plot([], [], [], 'yo')  # Moving point indicator
axis_line1, = ax.plot([], [], [], 'r--', linewidth=2)  # Line from origin to point
axis_line2, = ax.plot([], [], [], 'r-', linewidth=2)  # Line from origin to point
nadir_line, = ax.plot([], [], [], 'g-', linewidth=2)  # Line from origin to point
axis_line3, = ax.plot([], [], [], 'y-', linewidth=2)  # Line from origin to point

antenna_line, = ax.plot([], [], [], 'b-', linewidth=2)  # Line from origin to point

time_text1 = ax.text(0, 0, 0, '', fontsize=12, color='black')  # Time text annotation

# Set axis limits
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])


# Animation function
def update(frame):
    index = np.argmin(np.abs(t_vals - frame))  # Find closest index to the time value

    antenna_traj.set_data(antenna_x[:index], antenna_y[:index])
    antenna_traj.set_3d_properties(antenna_z[:index])

    e_c_3_traj.set_data(e_c_3_x[:index], e_c_3_y[:index])
    e_c_3_traj.set_3d_properties(e_c_3_z[:index])

    # Ensure point takes a sequence (using a list or array with one element)
    e_c_3_point.set_data([e_c_3_x[index-1]], [e_c_3_y[index-1]])
    e_c_3_point.set_3d_properties([e_c_3_z[index-1]])

    antenna_point.set_data([antenna_x[index-1]], [antenna_y[index-1]])
    antenna_point.set_3d_properties([antenna_z[index-1]])

    # Update line from origin to current point
    axis_line1.set_data([0, e_c_1_x[index-1]], [0, e_c_1_y[index-1]])
    axis_line1.set_3d_properties([0, e_c_1_z[index-1]])

    # Update line from origin to current point
    axis_line2.set_data([0, e_c_2_x[index-1]], [0, e_c_2_y[index-1]])
    axis_line2.set_3d_properties([0, e_c_2_z[index-1]])

    # Update line from origin to current point
    axis_line3.set_data([0, e_c_3_x[index-1]], [0, e_c_3_y[index-1]])
    axis_line3.set_3d_properties([0, e_c_3_z[index-1]])

    antenna_line.set_data([0, antenna_x[index-1]], [0, antenna_y[index-1]])
    antenna_line.set_3d_properties([0, antenna_z[index-1]])

    # Update time text
    time_text1.set_position((1, 0))  # Keep it near origin
    time_text1.set_text(f'Time: {t_vals[index-1]:.2f}s')
    time_text1.set_3d_properties(omega_z[index-1])  # Adjust text height dynamically

    return line, point


# Create animation
ani1 = animation.FuncAnimation(fig, update, frames=t_sub, interval=50, blit=False)
nadir_line.set_data([0, 0], [0,  np.sin(np.radians(5))])
nadir_line.set_3d_properties([0, np.cos(np.radians(5))])
plt.legend(['Antenna Pointing', 'Antenna', r'$e^c_3$', r'$e^c_3$', r'$e^c_1$', r'$e^c_2$', 'nadir'])
plt.show()

# Compute angle between nadir and antenna
antenna_offset = np.zeros(len(t_vals))
for i in range(0, len(t_vals)):
    antenna_offset[i] = np.acos(np.dot(antenna[i,:], [0,np.sin(np.radians(5)),np.cos(np.radians(5))]) /(
                             np.linalg.norm(antenna[i,:]) * np.linalg.norm([0,np.sin(np.radians(5)),np.cos(np.radians(5))])))


fig5, ax5 = plt.subplots(1, 1, figsize=(12,6))
fig5.canvas.manager.set_window_title('Antenna Offset')
ax5.plot(t_vals, np.degrees(antenna_offset))
ax5.set_title('Antenna Offset vs. Time')
ax5.legend([r'$\theta^\degree$'])
ax5.set_xlabel('time (sec)')
ax5.set_ylabel('Antenna Offset from Nadir pointing (degrees)')
plt.show()

fig5.savefig('Antenna_Offset.png')

print('The minimum antenna offset from nadir: {:.3f} deg'.format(np.degrees(np.min(antenna_offset))))
print('The maximum antenna offset from nadir: {:.5f} deg'.format(np.degrees(np.max(antenna_offset))))

ani.save('animation_body.gif', writer='pillow', fps=30)
ani1.save('animation_inertial.gif', writer='pillow', fps=10)
