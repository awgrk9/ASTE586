## ASTE 586 Computer Project
##      Part 3
##      Problem 2
## Andrew Gerth
## 20250403

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from Computer_Project.Part3.euler_angle_finder import euler_angle_finder
from Computer_Project.Part3.torqueFree_solver import torqueFree_solver
from three_d_omega_plot import *

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


## Define Initial State for three cases
I_1 = [3.0, 3.5, 6.0] # kg * m^2
I_2 = I_1
I_3 = [6.5, 6.0, 3.0] # kg * m^2
w0_1 = [1, 0, 4] # rad/s
w0_2 = [4, 0, 1] # rad/s
w0_3 = [4, 0, 3] # rad/s
q0_1 = [0, 0, 0, 1]
q0_2 = q0_1
q0_3 = q0_1
t_span_1 = (0, 4) # sec
t_span_2 = (0, 16) # sec
t_span_3 = (0, 11) # sec

## Compute State Models
t_vals_1, omega_vals_1, q_vals_1 = torqueFree_solver(I_1, w0_1, q0_1, t_span_1)
t_vals_2, omega_vals_2, q_vals_2 = torqueFree_solver(I_2, w0_2, q0_2, t_span_2)
t_vals_3, omega_vals_3, q_vals_3 = torqueFree_solver(I_3, w0_3, q0_3, t_span_3)

# Plot 1 for All Cases
fig1, ax1 = plt.subplots(3, 1, figsize=(12,16))
fig1.canvas.manager.set_window_title('Plot 1: Components of Angular Velocity in Body System')
plt.subplots_adjust(hspace=0.4)
ax1[0].plot(t_vals_1, omega_vals_1)
ax1[1].plot(t_vals_2, omega_vals_2)
ax1[2].plot(t_vals_3, omega_vals_3)
for i in range(0, len(ax1)):
    ax1[i].legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
    ax1[i].set_title('Angular Velocity in C Coordinate System: Case {}'.format(i+1))
    ax1[i].set_xlabel('time (s)')
    ax1[i].set_ylabel('rad/s')

plt.show()

# Produce 3d plots for each case, showing which axis is encircled by the angular velocity vector
three_d_omegaplot(t_vals_1, omega_vals_1, 1)
three_d_omegaplot(t_vals_2, omega_vals_2, 2)
three_d_omegaplot(t_vals_3, omega_vals_3, 3)

# Compute Euler Angles using method 2, contained in the function 'euler_angle_finder'
theta_1_a, psi_1_a, phi_1_a = euler_angle_finder(I_1, t_vals_1, omega_vals_1, q_vals_1, ref_axis=1)
theta_1_b, psi_1_b, phi_1_b = euler_angle_finder(I_1, t_vals_1, omega_vals_1, q_vals_1, ref_axis=3)
theta_2, psi_2, phi_2 = euler_angle_finder(I_2, t_vals_2, omega_vals_2, q_vals_2, ref_axis=1)
theta_3, psi_3, phi_3 = euler_angle_finder(I_3, t_vals_3, omega_vals_3, q_vals_3, ref_axis=3)

## Plot Euler Angles for Case 1a (Case 1 where ref axis is e_1)
fig2, ax2 = plt.subplots(3, 1, figsize=(12,16))
fig2.canvas.manager.set_window_title(r'Case 1; Reference Axis $\vec{e}^C_1$')
plt.subplots_adjust(hspace=0.4)
ax2[0].plot(t_vals_1, np.degrees(theta_1_a))
ax2[1].plot(t_vals_1, np.degrees(psi_1_a))
ax2[2].plot(t_vals_1, np.degrees(phi_1_a))
ax2[0].set_title(r'Nutation Angle $\theta$; Case 1; Reference Axis $\vec{e}^C_1$')
ax2[1].set_title(r'Precession Angle $\psi$; Case 1; Reference Axis $\vec{e}^C_1$')
ax2[2].set_title(r'Spin Angle $\phi$; Case 1; Reference Axis $\vec{e}^C_1$')

for i in range(0, len(ax2)):
    ax2[i].set_xlabel('time (s)')
    ax2[i].set_ylabel('degrees')


## Plot Euler Angles for Case 1b (Case 1 where ref axis is e_3)
fig3, ax3 = plt.subplots(3, 1, figsize=(12,16))
fig3.canvas.manager.set_window_title(r'Case 1; Reference Axis $\vec{e}^C_3$')
plt.subplots_adjust(hspace=0.4)
ax3[0].plot(t_vals_1, np.degrees(theta_1_b), 'orange')
ax3[1].plot(t_vals_1, np.degrees(psi_1_b), 'orange')
ax3[2].plot(t_vals_1, np.degrees(phi_1_b), 'orange')
ax3[0].set_title(r'Nutation Angle $\theta$; Case 1; Reference Axis $\vec{e}^C_3$')
ax3[1].set_title(r'Precession Angle $\psi$; Case 1; Reference Axis $\vec{e}^C_3$')
ax3[2].set_title(r'Spin Angle $\phi$; Case 1; Reference Axis $\vec{e}^C_3$')

for i in range(0, len(ax2)):
    ax3[i].set_xlabel('time (s)')
    ax3[i].set_ylabel('degrees')



## Plot Euler Angles for Case 2
fig4, ax4 = plt.subplots(3, 1, figsize=(12,16))
fig4.canvas.manager.set_window_title(r'Case 2; Reference Axis $\vec{e}^C_1$')
plt.subplots_adjust(hspace=0.4)
ax4[0].plot(t_vals_2, np.degrees(theta_2), 'green')
ax4[1].plot(t_vals_2, np.degrees(psi_2), 'green')
ax4[2].plot(t_vals_2, np.degrees(phi_2), 'green')
ax4[0].set_title(r'Nutation Angle $\theta$; Case 2; Reference Axis $\vec{e}^C_1$')
ax4[1].set_title(r'Precession Angle $\psi$; Case 2; Reference Axis $\vec{e}^C_1$')
ax4[2].set_title(r'Spin Angle $\phi$; Case 2; Reference Axis $\vec{e}^C_1$')

for i in range(0, len(ax2)):
    ax4[i].set_xlabel('time (s)')
    ax4[i].set_ylabel('degrees')


## Plot Euler Angles for Case 3
fig5, ax5 = plt.subplots(3, 1, figsize=(12,16))
fig5.canvas.manager.set_window_title(r'Case 3; Reference Axis $\vec{e}^C_3$')
plt.subplots_adjust(hspace=0.4)
ax5[0].plot(t_vals_3, np.degrees(theta_3), 'purple')
ax5[1].plot(t_vals_3, np.degrees(psi_3), 'purple')
ax5[2].plot(t_vals_3, np.degrees(phi_3), 'purple')
ax5[0].set_title(r'Nutation Angle $\theta$; Case 3; Reference Axis $\vec{e}^C_3$')
ax5[1].set_title(r'Precession Angle $\psi$; Case 3; Reference Axis $\vec{e}^C_3$')
ax5[2].set_title(r'Spin Angle $\phi$; Case 3; Reference Axis $\vec{e}^C_3$')

for i in range(0, len(ax2)):
    ax5[i].set_xlabel('time (s)')
    ax5[i].set_ylabel('degrees')

plt.show()

#fig1.savefig('Prob2_AngVel.png')
#fig2.savefig('Prob2_Case1a_Euler.png')
#fig3.savefig('Prob2_Case1b_Euler.png')
#fig4.savefig('Prob2_Case2_Euler.png')
#fig5.savefig('Prob2_Case3_Euler.png')


