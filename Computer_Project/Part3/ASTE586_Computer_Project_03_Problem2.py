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
t_span_1 = (0, 20) # sec
t_span_2 = (0, 20) # sec
t_span_3 = (0, 20) # sec

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
    ax1[i].set_title('Angular Velocity in C Coordinate System: Case {}'.format(i))
    ax1[i].set_xlabel('time (s)')
    ax1[i].set_ylabel('rad/s')

plt.show()

three_d_omegaplot(t_vals_1, omega_vals_1, 1)
three_d_omegaplot(t_vals_2, omega_vals_2, 2)
three_d_omegaplot(t_vals_3, omega_vals_3, 3)

