## ASTE 586 Computer Project
##      Part 4
##
## Andrew Gerth
## 20250408

import numpy as np

from Computer_Project.Part4.euler_angle_finder_damping import euler_angle_finder_damping
from Computer_Project.Part4.torqueFree_solver_damping import torqueFree_solver_damping
from three_d_omega_plot import *
from plotter import *

show_all_plots = False # True to show all plots in the interactive viewer as they are generated
save_all_plots = True # True to save plots to .png files in the root directory


## Define Initial State for three cases
I_1 = [5.0, 4.0, 5.0] # kg * m^2
I_2 = [300, 300, 1200] # kg * m^2
w0_1 = [0.0181844, 0.128911, 0] # rad/s
w0_2 = [14, 0, 50] # rad/s
q0_1 = [0, 0, 0.0871557, 0.996195]
q0_2 = [0, 0, 0, 1]
damper_props_1 = [0, 0, 1*10**(-4), 0, 0]   # [theta_d_0 (rad), big_omega_d_0 (rad/s), I_d(kg*m^2), D(Nm/(rad/s)), K (Nm/rad)
damper_props_2 = [0, 0, 0.5, 120, 1]        # [theta_d_0 (rad), big_omega_d_0 (rad/s), I_d(kg*m^2), D(Nm/(rad/s)), K (Nm/rad)

# Case 3 -- Section 4.1 Case (Explorer I)
I_3 = [4.388, 4.388, 0.0585] # kg * m^2
w0_3 = [0.0419, 0, 60] # rad/s
q0_3 = [0, 0, 0, 1]
damper_props_3 = [0, 0, 0.1, 1.5, 0.064]        # [theta_d_0 (rad), big_omega_d_0 (rad/s), I_d(kg*m^2), D(Nm/(rad/s)), K (Nm/rad)

# Case 4 -- Section 4.2 Case (Non-Symmetric)
I_4 = [45, 60, 31] # kg * m^2
w0_4 = [0, 40, 70] # rad/s
q0_4 = [0, 0, 0, 1]
damper_props_4 = [0, 0, 0.5, 540, 20]        # [theta_d_0 (rad), big_omega_d_0 (rad/s), I_d(kg*m^2), D(Nm/(rad/s)), K (Nm/rad)

# Case 5 -- Section 4.3 Case (Improvements on Intelsat)
I_5 = [300, 300, 1200] # kg * m^2
w0_5_3 = 40.85
w0_5_1 = (50**2 - w0_5_3**2)**0.5 # ensure magnitude of w0_5 == 50 rad/s
w0_5 = [w0_5_1, 0, w0_5_3] # rad/s
print('Initial angular velocity for improved Intelsat Case: {}'.format(w0_5))
q0_5 = [0, 0, 0, 1]
damper_props_5 = [0, 0, 2.34, 60, 1500]       # [theta_d_0 (rad), big_omega_d_0 (rad/s), I_d(kg*m^2), D(Nm/(rad/s)), K (Nm/rad)

# Case 6 -- Section 4.3 Case (Improvements on Intelsat)
I_6 = [300, 300, 1200] # kg * m^2
w0_6_3 = 40.85
w0_6_1 = (50**2 - w0_5_3**2)**0.5 # ensure magnitude of w0_5 == 50 rad/s
w0_6 = [w0_6_1, 0, w0_6_3] # rad/s
print('Initial angular velocity for improved Intelsat Case: {}'.format(w0_5))
q0_6 = [0, 0, 0, 1]
damper_props_6 = [0, 0, 3.5, 60, 1500]       # [theta_d_0 (rad), big_omega_d_0 (rad/s), I_d(kg*m^2), D(Nm/(rad/s)), K (Nm/rad)


t_span_1 = (0, 5) # sec
t_span_2 = (0, 5) # sec
#t_span_3 = (0, 7200) # sec
t_span_3 = (0, 1) # sec
t_span_4 = (0, 1) # sec
t_span_5 = (0, 5) # sec
t_span_6 = (0, 5) # sec


t_span = [t_span_1, t_span_2, t_span_3, t_span_4, t_span_5, t_span_6]
I = [I_1, I_2, I_3, I_4, I_5, I_6]
w_0 = [w0_1, w0_2, w0_3, w0_4, w0_5, w0_6]
q_0 = [q0_1, q0_2, q0_3, q0_4, q0_5, q0_6]

damper_props = [damper_props_1, damper_props_2, damper_props_3, damper_props_4, damper_props_5, damper_props_6]


## Compute State Models
# Initialize list of variables to contain each case.
t_vals      = [0] * len(I)
omega_vals  = [0] * len(I)
q_vals      = [0] * len(I)
theta_vals  = [0] * len(I)
big_omega_vals = [0] * len(I)
T_vals      = [0] * len(I)


for i in range(0, len(I)):
    t_vals[i], omega_vals[i], q_vals[i], theta_vals[i], big_omega_vals[i], T_vals[i] = (
        torqueFree_solver_damping(I[i], w_0[i], q_0[i], damper_props[i][0], damper_props[i][1],
                                  t_span[i], damper_props[i][2], damper_props[i][3], damper_props[i][4]))




# Initialize empty lists for theta, psi, phi, and angular momentum for each case
theta = [0] * len(t_vals)
psi = [0] * len(t_vals)
phi = [0] * len(t_vals)
H_f = [0] * len(t_vals)

for i in range(0, len(t_vals)):

    # Introduce separate variable for 3d plot animation rendering, as doing it all for
    #   this lengthy of a simulation can take a while...
    if i == 4: # i = desired_case - 1
        threed_save = True
    else:
        threed_save = False



    # Produce 3d plots for each case, showing which axis is encircled by the angular velocity vector
    three_d_omegaplot(t_vals[i], omega_vals[i], case=i+1, show=show_all_plots, save=threed_save)

    if i == 0:
        ref = 2
    else:
        ref = 3

    # Angular Velocity in Body Components Plot
    angular_velocity_plotter(t_vals[i], omega_vals[i], case=i+1, show=show_all_plots, save=save_all_plots)

    # Compute Euler Angles using method 2, contained in the function 'euler_angle_finder'
    theta[i], psi[i], phi[i], H_f[i] = euler_angle_finder_damping(
        I[i], t_vals[i], omega_vals[i], q_vals[i], damper_props[i][2], big_omega_vals[i], ref_axis=ref)






    # Plot Euler Angles
    euler_angle_plotter(theta[i][:], psi[i][:], phi[i][:], t_vals[i], case=i+1, ref_axis=ref,
                        show=show_all_plots, save=save_all_plots)

    if i == 4 or i == 5:
        print('Final Nutation Angle: {} deg'.format(np.degrees(theta[i][-1])))


    ## Plot Angular Momentum Vectors in Inertial Frame (should be zero)
    angular_momentum_error_plotter(t_vals[i], H_f[i], case=i+1, show=show_all_plots, save=save_all_plots)

    ## Plot Kinetic Energy
    energy_plotter(t_vals[i], T_vals[i], case=i+1, show=show_all_plots, save=save_all_plots)

    ## Plot Damper Position
    damper_pos_plotter(t_vals[i], theta_vals[i], case=i+1, show=show_all_plots, save=save_all_plots)




