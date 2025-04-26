import numpy as np

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

def three_d_omegaplot(t_vals_1, omega_vals_1, case, show, save):

    ## Create figure and 3D axis
    fig2_1 = plt.figure()
    ax2_1 = fig2_1.add_subplot(111, projection='3d')

    omega_x_1 = omega_vals_1[:,0]
    omega_y_1 = omega_vals_1[:,1]
    omega_z_1 = omega_vals_1[:,2]


    # Set axis labels
    ax2_1.set_xlabel(r'$e^C_1$')
    ax2_1.set_ylabel(r'$e^C_2$')
    ax2_1.set_zlabel(r'$e^C_3$')
    ax2_1.set_title('Body Fixed Reference Frame Motion, Case {} (Animated)'.format(case))

    # Initialize plot elements
    line_1,         = ax2_1.plot([], [], [], 'b', label=r'$\omega$ trajectory')  # Line for trajectory
    axis_line1,     = ax2_1.plot([], [], [], 'r-', linewidth=2)  # Line from origin to point
    axis_line2,     = ax2_1.plot([], [], [], 'g-', linewidth=2)  # Line from origin to point
    axis_line3,     = ax2_1.plot([], [], [], 'y-', linewidth=2)  # Line from origin to point
    point_1,        = ax2_1.plot([], [], [], 'bo')  # Moving point indicator
    origin_line_1,  = ax2_1.plot([], [], [], 'b--', linewidth=2)  # Line from origin to point
    time_text_1     = ax2_1.text(0, 0, 0, '', fontsize=12, color='black')  # Time text annotation

    # Set axis limits
    if min(omega_x_1) > 0:
        ax2_1.set_xlim([0, max(omega_x_1)])
    else:
        ax2_1.set_xlim([min(omega_x_1), max(omega_x_1)])

    if min(omega_y_1) > 0:
        ax2_1.set_ylim([0, max(omega_y_1)])
    else:
        ax2_1.set_ylim([min(omega_y_1), max(omega_y_1)])

    if min(omega_z_1) > 0:
        ax2_1.set_zlim([0, max(omega_z_1)])
    else:
        ax2_1.set_zlim([min(omega_z_1), max(omega_z_1)])

    def update_1(frame):
        # Find the index corresponding to the current time 'frame'
        index = np.argmin(np.abs(t_vals_1 - frame))  # Find closest index to the time value

        # Now use 'index' to update your data
        line_1.set_data(omega_x_1[:index], omega_y_1[:index])
        line_1.set_3d_properties(omega_z_1[:index])

        point_1.set_data([omega_x_1[index]], [omega_y_1[index]])
        point_1.set_3d_properties([omega_z_1[index]])

        origin_line_1.set_data([0, omega_x_1[index]], [0, omega_y_1[index]])
        origin_line_1.set_3d_properties([0, omega_z_1[index]])

        # Update time text
        time_text_1.set_position((1, 0))
        time_text_1.set_text(f'Time: {frame:.2f}s')
        time_text_1.set_3d_properties(omega_z_1[index])

        return line_1, point_1

    step = 20
    indices = np.arange(0, len(t_vals_1), step)
    t_sub = t_vals_1[indices]

    print(type(t_vals_1))
    print(np.shape(t_vals_1))
    # Create animation
    ani_1 = animation.FuncAnimation(fig2_1, update_1, frames=t_sub, interval=50, blit=False)
    axis_line1.set_data([0, max(omega_x_1)], [0, 0])
    axis_line1.set_3d_properties([0, 0])
    axis_line2.set_data([0, 0], [0, max(omega_y_1)])
    axis_line2.set_3d_properties([0, 0])
    axis_line3.set_data([0, 0], [0, 0])
    axis_line3.set_3d_properties([0, max(omega_z_1)])
    plt.legend([r'$\omega$', r'$e^c_1$', r'$e^c_2$', r'$e^c_3$'])

    if show is True:
        plt.show()

    if save is True:
        ani_1.save('case' + str(case) +'_animation.gif', writer='pillow', fps=30)

    plt.close()

    return