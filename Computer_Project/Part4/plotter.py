import numpy as np
from matplotlib import pyplot as plt


def euler_angle_plotter(theta_vals, psi_vals, phi_vals, t_vals, case, ref_axis):
    ## Plot Euler Angles for Case 1
    fig, ax = plt.subplots(3, 1, figsize=(12,16))
    fig.canvas.manager.set_window_title(r'Case ' + str(case) + '; Reference Axis ' + r'$\vec{e}^C$')
    plt.subplots_adjust(hspace=0.4)
    ax[0].plot(t_vals, np.degrees(theta_vals))
    ax[1].plot(t_vals, np.degrees(psi_vals))
    ax[2].plot(t_vals, np.degrees(phi_vals))
    ax[0].set_title(r'Nutation Angle $\theta$ Case ' + str(case) + r'; Reference Axis $\vec{e}^C$')
    ax[1].set_title(r'Precession Angle $\psi$ Case ' + str(case) + r'; Reference Axis $\vec{e}^C$')
    ax[2].set_title(r'Spin Angle $\phi$ Case ' + str(case) + r'; Reference Axis $\vec{e}^C$')

    for i in range(0, len(ax)):
        ax[i].set_xlabel('time (s)')
        ax[i].set_ylabel('degrees')

    plt.show()

    return
