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
    ax[0].set_title(r'Nutation Angle $\theta$ Case ' + str(case) + r'; Reference Axis $\vec{e}^C$' + str(ref_axis))
    ax[1].set_title(r'Precession Angle $\psi$ Case ' + str(case) + r'; Reference Axis $\vec{e}^C$' + str(ref_axis))
    ax[2].set_title(r'Spin Angle $\phi$ Case ' + str(case) + r'; Reference Axis $\vec{e}^C$' + str(ref_axis))

    for i in range(0, len(ax)):
        ax[i].set_xlabel('time (s)')
        ax[i].set_ylabel('degrees')

    plt.show()

    return

def angular_momentum_error_plotter(t, ang, case):
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    fig.canvas.manager.set_window_title(r'Case ' + str(case))
    ax.plot(t, ang-ang[0, :])
    ax.set_title('Angular Momentum difference from initial Angular Momentum\nRepresented in the Inertial Frame, F\n'
                 + r'Case ' + str(case))
    ax.legend([r'$h_1(t)-h_1(0) $', r'$h_2(t)-h_2(0)$', r'$h_3(t)-h_3(0)$'])
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'$kg*m^2/s$')

    plt.show()

    return

def energy_plotter(t, energy, case):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.canvas.manager.set_window_title(r'Case ' + str(case))
    ax.plot(t, energy)
    ax.set_title('Kinetic Energy\n'
                 + r'Case ' + str(case))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('energy (J)')

    plt.show()

    return

def damper_pos_plotter(t, damper_pos, case):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.canvas.manager.set_window_title(r'Case ' + str(case))
    ax.plot(t, damper_pos)
    ax.set_title('Damper Position\n'
                 + r'Case ' + str(case))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('angular position (rad)')

    plt.show()

    return
