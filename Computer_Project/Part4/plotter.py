import numpy as np
from matplotlib import pyplot as plt


def euler_angle_plotter(theta_vals, psi_vals, phi_vals, t_vals, case, ref_axis, show, save):
    ## Plot Euler Angles for Case 1
    fig, ax = plt.subplots(3, 1, figsize=(12,10))
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

    if show is True:
        plt.show()

    if save is True:
        fig.savefig('EulerAngle_Plot_Case_' + str(case) + '.png')

    plt.close()

    return

def angular_velocity_plotter(t, w, case, show, save):

    fig1, ax1 = plt.subplots(1, 1, figsize=(12,5))
    fig1.canvas.manager.set_window_title('Components of Angular Velocity in Body System')
    ax1.plot(t, w)
    ax1.legend([r'$\omega_1$', r'$\omega_2$', r'$\omega_3$'])
    ax1.set_title('Angular Velocity in C Coordinate System: Case {}'.format(case))
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('rad/s')

    if show is True:
        plt.show()

    if save is True:
        fig1.savefig('AngularVelocity_Plot_Case_' + str(case) + '.png')

    plt.close()

    return

def angular_momentum_error_plotter(t, ang, case, show, save):
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    fig.canvas.manager.set_window_title(r'Case ' + str(case))
    ax.plot(t, ang-ang[0, :])
    ax.set_title('Angular Momentum difference from initial Angular Momentum\nRepresented in the Inertial Frame, F\n'
                 + r'Case ' + str(case))
    ax.legend([r'$h_1(t)-h_1(0) $', r'$h_2(t)-h_2(0)$', r'$h_3(t)-h_3(0)$'])
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'$kg*m^2/s$')

    if show is True:
        plt.show()

    if save is True:
        fig.savefig('AngularMomentumError_Plot_Case_' + str(case) + '.png')

    plt.close()

    return

def energy_plotter(t, energy, case, show, save):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.canvas.manager.set_window_title(r'Case ' + str(case))
    ax.plot(t, energy)
    ax.set_title('Kinetic Energy\n'
                 + r'Case ' + str(case))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('energy (J)')

    if show is True:
        plt.show()

    if save is True:
        fig.savefig('Energy_Plot_Case_' + str(case) + '.png')

    plt.close()

    return

def damper_pos_plotter(t, damper_pos, case, show, save):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig.canvas.manager.set_window_title(r'Case ' + str(case))
    ax.plot(t, damper_pos)
    ax.set_title('Damper Position\n'
                 + r'Case ' + str(case))
    ax.set_xlabel('time (s)')
    ax.set_ylabel('angular position (rad)')

    if show is True:
        plt.show()

    if save is True:
        fig.savefig('DamperPos_Plot_Case_' + str(case) + '.png')

    plt.close()

    return
