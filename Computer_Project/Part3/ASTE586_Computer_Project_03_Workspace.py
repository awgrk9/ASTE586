## ASTE 586 Computer Project
##      Part 3 Workspace
## Andrew Gerth
## 20250401

import numpy as np
from numpy import sin as s
from numpy import cos as c
import scipy as sp
from matplotlib import pyplot as plt

gamma = np.radians(5)

omega_0 = 45 * 2 * 60**-1 * np.pi * np.array([0.0, np.sin(gamma), np.cos(gamma)])

I = [1.6975, 1.6975, 2.5272]

theta_0 = np.atan2(np.sqrt( (I[0] * omega_0[0])**2 + (I[1] * omega_0[1])**2 ), I[2] * omega_0[2])
phi_dot = -(I[2] - I[0]) / I[0] * omega_0[2]
psi_dot = -I[2] * phi_dot / ((I[2] - I[0]) * np.cos(theta_0))

print('Omega = {} rad/s'.format(omega_0))
print('Omega = {} deg/s'.format(np.degrees(omega_0)))
print('Nutation Angle: {:.3f} deg'.format(np.degrees(theta_0)))
print('Spin Rate: {:.3f} deg/s'.format(np.degrees(phi_dot)))
print('Precession Rate: {:.3f} deg/s'.format(np.degrees(psi_dot)))

t = np.linspace(0, 1500, 3000)

phi = phi_dot * t
psi = psi_dot * t
theta = np.zeros((len(t)))
theta[:] = theta_0
offset_angle = np.zeros((len(t)))
C_C_A = np.array([[1, 0, 0],
                  [0, c(-(gamma-theta_0)), -s(-(gamma-theta_0))],
                  [0, s(-(gamma-theta_0)), c(-(gamma-theta_0))]])

for i in range(0, len(t)):
    C_F_C = np.array([[-s(phi[i])*s(psi[i])*c(theta[i]) + c(phi[i])*c(psi[i]), -s(phi[i])*c(psi[i]) - s(phi[i])*c(phi[i])*c(theta[i]), s(psi[i])*s(theta[i])],
                      [ s(phi[i])*c(psi[i])*c(theta[i]) + s(psi[i])*c(phi[i]), -s(phi[i])*s(phi[i]) + c(phi[i])*c(psi[i])*c(theta[i]), -s(theta[i])*c(psi[i])],
                      [ s(phi[i])*s(theta[i]), s(theta[i])*c(phi[i]), c(theta[i])]])

    C_F_A = C_F_C @ C_C_A


    offset_angle[i] = np.acos((np.trace(C_F_A) - 1) / 2)

    if i == 0:
        print(C_F_C)
        print(C_F_A)
        print(np.trace(C_F_A))

fig1, ax1 = plt.subplots(1, 1, figsize=(12,16))
ax1.plot(t, np.degrees(offset_angle))

plt.show()

