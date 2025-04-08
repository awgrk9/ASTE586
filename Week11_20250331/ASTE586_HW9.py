import numpy as np

I = [50, 50, 54] # kg m^2
F = 10 # N
omega = 6 # rad/s
R = 0.6 # m
offset_angle = np.radians(30) # Angular Momentum Change in degrees

del_t = ( I[2]**2 * omega**2 * (1 - np.cos(offset_angle)**2) / (np.sin(offset_angle)**2 * F**2 * R**2) )**0.5
print('Delta_T ^2: {:.3f} s^2, Delta_T: {:.3f} s'.format(del_t**2, del_t))
