## ASTE 586 HW 10
##     Problem 2
## Andrew Gerth
## 20250413

import numpy as np
import sympy as sym


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

R_E = 6378.15 # km

q_c_o = [0.008628, 0.086293, 0.008628, -0.996195] # Conjugate of given quaternion (scalar last)
s_b = [0.00944, 0.00230, 0.9995] # given in body coordinate frame

s_o = rotate_vector_by_quaternion(s_b, q_c_o) # rotate into orbital frame
print('Sensor pointing Vector in Orbital Frame: {}'.format(s_o))

C_o_ecef = np.array([[0, 0, 1],
                     [0, 1, 0],
                     [-1, 0, 0]])
s_ecef = s_o @ C_o_ecef

print('Sensor pointing Vector in ECEF Frame: {}'.format(s_ecef))


t = sym.symbols('t')
r_sv = [R_E + 500, 0, 0] #km in ECEF
r_s = r_sv + s_ecef * t
print('r_s function of t:')
sym.pprint(r_s)
r_s_mat = sym.Matrix(r_s)
solved = sym.solve(sym.sqrt(r_s_mat.dot(r_s_mat)) - R_E, t) # no easy magnitude function in sympy, use sqrt of dotting by itself

print('roots of t')
sym.pprint(solved)

r_target = r_s_mat.subs(t, solved[0])
print('r_target:')
sym.pprint(r_target)

r_targetx = [0, 0, 0]
r_targetx[0], r_targetx[1], r_targetx[2] = float(r_target[0]), float(r_target[1]), float(r_target[2])

lat = np.asin(r_targetx[2]/R_E)
long = np.atan2(r_targetx[1], r_targetx[0])

print('Lat : {:.5f} degrees'.format(np.degrees(lat)))
print('Long: {:.5f} degrees'.format(np.degrees(long)))
