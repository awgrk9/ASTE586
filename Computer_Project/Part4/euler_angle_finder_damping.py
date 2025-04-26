import numpy as np
import scipy as sp

def euler_angle_finder_damping(I, t_vals, omega_vals, q_vals, I_d, big_omega_vals, ref_axis) :

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

    ## Compute Angular Momentum Components represented in Body Frame
    h1_c = I[0] * omega_vals[:, 0]
    h2_c = I[1] * omega_vals[:, 1] + I_d * big_omega_vals
    h3_c = I[2] * omega_vals[:, 2]
    H_c = np.zeros((len(t_vals), 3))
    H_c[:, 0] = h1_c
    H_c[:, 1] = h2_c
    H_c[:, 2] = h3_c
    # print(H_c[:10, :10])
    ## Use quaternion at each time step to rotate to Inertial Frame 'F'
    H_f = np.zeros((len(H_c), 3))

    for j in range(0, len(H_f)):
        H_f[j, :] = rotate_vector_by_quaternion(H_c[j, :], q_vals[j, :])


    ####################################################################################################
    ### Compute Euler Angles                                                                         ###
    ####################################################################################################
    # recognize which axes are symmetric
    if ref_axis == 3:
        symmetry_axis = 2
        first_axis = 0  # First  equal I_x axis
        second_axis = 1  # Second equal I_x axis
        e_ref = [0, 0, 1]

    elif ref_axis == 1:
        symmetry_axis = 0
        first_axis = 1  # First  equal I_x axis
        second_axis = 2  # Second equal I_x axis
        e_ref = [1, 0, 0]

    elif ref_axis == 2:
        symmetry_axis = 1
        first_axis = 0  # First  equal I_x axis
        second_axis = 2  # Second equal I_x axis
        e_ref = [0, 1, 0]

    theta = np.zeros((len(H_c), 1))
    for k in range(0, len(H_c)):
        theta[k] = np.acos(H_c[k, symmetry_axis] / np.linalg.norm(H_c[k, :]))

    ## Compute Spin Angle
    e_c_node = np.cross(H_c[0, :], e_ref) / np.linalg.norm(np.cross(H_c[0, :], e_ref))
    print('e_C_node: {}'.format(e_c_node))
    e_c_x = np.cross(e_ref, e_c_node)
    print('e_C_x:    {}'.format(e_c_x))
    u = np.zeros((len(t_vals), 3))
    for k in range(0, len(t_vals)):
        u[k, :] = np.cross(H_c[k, :], e_ref) / np.linalg.norm(np.cross(H_c[k, :], e_ref))

    phi = -np.atan2(np.dot(u, e_c_x), np.dot(u, e_c_node))

    ## Compute Precession

    ## Need to get all of the quantities in the inertial frame, F
    u_f = np.zeros((len(t_vals), 3))
    v_f = np.zeros((len(t_vals), 3))
    e_f_ref = np.zeros((len(t_vals), 3))
    for k in range(0, len(t_vals)):
        e_f_ref[k, :] = rotate_vector_by_quaternion(e_ref, q_vals[k, :])
        u_f[k, :] = np.cross(H_f[k, :], e_f_ref[k, :]) / np.linalg.norm(np.cross(H_f[k, :], e_f_ref[k, :]))
        v_f[k, :] = np.cross(H_f[k, :] / np.linalg.norm(H_f[k, :]), u_f[k, :])

    psi = np.atan2(np.dot(u_f, v_f[0, :]), np.dot(u_f, u_f[0, :]))



    return theta, psi, phi, H_f
